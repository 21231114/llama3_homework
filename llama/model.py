# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import flash_attn
import math
import time
from dataclasses import dataclass
from flash_attn.flash_attn_interface import flash_attn_unpadded_func
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_split_func
from typing import Optional, Tuple
import sys
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 3000


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)#对一个对象的每一个特征求均方根

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)#保持数据类型相同


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_seq_len = args.max_seq_len
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        flashattention: bool = False
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)#保证数据类型和设备与xq一致
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk#已经存过的kv,就不变了，因为前面的序列不受后面的序列影响
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        ###     数据准备结束  
        output = None
        if(flashattention):
            causal = (mask is not None)   
            # sum = 0 
            # cumulative_q = torch.tensor([0]+[sum := sum + xq[i].shape[0] for i in  range(0,bsz)]).to(torch.int32)
            # sum = 0
            # cumulative_k = torch.tensor([0]+[sum := sum + keys[i].shape[0] for i in  range(0,bsz)]).to(torch.int32)
            # xq = xq.view(-1,self.n_local_heads,self.head_dim)
            # keys = keys.view(-1,self.n_local_heads,self.head_dim)
            # values = values.view(-1,self.n_local_heads,self.head_dim)
            # output = flash_attn_unpadded_func(xq,keys,values,cumulative_q,cumulative_k,cumulative_q[1],cumulative_k[1],0.0,causal=causal)
            # output = torch.stack(torch.chunk(output,seqlen)).view(bsz,seqlen,-1) 
           
            output = flash_attn_unpadded_func(xq[0],keys[0],values[0],torch.tensor([0,xq[0].shape[0]]).to(torch.int32),torch.tensor([0,keys[0].shape[0]]).to(torch.int32),xq[0].shape[0],keys[0].shape[0],0.0,causal=causal)
            return self.wo( output.view(bsz,seqlen,-1)) 
            # Q_BLOCK_SIZE = 1
            # KV_BLOCK_SIZE = 1
            # Q_LEN = xq.shape[2]
            # K_LEN = keys.shape[2]
            # Tr = Q_LEN // Q_BLOCK_SIZE
            # Tc = K_LEN // KV_BLOCK_SIZE
            # O = torch.zeros_like(xq).type_as(xq)
            # l = torch.zeros(xq.shape[:-1])[..., None].type_as(xq)
            # m = (torch.ones(xq.shape[:-1])[..., None] * float("-inf")).type_as(xq)
            # Q_BLOCKS = torch.split(xq, Q_BLOCK_SIZE, dim=2)
            # K_BLOCKS = torch.split(keys, KV_BLOCK_SIZE, dim=2)
            # V_BLOCKS = torch.split(values, KV_BLOCK_SIZE, dim=2)
            # O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
            # l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
            # m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))
            # MASK_BLOCKS = None
            # for i in range(0,Tr):
            #     Qi = Q_BLOCKS[i]
            #     Oi = O_BLOCKS[i]
            #     li = l_BLOCKS[i]
            #     mi = m_BLOCKS[i]
            #     # S_i = torch.zeros(bsz, self.n_local_heads,Q_BLOCK_SIZE,K_LEN//KV_BLOCK_SIZE).type_as(xq)
            #     for j in range(0,Tc):
            #         Kj = K_BLOCKS[j]
            #         Vj = V_BLOCKS[j]
            #         S_ij = torch.matmul(Qi, Kj.transpose(2, 3)) / math.sqrt(self.head_dim)
            #         if seqlen > 1:#即有mask,先按默认块的宽度是1写着，后面有需要再改
            #             if(j-(start_pos)>i):
            #                 S_ij += -1e-10
            #         m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
            #         P_ij = torch.exp(S_ij - m_block_ij)
            #         # if((i==0 and j == 1) or (i==0 and j==0)):
            #         #      print("{},{}".format(i,j))
            #         #      print(P_ij)
            #         #      print(P_ij.shape)
            #         #      print(P_ij.dtype)
            #         #      print("------------------------")
            #         #P_ij计算的都是当前块内的
            #         l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True)
            #     #   P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)        
            #         mi_new = torch.maximum(m_block_ij, mi)
            #         li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij  
            #         Oi = (li / li_new) * torch.exp(mi - mi_new) * Oi \
            #             + (torch.exp(m_block_ij - mi_new) / li_new) * torch.matmul(P_ij, Vj) #结合下一个块的K,更新当前Q的O
            #         mi = mi_new
            #         li = li_new 
            #     O_BLOCKS[i] = Oi
            # #print(O_BLOCKS)
            # # print("--------------666----------------")
            # # sys.exit(0)
            # output = torch.stack(O_BLOCKS)
        else:
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            values = values.transpose(
            1, 2
            )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)#contigous返回具有连续内存的张量
            return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]#start_pos之前的都是已经计算过kv的
            ).type_as(h)#纵向是seqlen,对应新进来的tokens,计算（其之前）所有token对其的贡献score

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output
