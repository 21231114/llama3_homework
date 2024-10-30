# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 3000,
    max_gen_len: int = 1000,
    max_batch_size: int = 4,
):
    """
    Examples to run with the pre-trained models (no fine-tuning). Prompts are
    usually in the form of an incomplete text prefix that the model can then try to complete.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.
    `max_gen_len` is needed because pre-trained models usually do not stop completions naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts: List[str] = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        #  "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that ",
        # """A brief message congratulating the team on the launch:

        # Hi everyone,

        # I just """,
        # # Few shot prompt (providing a few examples before asking model to complete more);
        # """Translate English to French:

        # sea otter => loutre de mer
        # peppermint => menthe poivrée
        # plush girafe => girafe peluche
        # cheese =>""",
        #"""Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [18] and conditional computation [26], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains. Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 16]. In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network. In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs. 2 Background The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [11]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2. Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19]. End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28]. To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8]. 3 Model Architecture Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 29]. Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [9], consuming the previously generated symbols as additional input when generating the next. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 3.1 Encoder and Decoder Stacks Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position2 Figure 1: The Transformer - model architecture. wise fully connected feed-forward network. We employ a residual connection [10] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension dmodel = 512. Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i. 3.2 Attention An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key. 3.2.1 Scaled Dot-Product Attention We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the""",
#     '''The History of Artificial Intelligence
# Artificial Intelligence (AI) has a rich and complex history that spans several decades, marked by significant milestones, breakthroughs, and challenges. This journey reflects humanity’s enduring quest to create machines that can think, learn, and perform tasks typically requiring human intelligence. Below is an exploration of the key developments in the history of AI.

# Early Foundations (1940s - 1950s)
# The concept of artificial intelligence can be traced back to ancient myths and philosophical discussions about automatons and intelligent beings. However, the formal foundation for AI was laid in the mid-20th century with the advent of digital computers.

# 1. Turing and the Universal Machine
# In 1936, Alan Turing introduced the concept of a "universal machine," which could simulate any algorithmic process. His work laid the groundwork for the development of modern computers. In 1950, Turing published "Computing Machinery and Intelligence," where he posed the famous question, "Can machines think?" He also proposed the Turing Test, a method for evaluating a machine's ability to exhibit intelligent behavior indistinguishable from a human.

# 2. The Dartmouth Conference
# The term "artificial intelligence" was coined in 1956 at the Dartmouth Conference, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and Claude Shannon. This conference is widely regarded as the birth of AI as a field of study. Researchers aimed to develop machines that could perform tasks requiring human-like cognitive functions.

# The Formative Years (1950s - 1970s)
# The initial excitement around AI led to several pioneering projects and the development of early AI programs.

# 3. Symbolic AI and Early Programs
# In the 1950s and 1960s, much of AI research focused on symbolic AI, where researchers created programs that manipulated symbols to solve problems. Notable examples include:

# Logic Theorist (1955) by Allen Newell and Herbert A. Simon, which could prove mathematical theorems.
# General Problem Solver (1957), another Newell and Simon creation aimed at solving a wide range of problems.
# 4. Neural Networks and Perceptrons
# In 1958, Frank Rosenblatt developed the Perceptron, an early neural network model capable of pattern recognition. Although initially promising, the limitations of perceptrons became apparent, leading to a decline in interest in neural networks.

# The AI Winter (1970s - 1980s)
# Despite early successes, the field faced significant challenges, leading to periods of reduced funding and interest known as "AI winters."

# 5. Challenges and Criticism
# In the 1970s, AI faced criticism for its inability to solve complex real-world problems and for the limitations of symbolic AI. The high expectations set during the Dartmouth Conference were not met, leading to disillusionment among funders and researchers.

# 6. The Rise of Expert Systems
# Despite the setbacks, the late 1970s saw the emergence of expert systems—AI programs designed to mimic human expertise in specific domains. One of the most notable was MYCIN, developed in the early 1970s for diagnosing bacterial infections. Expert systems gained popularity in business applications, leading to renewed interest in AI.

# The Revival and Expansion (1980s - 1990s)
# The 1980s marked a resurgence in AI research, fueled by advances in computer hardware and software.

# 7. Machine Learning and Statistical Methods
# Researchers began exploring machine learning approaches, shifting from rule-based systems to data-driven methods. This period saw the development of algorithms that could learn from data, paving the way for modern AI.

# 8. Robotics and Computer Vision
# During this time, AI also made strides in robotics and computer vision. Notable projects included the development of autonomous robots and early image recognition systems. The integration of AI in various fields demonstrated its potential beyond theoretical applications.

# The Modern Era (2000s - Present)
# The 21st century has witnessed unprecedented growth in AI, driven by advancements in computing power, data availability, and algorithmic innovations.

# 9. Big Data and Deep Learning
# The explosion of data from the internet and the increasing power of GPUs facilitated the rise of deep learning, a subset of machine learning that uses neural networks with many layers. Breakthroughs such as AlexNet in 2012, which won the ImageNet competition, showcased the potential of deep learning in image recognition tasks.

# 10. Natural Language Processing (NLP)
# Natural Language Processing has also advanced significantly, with models like Google's BERT and OpenAI's GPT series demonstrating remarkable capabilities in understanding and generating human language. These technologies have transformed applications in chatbots, translation services, and content generation.

# 11. AI in Everyday Life
# Today, AI has permeated various aspects of daily life, including virtual assistants (like Siri and Alexa), recommendation systems (Netflix and Amazon), and autonomous vehicles. Businesses leverage AI for data analysis, customer service, and operational efficiencies.

# Ethical Considerations and Future Directions
# As AI continues to evolve, ethical considerations regarding its impact on society, jobs, privacy, and decision-making are increasingly critical. Discussions surrounding bias in AI systems, the potential for job displacement, and the need for regulation are central to the discourse on AI's future.

# 12. The Challenge of Bias
# AI systems often reflect the biases present in their training data, leading to concerns about fairness and discrimination. Addressing these biases is essential for building trust in AI technologies.

# 13. Regulation and Governance
# The rapid advancement of AI has prompted calls for regulatory frameworks to ensure its safe and ethical use. Governments and organizations worldwide are exploring guidelines to manage the development and deployment of AI technologies.

# Conclusion
# The history of artificial intelligence is a testament to human ingenuity and the relentless pursuit of knowledge. From its theoretical beginnings to its current applications, AI has''',
 '''1. Goals and ideas of doctoral learning
(1) Course objectives
We will pay attention to the cutting-edge theory of deep learning research, and at the same time, we will continue to pay more attention to the relevant knowledge of computer engineering. According to the characteristics of the discipline, we will focus on mastering the theories and methods of mathematical optimization in software engineering.
(2) Scientific research objectives
Focus on the deep learning at home and abroad and its application in the field of software engineering, improve the English writing ability, and summarize the suitable research methods. Through extensive reading of domestic and foreign literature, I will deepen my academic foundation, expand my academic vision, and find potential research gaps and deficiencies.
2. The subject direction of the proposed research project
Another interesting research direction for students is about using deep learning to solve combinatorial optimization problems.
(1) Research background and significance
Combination optimization has important applications in many fields, such as vehicle path planning and cargo loading in logistics and transportation, task arrangement of factory production lines in production scheduling, portfolio selection in finance, risk management, and resource allocation and layout of communication network in network design, etc. These real-life problems can all be modeled as combinatorial optimization models. At the same time, with the development of deep learning technology, it can extract valuable information from massive data. These techniques include cluster analysis for identifying natural groupings in data, labeling of data, classification algorithms to help decision making and association rule mining to find the relationship between data items.
The foothold of this topic is to find the modeling and application of deep learning in solving combinatorial optimization problems. Traditional combinatorial optimization methods, such as integer linear programming and heuristic algorithms, often face the problem of high computational complexity. However, deep learning can reduce the problem size and improve the solution speed through feature selection and dimension reduction, and use prediction models to predict future trends with historical data and assist decision-making. At the same time, it can solve the complexity and uncertainty problems. In real-time data analysis and multi-objective optimization, it can provide real-time decision support in a dynamic environment and consider multiple goals to help find better solutions.
Therefore, using deep learning to solve the combinatorial optimization problems can not only improve the solution efficiency and reduce the computing cost, but also provide more accurate decision support in practical applications. The in-depth exploration of this research field will bring an important impetus to the development of various industries.
(2) Research status at home and abroad
Combinatorial optimization involves selecting the optimal decision variable in a discrete decision space, with a natural similarity to the action selection process in reinforcement learning. At the same time, the characteristics of offline training and online decision-making available in deep reinforcement learning make the real-time online solution of combinatorial optimization problems become a reality. Therefore, deep reinforcement learning is often used at home and abroad when using deep learning to solve combinatorial optimization problems. The combinatorial optimization algorithm based on deep reinforcement learning shows significant advantages such as fast solution and strong generalization ability, which makes it more efficient and flexible in dealing with complex problems.
At present, the combinatorial optimization methods based on deep reinforcement learning are mainly divided into two categories: end-to-end algorithm and local search improvement algorithm. End-to-end algorithm mainly includes end-to-end method based on Pointer Network and end-to-end method based on graph neural network.
Pointer Network Based on the seq2seq model, the encoder is used to obtain the feature vector of the input sequence, and then the decoder is used to combine the attention mechanism to get the pointer to the input queue. In the current study, the model is usually trained in the way of reinforcement learning. Figure neural network is used to process graph data structure, by calculating the feature vector of each node node prediction, edge prediction, combined with reinforcement learning through the feature vector of each node further operation node Q value, choose the node expected long-term returns, and according to the Q value to train graph neural network model. The core of the improved local search method of deep reinforcement learning still belongs to heuristic search algorithms. Unlike traditional methods, it does not rely on manually designed search strategies, but autonomously learns these strategies through deep reinforcement learning algorithms. This method has powerful optimization capability and can go beyond traditional optimization algorithms in some cases. However, its solution time is significantly longer than the end-to-end model. Therefore, decision makers need to trade off between optimization effect and solution speed when choosing different methods.
This topic is based on Pointer Network, which is the theoretical basis of this research. Pointer Network based on Attention mechanism is the main method to solve the combinatorial optimization problems with sequence decision characteristics, such as TSP, VRP, etc. This topic focuses on how to improve the neural network model structure of encoders and decoders to improve the performance of models on combinatorial optimization problems.
(3) The main ideas to carry out the subject research
1. [] Theoretical basis and literature review
Definition of combination optimization problem: clarify the research objects, such as travel agent problems, backpack problems, etc.
Existing model analysis: Review the application of existing encoder-decoder model in combinatorial optimization and identify its advantages and disadvantages.
2. [] Model structure improvement
Encoder design: Introduce deeper layers or residual connections to enhance the feature extraction capability. Use attentional mechanisms, such as using self-attention or cross-attention, to focus on key input features.
Decoder design: adopt multi-head attention mechanism to improve the flexibility and accuracy of the generation process. Design a hierarchical decoding structure and generate solutions step by step to improve efficiency and feasibility.
3. [] Training strategy optimization
Loss function improvement: Design to optimize the specific loss function for the combination, considering the effectiveness and feasibility of the solution.
Data enhancement: Generate high-quality training samples using adversarial generative network (GAN) or reinforcement learning methods.
4.  Evaluation and experiment
Index setting: determine the indicators to evaluate the performance of the model, such as the quality of the solution, the convergence rate, etc.
Experimental design: Conduct extensive experiments on different combination optimization problems, compare the model performance before and after the improvement, and compare with the commonly used heuristic algorithm.
5. Results analysis and discussion
Performance comparison: the analysis of the new model and the traditional model'''
    ]
    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt)
        print(f"> {result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
