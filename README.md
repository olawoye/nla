# **KTH PhD Task 2026**





### **Introduction and Motivation**



LLMs are powerful, and the world is increasingly learning of its potential and promise. However, while, there are tools that target and improve its output or outcome, most of the internal thinking is a bit of a black box. LLM activations, a representation of the “internal mental state” of the LLM regarding a task, at any one time, is a high-dimensional vector. These opaque set of numbers are clearly difficult to understand for humans. In the last few years, efforts in mechanistic interpretability are trying to understand (or reverse-engineer) the internal workings of these powerful LLM models. Such efforts include breakthroughs such as Anthropic’s Sparse Autoencoders (SAEs) and Natural Language Autoencoders (NLAs). NLAs, the focus of this task, is an unsupervised way of generating natural language “explanations” of these activations earlier mentioned. Imagine the potential gains from the application of this, especially reduction of hallucinations from LLMs because if we can understand the internal workings of an LLM, then we can directly or indirectly reduce faulty thinking and improve correct thinking (or features).



NLAs, as proposed in Anthropics paper (https://transformer-circuits.pub/2026/nla/index.html) are basically 2 components that work together over an activation space. First, an Activation Verbalizer that converts an activation to a human-readable explanation and second, an Activation Reconstructor that can take a natural language description and “reconstruct” the activation. The unique thing about NLAs is that their human-readable “explanations” must be good enough to reconstruct the original activation. A way to measure or ensure this is the Fraction of Variance Explained (FVE)



The goal of this task is to re-impliment the NLA pipeline (and components) as documented by Anthropic, using a small open-source model and available compute, document our findings, and discuss our observations and learnings.



Additionally, this task tries to evaluate how this might perform in a software code domain and shares findings in those as well as possibly a way to see how the pipeline generalizes in this aspect and what to do better.



Through the task, I investigated how much of the NLA methodology can be leveraged by a small open-source model under compute and scale constraints. To be clear, the task did not reproduce the reinforcement learning part, or use claude-level performance. What I was able to achieve was to preserve meaningful activation and increasing improvements with a smaller model and easily accessible compute. In other words, I tried to answer the question: Can hidden activations ne compressed into natural language explanations and reconstructed back with decent fidelity?



I will now discuss:

* Model \& Architecture Choices
* Switching from MiniLM (a light-weight encoder) to Google Gemma (and what we learnt in doing that)
* The Activation Reconstruction loop and FVE Results
* Quantitative Analysis (of the verbalizer generations)
* Domain Comparison (Natural Language vs Python Code)
* Discussions
* Conclusion  





### **Model \& Architecture Choices**



The development environment of choice was Google Colab. Being cloud-based makes it easy to get up and running with minimal setup but full access to the Python runtime and GPU compute.



Model Choice: Initially, I had two base models in mind: Google Gemma-2-2B and Alibaba Qwen2.5-Coder-3B. They are both lightweight, and my excitement about how this would generalize in the software domain prompted the Qwen2.5-Coder-3B option. However, I eventually settled for the Google Gemma-2-2B, as the primary task is to first fully implement the NLA pipeline, and for ecosystem reasons, the Gemma-2-2B would be faster. See Section 2: Cells 5-6



Sample Scale: About half the task time was spent improving the algorithm using a starting size of 100 OpenWebText samples. Eventually, I went to 500, then 1,000, and finally 5,000. I aimed to only take on the cost (time and GPU resources) when I was sure the results were valid and improving in a logical way



Activation Data Metrics: the extraction was performed from

Layer 8 residual stream

5,000 OpenWebText samples

Each record contained

Input text

Layer number

Token Position

2304-dimensional activation vector

Checkpointing enabled a stable and resumeable pipeline (in case of Colab runtime interruptions)



Generated Artefacts:

* Item 1
* Item 2
* Item 3







The Encoder Upgrade: From MiniLM to Gemma: (350 words)







The Reconstruction Loop and FVE Results: (400 words)







Qualitative Analysis: What the Verbalizer Generates: (400 words)





Domain Comparison (Natural Language vs Python Code): (400 Words)



While the task and NLAs are generally broad, this submission tried to factor in application to the improvement of the software code generation output. Specifically, while the entire NLA pipeline was evaluated on OpenWebText, it was additionally extended to software code (Python Code from hugging face), to test how the pipeline generalises and with the goal that this will be of benefit to the larger AI4Code project’s perceived goal of mechanistic interpretability of software code generation.









Discussion: (250 words)



There were key findings throughout this task:



I started the verbalization with the MiniLM encoder. I later realized learning was capped and there was little improvement in FVE as we scaled. This was due to MiniLM’s 384-dimension capacity, and as such, a lot of rich data was not being used. I switched to the inbuilt encoder shipped with the Gemma-2-2B and took advantage of the full 2304-dimension vector capacity.

Linear Models outperformed the deeper MLPs (See Cells 24-26). It was mixed discovery for me because I expected that with higher samples, the MLP would adjust and have better performance, but it didn't. At 5,000 samples, the Linear Cosine Similarity = 0.524; FVE = 0.253, and the MLP Cosine Similarity = 0.383; FVE = 0.135. So it appears the activation geometry is substantially linear.

The paper’s warm-start FVE (before Reinforcement Learning, which I didn't do) is in the 0.3-0.4 range. After I increased sampling from 1,000 to 5,000, our (linear) FVE shot up 25% to 0.253 from 0.198 and Linear Cosine went from 0.487 to 0.524. This was really encouraging and made all the learning and tweaking worth it. A larger scale of 10,000+ samples may well improve this to the 0.3-0.4 range.





Conclusion: (100 words)











