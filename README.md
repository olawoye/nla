# **KTH PhD Task 2026**





### **Introduction and Motivation**



LLMs are powerful, and the world is increasingly learning of its potential and promise. However, while, there are tools that target and improve its output or outcome, most of the internal thinking is a bit of a black box. LLM activations, a representation of the “internal mental state” of the LLM regarding a task, at any one time, is a high-dimensional vector. These opaque set of numbers are clearly difficult to understand for humans. In the last few years, efforts in mechanistic interpretability are trying to understand (or reverse-engineer) the internal workings of these powerful LLM models. Such efforts include breakthroughs such as Anthropic’s Sparse Autoencoders (SAEs) and Natural Language Autoencoders (NLAs). NLAs, the focus of this task, is an unsupervised way of generating natural language “explanations” of these activations earlier mentioned. Imagine the potential gains from the application of this, especially reduction of hallucinations from LLMs because if we can understand the internal workings of an LLM, then we can directly or indirectly reduce faulty thinking and improve correct thinking (or features).



NLAs, as proposed in Anthropics paper (https://transformer-circuits.pub/2026/nla/index.html) are basically 2 components that work together over an activation space. First, an Activation Verbalizer that converts an activation to a human-readable explanation and second, an Activation Reconstructor that can take a natural language description and “reconstruct” the activation. The unique thing about NLAs is that their human-readable “explanations” must be good enough to reconstruct the original activation. A way to measure or ensure this is the Fraction of Variance Explained (FVE)



The goal of this task is to re-impliment the NLA pipeline (and components) as documented by Anthropic, using a small open-source model and available compute, document our findings, and discuss our observations and learnings.



Additionally, this task tries to evaluate how this might perform in a software code domain and shares findings in those as well as possibly a way to see how the pipeline generalizes in this aspect and what to do better.



Through the task, I investigated how much of the NLA methodology can be leveraged by a small open-source model under compute and scale constraints. To be clear, the task did not reproduce the reinforcement learning part, or use claude-level performance. What I was able to achieve was to preserve meaningful activation and increasing improvements with a smaller model and easily accessible compute. In other words, I tried to answer the question: Can hidden activations ne compressed into natural language explanations and reconstructed back with decent fidelity?



So, Anthropic's own NLA research saw them having an FVE of 0.6-0.8 (meaning 60% to 80% of the AI's internal mathematical thought process was successfully captured AND reconstructed into understandable English). They used their flagship Claude Opus 4.6. This is significant and amazing.



At the relatively small scale of this project (5,000 samples of OpenWebText using the open-source Google Gemma-2-2B on Google Colab L4 GPU), I got a post NLA FVE of 0.181. While this is much lower, but it did show the methodology in Anthropic's paper worked!



I will now discuss:

* Model \& Architecture Choices
* Switching from MiniLM (a light-weight encoder) to Google Gemma (and what we learnt in doing that)
* The Activation Reconstruction loop and FVE Results
* Quantitative Analysis (of the verbalizer generations)
* Domain Comparison (Natural Language vs Python Code)
* Discussions
* Conclusion





### **Model \& Architecture Choices**



The development environment of choice was Google Colab. Being cloud-based makes it easy to get up and running with minimal setup but full access to the Python 3 runtime and L4 GPU compute.



##### **Model Choice**:

Initially, I had two base models in mind:

* Google Gemma-2-2B and
* Alibaba Qwen2.5-Coder-3B.



They are both lightweight, and my excitement about how this would generalize in the software domain prompted the Qwen2.5-Coder-3B option. However, I eventually settled for **the Google Gemma-2-2B**, as the primary task is to first fully implement the NLA pipeline, and for ecosystem reasons, the Gemma-2-2B would be faster. See Section 2: Cells 5-6



##### Sample \& Scaling:

For the training, I sourced data samples from [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) samples (an open-source replication of the WebText dataset from OpenAI). The scaling was done in stages for the purpose of speed, efficiency and to save cost (in terms of deploying time and GPU resources ONLY when it really mattered:

* 100: About half the task time was spent building the modelling algorithm and debugging
* 500: This helped get the first set of sensible cosines and FVE values.
* 1,000: Moving to this showed improvements in FVE metrics and helped test our ability to resume checkpoints
* 5,000: This was the sample size used for the results and discussions
* I believe scaling to 10,000 and 50,000 would be the next logical step to further improve the process





Generated Artefacts:

* Item 1
* Item 2
* [Code Activations Paraquet](https://github.com/olawoye/nla_phd/blob/main/data/code_activations.parquet)







##### Activation Dataset:



From the OpenWebText samples, a dataset of 5,000 activation records were extracted.



The extraction was performed from the Layer 8 residual stream



Each record contained

* Input text
* Layer number
* Token Position
* 2304-dimensional activation vector



Checkpointing enabled a stable and resumeable pipeline (in case of Colab runtime interruptions)



This established the foundation for all subsequent experiments





### Baseline Activation Reconstruction



Before introducing natural language, a baseline activation reconstruction experiment was conducted. See Cells 13-26



The pipeline was 3 key stages:



##### Text (Gemma) Embedding

So, initially, the MiniLM was the first choice here. It was fast and it worked all the way to the end. However, due to its 384-dimensional capacity, it could not capture all necessary text information to transfer to Gemma. Eventually, I upgraded to Gemma-2-2Bs internal encoder which operated at a comfortable 2304-dimensions.



Reconstructor = 





Activation



This established an upper-bound on reconstruction performance BEFORE introducing the natural language "bottleneck".





##### Linear vs MLP Model Performance

Just to be sure, I did an A/B experiment to attempt to see if the activation relationship was better modelled by a linear or MLP pattern. See cells



Linear Model:

* Cosine Similarity = 0.524
* FVE = 0.253



MLP:

* Cosine Similarity = 0.383
* FVE = 0.135



The activation appears significantly linear at this scale. To be honest, I manually shuffled between Linear/MLP while varying scales (100, 500, 1,000 samples) hoping the flexibility will leverage the scale but the linear is generally preferred at all scales and was used as the primary reconstructor going forward.





##### Baseline Results



So The result from the pipeline was:

Cosine Similarity: 0.537

FVE: 0.267



So even before introducing natural language the activation reconstruction is far from perfect. It basically means that reconstructing activations is quite difficult and that NLA performance should be interpreted relative to this baseline rather than against a "perfect" reconstruction







### Natural Language Autoencoder Implementation



And next, to implement the NLA, we do the following:

* PCA Compression (to 64 dimensions)
* Natural Language Explanation



As I earlier mentioned, we simplified a lot of things:

* No Reinforcement Learning (RL)
* Small Gemma-2-2B model instead of Claude Opus 4.6





##### NLA Loop Results

Post NLA Roundtrip results are visible in Cell 



* NLA FVE = 0.181
* NLA Cosine Similarity = 0.450



Comparing this to baseline (pre NLA) FVE of 0.267



Introducing the natural language bottleneck caused a significant loss of information. But the reconstruction performance is significantly above zero. So activation information is preserved even after compression.



This supports Anthropic's work even thought it highlights the challenges of reproducing at scale and with decent accuracy but there is exciting promise.





##### The Elephant In The Room (The Gap)

So yes, Anthropic reports an FVE of 0.6 - 0.8.



This reproduction achieved an FVE of 0.18.



There were a lot of perceived differences in the experiments:

* Smaller model
* Smaller dataset
* No reinforcement learning
* PCA compression
* Simpler verbalizer architecture
* Limited training budget



However, the goal was not to match Anthropic's absolute performance but to investigate of the phenomenon exists at small models. These results suggests that it does!





### Qualitative Analysis of the Explanations

The most interesting (and sometimes shocking) aspects of the experiment was that a lot of times, the generated explanations. See Cells 35, 38 and 39



So at the earlier part of the algorithim, there were often blank or dot (.) explanations. These were debugged and fixed.



However, as the algorithm stabilized, there were still some explanations in fluent English but not semantically aligned with the source text.



For example:

* Original: Poverty statistics article
* Generated: "By Elizabeth Hill For many, the current position of State Department as a general is not quite correct."



So, the verbalizer frequently generated plausible news-like language without preserving the actual topic





### Domain Comparison (Natural Language vs Python Code)



While the task and NLAs are generally broad, I tried to factor in application to the improvement of the software code generation output. Specifically, while the entire NLA pipeline was evaluated on OpenWebText, I additionally extended to software code, to test how the pipeline generalizes and with the goal that this will be of benefit to the larger AI4Code project’s perceived goal of mechanistic interpretability of software code generation. See Section 6 (Cells 41-44).



To investigate generalization, the exact same NLA pipeline was applied to [Python Code from The Stack](https://huggingface.co/datasets/bigcode/the-stack-smol). Its a subset of [the stack](https://huggingface.co/datasets/bigcode/the-stack) dataset where each of the about 30 programming languages have 10,000 random samples from the original dataset.

&#x20;

##### Below are my findings:



The verbalizer did not always learn code-specific concepts. For example, Input: Python Code, Output: "The new season is ..."



This was a little bit dissappointing as I thought code, compared with OpenWebText samples, may be more structured.



But, looking at the setup, it would suggest:

* the verbalizer architechure is insifficient to deal with code-style algorithm
* training is inadequate



This is an aspect I'm determined to work on.









### Discussion



There were key findings throughout this task:



* I started the verbalization with the MiniLM encoder. I later realized learning was capped and there was little improvement in FVE as we scaled. This was due to MiniLM’s 384-dimension capacity, and as such, a lot of rich data was not being used. I switched to the inbuilt encoder shipped with the Gemma-2-2B and took advantage of the full 2304-dimension vector capacity.
* Linear Models outperformed the deeper MLPs (See Cells 24-26). It was mixed discovery for me because I expected that with higher samples, the MLP would adjust and have better performance, but it didn't. At 5,000 samples, the (pre NLA) Linear Cosine Similarity = 0.524; FVE = 0.253, and the MLP Cosine Similarity = 0.383; FVE = 0.135. So it appears the activation geometry is substantially linear. This was quite interesting
* The paper’s warm-start FVE (before Reinforcement Learning) is in the 0.3-0.4 range. After I increased sampling from 1,000 to 5,000, our (linear) pre-NLA FVE shot up 25% to 0.253 from 0.198 and Linear Cosine went from 0.487 to 0.524. The post-NLA FVE was 0.18 which is hopeful. This was really encouraging and made all the learning and tweaking worth it. A larger scale of 10,000+ samples may well improve this to the 0.3-0.4 range.
* The verbalizer performed poorly in performance on code. This gaps in reconstruction accuracy (or fidelity) were disappointing





### Conclusion



The task successfully demonstrated reproduction in the following areas:

* Activation extraction
* Activation reconstruction
* Natural-language bottleneck training
* FVE evaluation
* Cross-domain evaluation



The most exciting evidence is that Natural language explanations retained enough information to reconstruct activations with decent fidelity



In the code domain, the generated explanations often lacked far less semantic context relative to the OpenWebText samples



Future work and next steps will focus on:

* Reinforcement learning
* Better verbalizers
* Larger models
* Domain-specific (coding) experiments \& improvements





