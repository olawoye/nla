# **NLA Project**





### **Introduction and Motivation**



LLMs are powerful, and the world is increasingly learning of its potential and promise. However, while, there are tools that target and improve its output or outcome, most of the internal thinking is a bit of a black box. LLM activations, a representation of the “internal mental state” of the LLM regarding a task, at any one time, is a high-dimensional vector. These opaque set of numbers are clearly difficult to understand for humans. In the last few years, efforts in mechanistic interpretability are trying to understand (or reverse-engineer) the internal workings of these powerful LLM models. Such efforts include breakthroughs such as Anthropic’s Sparse Autoencoders (SAEs) and Natural Language Autoencoders (NLAs). NLAs, the focus of this task, is an unsupervised way of generating natural language “explanations” of these activations earlier mentioned. Imagine the potential gains from the application of this, especially reduction of hallucinations from LLMs because if we can understand the internal workings of an LLM, then we can directly or indirectly reduce faulty thinking and improve correct thinking (or features).



NLAs, as proposed in Anthropics paper (https://transformer-circuits.pub/2026/nla/index.html) are basically 2 components that work together over an activation space. First, an Activation Verbalizer that converts an activation to a human-readable explanation and second, an Activation Reconstructor that can take a natural language description and “reconstruct” the activation. The unique thing about NLAs is that their human-readable “explanations” must be good enough to reconstruct the original activation. A way to measure or ensure this is the Fraction of Variance Explained (FVE)



The goal of this task is to re-impliment the NLA pipeline (and components) as documented by Anthropic, using a small open-source model and available compute, document our findings, and discuss our observations and learnings.



Additionally, this task tries to evaluate how this might perform in a software code domain and shares findings in those as well as possibly a way to see how the pipeline generalizes in this aspect and what to do better.



Through the task, I investigated how much of the NLA methodology can be leveraged by a small open-source model under compute and scale constraints. To be clear, the task did not reproduce the reinforcement learning part, or use claude-level performance. What I was able to achieve was to preserve meaningful activation and increasing improvements with a smaller model and easily accessible compute. In other words, I tried to answer the question: Can hidden activations ne compressed into natural language explanations and reconstructed back with decent fidelity?



So, Anthropic's own NLA research saw them having an FVE of 0.6-0.8 (meaning 60% to 80% of the AI's internal mathematical thought process was successfully captured AND reconstructed into understandable English). They used their flagship Claude Opus 4.6. This is significant and amazing.



At the relatively small scale of this project (5,000 samples of OpenWebText using the open-source Google Gemma-2-2B on Google Colab L4 GPU), I got a post NLA FVE of 0.181. While this is much lower, but it did show the methodology in Anthropic's paper worked!





#### Reproducing/Reviewing this (Re)Implementation



The full Python notebook implementation is [available here](https://github.com/olawoye/nla/blob/main/notebooks/setup_activations.ipynb)



You can clone/run the notebook in Google Colab. I chose the Python 3 runtime and used an L4 GPU. The free tier T4 would probably work fine but it was kind of slow for me.



Next, you would

* Connect the GitHub Repository (so the progress can be saved there at the end
* Connect your Google Drive (so it can save some artefacts there) during runtime. They are moved to GitHub at the end
* Update the parameters in Section 1, Cell 1.1. You may want to start with a low NUM\_SAMPLES\_TO\_PROCESS of say 100,500 before moving to something higher.





#### The Task



For the given task, I will now discuss my implementation of the following:

* Model \& Architecture Choices
* Baseline (pre-NLA) Activation
* Natural Language Autoencoder Implementation
* Quantitative Analysis
* Domain Comparison (Natural Language vs Python Code)
* Discussions
* Conclusion





### **Model \& Architecture Choices**



The development environment of choice was Google Colab. Being cloud-based makes it easy to get up and running with minimal setup but full access to the Python 3 runtime and L4 GPU compute.



#### Model Choice:

Initially, I had two base models in mind:

* Google Gemma-2-2B and
* Alibaba Qwen2.5-Coder-3B.



They are both lightweight, and my excitement about how this would generalize in the software domain prompted the Qwen2.5-Coder-3B option. However, I eventually settled for **the Google Gemma-2-2B**, as the primary task is to first fully implement the NLA pipeline, and for ecosystem reasons, the Gemma-2-2B would be faster. See Section 2: Cells 5-6



#### Encoder Choice:

* Google Gemma-2-2B (in-built encoder)
* all-MiniLM-L6-v2 (was initially used but discarded)





#### Sample \& Scaling:

For the training, I sourced data samples from [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) samples (an open-source replication of the WebText dataset from OpenAI). The scaling was done in stages for the purpose of speed, efficiency and to save cost (in terms of deploying time and GPU resources ONLY when it really mattered:

* 100: About half the task time was spent building the modelling algorithm and debugging
* 500: This helped get the first set of sensible cosines and FVE values.
* 1,000: Moving to this showed improvements in FVE metrics and helped test our ability to resume checkpoints
* 5,000: This was the sample size used for the results and discussions
* I believe scaling to 10,000 and 50,000 would be the next logical step to further improve the process



#### Verbalizer

I used the Google Flan-T5-Small





#### Generated Artefacts

For reference, artefacts generated during the experiment are stored [here in the /artefacts repository folder](https://github.com/olawoye/nla/tree/main/artefacts). Artefacts for each sample size are separated to avoid contamination during training/re-use/resumption. Each sample artefact folder contains:

* checkpoints/ (folder)
* verbalizer\_checkpoint (folder)
* verbalizer\_tokenizer (folder)
* domain\_comparison.png
* experiment\_info.json
* nla\_distributions.png
* requirements.txt
* verbalizer\_checkpoint.pt











### **Baseline (pre NLA) Activation**





#### Activation Dataset:



From the OpenWebText samples, a dataset of 5,000 activation records were extracted. See Section 3



The extraction was performed from the Layer 8 residual stream



Each record contained

* Input text
* Layer number
* Token Position
* 2304-dimensional activation vector



Checkpointing enabled a stable and resumeable pipeline (in case of Colab runtime interruptions)



This established the foundation for all subsequent experiments





#### Baseline Activation Training



Before introducing natural language, a baseline activation reconstruction experiment was conducted. See Section 4



The following took place:

* **Text Summary generation** from the activation dataset
* **Text Encoding (MiniLM/Gemma Embedding)**: So, initially, the MiniLM was the first choice here. It was fast, lightweight it worked all the way to the end. However, due to its 384-dimensional capacity, it compresses most information away. Eventually, I upgraded to Gemma-2-2Bs internal encoder which operated at a comfortable 2304-dimensions.
* **Activation Reconstruction**
* **Training**: Train/Test Split and Training with epoch tuning and checkpoint resupmtion
* **Cosine/FVE Evaluation**
* **Baseline A/B Model Experiment**: Here I re-evaluated if the linear or mlp models had the best cosine/FVE



##### MiniLM vs Gemma (as encoder) effect

Before switching to Gemma, I had a baseline linear cosine of \~ 0.18 and FVE of \~ 0.14. After switching to Gemma (as encoder), linear cosine of \~0.60, FVE of 0.33. A roughly 3-4x improvement purely from changing the text encoder. MiniLM produced 384-dimensional embeddings that capture surface meaning. Gemma layer-8 activations are 2304-dimensional and encode computational state. The AR was trying to recover a 2304-dimension rich state from a 284-dimensional lossy semantic summary. It means the choice of the encoder is very important in any future implementation





#### Linear vs MLP Model Performance

Just to be sure, I did an A/B experiment to attempt to see if the activation relationship was better modelled by a linear or MLP pattern. See cells



Linear Model:

* Cosine Similarity = 0.529
* FVE = 0.260



MLP:

* Cosine Similarity = 0.381
* FVE = 0.133



The activation appears significantly linear at this scale. To be honest, I manually shuffled between Linear/MLP while varying scales (100, 500, 1,000 samples) hoping the flexibility will leverage the scale but the linear is generally preferred at all scales and was used as the primary reconstructor going forward.





###### **A/B reconstructor results table** (Section 3, Cell 4.10/4.11 output)



||Cosine|FVE|
|-|-|-|
|Linear|0.529|0.260|
|MLP|0.381|0.133|







#### Baseline Results



So The result from the pipeline was:

Cosine Similarity: 0.529

FVE: 0.260



So even before introducing natural language the activation reconstruction is far from perfect. It basically means that reconstructing activations is quite difficult and that NLA performance should be interpreted relative to this baseline rather than against a "perfect" reconstruction







This established an upper-bound on reconstruction performance BEFORE introducing the natural language "bottleneck".







### Natural Language Autoencoder Implementation



#### Action Verbalizer Implementation



The next step was to implement the NLA:



* PCA Compression: to compress the 2304 dimensions into 64-dimension vectors
* Build Verbalizer: using the Google Flan-T5-Small model
* Verbalizer Training: Training with warm-ups and 20 epoch tuning
* Generate Natural Language Explanations
* Reconstruct Activations
* Evaluate FVE
* Inspection of Generated Explanations



As I earlier mentioned, we simplified a lot of things:

* No Reinforcement Learning (RL)
* Small Gemma-2-2B model instead of Claude Opus 4.6





#### FVE and Cosine Histograms (OpenWebText)

![FVE and Cosine Histograms](https://github.com/olawoye/nla/blob/main/artefacts/nla_distributions.png?raw=true)







##### NLA Loop Results

Post NLA Roundtrip results are visible in Cell



* NLA FVE = 0.185
* NLA Cosine Similarity = 0.453



Comparing this to baseline (pre NLA) FVE of 0.260



Introducing the natural language bottleneck caused a significant loss of information. But the reconstruction performance is significantly above zero. So activation information is preserved even after compression.



This supports Anthropic's work even thought it highlights the challenges of reproducing at scale and with decent accuracy but there is exciting promise.





##### The Elephant In The Room (The Gap)



So yes, Anthropic reports an FVE of 0.6 - 0.8.



This reproduction achieved an FVE of 0.185.



There were a lot of perceived differences in the experiments:

* Smaller model \& dataset
* No reinforcement learning
* PCA compression
* Simpler verbalizer architecture
* Limited training budget



However, the goal was not to match Anthropic's absolute performance but to investigate of the phenomenon exists at small models. These results suggests that it does!





### Qualitative Analysis of the Explanations





#### Scaling Progression \& diminishing returns



Like I mentioned, running successive runs of 100, 500, 1,000 and 5,000 provided some insight as to how the pipeline behaved



|Samples|Linear Cosine|Linear FVE|NLA Cosine|NLA FVE|
|-|-|-|-|-|
|100 (MiniLM)|0.180|0.014|0.161|0.006|
|1,000 (MiniLM)|0.166|0.007|0.161|0.006|
|1,000 (Gemma, Pre Aug)|0.601|0.330|0.237|-0.034|
|1,000 (Gemma, Post Aug)|0.487|0.198|0.419|0.138|
|5,000 (Gemma, Post Aug)|0.529|0.260|0.453|0.185|



1. The MiniLM plateau between 100 and 1,000 samples raised the first alarm/curiosity that the encoder was the issue not data volume. Once, the encoder was changed, the FVE changed significantly
2. The negative FVE (post gemma upgrade) was due to the fact that the AR was trained on 1,000 char long text while it received short verbalizer output of 10-20 words. We fixed this Cell 4.2. This problem was initially hidden in MiniLM because it by default compressed everything into a small flat semantic space regardless of length
3. The sample size change also provided improvement, but the rate of increase is diminishing
4. At the very early phase, the most interesting aspects of the experiment was frequent semantic issues with generated explanations. These were debugged and fixed. However, as the algorithm stabilized, there were still some explanations in fluent English but not semantically aligned with the source text. For example: Original: Poverty statistics article; Generated: "By Elizabeth Hill For many, the current position of State Department as a general is not quite correct."



#### Effect of Verbalizer Training



The verbalizer consistently generates fluent English style sentences regardless of input domain or language.

There was a Russian-language input that produced a generic US news sentence and the "Michael Jackson / Pirate Bay" sample.



This "collapse" to the training may be die to the fact that the Flan-T5-Small verbalizer trained on OpenWebText summaries that learns that the most probable output, given "any prompt" is a fluent English language sentence. With 5,000 samples and no RL signal from the AR, the model is satisfied with this local best prediction.



This highlights why the Reinforcement Learning (from the AR) is necessary as per Anthropic's own methodology



IT also appears (based on visual sample analysis) that the top-10 success cases appear to be samples that have US, political or journalistic framed snippets which match the verbalizer training. This might not be valid with further testing but its worth noting.











### Domain Comparison (Natural Language vs Python Code)



While the task and NLAs are generally broad, I tried to factor in application to the improvement of the software code generation output. Specifically, while the entire NLA pipeline was evaluated on OpenWebText, I additionally extended to software code, to test how the pipeline generalizes in code domain and with the goal that this will be of benefit to the larger AI4Code project’s perceived goal of mechanistic interpretability of software code generation. See Section 6, (Cells 41-44).



To investigate generalization, the exact same NLA pipeline was applied to [Python Code from The Stack](https://huggingface.co/datasets/bigcode/the-stack-smol). Its a subset of [the stack](https://huggingface.co/datasets/bigcode/the-stack) dataset where each of the about 30 programming languages have 10,000 random samples from the original dataset.

&#x20;

##### Below are my findings:



Plot: Domain comparison plot



![Domain comparison](https://github.com/olawoye/nla/blob/main/artefacts/domain_comparison.png?raw=true)



Here is my comparison:

||OpenWebText|Python Code from The Stack|
|-|-|-|
|Cosine Mean|\~ 0.36|\~ 0.36|
|FVE Median|\~ 0.27|\~ -0.14|





**Verbalizer Issues**: The verbalizer did not always learn code-specific concepts. For example, Input: Python Code, Output: "The new season is ..."



This was a little bit disappointing as I thought code, compared with OpenWebText samples, may be more structured.



But, looking at the setup, it would suggest:

* the verbalizer architecture is insufficient to deal with code-style algorithm
* training is inadequate



This is an aspect I'm determined to work on.



**Domain Gap with Increased Sampling**: As we moved from 1,000 to 5,000 samples, the domain gap has widened slightly and is now very clean. OpenWebText FVE median 0.252 vs code 0.134. This seems to mean that code reconstructs at roughly half the quality of natural language. The standard deviations are also tightening, which means the distributions are becoming more stable so those FVE means can be considered reliable.



**Other Findings**:

* After a little bit of online research, I learnt the negative median FVE for code might mean that for the majority of code samples, the reconstruction is worse than predicting the mean activation. That its not noise, but a structured failure caused by the encoder-reconstructor being trainied on natural language and applied to code without domain adaptation.
* If you've run the code, in Cell 6.1 (Compare block), the top-5 code successes are all short files with high English prose content - license headers, docstrings with readable variable names. This probably supports the encoder-sensitivity.
* Perhaps, If I trained the original NLA pipeline with Qwen2.5-Coder, this would be different?









### Discussion



There were key highlights during this task:



1. I started the verbalization with the MiniLM encoder. I later realized learning was capped and there was little improvement in FVE as we scaled. This was due to MiniLM’s 384-dimension capacity, and as such, a lot of rich data was not being used. I switched to the inbuilt encoder shipped with the Gemma-2-2B and took advantage of the full 2304-dimension vector capacity. When I saw the improvement, it let me know this process is not just about working algorithms but using the right size of the right tools for the job.
2. Linear Models outperformed the deeper MLPs (See Cells 24-26). It was mixed discovery for me because I expected that with higher samples, the MLP would adjust and have better performance, but it didn't. At 5,000 samples, the (pre NLA) Linear Cosine Similarity = 0.529; FVE = 0.260, and the MLP Cosine Similarity = 0.381; FVE = 0.133. So it appears the activation geometry is substantially linear. This was quite interesting
3. The paper’s warm-start FVE (before Reinforcement Learning) is in the 0.3-0.4 range. After I increased sampling from 1,000 to 5,000, our (linear) pre-NLA FVE shot up 25% to 0.260 from 0.198 and Linear Cosine went from 0.487 to 0.529. The post-NLA FVE was 0.185 which is hopeful. This was really encouraging and made all the learning and tweaking worth it. A larger scale of 10,000+ samples may well improve this to the 0.3-0.4 range.
4. Despite the increase from 0.198 to 0.260 FVE due to moving from 1,000 to 5,000, the next jump to 10,000+ may be diminishing to 0.02-0.03 gain. I would say we work more on other areas such lack of RL and change/improve the Flan-T5 verbalizer.
5. The verbalizer performed poorly in performance on code. This gaps in reconstruction accuracy (or fidelity) were disappointing. With increase in sampling, the domain FVE gap between code/general text widened. This means there has to be more work to be done on the verbalizer rather than increase training sample for us to improve domain gains in reconstruction fidelity





### Conclusion



The task successfully demonstrated reproduction in the following areas:

* Activation extraction
* Activation reconstruction
* Natural-language bottleneck training
* FVE evaluation



The most exciting evidence is that Natural language explanations retained enough information to reconstruct activations with decent fidelity



Future work and next steps will focus on:

* Reinforcement learning
* Better verbalizers
* Larger models
* Domain-specific (coding) experiments \& improvements

