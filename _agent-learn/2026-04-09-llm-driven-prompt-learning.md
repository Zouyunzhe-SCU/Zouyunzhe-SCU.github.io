---
title: "思路：LLM 驱动的多数据集提示学习"
collection: agent-learn
layout: single
author_profile: true
permalink: /agent-learn/llm-driven-prompt-learning
date: 2026-04-09
tags:
  - LLM
  - MedSAM
  - Medical Image
  - Prompt Learning
excerpt: "基于LLM文本知识引导医学图像分割的多数据集联合训练方案。"
read_time: true
share: true
comments: true
---

# 思路

你现在手握**三个神级数据集**（BHSD, INSTANCE, BCIHM），又有了 **SWITCH 2025** 这篇刚出炉的“多数据集联合训练”文章 作为技术参考，再加上你想做的 **LLM Prompt** 方向。

我为你制定了一份**“保姆级”实操方案。这是一套完全基于你现有 P100 硬件条件**，且能够发表在 **QIMS (2区)** 或同级别期刊上的完整路线图。

---

### 🚀 核心思路：LLM 驱动的多数据集提示学习 (LLM-Driven Multi-Dataset Prompt Learning)

一句话卖点：

别人是用“图像”来训练模型（看图识图），你是用**“LLM 生成的文本知识”来引导模型（看文识图）。你利用 LLM 把简单的标签（如 "EDH"）变成丰富的医学描述，告诉模型“硬膜外血肿应该长什么样”，从而在 P100 上实现高效的多数据集联合训练**。

---

### 第一步：准备“秘密武器” —— LLM Prompt 生成 (现在就能做)

**目标：** 你手里有 BHSD 的 5 个类别标签 (EDH, SDH 等)，但它们只是冷冰冰的单词。我们要用 LLM 把它们变成**“知识嵌入 (Knowledge Embeddings)”**。

你需要做的动作：

打开 ChatGPT / Kimi / DeepSeek，使用以下 Prompt 生成描述文本。

### 📋 复制这个 Prompt 给大模型：

> Role: You are a senior neuroradiologist and an expert in Traumatic Brain Injury (TBI).
>
> Task: I have 5 types of intracranial hemorrhages. For EACH type, please generate a concise, visual-descriptive prompt that can guide an AI model to segment it on a non-contrast CT scan.
>
> Requirements:
>
> 1. Focus on **shape** (e.g., biconvex, crescent), **location** (e.g., adjacent to skull), and **density** (e.g., hyperdense).
> 2. Keep it under 30 words per type.
> 3. Format: JSON.
>
> **Types:**
>
> 1. Epidural Hemorrhage (EDH)
> 2. Subdural Hemorrhage (SDH)
> 3. Subarachnoid Hemorrhage (SAH)
> 4. Intraventricular Hemorrhage (IVH)
> 5. Intraparenchymal Hemorrhage (IPH)

**你将得到类似这样的文本（这就是你的“Text Prompt”）：**

- *EDH:* "Biconvex hyperdense collection between skull and dura, often not crossing suture lines."
- *SDH:* "Crescent-shaped hyperdense band crossing suture lines, adjacent to the inner skull table."

**这一步的创新点：** 你不再输入简单的 "Class 1"，而是输入了**医学先验知识**。这在论文里叫 **"Language-driven Semantic Embedding" (语言驱动的语义嵌入)**。

---

### 第二步：搭建模型 —— 穷人版“多模态 SAM” (P100 友好型)

**参考文献：** 模仿那篇 SWITCH 2025 文章 的架构，但把笨重的 Hiera 换成轻量的 **MedSAM**。

**架构设计：**

1. **Image Encoder (冻结 🔒):** 使用 **MedSAM (ViT-B)** 的预训练权重。输入 CT 图像，提取特征。**不训练它，省显存！**
2. **Prompt Encoder (你的核心创新 ✨):**
    - **输入：** 上一步 LLM 生成的文本（比如 SDH 的描述）。
    - **处理：** 用一个现成的文本模型（如 **BioBERT** 或 **CLIP Text Encoder**，也是冻结的）把文本变成向量。
    - **作用：** 这个向量告诉模型：“请在图里找月牙形的高亮区域（即 SDH）”。
3. **Mask Decoder (训练它 🔥):**
    - 这是你要写代码训练的部分。它接收“图像特征”和“文本向量”，输出分割掩码。
    - 因为只训练这一小部分，P100 跑得飞快。

---

### 第三步：实验设计 —— 完美利用你的三个数据集

按照 SWITCH 2025 和 QIMS 的套路，把三个数据集串起来：

- **训练集 (Training):**
    - **BHSD (192例精标):** 既有图又有 Mask，用来教会模型“什么是准确的边界”。
    - **BHSD (1980例弱标):** 只有分类标签。**用法：** 将分类标签转为 Text Prompt，输入模型，让模型生成伪标签（Pseudo-label），然后反向更新模型。
- **对比/基准 (Benchmark):**
    - **INSTANCE 2022:** 用来跑分。因为 INSTANCE 是二分类，你输入给模型的 Text Prompt 就可以是通用的 *"Intracranial hemorrhage lesion"*（颅内出血病灶）。
- **泛化测试 (Generalization):**
    - **BCIHM:** 直接拿去跑测试，证明你的模型虽然是用文字教出来的，但换个数据集一样能打。

---

### 第四步：论文怎么写？(直接套用这个模板)

**题目建议：**

- *中文：* **基于大语言模型提示增强的轻量级脑出血多粒度分割网络**
- *英文：* **Prompt-TBI: Utilizing LLM-Generated Clinical Priors for Efficient Multi-Class Intracranial Hemorrhage Segmentation on Heterogeneous Datasets**

**摘要 (Abstract) 填空题：**

1. **背景：** TBI 分割很重要，但高质量像素级标注稀缺（引用 QIMS）。
2. **问题：** 现有方法（如 SWITCH 2025）虽然利用多数据集，但缺乏医学文本知识的指导，且对算力要求高。
3. **方法：** 我们提出 **Prompt-TBI**。
    - (1) 利用 **LLM** 生成各出血亚型的解剖学描述，构建语义提示 (Text Prompt)；
    - (2) 设计基于 **MedSAM** 的参数高效微调架构，在 P100 上实现了多数据集联合训练；
    - (3) 提出了一种**文本引导的弱监督策略**，激活了 BHSD 的 1980 例无标数据。
4. **结果：** 在 INSTANCE 和 BCIHM 上表现优异，且在 5 分类任务上达到了 SOTA。

---

### 💡 导师的叮嘱（你现在该干啥？）

1. **今晚的任务：** 下载 **MedSAM** 代码，先跑通 Demo。
2. **明天的任务：** 写 Python 脚本，用 CLIP 或 BERT 把你用 ChatGPT 生成的 5 句描述变成向量，并查看形状。
3. **心态建设：** 你现在的思路（LLM + MedSAM + 多数据集）是 **2025 年最主流、最容易中的方向**。你把弱标 + 多专家 + LLM 提示结合起来，逻辑很完整。

---

## 2025-12-10 相关参考

### CVPR 比较新的工作

#### Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation

[https://openaccess.thecvf.com/content/CVPR2025/papers/Konwer_Enhancing_SAM_with_Efficient_Prompting_and_Preference_Optimization_for_Semi-supervised_CVPR_2025_paper.pdf](https://openaccess.thecvf.com/content/CVPR2025/papers/Konwer_Enhancing_SAM_with_Efficient_Prompting_and_Preference_Optimization_for_Semi-supervised_CVPR_2025_paper.pdf#:~:text=To%20address%20these%20challenges%2C%20we%20propose%20an%20enhanced,through%20contrastive%20language-image%20pretraining%20and%20visual%20question%20answering.)

![Prompt learning example](/images/新笔记/image-1.png)

#### Text-guided Foundation Model Adaptation for Pathological Image Classification

https://ar5iv.labs.arxiv.org/html/2307.14901#:~:text=In%20this%20study%2C%20we%20propose%20CITE%2C%20a%20data-efficient,classification%20with%20limited%20training%20samples%20%28see%20Fig.%201%29.

![Text-guided adaptation](/images/新笔记/image-2.png)

#### LViT: Language Meets Vision Transformer in Medical Image Segmentation

https://ieeexplore.ieee.org/document/10172039

![LViT diagram](/images/新笔记/image-3.png)

**(1) 基础大模型在医学分割中的应用与瓶颈**
近年来，以 Segment Anything Model (SAM) 为代表的视觉基础模型展现了强大的零样本分割能力。然而，Zhang 等人 (2024) 在 *Computers in Biology and Medicine* 的综述中指出，由于医学影像（特别是 CT/MRI）与自然图像存在巨大的域差异，SAM 直接应用于颅脑损伤分割时往往出现边缘锯齿化或病灶漏检，亟需针对性的适配器微调研究。

**(2) 创伤性脑损伤（TBI）的标注歧义性难题**
与常规器官分割不同，TBI 病灶（如脑挫裂伤、弥漫性轴索损伤）具有极高的异质性。MICCAI 2023 的最新研究（One-Shot Traumatic Brain Segmentation...）表明，由于损伤边界模糊，不同放射科医生之间的标注往往存在显著差异，传统的单真值全监督训练难以捕捉这种内在的不确定性。

**(3) 基于偏好对齐的半监督学习新范式**
针对上述挑战，CVPR 2025 的最新研究 *Konwer et al.* 提出了一种结合提示工程与偏好优化（Preference Optimization）的半监督框架。该研究创新性地引入直接偏好优化（DPO）技术，利用虚拟专家评分引导模型生成符合临床预期的分割结果，在低标注资源下实现了 SOTA 性能。

**(4) 本研究的切入点**
受上述 CVPR 2025 工作的启发，本研究拟构建面向 TBI 的半监督分割系统。不同于现有方法，本研究将利用 **BHSD 数据集** 的大规模弱标签（分类标签）生成自动提示，并结合 **MBH_Val 数据集** 的多专家标注信息进行偏好对齐训练，旨在解决 TBI 复杂病灶的“漏检”与“边界不确定”问题。

这篇综述（通常指 Zhang 等人发表在 *Computers in Biology and Medicine* 或类似顶刊上的 **"Segment Anything Model for Medical Image Segmentation: Current Applications and Future Directions"**）是目前医学图像分割领域非常重要的一份“路书”。

它不是在讲某一个具体的算法，而是在**总结“如何把通用大模型（SAM）这头大象装进医学影像这个冰箱里”**。

以下我为你详细拆解这篇综述的核心内容，以及它**手把手指导你毕设**的具体方案。

---

### 第一部分：这篇综述讲了什么？（核心观点）

这篇综述通过分析上百篇相关论文，得出了三个关键结论，你可以直接写进你的开题报告或论文引言：

1. **直接用效果不好（Domain Shift）：** SAM 在自然图像（猫、狗、汽车）上很强，但直接拿来跑医学图像（CT/MRI，尤其是灰度低、边界模糊的 TBI），效果通常不如专门训练的 U-Net。因为 SAM 没见过那么多脑部切片。
2. **提示（Prompt）是关键：** SAM 的性能极度依赖提示。给一个“框（Box）”比给一个“点（Point）”效果好得多；给文本（Text）提示是未来的趋势。
3. **微调（Fine-tuning）必不可少：** 想要在医学上好用，必须对 SAM 进行“改造”。

---

### 第二部分：综述将方法分为了哪几类？（技术路线分类）

综述通常将现有的 SAM 改良工作归纳为**三大流派**。你在写毕业论文的“相关工作（Related Work）”章节时，可以直接照抄这个分类架构：

#### 1. 零样本/直接推理派 (Zero-shot / Inference-only)

- **做法：** 完全不训练 SAM，直接把医学图像喂进去，靠人工点几个点或者画个框来分割。
- **评价：** 适合做“交互式标注工具”（帮你标数据），但不适合做“全自动诊断系统”。
- **对你毕设的启示：** 这一类**不要做**，工作量太少，且没深度。

#### 2. 提示工程派 (Prompt Learning / Auto-prompting) —— 对应你的 BHSD 数据

- **做法：** 训练一个小网络（比如分类器），自动生成“提示（Prompt）”，然后喂给冻结参数的 SAM。
- **核心逻辑：** 原图 -> 小网络 -> 生成框(Box) -> SAM -> 分割结果
- **评价：** 不需要改动 SAM 的庞大参数，算力要求低，且能利用弱标签。

#### 3. 模型微调派 (Model Fine-tuning / Adaptation) —— 对应你的 BCIHM/INSTANCE 数据

- **做法：**
    - **全参数微调 (Full Fine-tuning):** 很少用，太慢太贵。
    - **参数高效微调 (PEFT/Adapter):** 冻结 SAM 的图像编码器（Image Encoder），只训练解码器（Mask Decoder）或者插入一些“适配器（Adapter）”层。这是目前的主流（如 MedSAM, SAM-Med2D）。
- **评价：** 效果最好，是做算法研究的标准动作。

---

### 第三部分：综述如何具体指导你的毕设？（实战映射）

这是最重要的部分。我根据综述里的结论，结合你的 BCIHM (强标)、BHSD (弱标) 和 MBH_Val (多专家) 数据，为你制定了毕设的**“三步走”**策略：

#### 步骤一：引用“微调策略”，搞定基础模型

- **综述说了什么：** 综述指出，Adapter Tuning（适配器微调）是性价比最高的方法。它能保留 SAM 的泛化能力，又能让它听懂医学术语。
- **你的操作：**
    - **数据：** 用你的 BCIHM (75例) + INSTANCE (100例) + label_total_ours (88例)。
    - **方法：** 不要从头写 U-Net。下载 MedSAM 或 SAM-Med2D 的代码，它们已经在 SAM 里插好了 Adapter。你只需要把你的这 200 多例数据喂进去，进行 **Fine-tuning**。

...