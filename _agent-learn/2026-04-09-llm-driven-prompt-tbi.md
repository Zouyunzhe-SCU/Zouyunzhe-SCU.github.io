---
title: '思路：LLM 驱动的多数据集提示学习'
collection: agent-learn
category: research
date: 2026-04-09
permalink: /agent-learn/2026-04-09-llm-driven-prompt-tbi/
excerpt: '基于LLM文本知识引导医学图像分割的多数据集联合训练方案，适配P100与BHSD/BCIHM/INSTANCE数据集。'
tags:
  - 医学图像
  - LLM
  - Prompt Learning
  - MedSAM
  - P100
---

# 思路

核心思路：**LLM 驱动的多数据集提示学习 (LLM-Driven Multi-Dataset Prompt Learning)**。

一句话卖点：别人是用“图像”来训练模型（看图识图），你是用**“LLM 生成的文本知识”来引导模型（看文识图）**。你利用 LLM 把简单标签（如 EDH、SDH）变成丰富的医学描述，告诉模型“硬膜外血肿应该长什么样”，从而在 P100 上实现高效的多数据集联合训练。

你不再输入简单的“Class 1”，而是输入医学先验信息，这一步是语言驱动的语义嵌入。

---

## 第一步：准备“秘密武器” —— LLM Prompt 生成 (现在就能做)

目标：你手里有 BHSD 的 5 个类别标签（EDH、SDH 等），但它们只是冷冰冰的单词。我们要用 LLM 把它们变成**“知识嵌入 (Knowledge Embeddings)”**。

你需要做的动作：打开 ChatGPT / Kimi / DeepSeek，使用以下 Prompt 生成描述文本：

> Role: You are a senior neuroradiologist and an expert in Traumatic Brain Injury (TBI). Task: I have 5 types of intracranial hemorrhages. For EACH type, please generate a concise, visual-descriptive prompt that can guide an AI model to segment it on a non-contrast CT scan. Requirements:
> 1. Focus on shape (e.g., biconvex, crescent), location (e.g., adjacent to skull), and density (e.g., hyperdense).
> 2. Keep it under 30 words per type.
> 3. Format: JSON.
> Types:
> 1. Epidural Hemorrhage (EDH)
> 2. Subdural Hemorrhage (SDH)
> 3. Subarachnoid Hemorrhage (SAH)
> 4. Intraventricular Hemorrhage (IVH)
> 5. Intraparenchymal Hemorrhage (IPH)

你将得到类似这样的文本，这就是你的“Text Prompt”。

### 这一步的创新点

- 你不再输入简单标签，而是输入医学先验知识。  
- 这种方法叫“Language-driven Semantic Embedding”。

---

## 第二步：搭建模型 —— 穷人版“多模态 SAM” (P100 友好型)

参考文献：模仿 SWITCH 2025 文章的架构，但把笨重的 Hiera 换成轻量的 MedSAM。

### 架构设计

1. **Image Encoder (冻结)**
   - 使用 MedSAM (ViT-B) 的预训练权重
   - 输入 CT 图像，提取特征
   - 不训练它，省显存

2. **Prompt Encoder (你的核心创新)**
   - 输入：LLM 生成的文本描述，例如 SDH 的描述
   - 处理：用现成文本模型（如 BioBERT 或 CLIP Text Encoder，冻结）把文本变成向量
   - 作用：告诉模型“请在图里找月牙形的高亮区域，即 SDH”

3. **Mask Decoder (训练它)**
   - 接收图像特征和文本向量，输出分割掩码
   - 因为只训练这一小部分，P100 跑得快

---

## 第三步：实验设计 —— 完美利用你的三个数据集

按照 SWITCH 2025 和 QIMS 的套路，把三个数据集串起来：

- **训练集 (Training)**
  - BHSD (192例精标)：既有图又有 Mask，用来教会模型“什么是准确的边界”。
  - BHSD (1980例弱标)：只有分类标签。用法：将分类标签转为 Text Prompt，输入模型，让模型生成伪标签，然后反向更新模型。

- **对比/基准 (Benchmark)**
  - INSTANCE 2022：用来跑分。因为 INSTANCE 是二分类，你输入给模型的 Text Prompt 就可以是通用的“Intracranial hemorrhage region”或“bleeding region”。

- **多专家验证**
  - MBH_Val：用来验证不确定性、歧义性和偏好对齐。

---

## 第四步：论文怎么写？(直接套用这个模板)

题目建议：

- 中文：基于大语言模型提示增强的轻量级脑出血多粒度分割网络
- 英文：Prompt-TBI: Utilizing LLM-Generated Clinical Priors for Efficient Multi-Class Intracranial Hemorrhage Segmentation on Heterogeneous Datasets

摘要填空：

1. 背景：TBI 分割很重要，但高质量像素级标注稀缺。  
2. 问题：现有方法（如 SWITCH 2025）虽然利用多数据集，但缺乏医学文本知识指导，且对算力要求高。  
3. 方法：我们提出 Prompt-TBI。  
   - (1) 利用 LLM 生成各出血亚型的解剖学描述，构建语义提示 (Text Prompt)；  
   - (2) 设计基于 MedSAM 的参数高效多模态分割架构；  
   - (3) 将弱标数据转为提示驱动伪标签，并结合多专家数据做偏好对齐。

---

## 导师的叮嘱（你现在该干嘛？）

1. 今晚的任务：下载 MedSAM 代码，先跑通 Demo。  
2. 明天的任务：写 Python 脚本，用 CLIP 或 BERT 把你用 ChatGPT 生成的 5 句描述变成向量，并查看形状。  
3. 心态建设：你现在的思路（LLM + MedSAM + 多数据集）是 2025 年最主流、最容易中的方向。

---

## 2025-12-10 相关参考

### CVPR 比较新的工作

- Enhancing SAM with Efficient Prompting and Preference Optimization for Semi-supervised Medical Image Segmentation
- Text-guided Foundation Model Adaptation for Pathological Image Classification
- LViT: Language Meets Vision Transformer in Medical Image Segmentation

### 继续补充

- 这类综述指出：SAM 在自然图像上很强，但直接应用于医学图像效果差。  
- 提示是关键：框提示优于点提示，文本提示是未来趋势。  
- 微调必不可少：要让 SAM 在医学上好用，必须改造和适配。

---

## 关键结论

- 你现在手握三个数据集（BHSD, INSTANCE, BCIHM），又有 SWITCH 2025 作为参考，再加上 LLM Prompt 方向，已经具备一套可落地的方案。  
- 这是一条适合 P100 的路线：先用冻结 encoding 的 MedSAM，再训练小型 decoder / prompt adapter。  
- 通过弱标签生成提示驱动伪标签、利用多专家数据做偏好对齐，你的研究具备发表潜力。
