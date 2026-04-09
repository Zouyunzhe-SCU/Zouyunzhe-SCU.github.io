---
title: '文献调研：GPT与医学图像分割初步研究'
collection: agent-learn
category: research
date: 2026-04-09
permalink: /agent-learn/2026-04-09-literature-review-gpt-medical/
excerpt: '关于SAM、MedSAM、Vision-Language Pretraining在医学图像分割中应用的深度文献调研笔记'
tags:
  - 医学图像分割
  - SAM
  - MedSAM
  - Vision-Language
  - PEFT
  - 弱监督学习
---

# 文献调研_gpt初步

## 概述

这份笔记系统总结了在医学图像分割（特别是颅内出血分割）中应用GPT、SAM、MedSAM及多模态学习的最新研究方向。本调研覆盖以下核心主题：

1. **SAM 与 MedSAM** - 基础模型与医学领域适配
2. **Vision-Language Pretraining (VLP)** - CLIP及医学VLP应用
3. **文本引导的医学图像分割** - 多模态Transformer架构
4. **参数高效微调 (PEFT)** - Adapter、LoRA、Visual Prompt Tuning
5. **弱监督与多数据集联合训练** - 实用策略
6. **LLM辅助医学分析** - 临床先验作为提示
7. **多粒度监督与SWITCH类方法** - 最新融合方案

---

## 1) 查找 SAM 与 MedSAM：论文 + 代码 + image-encoder 要点

### 关键来源

- **SAM 原始论文** (Kirillov et al., "Segment Anything", ICCV 2023)  
  - [arXiv](https://arxiv.org/abs/2304.02643)
  - [CVF开放获取](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)

- **MedSAM** (Ma et al., "Segment anything in medical images", Nature Communications 2024)
  - [Nature](https://www.nature.com/articles/s41467-024-44824-z)
  - [GitHub: bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM)

### 与你最相关的技术要点 (Image Encoder / 预训练策略)

#### SAM 的 Image Encoder

- **架构**：基于 ViT（通常是 ViT-B）
  - Patch embedding + Transformer encoder 提取多尺度/多层次视觉特征
  - 输出为 decoder 可消费的多分辨率特征图
  - SAM 在论文与实现中对 encoder 输出进行了多尺度处理以供 mask decoder 使用

- **训练策略**
  - 使用了大规模掩码数据集（SA-1B）
  - 目标是 promptable segmentation
  - Encoder 被训练成通用的视觉表征

#### MedSAM 的改动

- 在医学影像上做了进一步的微调/重训练
- 或在专门的医学数据集上做 domain-specific pretraining/fine-tuning
- 在 decoder/prompt pipeline 上对医学 prompt 做了适配
- 目的是弥补 CT/MR 与自然图像分布差异
- 提供了模型权重与 fine-tune 脚本（利于在P100上做轻量训练或PEFT）

### 实操建议（针对 P100）

- **冻结策略**：在 P100 上尽量冻结 ViT encoder（只训练 decoder / prompt adapter）
  - 这与 MedSAM 的"冻结大部分权重只微调少量参数"的思路一致
  - 显存友好且效果好
  - MedSAM 官方也提供了"只更新少数模块"的配置

---

## 2) VLP（Vision-Language Pretraining）在医学中的应用；CLIP 及 Bio-VLPs

### 关键来源

- **CLIP** (Radford et al., 2021)
  - [arXiv](https://arxiv.org/abs/2103.00020)
  - [论文PDF](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)

- **医学 VLP 综述与代表工作** (BioViL / GLoRIA / ConVIRT / Med-VLP)
  - [arXiv HTML](https://arxiv.org/html/2312.06224v1)

### 要点速记（与 Prompt → 图像特征对齐直接相关）

#### CLIP 思路与应用

- **双编码器架构**：image encoder + text encoder
- 通过对比学习对齐图像与文本表征
- 支持"给定文本描述 → 直接检索/定位图像区域或分类"
- 在自然图像上非常成功

#### 医学领域的挑战与适配

- 医学图像/文本对的统计分布差异很大
- 需要 domain-specific pretraining 或使用医学文本/报告做二次对齐

#### 医学 VLP 案例：GLoRIA / ConVIRT / BioViL

常见策略：

1. **双流对比（image/report）** - 对齐全局/局部特征
2. **融合 encoder（fusion encoder）** - 或用局部对齐（report 中的句子段落与图像区域做对齐）
3. 这些方法证明：**把放射科报告/病历文本纳入训练能显著提升图像表征的临床可用性**

### 实操建议（与你的 Prompt→Mask 思路）

- **不要直接用原始 CLIP 的通用权重 去 encode CT report 文本**
- **优先使用**：医学文本/临床 BERT（如 ClinicalBERT）或医学版 embedding
- 这样文本先验（你的 prompt）与图像特征的语义对齐会更可靠

---

## 3) "Text-Guided Medical Image Segmentation" 的相关架构

### 关键检索与代表工作

- 医学领域内越来越多以"text prompt 控制分割"的工作
- 把 report/标签作为 prompt 送入 decoder
- 把 text embedding 作为 query token 融入 decoder
- 最近 MICCAI/ICCV/NeurIPS workshop 论文实现了"language → segmentation"控制器
- [MICCAI Papers](https://papers.miccai.org/miccai-2024/paper/2014_paper.pdf)

### 要点速记

#### 常见设计模式

1. **双编码器 + 融合 decoder**
   - Image encoder 输出 + text embedding concatenated / cross-attention to decoder

2. **Text embedding 注入为 learnable prompt tokens**
   - 类似 VPT，但用文本作为初始化或条件

3. **Cross-modal transformer**
   - Takes image patches + text tokens jointly
   - 更重但能更好地融合模态信息

#### 关键结论

- 在 P100 上，建议采用方案 (1)+(2) 的组合
- 只冻结 image encoder 更节省显存
- 实验证明可行

---

## 4) PEFT（参数高效微调）在 ViT 上的应用

### 关键来源（必须读）

- **Visual Prompt Tuning (VPT)** (Jia et al., ECCV 2022)
  - 提出用可学的视觉 prompt tokens 去调优 ViT，而不更新大部分 backbone
  - [arXiv](https://arxiv.org/abs/2203.12119)
  - [PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf)

- **LoRA / Adapter 在 ViT 的应用综述与实证** (包括 AAAI 2023 PEFT on ViT)
  - [AAAI Open Access](https://ojs.aaai.org/index.php/AAAI/article/view/25160/24932)

### 三种主流方法对比

#### Visual Prompt Tuning (VPT)

- 在输入处添加可学的 prompt token（或少量层）
- 对显存友好且在少样本场景中表现好

#### LoRA（低秩适配）

- 在 transformer 的关键线性层插入低秩矩阵
- 只训练这些小矩阵
- 非常节省参数与显存，适合 P100
- 多篇近期工作证明 LoRA 在 ViT 上同样有效
- [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11118906/)

#### Adapters

- 在每个 transformer block 中插入小的瓶颈结构
- 训练成本低
- 适合做跨任务微调

### 实操建议（针对你 P100）

**推荐的尝试顺序**：

1. 先试 VPT（最简单）
2. 若需要更强表示能力，再用 LoRA
3. 若还需要进一步性能提升，再加 Adapter

> 设显存消耗和训练时间都会可控

---

## 5) 弱监督医学分割 与 多数据集联合训练

### 关键来源

- 弱监督分割综述与医学方向案例 (2021–2024)
- "MultiTalent / Multi-dataset"类型的工作（MICCAI 2023）
- WeakMedSAM 方案
- [ResearchGate](https://www.researchgate.net/publication/353863363_Weakly_Supervised_Medical_Image_Segmentation)

### 要点速记（对你最有价值）

#### 混合监督策略常用模式

1. **有 Mask → 强监督** 
   - 使用 Dice / BCE loss

2. **无 Mask 但有图级标签 → 弱监督**
   - 使用 CAM / pseudo-label
   - 或约束面积/形状的弱监督损失

3. **完全无标签 → 自训练**
   - Self-training 的范式，是可发表的方向

#### 多数据集联合训练（MultiTalent 等）

**处理"标签集合不一致/类别冲突"的常用做法**：

- 采用任务式损失：只在该数据集标签存在时计算对应分支损失
- 把标签集合映射到统一的大类空间再训练

**针对你的具体场景**（INSTANCE / BCIHM / 自有标注数据）：

- 对有精标签的数据集：强监督训练
- 对无标签数据：先用 MedSAM + 5 条 LLM prompt 生成伪标签
- 对伪标签+精标混合训练（按不同权重/不同 loss）
- 对于 INSTANCE / BCIHM：把它们作为验证/测试或少量微调集
- 采用 task-aware loss（只对存在的标签计算损失）
- [arXiv](https://arxiv.org/abs/2303.14444)

---

## 6) LLM 辅助医学图像分析：Clinical Priors 作为 Prompt

### 关键来源

- Recent med-VLP 与 SAM-based weak supervision 研究
- 文献中开始出现"LLM→生成structured priors/描述→作为视觉模型条件"的方法
- [arXiv Survey](https://arxiv.org/html/2312.06224v1)

### 要点速记

#### LLM 可生成两类有用信息

1. **结构化临床描述**
   - 短句描述病灶位置/形态/密度
   - 可作为 text prompt embedding 直接输入 decoder

2. **弱监督规则 / 决策阈值**
   - 如何筛选伪标签
   - 先验知识指导

### 实操建议（与你的伪标签 + 文本 prompt 流程直接相关）

- **采集 5–10 条临床模板**（例如由 LLM 生成）
- 每条描述一类典型病灶（"颅内出血的典型位置"、"脑水肿表现"等）
- 用 ClinicalBERT 或 BioClinicalBERT 编码这些描述
- **把编码结果存为缓存**（.pt 文件）

**实际推理流程**：

- 对每张无标 CT：加载 5 条缓存的 prompt embedding
- 逐条输入 MedSAM decoder
- 得到 5 个 mask 输出
- 计算置信度 score → 选择 top 1 或融合多个

---

## 7) "SWITCH 2025" 或多粒度/多数据集联合训练案例

### 查找结果

虽然没找到特指"SWITCH 2025"的论文，但找到了多个 2023–2025 年的相关工作：

- 多数据集联合训练 / multi-granularity supervision
- MultiTalent
- "Overcoming data scarcity in biomedical imaging with a multi-task strategy"
- 2024–2025 年一系列以 SAM/MedSAM 为基础的 weak/semi-supervised 医学分割论文

这些都可被视为与"多粒度监督"思路同类/可借鉴的工作。

### 补充信息

如果你确信"SWITCH 2025"是特指某篇论文或会议投稿，请提供完整题目或作者，可以直接检索并给出精确引用。

---

## 直接可运行的下一步（5 个立刻可做的动作）

### 1. 读 2 篇必读论文（各花 10–20 分钟）

- **SAM** (Kirillov et al., 2023)  
  [CVF开放获取](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)

- **MedSAM** (Ma et al., Nature Commun., 2024)  
  在 [bowang-lab/MedSAM](https://github.com/bowang-lab/MedSAM) 看 README 与获取 checkpoint

### 2. 克隆／跑通代码（30–90 分钟）

```bash
git clone https://github.com/bowang-lab/MedSAM.git
# 跑 demo（先在一台 GPU 上试用 checkpoint）
```

### 3. 准备文本 Prompt 编码（30 分钟）

- 把 5 条 LLM 生成的 prompt 用 ClinicalBERT / BioClinicalBERT 编码
- 保存为 `.pt` 供训练时 fast load

### 4. 搭建 PEFT 框架（1–2 小时）

- 学习 Visual Prompt Tuning / LoRA 论文与 github 示例
  - [VPT ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf)

### 5. 实现伪标签流水线（自动化）（2–4 小时）

对每张无标 CT：

- 用 5 条 prompt 逐个推理 → 得到 5 个 mask
- 计算置信度 score（例如 `mean sigmoid × log(area+1)`）
- 选择 top 1 作为伪标签（或设置阈值只保留高置信样本）
- 把伪标签和原始精标混合训练（强监督与伪监督分阶段/分权重）
- 参考 weak/self-training 文献

---

## 我为你挑的 5 篇 最重要参考（强烈建议先看）

### 核心模型与方法

1. **Kirillov A. et al.** - "Segment Anything" (ICCV 2023 / arXiv)  
   *（SAM 基石）*  
   [CVF开放获取](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)

2. **Ma J. et al.** - "Segment anything in medical images" (Nature Communications, 2024)  
   *（MedSAM 官方论文）*  
   [Nature](https://www.nature.com/articles/s41467-024-44824-z)

3. **Radford A. et al.** - "Learning Transferable Visual Models From Natural Language Supervision / CLIP" (2021)  
   [PMLR](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf)

4. **Jia M. et al.** - "Visual Prompt Tuning" (ECCV 2022)  
   *（VPT，PEFT 的重要方法）*  
   [ECVA PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136930696.pdf)

5. **He X. et al.** - "Parameter-Efficient Model Adaptation for Vision Transformers" (AAAI 2023)  
   *（Adapters/PEFT 在 ViT 的系统性讨论）*  
   [AAAI Open Access](https://ojs.aaai.org/index.php/AAAI/article/view/25160/24932)

### 综述与理论框架

- **Li 等, 2025** - "A Survey on Prompt Tuning"  
  将提示调优方法分为"直接学习"和"迁移学习"两大类   
  [arxiv.org](https://arxiv.org/html/2507.06085v1)

- **Vatsal 等, 2024** - 提示工程方法综述  
  覆盖44篇论文中的39种提示策略  
  [arXiv](https://arxiv.org/abs/2407.12994)

- **Lou 等, 2024** - "Large Language Model Instruction Following: A Survey"  
  系统总结了指令跟随相关研究  
  [arXiv](https://arxiv.org/abs/2303.10475)

- **Zhang 等, 2025** - "Instruction Tuning for LLMs: A Survey"  
  综述了监督微调/指令微调领域的研究现状  
  [arXiv](https://arxiv.org/abs/2308.10792)

### 自动提示工程与优化

- **Wang 等, 2025** - 序贸最优学习  
  基于知识梯度的序贸优化框架来自动生成高质量提示  
  [arXiv](https://arxiv.org/html/2501.03508v1)

- **Zhang 等, 2024** - "APE: 自动提示改进"  
  针对检索排序任务的 APEE 算法  
  [arXiv](https://arxiv.org/html/2406.14449v1)

- **Mihindukulasooriya 等, 2024** - 知识图谱提示优化  
  知识图谱三元组抽取任务的自动提示优化实证研究  
  [arXiv](https://arxiv.org/html/2506.19773v1)

---

## 后续建议

- 定期跟踪 arXiv 上 medical image segmentation + prompt / LLM 的最新论文
- 关注 MICCAI、ICCV、ECCV、NeurIPS 会议上的相关工作
- 在 GitHub 上 star MedSAM、SAM 及相关 PEFT 库，及时获取更新
- 根据P100的显存限制，优先从 VPT 开始原型验证

