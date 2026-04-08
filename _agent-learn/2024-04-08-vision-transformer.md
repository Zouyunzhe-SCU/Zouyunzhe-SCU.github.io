---
title: "Vision Transformer (ViT) 详解"
collection: agent-learn
permalink: /agent-learn/vision-transformer-explained
date: 2024-04-08
tags:
  - ComputerVision
  - DeepLearning
  - Transformer
excerpt: '从零开始理解Vision Transformer的架构原理、自注意力机制在图像领域的应用，以及相比CNN的优势。'
---

## Vision Transformer 概述

Vision Transformer (ViT) 是 Google 在 2020 年提出的创新架构，首次将纯 Transformer 应用到计算机视觉任务中。

### 核心创新点

**1. Patch Embedding**
- 将图像分割为 16×16 的 patch（图像块）
- 每个 patch 展平后通过线性映射转为 embedding
- 类似 NLP 中的 tokenization

**2. Position Encoding**
- 添加位置编码保留空间信息
- 可学习的位置向量

**3. Transformer Encoder**
- 多层 Multi-Head Self-Attention
- Feed-Forward Network

### vs CNN 的优势

| 特性 | CNN | ViT |
|------|-----|-----|
| 感受野 | 局部→全局 | 全局注意力 |
| 参数联系 | 卷积局部 | 自注意力全局 |
| 可扩展性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 数据需求 | 小数据可用 | 需要大数据 |

## 实现要点

```python
class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=1000):
        # Patch embedding
        self.patch_embed = PatchEmbeding(image_size, patch_size)
        # Transformer encoder
        self.transformer = TransformerEncoder()
        # Classification head
        self.cls_head = nn.Linear(768, num_classes)
    
    def forward(self, x):
        x = self.patch_embed(x)  # (B, 196, 768)
        x = self.transformer(x)
        x = x[:, 0]  # CLS token
        x = self.cls_head(x)
        return x
```

## 医学图像应用

在 CT/MRI 图像分析中，ViT 展现出优势：
- 全局上下文理解
- 灵活的长距离依赖
- 更好的泛化性能

## 参考资源
- 原论文：An Image is Worth 16×16 Words
- PyTorch 实现：timm 库
- 医学影像应用：TransUNet
