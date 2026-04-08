---
title: "自注意力机制（Self-Attention）从零开始理解"
collection: agent-learn
layout: single
author_profile: true
permalink: /agent-learn/self-attention-explained
date: 2024-04-06
tags:
  - Transformer
  - DeepLearning
  - Attention
excerpt: '深入浅出讲解 Self-Attention 的核心概念、数学原理、与传统卷积的对比，以及在图像、文本中的应用。'
read_time: 9
share: true
comments: true
---

## 什么是 Self-Attention？

Self-Attention（自注意力）是 Transformer 的核心机制，用于**捕捉序列中任意两个位置之间的依赖关系**。

### 直观理解

> "当你看一张图片时，你会自动关注突出的部分。自注意力就是让模型学会这样的选择性关注。"

## 数学原理

### Query, Key, Value (Q, K, V)

```
输入向量序列: x₁, x₂, ..., xₙ
       ↓
创建 Q, K, V 向量:
- Query: Wq × x
- Key: Wk × x  
- Value: Wv × x
       ↓
计算注意力权重:
α_ij = softmax(Qᵢ · Kⱼ / √d_k)
       ↓
加权求和:
Output_i = Σ α_ij × Vⱼ
```

### 完整过程

```
输入: X (n × d)
    ↓
Q = XW^Q, K = XW^K, V = XW^V
    ↓
Attention(Q,K,V) = softmax(QK^T/√d_k)V
    ↓
输出: Z (n × d)
```

## 为什么是 √d_k？

- **缩放** - 防止注意力权重过于集中
- **稳定训练** - 避免梯度过小或过大
- `d_k` 是 Key 的维度

## PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # 线性变换
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # 1. 线性变换并分头
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # 3. 应用到值向量
        context = torch.matmul(attention_weights, V)
        
        # 4. 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.embed_dim)
        
        # 5. 最后的线性变换
        output = self.W_o(context)
        
        return output, attention_weights
```

## Self-Attention vs CNN

| 特性 | CNN | Self-Attention |
|------|-----|----------------|
| **感受野** | 局部固定 | 全局动态 |
| **参数** | 固定(filter size) | 动态(数据驱动) |
| **长距离依赖** | 需多层堆叠 | 一步到位 |
| **计算复杂度** | O(n) | O(n²) |
| **可解释性** | 中等 | 高(可视化注意力) |

## 应用领域

### NLP（自然语言处理）
- BERT, GPT 等语言模型
- 机器翻译
- 问答系统

### CV（计算机视觉）
- Vision Transformer (ViT)
- 医学图像分析
- 视频理解

### 多模态
- CLIP（图文匹配）
- Flamingo（多模态LLM）

## 训练技巧

1. **多头注意力** - 并行学习多个子空间
2. **Layer Norm** - 注意力前后的正则化
3. **Dropout** - 防止过拟合
4. **Warm-up** - 学习率预热

## 参考资源
- 论文: Attention is All You Need
- 博客: The Illustrated Transformer
