# Object-Centric Discriminative Learning for Text-Based Person Retrieval (OCDL)
Pytorch implementation of the ICASSP 2025 paper "Object-Centric Discriminative Learning for Text-Based Person Retrieval" [Paper](https://ieeexplore.ieee.org/document/10887901)\
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/JThuge/OCDL/blob/main/LICENSE)\

## Highlights
In this paper, we propose a novel framework for text-based person retrieval, termed Object-Centric Discriminative Learning (OCDL), which incorporates person masks to indicate attentive regions, thereby enhancing the model’s focus on the pedestrians in images while suppressing the background noise. Additionally, a novel cross-modal matching loss, namely Soft Angular Distribution Matching (SADM), is introduced to learn discriminative visual and textual representations. Extensive experiments on three widely-used TBPR datasets demonstrate the effectiveness of our approach.

## Usage
### Requirements
We use a single NVIDIA A100 GPU for training and evaluation.
```
pytorch 1.9.0
torchvision 0.10.0
prettytable
easydict
```
### Prepare Datasets
