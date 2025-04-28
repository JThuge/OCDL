# Object-Centric Discriminative Learning for Text-Based Person Retrieval (OCDL)
Pytorch implementation of the ICASSP 2025 paper "Object-Centric Discriminative Learning for Text-Based Person Retrieval" [Paper](https://ieeexplore.ieee.org/document/10887901)

## Highlights
We propose a novel framework for text-based person retrieval, Object-Centric Discriminative Learning (OCDL), which incorporates person masks to indicate attentive regions, thereby enhancing the model’s focus on the pedestrians in images while suppressing the background noise. Additionally, a novel cross-modal matching loss, namely Soft Angular Distribution Matching (SADM), is introduced to learn discriminative visual and textual representations. Experiments on three widely-used TBPR datasets demonstrate the effectiveness of our approach.


![main](assets/main.png)

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
To be done

### Pretrained Weights
|   model   |                      google drive link                       |                        openxlab link                         |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| AlphaCLIP-B/16 | [clip_b16_grit1m+mim_fultune_4xe](https://drive.google.com/file/d/11iDlSAYI_BAi1A_Qz6LTWYHNgPe-UY7I/view?usp=sharing) | [clip_b16_grit1m+mim_fultune_4xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit+mim_fultune_4xe.pth) |
| AlphaCLIP-L/14 | [clip_l14_grit1m+mim_fultune_6xe](https://drive.google.com/file/d/1JfzOTvjf0tqBtKWwpBJtjYxdHi-06dbk/view?usp=sharing) | [clip_l14_grit1m+mim_fultune_6xe](https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit+mim_fultune_6xe.pth) |


## Training

```python
python train_ocdl.py \
--root_dir 'YOUR DATASET ROOT DIR' \
--name OCDL \
--batch_size 128 \
--dataset_name 'CUHK-PEDES' \
--loss_names 'sadm+id' \
--img_aug \
--lr 1e-5 \
--num_epoch 60 \
--pretrain_choice 'ViT-B/16' \
--sampler 'identity' \
--num_cls 4
```

## Acknowledgments
Some components of this code implementation are adapted from [CLIP](https://github.com/openai/CLIP), [IRRA](https://github.com/anosorae/IRRA) and [AlphaCLIP](https://github.com/SunzeY/AlphaCLIP). We sincerely appreciate for their contributions.

## Citation
If you find our work useful for your research, please cite our paper.

```tex
@inproceedings{li2025object,
  title={Object-Centric Discriminative Learning for Text-Based Person Retrieval},
  author={Li, Haiwen and Liu, Delong and Su, Fei and Zhao, Zhicheng},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## Contact
If you have any question, please contact us. E-mail: [lihaiwen@bupt.edu.cn](mailto:lihaiwen@bupt.edu.cn), [liudelong@bupt.edu.cn](mailto:liudelong@bupt.edu.cn).
