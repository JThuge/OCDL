#!/bin/bash
YOUR_DATA_ROOT="data"
DATASET_NAME="CUHK-PEDES"

CUDA_VISIBLE_DEVICES=0 \
python train_ocdl.py \
--root_dir $YOUR_DATA_ROOT \
--name OCDL \
--batch_size 128 \
--dataset_name $DATASET_NAME \
--loss_names 'sadm+id' \
--img_aug \
--lr 1e-5 \
--num_epoch 60 \
--pretrain_choice 'ViT-B/16' \
--sampler 'identity' \
--num_cls 4
