from prettytable import PrettyTable
import os
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader, build_dataloader_ocdl
from processor.processor import do_inference, do_inference_ocdl
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model, build_model_OCDL
import argparse
from utils.iotools import load_train_configs

CONFIG = "" # Your config file -> e.g. 'logs/CUHK-PEDES/OCDL/configs.yaml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OCDL Test")
    parser.add_argument("--config_file", default=CONFIG)
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('OCDL', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.root_dir = "" # Your data root
    args.dataset_name = "CUHK-PEDES" # The dataset you want to evaluate on
    test_img_loader, test_txt_loader, num_classes = build_dataloader_ocdl(args)
    model = build_model_OCDL(args, num_classes=args.num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model.to(device)

    do_inference_ocdl(model, test_img_loader, test_txt_loader)