import logging
import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler, RandomIdnumSampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP
from torch.utils.data.distributed import DistributedSampler
from PIL import ImageFilter
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset, ImageTextMLMDataset, OCDLImageDataset, OCDLDataset

from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid

import random
import numpy as np

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform

    # transform for training
    if aug:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            # T.RandomErasing(scale=(0.02, 0.4), value=mean),
        ])
    else:
        transform = T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    return transform

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def build_transforms_ocdl(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize(img_size, interpolation=BICUBIC),
            T.CenterCrop(img_size),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        mask_transform = T.Compose([
            T.ToTensor(), 
            T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
            T.Normalize(0.5, 0.26)
        ])
        return transform, mask_transform

    # transform for training
    else:
        transform = T.Compose([
            T.Resize(img_size, interpolation=BICUBIC),
            T.CenterCrop(img_size),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        mask_transform = T.Compose([
                T.ToTensor(), 
                T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
                T.Normalize(0.5, 0.26)
            ])
        return transform, mask_transform


def build_transforms_ocdl_com(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        transform = T.Compose([
            T.Resize(img_size, interpolation=BICUBIC),
            T.CenterCrop(img_size),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        
        mask_transform = T.Compose([
            T.ToTensor(), 
            T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
            T.Normalize(0.5, 0.26)
        ])
        
        return transform, mask_transform
    

    # transform for training
    else:
        if aug:
            transform = T.Compose([
            T.Resize(img_size, interpolation=BICUBIC),
            T.CenterCrop(img_size),
            _convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(scale=(0.02, 0.4), value=mean)
            ])

            mask_transform = T.Compose([
                    T.ToTensor(), 
                    T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
                    T.Normalize(0.5, 0.26)
                ])
            
            bin_mask_transform = T.Compose([
                    T.ToTensor(), 
                    T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
                ])
            
            com_transform = Compose([
                Resize((height, width)),
                RandomHorizontalFlip(0.5),
                Pad(10),
                RandomCrop((height, width))
            ])
            return transform, mask_transform, com_transform, bin_mask_transform
        else:
            transform = T.Compose([
                T.Resize(img_size, interpolation=BICUBIC),
                T.CenterCrop(img_size),
                _convert_image_to_rgb,
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

            mask_transform = T.Compose([
                    T.ToTensor(), 
                    T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
                    T.Normalize(0.5, 0.26)
                ])
            
            bin_mask_transform = T.Compose([
                    T.ToTensor(), 
                    T.Resize(img_size), # change to (336,336) when using ViT-L/14@336px
                ])
            
            com_transform = Compose([
                Resize((height, width))
            ])
        return transform, mask_transform, com_transform, bin_mask_transform

def collate(batch):
    keys = set([key for b in batch for key in b.keys()])
    # turn list of dicts data structure to dict of lists data structure
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        if isinstance(v[0], int):
            batch_tensor_dict.update({k: torch.tensor(v)})
        elif torch.is_tensor(v[0]):
             batch_tensor_dict.update({k: torch.stack(v)})
        else:
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")

    return batch_tensor_dict

def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("OCDL.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    
    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            train_set = ImageTextDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)

            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
                
        elif args.sampler == 'idrannum':
            train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdnumSampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
            
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms = build_transforms(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_img_loader, test_txt_loader, num_classes

def build_dataloader_ocdl(args, tranforms=None):
    logger = logging.getLogger("OCDL.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)
    if args.dataset_name == 'ALLREID' or args.dataset_name == 'CUHK-PEDES':
        num_classes = 11003
    else:
        num_classes = 11003
    
    if args.training:
        train_transforms, train_mask_transforms, com_transforms, bin_mask_transform = build_transforms_ocdl_com(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms, val_mask_transforms = build_transforms_ocdl(img_size=args.img_size,
                                          is_train=False)

        if args.MLM:
            train_set = ImageTextMLMDataset(dataset.train,
                                     train_transforms,
                                     text_length=args.text_length)
        else:
            train_set = OCDLDataset(dataset.train,
                                     train_transforms,
                                     train_mask_transforms,
                                     com_transforms,
                                     bin_mask_transform,
                                     text_length=args.text_length)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')
                mini_batch_size = args.batch_size // get_world_size()
                # TODO wait to fix bugs
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = torch.utils.data.sampler.BatchSampler(
                    data_sampler, mini_batch_size, True)
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        elif args.sampler == 'random':
            # TODO add distributed condition
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate)
            
        elif args.sampler == 'idrannum':
            train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdnumSampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate)
        else:
            logger.error('unsupported sampler! expected softmax or triplet but got {}'.format(args.sampler))

        # use test set as validate set
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_ocdl_set = OCDLImageDataset(ds['image_pids'], ds['img_paths'], ds['alpha_paths'],
                                    val_transforms, val_mask_transforms)
        val_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

       
        val_ocdl_loader = DataLoader(val_ocdl_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        val_txt_loader = DataLoader(val_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)

        return train_loader, val_ocdl_loader, val_txt_loader, num_classes

    else:
        # build dataloader for testing
        if tranforms:
            test_transforms = tranforms
        else:
            test_transforms, test_mask_transforms = build_transforms_ocdl(img_size=args.img_size,
                                               is_train=False)

        ds = dataset.test
        test_ocdl_set = OCDLImageDataset(ds['image_pids'], ds['img_paths'], ds['alpha_paths'],
                                    test_transforms, test_mask_transforms)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)
        test_ocdl_loader = DataLoader(test_ocdl_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers)
        return test_ocdl_loader, test_txt_loader, num_classes
