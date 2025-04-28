from model import objectives
from .clip_model import Transformer, QuickGELU, LayerNorm, convert_weights
import torch
import torch.nn as nn
from collections import OrderedDict

import alpha_clip
import torch.nn.functional as F
import sys

class OCDLPer(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        if args.pretrain_choice == 'ViT-L/14':
            self.base_model, _ = alpha_clip.load("ViT-L/14", alpha_vision_ckpt_pth=args.alpha_ckpt, device="cuda", image_size=args.img_size, stride_size=args.stride_size, 
                                                 num_cls=args.num_cls, vision_mask=args.multi_cls_mask)
            self.embed_dim = 768
        else:
            self.base_model, _ = alpha_clip.load("ViT-B/16", alpha_vision_ckpt_pth=args.alpha_ckpt, device="cuda", image_size=args.img_size, stride_size=args.stride_size, 
                                                 num_cls=args.num_cls, vision_mask=args.multi_cls_mask)    
            self.embed_dim = 512
            
        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
            
        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = Transformer(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5
            
            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            # init cross attn
            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            # init mlm head
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)
            
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image, alpha):
        image_feats = self.base_model.visual(image.half(), alpha.half(), return_attn=False)
        i_feats = image_feats[:, 0:self.args.num_cls, :].float().mean(dim=1)
        return i_feats

    def encode_text(self, text):
        x = self.base_model.encode_text(text)
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()
        
        images = batch['images']
        caption_ids = batch['caption_ids']
        alphas = batch['alphas']
        binary_mask = batch['binary_mask'].half()
        
        # compute image features
        image_feats = self.base_model.visual(images.half(), alphas.half(), return_attn=False)
        i_feats = image_feats[:, 0:self.args.num_cls, :]
        i_patches = image_feats[:, self.args.num_cls:, :]
        
        # compute textual features
        text_feats = self.base_model.encode_text(caption_ids)
        t_feats=text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)]
        bs = image_feats.shape[0]

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})
        
        if 'sadm' in self.current_task:
            sadm_loss = objectives.compute_sadm(i_feats, t_feats, batch['pids'], logit_scale, image_id=batch['image_ids'], factor=0.8)
            ret.update({'sadm_loss': sadm_loss})
            
        
        if 'id' in self.current_task:
            image_logits = self.classifier((i_feats.mean(dim=1)).half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})
        

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})
        

        return ret


def build_model_OCDL(args, num_classes=11003):
    model = OCDLPer(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
