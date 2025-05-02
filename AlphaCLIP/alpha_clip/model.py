from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import loralib as lora
import math
import collections
from einops import rearrange


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1_alpha = nn.Conv2d(in_channels=1, out_channels=width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, alpha=None):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x) + self.conv1_alpha(alpha)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.,
            lora_adapt=False, 
            rank=16
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        if lora_adapt:
            print("!!!!!!!!!!using lora for qkv projection!!!!!!!!!!")
            self.in_proj = lora.MergedLinear(dim, 3*dim, r=rank, enable_lora=[True, False, True])
        else:
            self.in_proj = nn.Linear(dim, dim * 3)
        # self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        # if qkv_bias:
        #     self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        # else:
        #     self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim) if not lora_adapt else lora.Linear(dim, dim, r=rank)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask = None):
        L, N, C = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-2, -1))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask
            # print(attn.shape)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x, attn


class CustomResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, lora_adapt=False, rank=16):
        super().__init__()
        
        self.attn = Attention(d_model, n_head, lora_adapt=lora_adapt, rank=rank)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4) if not lora_adapt else lora.Linear(d_model, d_model*4, r=rank)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model) if not lora_adapt else lora.Linear(d_model*4, d_model, r=rank))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor, return_attn=False, attn_mask=None):
        if attn_mask is not None:
            self.attn_mask = attn_mask
        attn_out, attn = self.attention(self.ln_1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        if return_attn:
            return x, attn
        else:
            return x


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, return_attn: bool = False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if return_attn:
            return self.attn(x, x, x, attn_mask=self.attn_mask)
        else:
            return self.attn(x, x, x, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, return_attn=False):
        x = x + self.attention(self.ln_1(x), return_attn)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        
    def forward(self, x: torch.Tensor, return_attn: bool = False):
        if return_attn:
            for i, block in enumerate(self.resblocks):
                if i == self.layers - 1:
                    x, attn = block(x, return_attn=True)
                    attn = rearrange(attn, '(b h) l k -> b h l k', h=12)
                    return x, attn
                else:
                    x = block(x)
        else:
            return self.resblocks(x)


class CustomTransformer(nn.Module):
    
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, lora_adapt=False, rank=16, 
                 fine_grained=False, fg_mode='att', fg_tokens=1, num_cls=1, mul_lis=None, mask_mode='criss_cross'):
        super().__init__()
        self.width = width
        self.layers = layers
        
        self.mask = True if attn_mask is not None else False

        reslist = []
        for i in range(layers):
            if i == layers - 1 and fine_grained:
                reslist.append(CustomResidualAttentionBlock(width, heads, attn_mask=None, lora_adapt=lora_adapt, rank=rank))
            else:    
                reslist.append(CustomResidualAttentionBlock(width, heads, attn_mask, lora_adapt=lora_adapt, rank=rank))
        
        self.resblocks = nn.Sequential(*reslist)
        self.fine_grained = fine_grained
        self.fg_mode = fg_mode
        self.mul_lis = mul_lis
        self.fg_tokens = fg_tokens
        self.mask_mode = mask_mode
        self.num_cls = num_cls
        self.act_layer = 10
        
    def part_selection(self, attn, num_tokens, num_cls, batch_size):
        length = len(attn)
        last_map = attn[0]
        for i in range(1, length):
        # 这个attn map相乘是有物理意义的，5/9明白啦，自己思考
            last_map = torch.matmul(attn[i], last_map)
            
        max_inx = []
        for i in range(num_cls):
            # B,h,L-1 -> 选出每个头中对cls贡献最大的patch token
            cur_map = last_map[:, :, i, num_cls:]
            _, max_ind = cur_map.topk(dim=2, k=num_tokens) # B,h
            max_ind = max_ind.reshape(max_ind.size(0), -1)
            cls_index = (i * torch.ones([batch_size, 1])).long().to(attn[0].device)
            # print(max_ind.shape, cls_index.shape)
            max_inx.append(torch.cat([cls_index, max_ind+num_cls], dim=1))
        
        return max_inx
    
    def seq_selection(self, seq_len, num_cls, batch_size, device):
        seq_inx = []
        N = (seq_len-num_cls) // num_cls
        
        for i in range(num_cls):
            seq_ind = (torch.arange(i*N, (i+1)*N)).repeat(batch_size, 1).long().cuda()
            cls_index = (i * torch.ones([batch_size, 1])).long().to(device)
            seq_inx.append(torch.cat([cls_index, seq_ind+num_cls], dim=1))
        
        return seq_inx
    
    def mul_selection(self, attn, num_cls, batch_size, seq_len):
        length = len(attn)
        last_map = attn[0]
        head = attn[0].shape[1]
        device = attn[0].device
        for i in range(1, length):
        # 这个attn map相乘是有物理意义的，5/9明白啦，自己思考
            last_map = torch.matmul(attn[i], last_map)
            
        mul_inx = []
        for i in range(num_cls):
            if i == 0:
                assert self.mul_lis[i] == 1.0 and len(self.mul_lis) == num_cls, "The first token must have a whole view and its length must match num_cls"
                mul_ind = (torch.arange(num_cls, seq_len)).repeat(batch_size, 1).long().to(device)
                cls_index = (torch.zeros([batch_size, 1])).long().to(device)
                mul_inx.append(torch.cat([cls_index, mul_ind], dim=1))
            else:
                # how to construct a non-overlapping index list
                num_tokens = int((self.mul_lis[i] * (seq_len - num_cls)) // head)
                # B,h,L-1 -> 选出每个头中对cls贡献最大的patch token
                cur_map = last_map[:, :, i, num_cls:]
                _, mul_ind = cur_map.topk(dim=2, k=num_tokens) # B,h,k
                mul_ind, _ = (mul_ind.reshape(mul_ind.size(0), -1)).sort(dim=1)
                cls_index = (i * torch.ones([batch_size, 1])).long().to(device)
                # print(max_ind.shape, cls_index.shape)
                mul_inx.append(torch.cat([cls_index, mul_ind+num_cls], dim=1))
        # torch.set_printoptions(profile="full")
        # print(mul_inx[3])
        return mul_inx
    
    def mask_selection(self, attn, num_cls, mode):
        length = len(attn)
        batch_size, head, seq_len, _ = attn[0].size()
        last_map = attn[0]
        for i in range(1, length):
        # 这个attn map相乘是有物理意义的，5/9明白啦，自己思考
            last_map = torch.matmul(attn[i], last_map) # B,h,L,L
        
        last_map = last_map.mean(dim=0) # h,L,L
        
        attn_mask = torch.zeros_like(last_map)
        if self.mask:
            for i in range(num_cls):
                for j in range(num_cls):
                    if i != j:
                        attn_mask[:, i, j] = float('-inf')
        assert self.mul_lis[0] == 1.0 and len(self.mul_lis) == num_cls, \
            "The first token must have a whole view and its length must match num_cls"
        
        for j in range(head):
            for i in range(1, num_cls):
                num_tokens = int((self.mul_lis[i] * (seq_len - num_cls)))
                _, ind = last_map[j, i, num_cls:].sort(descending=True) # [240]
                ind = ind + num_cls
                if mode == 'criss_cross':
                    attn_mask[j, i, ind[num_tokens:]] = float('-inf')
                    attn_mask[j, ind[num_tokens:], i] = float('-inf')
                elif mode == 'axis_row':
                    attn_mask[j, i, ind[num_tokens:]] = float('-inf')
                elif mode == 'axis_col':
                    attn_mask[j, ind[num_tokens:], i] = float('-inf')
                else:
                    raise ValueError("The mask selection mode {} is not implemented yet".format(mode))
                # torch.set_printoptions(profile='full')
                # print(attn_mask[0:4])
        return attn_mask.repeat(batch_size, 1, 1)

    def forward(self, x: torch.Tensor, return_attn=False):
        if self.fine_grained:
            if self.fg_mode == 'att':
                attn_lis = []
                for i, block in enumerate(self.resblocks):
                    if i == self.layers - 1:
                        _, B, _ = x.size()
                        # L,B,D -> B,L,D
                        x = x.permute(1, 0, 2)
                        index = self.part_selection(attn=attn_lis, num_tokens=self.fg_tokens, 
                                                    num_cls=self.num_cls, batch_size=B)
                        ret_lis = []
                        for i in range(len(index)):
                            new_x = []
                            for j in range(B):
                                new_x.append(x[j, index[i][j], :]) # h*k+1, D
                            # h*k+1,B,D
                            new_x = torch.stack(new_x).permute(1, 0, 2)
                            ret_lis.append(block(new_x)[0,:,:]) # B,D
                        res = torch.stack(ret_lis) # num_cls,B,D
                        return res
                    else:
                        x, attn = block(x, return_attn=True)
                        attn = rearrange(attn, '(b h) l k -> b h l k', h=12)
                        attn_lis.append(attn)
            elif self.fg_mode == 'seq':
                for i, block in enumerate(self.resblocks):
                    if i == self.layers - 1:
                        L, B, _ = x.size()
                        index = self.seq_selection(L, self.num_cls, B, device=x.device)
                        ret_lis = []
                        for i in range(len(index)):
                            new_x = []
                            for j in range(B):
                                new_x.append(x[index[i][j], j, :]) # hk+1,D
                            # hk+1,B,D
                            new_x = torch.stack(new_x, dim=1)
                            ret_lis.append(block(new_x)[0,:,:]) # B,D
                        # num_cls,B,D
                        res = torch.stack(ret_lis)
                        return res
                    else:
                        x = block(x, return_attn=False)
            elif self.fg_mode == 'mul':
                attn_lis = []
                for i, block in enumerate(self.resblocks):
                    if i == self.layers - 1:
                        L, B, _ = x.size()
                        # L,B,D->B,L,D
                        x = x.permute(1, 0, 2)
                        index = self.mul_selection(attn=attn_lis, num_cls=self.num_cls, batch_size=B, seq_len=L)
                        ret_lis = []
                        for i in range(len(index)):
                            new_x = []
                            for j in range(B):
                                new_x.append(x[j, index[i][j], :]) # h*k+1, D
                            # h*k+1,B,D
                            new_x = torch.stack(new_x).permute(1, 0, 2)
                            ret_lis.append(block(new_x)[0,:,:]) # B,D
                        res = torch.stack(ret_lis) # num_cls,B,D
                        return res
                    else:
                        x, attn = block(x, return_attn=True)
                        attn = rearrange(attn, '(b h) l k -> b h l k', h=12)
                        attn_lis.append(attn)
            elif self.fg_mode == 'mask':
                attn_lis = []
                for i, block in enumerate(self.resblocks):
                    if i > self.act_layer:
                        if i == self.layers - 1:
                            x, attn = block(x, return_attn=True)
                            attn = (attn.mean(dim=[0,1]))[0:self.num_cls].softmax(dim=-1) # _,_,L
                            return x[0:self.num_cls,:,:] * attn.unsqueeze(dim=1).unsqueeze(dim=2)
                        else:
                            x = block(x, return_attn=False)
                    else:
                        if i == self.act_layer:
                            batch_mask = self.mask_selection(attn=attn_lis, num_cls=self.num_cls, mode=self.mask_mode)
                            x = block(x, return_attn=False, attn_mask=batch_mask)
                            if i == self.layers - 1:
                                raise ValueError('Act Layer can not be the last layer')
                        else:
                            x, attn = block(x, return_attn=True)
                            attn = rearrange(attn, '(b h) l k -> b h l k', h=12)
                            attn_lis.append(attn)        
            else:
                raise ValueError("The fine-grained mode {} is not implemented yet".format(self.fg_mode))
        else:
            if return_attn:
                attn_lis = []
                for _, block in enumerate(self.resblocks):
                    x, attn = block(x, return_attn=True)
                    attn = rearrange(attn, '(b h) l k -> b h l k', h=12)
                    attn_lis.append(attn)  
                return x, attn_lis

            return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: Tuple[int, int], patch_size: int, stride_size: int, width: int, layers: int, 
                 heads: int, output_dim: int, lora_adapt=False, rank=16, num_cls=1, mask=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // stride_size + 1
        self.num_y = (input_resolution[0] - patch_size) // stride_size + 1
        self.num_patches = self.num_x * self.num_y
        
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)
        self.conv1_alpha = nn.Conv2d(in_channels=1, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5
        
        self.num_cls = num_cls
        self.class_embedding = nn.Parameter(scale * torch.randn(num_cls, width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.num_patches + self.num_cls, width))
        
        self.ln_pre = LayerNorm(width)
        
        attn_mask = None
        if mask:
            attn_mask = torch.zeros([self.num_patches + num_cls, self.num_patches + num_cls])
            for i in range(num_cls):
                for j in range(num_cls):
                    if i != j:
                        attn_mask[i, j] = float('-inf')
        # print(attn_mask.shape, attn_mask[0:num_cls, 0:num_cls])
        self.transformer = CustomTransformer(width, layers, heads, lora_adapt=lora_adapt, rank=rank, attn_mask=attn_mask, num_cls=num_cls)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        
    def forward(self, x: torch.Tensor, alpha=None, return_attn=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # ASSUME alpha is always not None!
        x = x + self.conv1_alpha(alpha)
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], self.num_cls, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        if return_attn:
            x, attn_last = self.transformer(x, return_attn=True)
        else:
            x = self.transformer(x, return_attn=False)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        if return_attn:
            return x, attn_last
        else:
            return x
        
    def load_param(self, state_dict):
        # 将pretrained_dict里不属于model_dict的键剔除掉
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_pos_embed(v, self.positional_embedding, self.num_y, self.num_x, self.num_cls)
            elif k == 'class_embedding':
                if all([self.num_cls > 1 and v.ndim == 2 and v.shape[0] == 1]):
                    v = v.repeat(self.class_embedding.shape[0], 1)
                    print('load cls embeddings')
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: Union[int, Tuple[int, int]], 
                 stride_size: int, 
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_mask: bool,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 lora_adapt = False,
                 rank = 16,
                 num_cls=1
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                stride_size=stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                lora_adapt=lora_adapt,
                rank=rank,
                num_cls = num_cls,
                mask = vision_mask
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        if not hasattr(self.visual, "conv1"):
            return self.visual.module.conv1.weight.dtype
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, alpha):
        assert alpha is not None
        return self.visual(image.type(self.dtype), alpha.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x @ self.text_projection
        return x

    def forward(self, image, text, alpha):
        image_features = self.encode_image(image, alpha)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

    def load_param(self, state_dict):
        # 将pretrained_dict里不属于model_dict的键剔除掉
        param_dict =  {k: v for k, v in state_dict.items() if k in self.state_dict()}

        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if k == 'visual.positional_embedding' and v.shape != self.visual.positional_embedding.shape:
                v = resize_pos_embed(v, self.visual.positional_embedding, self.visual.num_y, self.visual.num_x, self.visual.num_cls)
            elif k == 'positional_embedding' and v.shape != self.positional_embedding.shape:
                v = resize_text_pos_embed(v, self.context_length)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print(f'===========================ERROR occur in copy {k}, {v.shape}=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))
    

def resize_pos_embed(posemb, posemb_new, hight, width, num_cls):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    posemb = posemb.unsqueeze(0) # 1,N+1,D
    posemb_new = posemb_new.unsqueeze(0) # 1,N+num_cls,D

    # 1,1,D for cls 和 N,D for others
    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb_token_cls = posemb_token.repeat(1, num_cls, 1)
    posemb = torch.cat([posemb_token_cls, posemb_grid], dim=1)
    
    return posemb.squeeze(0)


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, image_size: Union[int, Tuple[int, int]], stride_size: int, lora_adapt=False, rank=16, 
                num_cls=1, vision_mask=False):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # always load lora version
    model = CLIP(
        embed_dim, 
        image_size, stride_size, vision_layers, vision_width, vision_patch_size, vision_mask,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        lora_adapt=lora_adapt, rank=rank, num_cls=num_cls
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    # para_wb to linear
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        if 'visual' in k:
            if 'in_proj_weight' in k:
                new_state_dict[k.replace('in_proj_weight', 'in_proj.weight')] = v
            elif 'in_proj_bias' in k:
                new_state_dict[k.replace('in_proj_bias', 'in_proj.bias')] = v
            else:
                new_state_dict[k] = v
        else:
            new_state_dict[k] = v
                
    state_dict = new_state_dict
    # add rgba_conv_weight
    if 'visual.conv1_alpha.weight' not in state_dict.keys(): # zero initialization on alpha channel
        rgb_weight = state_dict['visual.conv1.weight'].clone().detach()
        rgba_weigth = torch.zeros_like(rgb_weight)[:, 0:1, :, :]
        state_dict['visual.conv1_alpha.weight'] = rgba_weigth
    convert_weights(model)
    # model.load_state_dict(state_dict, strict=False)
    model.load_param(state_dict)
    return model
