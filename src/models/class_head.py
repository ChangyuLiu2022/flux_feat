# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
#from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from ..utils.download import find_model
from torch.nn import Conv2d, Dropout
from functools import reduce
from operator import mul
from .mlp import MLP
from .attention_fusion import AttentionFusion, ConvFusion
from einops import rearrange
import sys



import time

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)



class classifier(nn.Module):

    def __init__(
        self, cfg, load_pretrain=True, vis=False,
    ):
        super().__init__()   
        #self.weight_dtype = torch.float16
        self.weight_dtype = torch.float16
        self.cfg = cfg
        if self.cfg.MODEL.FUSION_TYPE == "conv":
            self.merge_module = ConvFusion(cfg)
        elif self.cfg.MODEL.FUSION_TYPE == "linear":
            raise ValueError("Fusion type not supported")
        elif self.cfg.MODEL.FUSION_TYPE == "attention":
            self.merge_module = AttentionFusion(cfg)
        else:
            raise ValueError("Fusion type not supported")
        
        #add downstream head
        self.setup_head(cfg)

        
        
    def setup_head(self, cfg):
        self.head = MLP(
            # input_dim= self.encoder.x_embedder.num_patches,
            # mlp_dims=[self.encoder.x_embedder.num_patches * 4] * self.cfg.MODEL.MLP_NUM + \
            #     [cfg.DATA.NUMBER_CLASSES], # noqa
            input_dim= self.merge_module.feature_dims,  
            mlp_dims=[self.merge_module.feature_dims * 4] * self.cfg.MODEL.MLP_NUM + \
                [cfg.DATA.NUMBER_CLASSES], # noqa
            special_bias=True
        )


    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            self.head.train()
            self.merge_module.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def eval(self):
        self.head.eval()
        self.merge_module.eval()

    def downstream_head(self, x):
        x = self.head(x)
        return x

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False
    
    def merge_features(self, features):
        features = self.merge_module(features)
        return  features



    def forward(self, 
                features_list
                ):
        features_list = self.flatten_list(features_list)
        for i in range(len(features_list)):
            features_list[i] = torch.randn_like(features_list[i]).to(features_list[i].device)
        features = self.merge_features(features_list)
        result = self.head(features)
        return result

    def save_featues(self, features_list, timesteps_list,  path="./features"):
        for idx, timestep in enumerate(timesteps_list):
            features_dict = features_list[idx]
            for key, features in features_dict.items():
                features = torch.cat(features, dim=0)
                features = features.cpu().detach().numpy()
                np.save("/home/local/ASUAD/changyu2/prompt_DiT/features/base_feature_SD3/"+ f"t_{int(timestep.item())}_{key}.npy", features)
    
    def flatten_list(self, features_list):
        features_list_flatten = []
        for feature_t in features_list:
            for key, features in feature_t.items():
                features_list_flatten.extend(features)
        features_list_flatten = [feature.to(torch.float32) for feature in features_list_flatten]
        return features_list_flatten
    
