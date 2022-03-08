import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce


class TwoDtoThreeDWarp(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x.shape [b, c, t, h, w]
        batch_size = x.shape[0]

        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x_dict = self.model(x)
        for k, v in x_dict.items():
            x_dict[k] = rearrange(v, '(b t) c h w -> b c t h w', b=batch_size)

        return x_dict
