# modified from: https://github.com/SwinTransformer/Video-Swin-Transformer

# backbone download links
download_url = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window1677_sthv2.pth'

import os

import numpy as np
import torch
from einops import rearrange

from .swin_transformer import SwinTransformer3D


def modify_swin_transformer_3d_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)

        ret_dict = {}
        for i, layer in enumerate(self.layers):
            # drop layers to run faster
            if i + 1 > max_depth:
                break

            x = layer(x.contiguous())

            if i + 1 == 4:
                x = rearrange(x, 'n c d h w -> n d h w c')
                x = self.norm(x)
                x = rearrange(x, 'n d h w c -> n c d h w')

            ret_dict[f'x{i + 1}'] = x

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def load_backbone(model, ckpt):
    backbone_dict = {k.replace('backbone.', ''): v for k, v in ckpt['state_dict'].items() if
                     k.split('.')[0] == 'backbone'}

    model_dict = model.state_dict()
    # 2. overwrite entries in the existing state dict
    model_dict.update(backbone_dict)
    # 3. load the new state dict
    model.load_state_dict(backbone_dict)

    return model


def sthv2_video_swin_b(pretrained=True, cache_dir='~/.cache/'):
    model = SwinTransformer3D(
        depths=[2, 2, 18, 2],
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.,
        patch_size=(2, 4, 4),
        window_size=(16, 7, 7),
        drop_path_rate=0.4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        patch_norm=True
    )
    if pretrained:
        stat_dict = torch.hub.load_state_dict_from_url(download_url)
        model = load_backbone(model, stat_dict)

    return model
