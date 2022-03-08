# source: https://github.com/SwinTransformer/Transformer-SSL

# backbone download link:
download_url = 'https://drive.google.com/u/0/uc?id=1PS1Q0tAnUfBWLRPxh9iUrinAxeq7Y--u&export=download'

import os

import numpy as np
import torch
from einops import rearrange

from .config import get_default_config
from .models.build import build_model


def modify_swin_partial(model, layers=('x1', 'x2', 'x3', 'x4')):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        ret_dict = {}
        for i, layer in enumerate(self.layers):
            # drop layers to run faster
            if i + 1 > max_depth:
                break

            x = layer(x)

            if i + 1 == 4:
                x = self.norm(x)  # B L C

            h = int(np.sqrt(x.shape[1]))
            x_out = rearrange(x, 'b (h w) c -> b c h w', h=h)
            ret_dict[f'x{i + 1}'] = x_out

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def get_moby_swin_t_2d(pretrained=True, cache_dir='~/.cache/'):
    c = get_default_config()
    c.MODEL.TYPE = 'moby'
    c.MODEL.NAME = 'moby__swin_tiny__patch4_window7_224__odpr02_tdpr0_cm099_ct02_queue4096_proj2_pred2'
    c.MODEL.SWIN.EMBED_DIM = 96
    c.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    c.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    c.MODEL.SWIN.WINDOW_SIZE = 7
    c.MODEL.MOBY.ENCODER = 'swin'
    c.MODEL.MOBY.ONLINE_DROP_PATH_RATE = 0.2
    c.MODEL.MOBY.TARGET_DROP_PATH_RATE = 0.
    c.MODEL.MOBY.CONTRAST_MOMENTUM = 0.99
    c.MODEL.MOBY.CONTRAST_TEMPERATURE = 0.2
    c.MODEL.MOBY.CONTRAST_NUM_NEGATIVE = 4096
    c.MODEL.MOBY.PROJ_NUM_LAYERS = 2
    c.MODEL.MOBY.PRED_NUM_LAYERS = 2

    # c.DATA.TRAINING_IMAGES = 233

    model = build_model(c)

    if pretrained:
        cache_dir = os.path.expanduser(cache_dir)
        path = os.path.join(cache_dir, 'moby_swin_t_300ep_pretrained.pth')
        if not os.path.exists(path):
            raise Exception(f'{path} not exists! please download from: {download_url}')
        state_dict = torch.load(path)
        model.load_state_dict(state_dict['model'])

    model = model.encoder

    return model
