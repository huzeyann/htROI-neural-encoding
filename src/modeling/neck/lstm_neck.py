from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.config import get_cfg_defaults
from src.modeling.components.pyramidpoolingnd import SpatialPyramidPoolingND, SpatialPyramidInterpolationND

from src.modeling.components.fc import build_fc, FcFusion


class LSTMBlock(nn.Module):

    def __init__(self, input_size, hidden_size=2048, num_layers=1, bidirectional=True):
        super().__init__()

        # will introduce 1 extra layer, but reduce 4x parameters
        self.dim_reduction = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=False)
        self.output_h_size = hidden_size * num_layers * (2 if bidirectional else 1)

    def forward(self, x):
        # will introduce 1 extra layer, but reduce 4x parameters
        t = x.shape[1]
        x = rearrange(x, 'b t d -> (b t) d')
        x = self.dim_reduction(x)
        x = rearrange(x, '(b t) d -> t b d', t=t)

        # x = rearrange(x, 'b t d -> t b d')
        output, (hidden, cell) = self.lstm(x)
        hidden = rearrange(hidden, 'd b h -> b (d h)')
        return hidden


class ThreeDPoolingWarp(nn.Module):
    def __init__(self, spp_level, mode='avg'):
        super().__init__()

        self.spp = SpatialPyramidPoolingND(spp_level, mode=mode)

    def forward(self, x):
        # x.shape (b c t h w)
        batch_size = x.shape[0]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.spp(x)
        x = rearrange(x, '(b t) d -> b t d', b=batch_size)

        return x


class TwoDLSTMNeck(nn.Module):

    def __init__(self, cfg=get_cfg_defaults()):
        super().__init__()
        self.cfg = cfg
        channel_dict = {
            '2d_simclr_warp_3d': [256, 512, 1024, 2048],
            '2d_densnet_warp_3d': [128, 256, 640, 1664],
            '2d_pyconvsegnet_warp_3d': [256, 512, 1024, 2048],
            '2d_bdcnvgg_warp_3d': [1, 1, 1, 1],  # edge prediction at different level
            '2d_moby_swin_warp_3d': [192, 384, 768, 768],
            '2d_seg_swin_warp_3d': [128, 256, 512, 1024],
            '2d_colorizer_warp_3d': [256, 512, 512, 128],
        }

        self.c_dict = {f'x{i + 1}': c for i, c in enumerate(channel_dict[self.cfg.MODEL.BACKBONE.NAME])}
        self.pyramid_layers = [xi for xi in self.cfg.MODEL.BACKBONE.LAYERS]  # x1,x2,x3,x4
        self.pyramid_layers.sort()
        self.pathways = self.cfg.MODEL.BACKBONE.LAYER_PATHWAYS.split(
            ',')  # ['topdown', 'bottomup'] aka 'parallel', or "none"
        self.is_pyramid = False if self.pathways[0] == 'none' else True
        self.output_size = len(torch.load(Path.joinpath(Path(self.cfg.DATASET.VOXEL_INDEX_DIR),
                                                        Path(self.cfg.DATASET.ROI + '.pt'))))
        self.spp_level = [(i, i) for i in self.cfg.MODEL.NECK.SPP_LEVELS]  # this will not pool time dimension

        self.planes = {k: np.min(
            [v, self.cfg.MODEL.NECK.FIRST_CONV_SIZE]) if not self.is_pyramid else self.cfg.MODEL.NECK.FIRST_CONV_SIZE
                       for k, v in self.c_dict.items()}

        if self.is_pyramid:
            assert len(self.pathways) >= 1 and self.pathways[0] != 'none'

        self.first_convs = nn.ModuleDict()
        self.smooths = nn.ModuleDict()
        self.poolings = nn.ModuleDict()
        self.lstm_input_dims = {}
        self.ch_response = nn.ModuleDict()

        self.num_chs = len(self.pyramid_layers) * len(self.pathways)

        for x_i in self.pyramid_layers:
            for pathway in self.pathways:
                k = f'{pathway}_{x_i}'

                # reduce conv channel dimension
                self.first_convs.update(
                    {k: nn.Conv3d(self.c_dict[x_i], self.planes[x_i], kernel_size=1, stride=1)})

                # optional pathways
                if self.is_pyramid:
                    self.smooths.update(
                        {k: nn.Conv3d(self.planes[x_i], self.planes[x_i], kernel_size=(1, 3, 3), stride=(1, 1, 1),
                                      padding='same')})

                # SPP
                pooling = ThreeDPoolingWarp(self.spp_level, self.cfg.MODEL.NECK.POOLING_MODE)
                self.poolings.update({k: pooling})
                self.lstm_input_dims.update({k: pooling.spp.get_output_size(self.planes[x_i])})

                # LSTM
                lstm = LSTMBlock(input_size=self.lstm_input_dims[k],
                                 hidden_size=self.cfg.MODEL.NECK.LSTM.HIDDEN_SIZE,
                                 num_layers=self.cfg.MODEL.NECK.LSTM.NUM_LAYERS,
                                 bidirectional=self.cfg.MODEL.NECK.LSTM.BIDIRECTIONAL)
                self.ch_response.update({k: lstm})

        self.final_fusions = FcFusion(fusion_type='concat')
        in_size = sum([lstm.output_h_size for lstm in self.ch_response.values()], 0)
        self.final_fc = build_fc(self.cfg, in_size, self.output_size)

    def forward(self, x):

        # x.shape (b c t h w)
        out = {}
        for x_i in self.pyramid_layers:
            for pathway in self.pathways:
                k = f'{pathway}_{x_i}'
                out[k] = x[x_i].clone()
        x = out
        x = {k: self.first_convs[k](v) for k, v in x.items()}
        if self.is_pyramid:
            x = self.pyramid_pathway(x, self.pyramid_layers, self.pathways)
        x = {k: self.poolings[k](v) for k, v in x.items()}
        x = {k: self.ch_response[k](v) for k, v in x.items()}

        x = list(x.values())

        x = self.final_fusions(x)
        out = self.final_fc(x)

        return out

    def pyramid_pathway(self, x, layers, pathways):
        """
        add pathway between layers, for multi-layer modeling
        """
        for pathway in pathways:

            if pathway == 'bottomup':
                layers_iter = layers
            elif pathway == 'topdown':
                layers_iter = reversed(layers)
            elif pathway == 'none':
                continue
            else:
                NotImplementedError()

            for i, x_i in enumerate(layers_iter):
                k = f'{pathway}_{x_i}'
                if i == 0:
                    pass
                else:
                    x[k] = self.resample_and_add(prev, x[k])
                x[k] = self.smooths[k](x[k])
                prev = x[k]
        return x

    @staticmethod
    def resample_and_add(x, y):
        target_shape = y.shape[2:]
        out = F.interpolate(x, size=target_shape, mode='nearest')
        return out + y
