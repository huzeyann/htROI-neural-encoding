from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_cfg_defaults
from src.modeling.components.pyramidpoolingnd import SpatialPyramidPoolingND, SpatialPyramidInterpolationND

from src.modeling.components.fc import build_fc, FcFusion

class I3DNeck(nn.Module):

    def __init__(self, cfg=get_cfg_defaults()):
        super().__init__()
        self.cfg = cfg

        def roundup(x):
            return int(np.ceil(x))

        # video_size = self.cfg.DATASET.RESOLUTION
        # video_frames = self.cfg.DATASET.FRAMES
        if self.cfg.MODEL.BACKBONE.NAME == 'i3d_rgb':
            cs = [256, 512, 1024, 2048]
        elif self.cfg.MODEL.BACKBONE.NAME == 'i3d_flow':
            cs = [192, 480, 832, 1024]
        elif self.cfg.MODEL.BACKBONE.NAME == '3d_swin':
            cs = [256, 512, 1024, 1024]
        else:
            NotImplementedError()

        self.c_dict = {f'x{i+1}': c for i, c in enumerate(cs)}
        self.planes = self.cfg.MODEL.NECK.FIRST_CONV_SIZE
        self.pyramid_layers = [xi for xi in self.cfg.MODEL.BACKBONE.LAYERS]  # x1,x2,x3,x4
        self.pyramid_layers.sort()
        self.pathways = self.cfg.MODEL.BACKBONE.LAYER_PATHWAYS.split(
            ',')  # ['topdown', 'bottomup'] aka 'parallel', or "none"
        self.is_pyramid = False if self.pathways[0] == 'none' else True
        self.output_size = len(torch.load(Path.joinpath(Path(self.cfg.DATASET.VOXEL_INDEX_DIR),
                                                        Path(self.cfg.DATASET.ROI + '.pt'))))
        self.spp_level = [(1, i, i) for i in self.cfg.MODEL.NECK.SPP_LEVELS]

        if self.is_pyramid:
            assert len(self.pathways) >= 1 and self.pathways[0] != 'none'

        self.first_convs = nn.ModuleDict()
        self.smooths = nn.ModuleDict()
        self.poolings = nn.ModuleDict()
        self.fc_input_dims = {}
        self.ch_response = nn.ModuleDict()

        self.num_chs = len(self.pyramid_layers) * len(self.pathways)

        for x_i in self.pyramid_layers:
            for pathway in self.pathways:
                k = f'{pathway}_{x_i}'

                # reduce conv channel dimension
                self.first_convs.update(
                    {k: nn.Conv3d(self.c_dict[x_i], self.planes, kernel_size=1, stride=1)})

                # optional pathways
                if self.is_pyramid:
                    self.smooths.update(
                        {k: nn.Conv3d(self.planes, self.planes, kernel_size=3, stride=1, padding='same')})

                # SPP
                spp = SpatialPyramidPoolingND(self.spp_level, self.cfg.MODEL.NECK.POOLING_MODE)
                self.poolings.update({k: spp})
                self.fc_input_dims.update({k: spp.get_output_size(self.planes)})

                # FC first part
                self.ch_response.update({k: build_fc(
                    self.cfg, self.fc_input_dims[k], self.output_size, part='first')})

        self.final_fusions = FcFusion(fusion_type='concat')
        in_size = self.cfg.MODEL.NECK.FC_HIDDEN_DIM * self.num_chs
        self.final_fc = build_fc(self.cfg, in_size, self.output_size)

    def forward(self, x):

        # vid
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