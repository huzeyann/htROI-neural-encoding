from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_cfg_defaults
from src.modeling.components.pyramidpoolingnd import SpatialPyramidPoolingND

from src.modeling.components.fc import build_fc, FcFusion
from src.modeling.backbone.build import CHANNEL_DICT


class AudioNeck(nn.Module):

    def __init__(self, cfg=get_cfg_defaults()):
        super().__init__()
        self.cfg = cfg

        self.c_dict = CHANNEL_DICT[self.cfg.MODEL.BACKBONE.NAME]

        self.planes = {k: np.min([v, self.cfg.MODEL.NECK.FIRST_CONV_SIZE]) for k, v in self.c_dict.items()}
        self.pyramid_layers = [xi for xi in self.cfg.MODEL.BACKBONE.LAYERS]  # x1,x2,x3,x4
        self.pyramid_layers.sort()
        self.pathways = self.cfg.MODEL.BACKBONE.LAYER_PATHWAYS.split(
            ',')  # ['topdown', 'bottomup'] aka 'parallel', or "none"
        self.is_pyramid = False if self.pathways[0] == 'none' else True
        assert self.is_pyramid == False
        self.output_size = len(torch.load(Path.joinpath(Path(self.cfg.DATASET.VOXEL_INDEX_DIR),
                                                        Path(self.cfg.DATASET.ROI + '.pt'))))
        self.spp_level = [(i, i) for i in self.cfg.MODEL.NECK.SPP_LEVELS]

        self.first_convs = nn.ModuleDict()
        self.poolings = nn.ModuleDict()
        self.fc_input_dims = {}
        self.ch_response = nn.ModuleDict()

        self.num_chs = len(self.pyramid_layers) * len(self.pathways)

        for x_i in self.pyramid_layers:
            k = f'{x_i}'

            if x_i != 'x_label':

                # reduce conv channel dimension
                self.first_convs.update(
                    {k: nn.Conv2d(self.c_dict[x_i], self.planes[x_i], kernel_size=1, stride=1)})

                # SPP
                spp = SpatialPyramidPoolingND(self.spp_level, mode=self.cfg.MODEL.NECK.POOLING_MODE)
                self.poolings.update({k: spp})
                self.fc_input_dims.update({k: spp.get_output_size(self.planes[x_i])})

                # FC first part
                self.ch_response.update({k: build_fc(
                    self.cfg, self.fc_input_dims[k], self.output_size, part='first')})

            else:
                # x_label
                self.first_convs.update({k: nn.Flatten()})
                self.poolings.update({k: nn.Flatten()})
                self.ch_response.update({k: build_fc(
                    self.cfg, self.c_dict['x_label'], self.output_size, part='first')})

        self.final_fusions = FcFusion(fusion_type='concat')
        in_size = self.cfg.MODEL.NECK.FC_HIDDEN_DIM * self.num_chs
        self.final_fc = build_fc(self.cfg, in_size, self.output_size)

    def forward(self, x):
        x = {k: v for k, v in x.items() if k in self.pyramid_layers}
        # x: dict
        x = {k: self.first_convs[k](v) for k, v in x.items()}
        x = {k: self.poolings[k](v) for k, v in x.items()}
        x = {k: self.ch_response[k](v) for k, v in x.items()}

        x = list(x.values())

        x = self.final_fusions(x)
        out = self.final_fc(x)

        return out
