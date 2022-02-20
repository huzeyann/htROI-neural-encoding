from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_cfg_defaults
from src.modeling.components.pyramidpooling3d import SpatialPyramidPooling3D


class I3DNeck(nn.Module):

    def __init__(self, cfg=get_cfg_defaults()):
        super().__init__()
        self.cfg = cfg

        def roundup(x):
            return int(np.ceil(x))

        video_size = self.cfg.DATASET.RESOLUTION
        video_frames = self.cfg.DATASET.FRAMES
        if self.cfg.MODEL.BACKBONE.NAME == 'i3d_rgb':
            self.x1_twh = (roundup(video_frames / 2), roundup(video_size / 4), roundup(video_size / 4))
            self.x2_twh = tuple(map(lambda x: roundup(x / 2), self.x1_twh))
            self.x3_twh = tuple(map(lambda x: roundup(x / 2), self.x2_twh))
            self.x4_twh = tuple(map(lambda x: roundup(x / 2), self.x3_twh))
            self.x1_c, self.x2_c, self.x3_c, self.x4_c = 256, 512, 1024, 2048
        elif self.cfg.MODEL.BACKBONE.NAME == 'i3d_flow':
            self.x1_twh = (roundup(video_frames / 2), roundup(video_size / 8), roundup(video_size / 8))
            self.x2_twh = (roundup(self.x1_twh[0] / 2), roundup(self.x1_twh[1] / 2), roundup(self.x1_twh[2] / 2))
            self.x3_twh = (roundup(self.x2_twh[0] / 2), roundup(self.x2_twh[1] / 2), roundup(self.x2_twh[2] / 2))
            self.x4_twh = (roundup(self.x2_twh[0] / 2), roundup(self.x2_twh[1] / 2), roundup(self.x2_twh[2] / 2))
            self.x1_c, self.x2_c, self.x3_c, self.x4_c = 192, 480, 832, 1024
        else:
            NotImplementedError()

        self.twh_dict = {'x1': self.x1_twh, 'x2': self.x2_twh, 'x3': self.x3_twh, 'x4': self.x4_twh}
        self.c_dict = {'x1': self.x1_c, 'x2': self.x2_c, 'x3': self.x3_c, 'x4': self.x4_c}
        self.planes = self.cfg.MODEL.NECK.FIRST_CONV_SIZE
        self.pyramid_layers = self.cfg.MODEL.BACKBONE.LAYERS.split(',')  # x1,x2,x3,x4
        self.pyramid_layers.sort()
        self.pathways = self.cfg.MODEL.BACKBONE.LAYER_PATHWAYS.split(',')  # ['topdown', 'bottomup'] aka 'parallel', or "none"
        self.is_pyramid = False if self.pathways[0] == 'none' else True
        self.output_size = len(torch.load(Path.joinpath(Path(self.cfg.DATASET.VOXEL_INDEX_DIR),
                                                        Path(self.cfg.DATASET.ROI + '.pt'))))
        self.pooling_mode = self.cfg.MODEL.NECK.POOLING_MODE
        self.spp_level = self.cfg.MODEL.NECK.SPP_LEVELS
        self.spp_level = [int(i) for i in self.spp_level.split(',')]
        self.spp_level = np.array([[1 for _ in self.spp_level], self.spp_level, self.spp_level])

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

                self.first_convs.update(
                    {k: nn.Conv3d(self.c_dict[x_i], self.planes, kernel_size=1, stride=1)})

                if self.is_pyramid:
                    self.smooths.update(
                        {k: nn.Conv3d(self.planes, self.planes, kernel_size=3, stride=1, padding='same')})

                # SPP
                self.poolings.update({k: nn.Sequential(
                    SpatialPyramidPooling3D(self.spp_level,
                                            self.cfg.MODEL.NECK.POOLING_MODE),
                    nn.Flatten())})
                self.fc_input_dims.update({k: np.sum(
                    self.spp_level[0] * self.spp_level[1] * self.spp_level[2]) * self.planes})



                self.ch_response.update({k: build_fc(
                    self.cfg, self.fc_input_dims[k], self.output_size, part='first')})

        in_size = self.cfg.MODEL.NECK.FC_HIDDEN_DIM * self.num_chs

        self.final_fusions = FcFusion(fusion_type='concat')

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


def build_fc(cfg, input_dim, output_dim, part='full'):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }

    layer_hidden = cfg.MODEL.NECK.FC_HIDDEN_DIM

    module_list = []
    for i in range(cfg.MODEL.NECK.FC_NUM_LAYERS):
        if i == 0:
            size1, size2 = input_dim, layer_hidden
        else:
            size1, size2 = layer_hidden, layer_hidden
        module_list.append(nn.Linear(size1, size2))
        if cfg.MODEL.NECK.FC_BATCH_NORM:
            module_list.append(nn.BatchNorm1d(size2))
        module_list.append(activations[cfg.MODEL.NECK.FC_ACTIVATION])
        if cfg.MODEL.NECK.FC_DROPOUT > 0:
            module_list.append(nn.Dropout(cfg.MODEL.NECK.FC_DROPOUT))
        if i == 0:
            layer_one_size = len(module_list)

    # last layer
    module_list.append(nn.Linear(size2, output_dim))

    if part == 'full':
        module_list = module_list
    elif part == 'first':
        module_list = module_list[:layer_one_size]
    elif part == 'last':
        module_list = module_list[layer_one_size:]
    else:
        NotImplementedError()

    return nn.Sequential(*module_list)


class FcFusion(nn.Module):
    def __init__(self, fusion_type='concat'):
        super(FcFusion, self).__init__()
        assert fusion_type in ['add', 'avg', 'concat', ]
        self.fusion_type = fusion_type

    def init_weights(self):
        pass

    def forward(self, input):
        assert (isinstance(input, tuple)) or (isinstance(input, dict)) or (isinstance(input, list))
        if isinstance(input, dict):
            input = tuple(input.values())

        if self.fusion_type == 'add':
            out = torch.sum(torch.stack(input, -1), -1, keepdim=False)

        elif self.fusion_type == 'avg':
            out = torch.mean(torch.stack(input, -1), -1, keepdim=False)

        elif self.fusion_type == 'concat':
            out = torch.cat(input, -1)

        else:
            raise ValueError

        return out
