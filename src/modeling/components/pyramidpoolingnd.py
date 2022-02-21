import numpy as np
import torch
import torch.nn as nn


class SpatialPyramidPoolingND(nn.Module):
    def __init__(self, levels, mode="max"):
        """
                Spatial Pyramid Pooling Module, which divides the input Tensor horizontally and horizontally
                (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
                Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
                In other words: It divides the Input Tensor in level*level rectangles width of roughly (previous_conv.size(3) / level)
                and height of roughly (previous_conv.size(2) / level) and pools its value. (pads input to fit)
                :param levels defines the different divisions to be made in the width dimension
                :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
                :returns (forward) a tensor vector with shape [batch x 1 x n],
                                                    where n: sum(filter_amount*level*level) for each level in levels
                                                    which is the concentration of multi-level pooling
                """
        super().__init__()
        self.levels = levels
        self.n_dim = len(levels[0])
        self.mode = mode

        if self.n_dim == 3:
            if self.mode == 'max':
                pooling_cls = nn.AdaptiveMaxPool3d
            elif self.mode == 'avg':
                pooling_cls = nn.AdaptiveAvgPool3d
            else:
                NotImplementedError()
        elif self.n_dim == 2:
            if self.mode == 'max':
                pooling_cls = nn.AdaptiveMaxPool2d
            elif self.mode == 'avg':
                pooling_cls = nn.AdaptiveAvgPool2d
            else:
                NotImplementedError()
        else:
            NotImplementedError()

        self.poolings = nn.ModuleList()
        for level in self.levels:
            self.poolings.append(nn.Sequential(
                pooling_cls(level),
                nn.Flatten()
            ))

    def forward(self, x):
        outs = []
        for i in range(len(self.levels)):
            out = self.poolings[i](x)
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs

    def get_output_size(self, filters):
        """
                Calculates the output shape given a filter_amount: sum(filter_amount*level*level) for each level in levels
                Can be used to x.view(-1, spp.get_output_size(filter_amount)) for the fully-connected layers
                :param filters: the amount of filter of output fed into the spatial pyramid pooling
                :return: sum(filter_amount*level*level)
        """
        out = 0
        for level in self.levels:
            out += np.prod(level) * filters
        return out
