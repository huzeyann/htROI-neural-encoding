# modified from: https://github.com/huzeyann/video_features/tree/master/models/i3d

# backbone download link
#
import os

download_url = 'https://raw.githubusercontent.com/v-iashin/video_features/master/models/i3d/checkpoints/i3d_flow.pt'

import numpy as np
import torch

from .model import I3D


def modify_i3d_flow_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, inp):
        ret_dict = {}
        # Preprocessing
        out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        x1 = self.maxPool3d_3a_3x3(out)
        ret_dict['x1'] = x1
        if max_depth >= 2:
            out = self.mixed_3b(x1)
            out = self.mixed_3c(out)
            x2 = self.maxPool3d_4a_3x3(out)
            ret_dict['x2'] = x2
        if max_depth >= 3:
            out = self.mixed_4b(x2)
            out = self.mixed_4c(out)
            out = self.mixed_4d(out)
            out = self.mixed_4e(out)
            out = self.mixed_4f(out)
            x3 = self.maxPool3d_5a_2x2(out)
            ret_dict['x3'] = x3
        if max_depth >= 4:
            out = self.mixed_5b(x3)
            x4 = self.mixed_5c(out)  # <- [1,  832, 8 (for T=64) or 3 (for T=24), 1, 1]
            ret_dict['x4'] = x4
        if max_depth >= 5:
            out = self.avg_pool(x4)  # <- [1, 1024, 8 (for T=64) or 3 (for T=24), 1, 1]
            # out = self.dropout(out)
            # out = self.conv3d_0c_1x1(out)
            out = out.squeeze(3)
            out = out.squeeze(3)
            out = out.mean(2)
            # out_logits = out
            # out = self.softmax(out_logits)
            ret_dict['x5'] = out

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def load_i3d_flow(pretrained=True, cache_dir='~/.cache/'):
    model = I3D(num_classes=400, modality='flow')
    if pretrained:
        cache_dir = os.path.expanduser(cache_dir)
        path = os.path.join(cache_dir, 'i3d_flow.pt')
        if not os.path.exists(path):
            raise Exception(f'{path} not exists! please download from: {download_url}')

        stat_dict = torch.load(path)
        model.load_state_dict(stat_dict)
    return model
