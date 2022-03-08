# modified from: https://github.com/Separius/SimCLRv2-Pytorch

download_cmd = '''
git clone https://github.com/Separius/SimCLRv2-Pytorch
cd SimCLRv2-Pytorch
python download.py r50_1x_sk1
python convert.py r50_1x_sk1/model.ckpt-250228 --ema
cp r50_1x_sk1_ema.pth ~/.cache/
rm -r r50_1x_sk1
'''
import os

import numpy as np
import torch

from .resnet import ResNet


def modify_simclr_resnet_2d_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        """Forward function."""
        x = self.net[0](x)

        ret_dict = {}
        for i, layer in enumerate(self.net[1:]):
            # drop layers to run faster
            if i + 1 > max_depth:
                break
            x = layer(x)
            ret_dict[f'x{i + 1}'] = x

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def simclrv2_resnet_50_1x_sk1(pretrained=True, cache_dir='~/.cache/'):
    model = ResNet([3, 4, 6, 3], 1, 0.0625)
    cache_dir = os.path.expanduser(cache_dir)
    pth_path = os.path.join(cache_dir, 'r50_1x_sk1_ema.pth')
    if pretrained:
        if not os.path.exists(pth_path):
            raise Exception(f'{pth_path} not exists! please download from: \n {download_cmd}')
        model.load_state_dict(torch.load(pth_path)['resnet'])
    return model
