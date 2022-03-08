# source: https://github.com/iduta/pyconvsegnet

# backbone download link:
download_url = 'https://drive.google.com/u/0/uc?id=1P2qJNt72bCCDEO9FeKkWY4uakFLz-Ncb&export=download'

import os

import numpy as np
import torch

from .pyconvsegnet import PyConvSegNet


def modify_pyconvsegnet_2d_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        x = self.layer0(x)

        ret_dict = {}

        x1 = self.layer1(x)
        ret_dict['x1'] = x1
        if max_depth >= 2:
            x2 = self.layer2(x1)
            ret_dict['x2'] = x2
        if max_depth >= 3:
            x3 = self.layer3(x2)
            ret_dict['x3'] = x3
        if max_depth >= 4:
            x4 = self.layer4(x3)
            ret_dict['x4'] = x4

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def get_pyconvsegnet(pretrained=True, cache_dir='~/.cache/'):
    model = PyConvSegNet(layers=152, classes=150, zoom_factor=8,
                         pretrained=False,
                         backbone_net='pyconvresnet',
                         backbone_output_stride=8)
    if pretrained:
        cache_dir = os.path.expanduser(cache_dir)
        path = os.path.join(cache_dir, 'ade20k_pyconvresnet152_pyconvsegnet.pth')
        if not os.path.exists(path):
            raise Exception(f'{path} not exists! please download from: {download_url}')

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
    return model
