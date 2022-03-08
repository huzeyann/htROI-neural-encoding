# modified from: https://github.com/zhoubolei/moments_models

# backbone download links
download_url = 'http://moments.csail.mit.edu/moments_models/multi_moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar'

import os

import numpy as np
import torch
import torchvision.models as models
from filelock import FileLock

from .model import ResNet3D, Bottleneck


def modify_resnets(model):
    # Modify attributs
    model.last_linear, model.fc = model.fc, None

    def features(self, input):
        x = self.conv1(input)
        # print("conv, ", x.view(-1)[:10])
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # torch.Size([1, 64, 8, 56, 56])

        x = self.layer1(x)  # torch.Size([1, 256, 8, 56, 56])
        x = self.layer2(x)  # torch.Size([1, 512, 4, 28, 28])
        x = self.layer3(x)  # torch.Size([1, 1024, 2, 14, 14])
        x = self.layer4(x)  # torch.Size([1, 2048, 1, 7, 7])
        return x

    def logits(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

    # Modify methods
    setattr(model.__class__, 'features', features)
    setattr(model.__class__, 'logits', logits)
    setattr(model.__class__, 'forward', forward)
    return model


def modify_resnets_patrial(model, layers):
    del model.fc
    # del modeling.last_linear
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

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
        if max_depth >= 5:
            x_label = self.logits(x4)
            ret_dict['x_label'] = x_label

        # print(ret_dict.keys())
        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


ROOT_URL = 'http://moments.csail.mit.edu/moments_models'
weights = {
    'resnet50': 'moments_v2_RGB_resnet50_imagenetpretrained.pth.tar',
    'resnet3d50': 'moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar',
    'multi_resnet3d50': 'multi_moments_v2_RGB_imagenet_resnet3d50_segment16.pth.tar',
}


def load_checkpoint(weight_file):
    if not os.access(weight_file, os.W_OK):
        weight_url = os.path.join(ROOT_URL, weight_file)
        os.system('wget ' + weight_url)
    checkpoint = torch.load(weight_file, map_location=lambda storage, loc: storage)  # Load on cpu
    return {str.replace(str(k), 'module.', ''): v for k, v in checkpoint['state_dict'].items()}


def multi_resnet3d50(num_classes=292, pretrained=True, cache_dir='~/.cache/', **kwargs):
    """Constructs a ResNet3D-50 modeling."""
    model = modify_resnets(ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs))
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(download_url)
        model.load_state_dict({str.replace(str(k), 'module.', ''): v for k, v in checkpoint['state_dict'].items()})
    return model
