from src.data.utils import TwoFiveFive, RGB2LAB_L, RGB2BGR
from torchvision import transforms

from src.utils.rigistry import Registry

TRANSFORM_REGISTRY = Registry()


@TRANSFORM_REGISTRY.register('standard_rgb')
def get_i3d_rgb_mmit_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@TRANSFORM_REGISTRY.register('swin')
def get_swin_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        TwoFiveFive(),
        transforms.Normalize([123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),
    ])


@TRANSFORM_REGISTRY.register('greyscale')
def get_greyscale_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        RGB2LAB_L(),
    ])


@TRANSFORM_REGISTRY.register('simclr')
def get_simclr_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


@TRANSFORM_REGISTRY.register('bdcn')
def get_bdcnvgg_2d_transform(resolution):
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.4810938, 0.45752459, 0.40787055], [1, 1, 1]),
        RGB2BGR(),
        TwoFiveFive(),
    ])


def get_transform(key):
    return TRANSFORM_REGISTRY[key]
