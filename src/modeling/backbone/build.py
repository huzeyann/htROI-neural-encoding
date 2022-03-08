import os

import torchvision
from filelock import FileLock

from .bdcnvgg_2d.my_api import get_bdcn, modify_bdcnvgg_2d_partial
from .densnet_2d.my_api import modify_densnet_2d_partial
from .i3d_flow.my_api import load_i3d_flow, modify_i3d_flow_partial
from .i3d_rgb.my_api import multi_resnet3d50, modify_resnets_patrial
from .pyconvsegnet_2d.my_api import get_pyconvsegnet, modify_pyconvsegnet_2d_partial
from .seg_swin_b_2d.my_api import get_mmseg, modify_seg_swin_2d_partial
from .simclr_resnet_2d.my_api import simclrv2_resnet_50_1x_sk1, modify_simclr_resnet_2d_partial
from .swin_b_3d.my_api import sthv2_video_swin_b, modify_swin_transformer_3d_partial
from .moby_swin_t_2d.my_api import get_moby_swin_t_2d, modify_swin_partial
from .colorizer_2d.my_api import get_colorizer, modify_colorizer_partial
from .vggish_audio.my_api import get_vggish_torch, modify_vggish_audio_partial
from .warp_2Dto3D import TwoDtoThreeDWarp
from src.utils.rigistry import Registry

BACKBONE_REGISTRY = Registry()


@BACKBONE_REGISTRY.register('i3d_rgb')
def get_i3d_rgb_backbone(cfg):
    model = multi_resnet3d50(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                             cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_resnets_patrial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('i3d_flow')
def get_i3d_flow_backbone(cfg):
    model = load_i3d_flow(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                          cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_i3d_flow_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('3d_swin')
def get_swin_transformer_3d(cfg):
    model = sthv2_video_swin_b(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                               cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_swin_transformer_3d_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_colorizer')
def get_2d_colorizer(cfg):
    model = get_colorizer(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)
    model = modify_colorizer_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_colorizer_warp_3d')
def get_2d_colorizer_warp_3d(cfg):
    model = get_2d_colorizer(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('2d_seg_swin')
def get_2d_seg_swin(cfg):
    model = get_mmseg(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)
    model = modify_seg_swin_2d_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_seg_swin_warp_3d')
def get_2d_seg_swin_warp_3d(cfg):
    model = get_2d_seg_swin(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('2d_moby_swin')
def get_2d_moby_swin(cfg):
    model = get_moby_swin_t_2d(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                               cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_swin_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_moby_swin_warp_3d')
def get_2d_moby_swin_warp_3d(cfg):
    model = get_2d_moby_swin(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('2d_bdcnvgg')
def get_2d_bdcnvgg(cfg):
    model = get_bdcn(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                     cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_bdcnvgg_2d_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_bdcnvgg_warp_3d')
def get_2d_bdcnvgg_warp_3d(cfg):
    model = get_2d_bdcnvgg(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('2d_pyconvsegnet')
def get_2d_pyconvsegnet(cfg):
    model = get_pyconvsegnet(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                             cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_pyconvsegnet_2d_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_pyconvsegnet_warp_3d')
def get_2d_pyconvsegnet_warp_3d(cfg):
    model = get_2d_pyconvsegnet(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('2d_densnet')
def get_2d_densnet(cfg):
    model = torchvision.models.densenet169(pretrained=cfg.MODEL.BACKBONE.PRETRAINED)
    model = modify_densnet_2d_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_densnet_warp_3d')
def get_2d_densnet_warp_3d(cfg):
    model = get_2d_densnet(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('2d_simclr')
def get_2d_simclr(cfg):
    model = simclrv2_resnet_50_1x_sk1(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                                      cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_simclr_resnet_2d_partial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('2d_simclr_warp_3d')
def get_2d_simclr_warp_3d(cfg):
    model = get_2d_simclr(cfg)
    model = TwoDtoThreeDWarp(model)
    return model


@BACKBONE_REGISTRY.register('audio_vggish')
def get_audio_vggish(cfg):
    model = get_vggish_torch(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                             cache_dir=cfg.MODEL.BACKBONE.PRETRAINED_WEIGHT_DIR)
    model = modify_vggish_audio_partial(model)
    return model


def build_backbone(cfg):
    with FileLock(os.path.expanduser("~/.model.lock")):  # multi-processing lock to avoid duplicated download
        return BACKBONE_REGISTRY[cfg.MODEL.BACKBONE.NAME](cfg)
