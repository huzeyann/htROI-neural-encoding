from src.modeling.backbone.i3d_flow import load_i3d_flow, modify_i3d_flow
from src.modeling.backbone.i3d_rgb import multi_resnet3d50, modify_resnets_patrial
from src.utils.rigistry import Registry

BACKBONE_REGISTRY = Registry()


@BACKBONE_REGISTRY.register('i3d_rgb')
def get_i3d_rgb_backbone(cfg):
    model = multi_resnet3d50(pretrained=cfg.MODEL.BACKBONE.PRETRAINED,
                             cache_dir=cfg.MODEL.PATH.I3D_RGB_CACHE_DIR)
    model = modify_resnets_patrial(model, cfg.MODEL.BACKBONE.LAYERS)
    return model


@BACKBONE_REGISTRY.register('i3d_flow')
def get_i3d_flow_backbone(cfg):
    model = load_i3d_flow(path=cfg.MODEL.PATH.I3D_FLOW_FILE_PATH,
                          pretrained=cfg.MODEL.BACKBONE.PRETRAINED)
    model = modify_i3d_flow(model, cfg.MODEL.BACKBONE.LAYERS)
    return model



def build_backbone(cfg):
    return BACKBONE_REGISTRY[cfg.MODEL.BACKBONE.NAME](cfg)