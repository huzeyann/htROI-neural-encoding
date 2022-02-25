from src.modeling.neck.i3dneck import I3DNeck
from src.utils.rigistry import Registry

NECK_REGISTRY = Registry()


@NECK_REGISTRY.register('i3dneck')
def get_i3dneck(cfg):
    model = I3DNeck(cfg)
    return model

def build_neck(cfg):
    return NECK_REGISTRY[cfg.MODEL.NECK.NECK_TYPE](cfg)