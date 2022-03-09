from .audio_neck import AudioNeck
from .i2d_neck import I2DNeck
from .i3d_neck import I3DNeck
from .lstm_neck import TwoDLSTMNeck
from src.utils.rigistry import Registry

NECK_REGISTRY = Registry()


@NECK_REGISTRY.register('i3d_neck')
def get_i3d_neck(cfg):
    model = I3DNeck(cfg)
    return model


@NECK_REGISTRY.register('lstm_neck')
def get_lstm_neck(cfg):
    model = TwoDLSTMNeck(cfg)
    return model


@NECK_REGISTRY.register('audio_neck')
def get_audio_neck(cfg):
    model = AudioNeck(cfg)
    return model


@NECK_REGISTRY.register('i2d_neck')
def get_audio_neck(cfg):
    model = I2DNeck(cfg)
    return model


def build_neck(cfg):
    return NECK_REGISTRY[cfg.MODEL.NECK.NECK_TYPE](cfg)
