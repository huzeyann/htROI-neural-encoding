from src.data.datasets.algonauts2021 import Algonauts2021Dataset
from src.utils.rigistry import Registry

DATASETS_REGISTRY = Registry()


@DATASETS_REGISTRY.register('Algonauts2021')
def build_algonauts_rgb_dataset(dataset_cfg, kwargs):
    return Algonauts2021Dataset(dataset_cfg, is_train=kwargs['is_train'])


def build_dataset(cfg, **kwargs):
    return DATASETS_REGISTRY[cfg.DATASET.NAME](cfg, kwargs)
