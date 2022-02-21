from adabelief_pytorch import AdaBelief
from torch.optim import AdamW

from src.utils.rigistry import Registry

OPTIMIZER_REGISTRY = Registry()


@OPTIMIZER_REGISTRY.register('AdaBelief')
def get_adabelief(cfg, optimizer_grouped_parameters):
    return AdaBelief(
        optimizer_grouped_parameters,
        lr=cfg.OPTIMIZER.LR,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
        print_change_log=False,
    )


@OPTIMIZER_REGISTRY.register('AdamW')
def get_adamw(cfg, optimizer_grouped_parameters):
    return AdamW(
        optimizer_grouped_parameters,
        lr=cfg.OPTIMIZER.LR,
        weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
    )


def build_optimizer(cfg, optimizer_grouped_parameters):
    return OPTIMIZER_REGISTRY[cfg.OPTIMIZER.NAME](cfg, optimizer_grouped_parameters)
