import logging
from typing import Callable, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BaseFinetuning
from pytorch_lightning.callbacks.finetuning import multiplicative
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim.optimizer import Optimizer

log = logging.getLogger(__name__)

class MyScoreFinetuning(BaseFinetuning):
    r"""

    Finetune a backbone modeling based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current modeling learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.

    Args:

        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.

        lambda_func: Scheduling function for increasing backbone learning rate.

        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of modeling

        backbone_initial_lr: Optional, Inital learning rate for the backbone.
            By default, we will use current_learning /  backbone_initial_ratio_lr

        should_align: Wheter to align with current learning rate when backbone learning
            reaches it.

        initial_denom_lr: When unfreezing the backbone, the intial learning rate will
            current_learning_rate /  initial_denom_lr.

        train_bn: Wheter to make Batch Normalization trainable.

        verbose: Display current learning rate for modeling and backbone

        round: Precision for displaying learning rate

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])

    """

    def __init__(
        self,
        unfreeze_backbone_at_val_score: float = 0.,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.,
        train_bn: bool = True,
        verbose: bool = False,
        round: int = 12,
    ):
        super().__init__()

        self.unfreeze_backbone_at_val_score = unfreeze_backbone_at_val_score
        self.backbone_initial_lr = backbone_initial_lr
        self.lambda_func = lambda_func
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.should_align = should_align
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn
        self.round = round
        self.verbose = verbose

        self.freeze_flag = True

    def on_fit_start(self, trainer, pl_module):
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        if not hasattr(pl_module, "current_val_score"):
            raise MisconfigurationException("The LightningModule should have a float `current_val_score` attribute")
        if not (hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module)):
            raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: 'pl.LightningModule'):
        self.freeze(pl_module.backbone)

    def finetune_function(self, pl_module: 'pl.LightningModule', epoch: int, optimizer: Optimizer, opt_idx: int):
        """Called when the epoch begins."""
        if self.freeze_flag:
            if pl_module.current_val_score >= self.unfreeze_backbone_at_val_score:
                self.freeze_flag = False
                current_lr = optimizer.param_groups[0]['lr']
                initial_backbone_lr = self.backbone_initial_lr if self.backbone_initial_lr is not None \
                    else current_lr * self.backbone_initial_ratio_lr
                self.previous_backbone_lr = initial_backbone_lr
                self.unfreeze_and_add_param_group(
                    pl_module.backbone,
                    optimizer,
                    initial_backbone_lr,
                    train_bn=self.train_bn,
                    initial_denom_lr=self.initial_denom_lr
                )
                if self.verbose:
                    log.info(
                        f"Current lr: {round(current_lr, self.round)}, "
                        f"Backbone lr: {round(initial_backbone_lr, self.round)}"
                    )

        else:
            current_lr = optimizer.param_groups[0]['lr']
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = current_lr if (self.should_align and next_current_backbone_lr > current_lr) \
                else next_current_backbone_lr
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.round)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.round)}"
                )