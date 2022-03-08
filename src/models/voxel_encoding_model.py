from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from src.config import get_cfg_defaults
from src.config.config import convert_to_dict
from src.modeling.backbone.build import build_backbone
from src.modeling.neck.build import build_neck
from src.modeling.components.build_optimizer import build_optimizer
from src.utils.metrics import vectorized_correlation


class VoxelEncodingModel(pl.LightningModule):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg
        self._cfg_hparams = convert_to_dict(self.cfg.clone())
        self.save_hyperparameters(self._cfg_hparams)

        self.backbone = build_backbone(self.cfg)
        self.neck = build_neck(self.cfg)

        self.current_val_score = 0  # dirty finetune callback

    def forward(self, x):
        x = self.backbone(x)
        out = self.neck(x)
        return out

    def on_train_start(self):
        self.logger.log_hyperparams(self._cfg_hparams)
        ...

    def _shared_train_val(self, batch, batch_idx, prefix, is_log=True):
        x, y = batch
        out = self(x)
        loss = F.mse_loss(out, y)
        if is_log:
            self.log(f'{prefix}_mse_loss/final', loss,
                     on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=False)
        return out, loss

    def training_step(self, batch, batch_idx):
        if self.cfg.MODEL.BACKBONE.DISABLE_BN:
            def disable_bn(model):
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm3d) \
                            or isinstance(module, nn.BatchNorm2d) \
                            or isinstance(module, nn.BatchNorm1d):
                        module.eval()

            self.backbone.apply(disable_bn)

        out, loss = self._shared_train_val(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        out, loss = self._shared_train_val(batch, batch_idx, 'val')
        y = batch[-1]
        return {'out': out, 'y': y}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        x = batch
        return self(x)

    def validation_epoch_end(self, val_step_outputs) -> None:
        with torch.no_grad():
            val_outs = torch.cat([out['out'] for out in val_step_outputs], 0).to(self.device)
            val_ys = torch.cat([out['y'] for out in val_step_outputs], 0).to(self.device)
            corr = vectorized_correlation(val_outs, val_ys)
            current_corr = corr.mean().item()
        self.current_val_score = current_corr  # for my dirty finetune callback
        self.log(f'val_corr', current_corr, prog_bar=True, logger=True, sync_dist=False)

        best_score = self.trainer.checkpoint_callback.best_model_score
        best_score = best_score if best_score is not None else -1
        best_score = best_score if best_score > current_corr else current_corr
        self.log("hp_metric", best_score, prog_bar=True, logger=True, sync_dist=False)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (warmup backbone)"""
        # backbone does not requires_grad at start
        optimizer = build_optimizer(self.cfg, filter(lambda p: p.requires_grad, self.parameters()))

        return optimizer


if __name__ == '__main__':
    model = VoxelEncodingModel(get_cfg_defaults())
    ...
