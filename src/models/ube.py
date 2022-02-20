import os
from abc import ABC
from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from src.config import get_cfg_defaults
from src.models.build_backbone import build_backbone
from src.models.build_neck import build_neck
from src.models.build_optimizer import build_optimizer
from src.utils.metrics import vectorized_correlation


class UBE(pl.LightningModule):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__()
        self.cfg = cfg

        self.automatic_optimization = True

        self.backbone = build_backbone(self.cfg)
        self.neck = build_neck(self.cfg)

        # self.save_hyperparameters(self.cfg)

        self.current_val_score = 0  # dirty finetune callback

    def on_train_start(self):
        # save hparams
        with open(os.path.join(self.logger[0].log_dir, 'hparams.yaml'), 'w') as f:
            f.write(self.cfg.clone().dump())
        # self.logger.log_hyperparams(self.cfg)
        ...

    def forward(self, x):
        x = self.backbone(x)
        out = self.neck(x)
        return out

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
                    if isinstance(module, nn.BatchNorm3d):
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
            mean_corr = corr.mean().item()
        # print(self.neck.final_fc[-1].weight.data.flatten()[:10])
        self.current_val_score = mean_corr  # dirty finetune callback
        self.log(f'val_corr', mean_corr, prog_bar=True, logger=True, sync_dist=False)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup backbone)"""
        # no_decay = ["bias", "BatchNorm3D.weight", "BatchNorm1D.weight", "BatchNorm2D.weight"]
        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.backbone.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.cfg.OPTIMIZER.WEIGHT_DECAY,
        #         'lr': self.cfg.OPTIMIZER.LR * self.cfg.TRAINER.CALLBACKS.BACKBONE.INITIAL_RATIO_LR,
        #     },
        #     {
        #         "params": [p for n, p in self.backbone.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #         'lr': self.cfg.OPTIMIZER.LR * self.cfg.TRAINER.CALLBACKS.BACKBONE.INITIAL_RATIO_LR,
        #     },
        #     {
        #         "params": [p for n, p in self.neck.named_parameters() if not any(nd in n for nd in no_decay)],
        #         "weight_decay": self.cfg.OPTIMIZER.WEIGHT_DECAY,
        #         'lr': self.cfg.OPTIMIZER.LR,
        #     },
        #     {
        #         "params": [p for n, p in self.neck.named_parameters() if any(nd in n for nd in no_decay)],
        #         "weight_decay": 0.0,
        #         'lr': self.cfg.OPTIMIZER.LR,
        #     },
        # ]

        optimizer = build_optimizer(self.cfg, filter(lambda p: p.requires_grad, self.parameters()))
        # optimizer = build_optimizer(self.cfg, optimizer_grouped_parameters)
                                                            # backbone does not requires_grad at start

        return optimizer


if __name__ == '__main__':
    model = UBE(get_cfg_defaults())
    ...