import json
import os
from typing import Dict

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from yacs.config import CfgNode

from src.config import get_cfg_defaults
from src.data.datamodule import MyDataModule
from src.models.ube import UBE
from src.utils.callbacks import MyScoreFinetuning


def build_callbacks(cfg):
    callbacks = [
        EarlyStopping(
            monitor='val_corr',
            min_delta=0.00,
            patience=int(cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE / cfg.TRAINER.VAL_CHECK_INTERVAL),
            verbose=False,
            mode='max'
        ),
        MyScoreFinetuning(
            unfreeze_backbone_at_val_score=cfg.TRAINER.CALLBACKS.BACKBONE.DEFROST_SCORE if not cfg.DEBUG else -1,
            lambda_func=lambda x: cfg.TRAINER.CALLBACKS.BACKBONE.LR_MULTIPLY_EFFICIENT,
            backbone_initial_ratio_lr=cfg.TRAINER.CALLBACKS.BACKBONE.INITIAL_RATIO_LR,
            should_align=cfg.TRAINER.CALLBACKS.BACKBONE.SHOULD_ALIGN,
            train_bn=cfg.TRAINER.CALLBACKS.BACKBONE.TRAIN_BN,
            verbose=cfg.TRAINER.CALLBACKS.BACKBONE.VERBOSE,
        ),
        ModelCheckpoint(
            monitor='val_corr',
            dirpath=cfg.TRAINER.CALLBACKS.CHECKPOINT.ROOT_DIR,
            filename='{epoch:02d}-{val_corr:.6f}',
            auto_insert_metric_name=True,
            save_weights_only=True,
            save_top_k=1,
            mode='max',
        ),
        # this does not report max score...
        TuneReportCallback(
            ['val_corr', 'hp_metric'],
            on="validation_end",
        )
    ]
    return callbacks


def build_dm(cfg):
    datamodule = MyDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def run_single_train(cfg: CfgNode):
    cfg.freeze()
    if cfg.DEBUG:
        torch.set_printoptions(10)

    datamodule = build_dm(cfg)
    plmodel = UBE(cfg)
    loggers = pl_loggers.TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version=".",
                                           default_hp_metric=False)
    callbacks = build_callbacks(cfg)

    trainer = pl.Trainer(
        precision=16 if cfg.TRAINER.FP16 else 32,
        gpus=cfg.TRAINER.GPUS,
        accumulate_grad_batches=cfg.TRAINER.ACCUMULATE_GRAD_BATCHES,
        # accelerator='ddp',
        # plugins=DDPPlugin(find_unused_parameters=False),
        limit_train_batches=1.0 if not cfg.DEBUG else 0.01,
        limit_val_batches=1.0 if not cfg.DEBUG else 0.1,
        limit_predict_batches=1.0 if not cfg.DEBUG else 0.03,
        max_epochs=cfg.TRAINER.MAX_EPOCHS if not cfg.DEBUG else 3,
        val_check_interval=cfg.TRAINER.VAL_CHECK_INTERVAL if not cfg.DEBUG else 1.0,
        callbacks=callbacks,
        logger=loggers,
    )

    ### TRAIN
    trainer.fit(plmodel, datamodule=datamodule)

    # trainer.checkpoint_callback.to_yaml(os.path.join(tune.get_trial_dir(), 'checkpoints.yaml'))
    # tune.report({'best_val_corr': trainer.checkpoint_callback.best_model_score})
    # for logger in loggers:
    #     logger.log_metrics({"hp_metric", trainer.checkpoint_callback.best_model_score})

    ### PREDICT
    plmodel = plmodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, cfg=cfg)
    predictions = trainer.predict(plmodel, datamodule=datamodule)
    prediction = torch.cat([p for p in predictions], 0)
    np.save(os.path.join(tune.get_trial_dir(), 'prediction.npy'), prediction.cpu().numpy())

    ### VOXEL EMBEDDING
    last_fc_weight = plmodel.neck.final_fc[-1].weight.data
    np.save(os.path.join(tune.get_trial_dir(), 'voxel_embedding.npy'), last_fc_weight.cpu().numpy())

    ### delete ckpt to save disk space (default True)
    if cfg.TRAINER.CALLBACKS.CHECKPOINT.RM_AT_DONE:
        os.remove(trainer.checkpoint_callback.best_model_path)


if __name__ == '__main__':
    ...
