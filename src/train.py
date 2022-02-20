import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.config import get_cfg_defaults
from src.data.datamodule import MyDataModule
from src.models.ube import UBE
from src.utils.callbacks import MyScoreFinetuning


def build_callbacks(cfg, task_id='debug'):
    callbacks = [
        EarlyStopping(
            monitor='val_corr',
            min_delta=0.00,
            patience=int(cfg.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE / cfg.TRAINER.VAL_CHECK_INTERVAL),
            verbose=False,
            mode='max'
        ),
        MyScoreFinetuning(
            unfreeze_backbone_at_val_score=cfg.TRAINER.CALLBACKS.BACKBONE.DEFROST_SCORE if not cfg.DEBUG else 1e-10,
            lambda_func=lambda x: cfg.TRAINER.CALLBACKS.BACKBONE.LR_MULTIPLY_EFFICIENT,
            backbone_initial_ratio_lr=cfg.TRAINER.CALLBACKS.BACKBONE.INITIAL_RATIO_LR,
            should_align=cfg.TRAINER.CALLBACKS.BACKBONE.SHOULD_ALIGN,
            train_bn=cfg.TRAINER.CALLBACKS.BACKBONE.TRAIN_BN,
            verbose=cfg.TRAINER.CALLBACKS.BACKBONE.VERBOSE,
        ),
        ModelCheckpoint(
            monitor='val_corr',
            dirpath=Path.joinpath(Path(cfg.TRAINER.CALLBACKS.CHECKPOINT.ROOT_DIR), Path(task_id)),
            filename='{epoch:02d}-{val_corr:.6f}',
            auto_insert_metric_name=True,
            save_weights_only=True,
            save_top_k=1,
            mode='max',
        )
    ]
    return callbacks


def build_loggers(cfg, task_id='debug'):
    loggers = [
        pl_loggers.TensorBoardLogger(str(Path.joinpath(Path(cfg.TRAINER.CALLBACKS.LOGGER.ROOT_DIR), Path(task_id)))),
        pl_loggers.CSVLogger(str(Path.joinpath(Path(cfg.TRAINER.CALLBACKS.LOGGER.ROOT_DIR), Path(task_id))))
    ]
    return loggers


def build_dm(cfg):
    datamodule = MyDataModule(cfg)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def build_model(cfg):
    return UBE(cfg)


def train_main(cfg, task_id='debug'):
    cfg.freeze()
    if cfg.DEBUG:
        torch.set_printoptions(10)

    datamodule = build_dm(cfg)
    plmodel = build_model(cfg)
    callbacks = build_callbacks(cfg, task_id)
    loggers = build_loggers(cfg, task_id)

    trainer = pl.Trainer(
        precision=16 if cfg.TRAINER.FP16 else 32,
        gpus=[cfg.TRAINER.GPU_DEVICE_ID],
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

    ### PREDICT
    prediction_dir = Path.joinpath(Path(cfg.PREDICTION_DIR), Path(task_id))
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = Path.joinpath(prediction_dir, Path('prediction.pt'))

    plmodel = plmodel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, cfg=cfg)
    predictions = trainer.predict(plmodel, datamodule=datamodule)
    prediction = torch.cat([p for p in predictions], 0)
    print(prediction.shape)
    torch.save(prediction.cpu(), prediction_path)

    ### delete ckpt to save disk space (default False)
    if cfg.TRAINER.CALLBACKS.CHECKPOINT.RM_AT_DONE:
        os.remove(trainer.checkpoint_callback.best_model_path)


if __name__ == '__main__':
    C = get_cfg_defaults()
    C.DEBUG = True
    train_main(C)
