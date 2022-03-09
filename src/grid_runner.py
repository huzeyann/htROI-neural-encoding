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
from src.utils.misc import dict_to_list
from yacs.config import CfgNode

from src.config import get_cfg_defaults, combine_cfgs
from src.data.datamodule import MyDataModule
from src.models.voxel_encoding_model import VoxelEncodingModel
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


def run_single_train(cfg: CfgNode = get_cfg_defaults()):
    cfg.freeze()
    if cfg.DEBUG:
        torch.set_printoptions(10)

    datamodule = build_dm(cfg)
    plmodel = VoxelEncodingModel(cfg)
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

    if not cfg.DEBUG:
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


def run_single_tune_config(tune_dict: Dict, cfg: CfgNode):
    cfg.merge_from_list(dict_to_list(tune_dict))
    run_single_train(cfg)


if __name__ == '__main__':
    from ray.tune import CLIReporter
    import ray

    # ray.init(local_mode=True)

    # reporter = CLIReporter()
    # reporter.logdir = '/home/huze/.cache/debug/'
    # tune.session.init(reporter)

    exp_config = '/data_smr/huze/projects/kROI-voxel-encoding/src/config/experiments/algonauts2021/algonauts2021_3d_resnet.yml'

    cfg = combine_cfgs(
        path_cfg_data=exp_config,
        list_cfg_override=['DEBUG', False]
    )
    # tune.run(
    #     tune.with_parameters(
    #         run_single_tune_config,
    #         cfg=cfg,
    #     ),
    #     local_dir='/home/huze/ray_results/debug/'
    # )

    tune_config = {
        'DATASET.ROI': tune.grid_search(['LC2']),
        # 'DATASET.FRAMES': tune.grid_search([4, 8, 12, 16]),
        # 'MODEL.NECK.LSTM.BIDIRECTIONAL': tune.grid_search([False]),
        # 'MODEL.NECK.LSTM.NUM_LAYERS': tune.grid_search([1]),
        # 'MODEL.NECK.NECK_TYPE': tune.grid_search(['i3d_neck', 'lstm_neck']),
        # 'MODEL.NECK.NECK_TYPE': tune.grid_search(['i3d_neck']),
        'MODEL.BACKBONE.LAYERS': tune.grid_search([('x3',), ]),
        # 'MODEL.NECK.SPP_LEVELS': tune.grid_search([[3], ]),
        # 'MODEL.NECK.POOLING_MODE': tune.grid_search(['max']),
        'TRAINER.CALLBACKS.BACKBONE.DEFROST_SCORE': tune.grid_search([0.12, 0.18]),
    }

    tune.run(
        tune.with_parameters(
            run_single_tune_config,
            cfg=cfg
        ),
        local_dir='/home/huze/ray_results/',
        resources_per_trial={"cpu": 2, "gpu": 1},
        mode="max",
        metric='hp_metric',
        config=tune_config,
        num_samples=1,
        progress_reporter=CLIReporter(
            parameter_columns=list(tune_config.keys()),
            metric_columns=["val_corr", 'hp_metric']
        ),
        name='debug',
        verbose=3,
    )
