import argparse
from pathlib import Path

import numpy as np
import ray
from ray import tune

from src.config import combine_cfgs
from src.grid_runner import run_single_tune_config


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config-dir", "-e",
        dest='exp_config_dir',
        required=True,
        help="dir path) to config yaml containing info about experiment. "
             "example: `--exp-config-dir src/config/experiments/algonauts2021`",
    )

    parser.add_argument(
        '--resume',
        choices=["AUTO", "LOCAL", "REMOTE", "PROMPT", "ERRORED_ONLY"],
        default='AUTO',
        dest='resume',
        help='ray.tune(resume=resume).'
    )

    parser.add_argument('--debug', action='store_true')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    launch_grid(**vars(args))


def launch_grid(exp_config_dir: str, resume: str, debug: bool):
    exp_configs = [p for p in Path(exp_config_dir).iterdir() if p.name.endswith('.yml')]

    # ray.init(local_mode=True)
    # if debug:
    #     ray.init(local_mode=True)

    for exp_config in exp_configs:
        cfg = combine_cfgs(
            path_cfg_data=exp_config,
            list_cfg_override=['DEBUG', debug]
        )

        name = exp_config.name.replace('.yml', '')

        # multi-layer 1 run
        tune.run(
            tune.with_parameters(
                run_single_tune_config,
                cfg=cfg
            ),
            config={
                'DATASET.ROI': tune.grid_search(['WB']),
                'MODEL.BACKBONE.LAYERS': tune.grid_search([cfg.MODEL.BACKBONE.LAYERS]),
                'MODEL.NECK.SPP_LEVELS': tune.grid_search([cfg.MODEL.NECK.SPP_LEVELS]),
                'MODEL.NECK.FIRST_CONV_SIZE': tune.sample_from(
                    lambda spec: {1: 2048, 2: 1024, 3: 512, 6: 256, 7: 256, 16: 1, 32: 1, 48: 1, 64: 1}[
                        np.max(spec.config['MODEL.NECK.SPP_LEVELS'])]),
            },
            local_dir=cfg.RESULTS_DIR,
            resources_per_trial={"cpu": 4, "gpu": 1},
            name=name + '_' + 'multilayer',
            # verbose=1,
            resume=resume,
        )

        # single-layer 16 run
        tune.run(
            tune.with_parameters(
                run_single_tune_config,
                cfg=cfg
            ),
            config={
                'DATASET.ROI': tune.grid_search(['WB']),
                'MODEL.BACKBONE.LAYERS': tune.grid_search([[i] for i in cfg.MODEL.BACKBONE.LAYERS]),
                'MODEL.NECK.SPP_LEVELS': tune.grid_search([[i] for i in cfg.MODEL.NECK.SPP_LEVELS]),
                'MODEL.NECK.FIRST_CONV_SIZE': tune.sample_from(
                    lambda spec: {1: 2048, 2: 1024, 3: 512, 6: 256, 7: 256, 16: 1, 32: 1, 48: 1, 64: 1}[
                        np.max(spec.config['MODEL.NECK.SPP_LEVELS'])]),
            },
            local_dir=cfg.RESULTS_DIR,
            resources_per_trial={"cpu": 4, "gpu": 1},
            name=name + '_' + 'singlelayer',
            # verbose=1,
            resume=resume,
        )


if __name__ == '__main__':
    main()
