import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm
from yaml import CLoader

from src.config.config import flatten


def load_run_df(results_dir: Union[Path, str]):
    results_dir = Path(results_dir)
    finished_runs = [path.parent for path in
                     results_dir.glob('**/prediction.npy')]  # a success run always save prediction

    run_meta_infos = []
    for run_dir in tqdm(finished_runs):
        hparams = yaml.load(run_dir.joinpath('hparams.yaml').open(), Loader=CLoader)
        run_meta_info = flatten(hparams)
        run_meta_info['path'] = run_dir

        data = [json.loads(line) for line in run_dir.joinpath('result.json').open()]
        ddf = pd.DataFrame(data)
        run_meta_info['score'] = ddf.val_corr.max()
        run_meta_info['time'] = ddf.time_total_s.max()

        run_meta_infos.append(run_meta_info)

    run_df = pd.DataFrame(run_meta_infos)

    # fix list unhashable
    run_df['MODEL.BACKBONE.LAYERS'] = run_df['MODEL.BACKBONE.LAYERS'].apply(lambda x: tuple(x))
    run_df['MODEL.NECK.SPP_LEVELS'] = run_df['MODEL.NECK.SPP_LEVELS'].apply(lambda x: tuple(x))

    return run_df


def my_query_df(df: pd.DataFrame, equal_dict: dict = {}, isin_dict: dict = {}) -> pd.DataFrame:
    ret_df = df
    for k, v in equal_dict.items():
        ret_df = ret_df.loc[ret_df[k] == v]
    for k, v in isin_dict.items():
        ret_df = ret_df.loc[ret_df[k].isin(v)]
    return ret_df


def filter_single_layer_runs(run_df: pd.DataFrame):
    return run_df[run_df.apply(lambda row: len(row['MODEL.BACKBONE.LAYERS']) == 1, axis=1)],


def filter_multi_layer_runs(run_df: pd.DataFrame):
    return run_df[run_df.apply(lambda row: len(row['MODEL.BACKBONE.LAYERS']) > 1, axis=1)],


def mahou_list(query):
    mahou = np.arange(0, 1, 0.003)
    return mahou[min(range(len(mahou)), key=lambda i: abs(mahou[i] - query))]
