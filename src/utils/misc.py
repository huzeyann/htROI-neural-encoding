import pandas as pd

import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)


def my_query_df(df: pd.DataFrame, equal_dict: dict = {}, isin_dict: dict = {}) -> pd.DataFrame:
    ret_df = df
    for k, v in equal_dict.items():
        ret_df = ret_df.loc[ret_df[k] == v]
    for k, v in isin_dict.items():
        ret_df = ret_df.loc[ret_df[k].isin(v)]
    return ret_df


def dict_to_list(config):
    config_list = []
    for key, val in config.items():
        # print(key, val, type(val))
        config_list.append(key)
        config_list.append(val)
    return config_list


def dokodemo_hsplit(x, idxs):
    ret = []
    for i in range(len(idxs)):
        if i == 0:
            ret.append(x[:, :idxs[i]])
        else:
            ret.append(x[:, idxs[i - 1]:idxs[i]])
    return ret
