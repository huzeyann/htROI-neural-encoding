import pandas as pd


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
