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
