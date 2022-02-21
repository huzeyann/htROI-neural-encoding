import os
import warnings
from pathlib import Path
from typing import List

from dotenv import find_dotenv, load_dotenv
from yacs.config import CfgNode as ConfigurationNode, CfgNode
from yacs.config import _assert_with_logging, _valid_type

import logging

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

__C = ConfigurationNode()

# importing default as a global singleton
cfg = __C
__C.DESCRIPTION = 'Default config'

__C.DATAMODULE = ConfigurationNode()
__C.DATAMODULE.SPLIT_SCHEMATIC = 'UBE'
__C.DATAMODULE.NUM_CV_SPLITS = 10
__C.DATAMODULE.I_CV_FOLD = -1

__C.DATASET = ConfigurationNode()
__C.DATASET.NAME = 'Algonauts2021'
__C.DATASET.ROOT_DIR = os.path.expanduser("~/Algonauts_2021_data/") # overwrite by .env
__C.DATASET.LOAD_PRECOMPUTED_FLOW = False
__C.DATASET.TRANSFORM = 'i3d_rgb'
__C.DATASET.RESOLUTION = 224
__C.DATASET.FRAMES = 16
__C.DATASET.VOXEL_INDEX_DIR = os.path.expanduser("~/Algonauts_2021_data/voxel_indexs/") # overwrite by .env
__C.DATASET.ROI = 'WB'

__C.MODEL = ConfigurationNode()

__C.MODEL.BACKBONE = ConfigurationNode()
__C.MODEL.BACKBONE.NAME = 'i3d_rgb'
__C.MODEL.BACKBONE.PRETRAINED = True
__C.MODEL.BACKBONE.DISABLE_BN = True
__C.MODEL.BACKBONE.LAYERS = ('x2', 'x3', 'x4')  # 'x1,x2,x3,x4'
__C.MODEL.BACKBONE.LAYER_PATHWAYS = 'topdown'  # 'none,topdown,bottomup'

__C.MODEL.NECK = ConfigurationNode()
__C.MODEL.NECK.NECK_TYPE = 'i3dneck'
__C.MODEL.NECK.FIRST_CONV_SIZE = 256
__C.MODEL.NECK.POOLING_MODE = 'max'
__C.MODEL.NECK.SPP_LEVELS = (1, 3, 5)  # '1-3-5'
__C.MODEL.NECK.FC_ACTIVATION = 'elu'
__C.MODEL.NECK.FC_HIDDEN_DIM = 2048
__C.MODEL.NECK.FC_NUM_LAYERS = 2
__C.MODEL.NECK.FC_BATCH_NORM = False
__C.MODEL.NECK.FC_DROPOUT = 0.

__C.MODEL.PATH = ConfigurationNode()
__C.MODEL.PATH.I3D_RGB_CACHE_DIR = os.path.expanduser("~/.cache/") # overwrite by .env
__C.MODEL.PATH.I3D_FLOW_FILE_PATH = os.path.expanduser("~/.cache/i3d_flow.pt") # overwrite by .env

# __C.LOGGING_ROI = None

__C.OPTIMIZER = ConfigurationNode()
__C.OPTIMIZER.NAME = 'AdaBelief'
__C.OPTIMIZER.LR = 1e-4
__C.OPTIMIZER.WEIGHT_DECAY = 1e-2

__C.SCHEDULER = ConfigurationNode()
__C.SCHEDULER.NAME = 'no'

__C.TRAINER = ConfigurationNode()
__C.TRAINER.GPUS = 1
__C.TRAINER.FP16 = True
__C.TRAINER.MAX_EPOCHS = 100
__C.TRAINER.ACCUMULATE_GRAD_BATCHES = 4
__C.TRAINER.BATCH_SIZE = 8
__C.TRAINER.VAL_CHECK_INTERVAL = 1.

__C.TRAINER.CALLBACKS = ConfigurationNode()

__C.TRAINER.CALLBACKS.BACKBONE = ConfigurationNode()
__C.TRAINER.CALLBACKS.BACKBONE.INITIAL_RATIO_LR = 0.1
__C.TRAINER.CALLBACKS.BACKBONE.LR_MULTIPLY_EFFICIENT = 1.6
__C.TRAINER.CALLBACKS.BACKBONE.DEFROST_SCORE = 0.06
__C.TRAINER.CALLBACKS.BACKBONE.SHOULD_ALIGN = True
__C.TRAINER.CALLBACKS.BACKBONE.TRAIN_BN = False
__C.TRAINER.CALLBACKS.BACKBONE.VERBOSE = True

__C.TRAINER.CALLBACKS.EARLY_STOP = ConfigurationNode()
__C.TRAINER.CALLBACKS.EARLY_STOP.PATIENCE = 9

__C.TRAINER.CALLBACKS.CHECKPOINT = ConfigurationNode()
__C.TRAINER.CALLBACKS.CHECKPOINT.ROOT_DIR = os.path.expanduser("~/.cache/checkpoints/") # overwrite by .env
__C.TRAINER.CALLBACKS.CHECKPOINT.RM_AT_DONE = True

__C.TRAINER.CALLBACKS.LOGGER = ConfigurationNode()

__C.RESULTS_DIR = '/data/huze/ray_results/algonauts2021/' # overwrite by .env

__C.DEBUG = True

def check_cfg(C):
    if C.DATASET.NAME == 'Algonauts2021':
        if C.DATASET.LOAD_PRECOMPUTED_FLOW:
            assert C.DATASET.RESOLUTION == 224
            assert C.DATASET.FRAMES == 64
        if C.MODEL.BACKBONE.NAME == 'i3d_flow':
            assert C.DATASET.LOAD_PRECOMPUTED_FLOW

        if C.DATAMODULE.SPLIT_SCHEMATIC == 'UBE':
            assert C.DATAMODULE.NUM_CV_SPLITS == 10
            assert C.DATAMODULE.I_CV_FOLD == -1

    if C.TRAINER.ACCUMULATE_GRAD_BATCHES > 1:
        assert C.MODEL.BACKBONE.DISABLE_BN


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()


def convert_to_dict(cfg_node):
    def _convert_to_dict(cfg_node, key_list):
        _VALID_TYPES = {tuple, list, str, int, float, bool, type(None)}
        if not isinstance(cfg_node, CfgNode):
            _assert_with_logging(
                _valid_type(cfg_node),
                "Key {} with value {} is not a valid type; valid types: {}".format(
                    ".".join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
            return cfg_node
        else:
            cfg_dict = dict(cfg_node)
            for k, v in cfg_dict.items():
                cfg_dict[k] = _convert_to_dict(v, key_list + [k])
            return cfg_dict

    return _convert_to_dict(cfg_node, [])


def combine_cfgs(path_cfg_data: Path = None,
                 path_cfg_override: Path = None,
                 list_cfg_override: List = None,
                 ):
    """
    An internal facing routine thaat combined CFG in the order provided.
    :param path_output: path to output files
    :param path_cfg_data: path to path_cfg_data files
    :param path_cfg_override: path to path_cfg_override actual
    :param list_cfg_override: [key1, value1, key2, value2, ...]
    :return: cfg_base incorporating the overwrite.
    """
    if path_cfg_data is not None:
        path_cfg_data = Path(path_cfg_data)
    if path_cfg_override is not None:
        path_cfg_override = Path(path_cfg_override)
    # Path order of precedence is:
    # Priority 1, 2, 3, 4, 5 respectively
    # .env > List > other CFG YAML > data.yaml > default.yaml

    # Load default lowest tier one:
    # Priority 5:
    cfg_base = get_cfg_defaults()

    # Merge from the path_data
    # Priority 4:
    if path_cfg_data is not None and path_cfg_data.exists():
        cfg_base.merge_from_file(path_cfg_data.absolute())

    # Merge from other cfg_path files to further reduce effort
    # Priority 3:
    if path_cfg_override is not None and path_cfg_override.exists():
        cfg_base.merge_from_file(path_cfg_override.absolute())

    # Merge from List
    # Priority 2:
    if list_cfg_override is not None:
        cfg_base.merge_from_list(list_cfg_override)

    # Merge from .env
    # Priority 1:
    list_cfg = update_cfg_using_dotenv()
    if list_cfg is not []:
        cfg_base.merge_from_list(list_cfg)

    return cfg_base


def update_cfg_using_dotenv() -> list:
    """
    In case when there are dotenvs, try to return list of them.
    # It is returning a list of hard overwrite.
    :return: empty list or overwriting information
    """
    # If .env not found, bail
    if find_dotenv() == '':
        warnings.warn(".env files not found. YACS config file merging aborted.")
        return []

    # Load env.
    load_dotenv(find_dotenv(), verbose=True)

    # Load variables
    list_key_env = {
        "DATASET.ROOT_DIR",
        "DATASET.VOXEL_INDEX_DIR",
        "MODEL.PATH.I3D_RGB_CACHE_DIR",
        "MODEL.PATH.I3D_FLOW_FILE_PATH",
        "TRAINER.CALLBACKS.CHECKPOINT.ROOT_DIR",
        "RESULTS_DIR",
    }

    # Instantiate return list.
    path_overwrite_keys = []
    logging.info('merge from .env')
    # Go through the list of key to be overwritten.
    for key in list_key_env:

        # Get value from the env.
        value = os.getenv(key)
        logging.info(f'{key}={value}')
        # If it is none, skip. As some keys are only needed during training and others during the prediction stage.
        if value is None:
            continue

        # Otherwise, adding the key and the value to the dictionary.
        path_overwrite_keys.append(key)
        path_overwrite_keys.append(value)

    return path_overwrite_keys


# The flatten and unflatten snippets are from an internal lfads_tf2 implementation.

def flatten(dictionary, level=[]):
    """ Flattens a dictionary by placing '.' between levels.

    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.

    Returns
    -------
    dict
        The flattened dictionary.

    See Also
    --------
    lfads_tf2.utils.unflatten : Performs the opposite of this operation.

    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict


def unflatten(dictionary):
    """ Unflattens a dictionary by splitting keys at '.'s.

    This function unflattens a hierarchical dictionary by splitting
    its keys at '.'s. It is used internally for converting the
    configuration dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.

    Parameters
    ----------
    dictionary : dict
        The flat dictionary to be unflattened.

    Returns
    -------
    dict
        The unflattened dictionary.

    See Also
    --------
    lfads_tf2.utils.flatten : Performs the opposite of this operation.

    """

    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict
