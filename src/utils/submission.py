import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        ret_di = u.load()
    return ret_di


def algonauts2021_submission_from_whole_brain_prediction(meta_data_dir,
                                                         zip_file_name,
                                                         prediction: np.ndarray,
                                                         output_dir='./submissions'):
    """
    prediction: Numpy Array, shape [102, num_voxels]
    """
    assert prediction.shape == (102, 161326)
    assert type(prediction) == np.ndarray
    assert prediction.dtype == np.float32

    meta_data_dir = Path(meta_data_dir)
    voxel_config = json.load(meta_data_dir.joinpath(Path('algonauts2021_mini_track_roi_voxel_index.json')).open())
    sub_config = json.load(meta_data_dir.joinpath(Path('algonauts2021_mini_track_subject_voxel_index.json')).open())
    sub_ends = [18222, 39795, 55020, 74465, 87805, 107623, 118459, 130806, 148376, 161326]

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # full track
    res_dict = {'WB': {}}
    outs = np.hsplit(prediction, sub_ends)
    for i in range(10):
        out = outs[i]
        res_dict['WB'][f'sub{i + 1:02d}'] = out
    pkl_file_path = 'full_track.pkl'
    save_dict(res_dict, str(pkl_file_path))
    zipped_results = zipfile.ZipFile(str(output_dir.joinpath(Path(zip_file_name + '_full_track.zip'))), 'w')
    zipped_results.write(pkl_file_path)
    zipped_results.close()
    os.remove(pkl_file_path)

    # mini track
    res_dict = {roi: {} for roi in sub_config.keys()}
    for roi, sub_lens in sub_config.items():
        outs = np.hsplit(prediction[:, voxel_config[roi]], np.cumsum(sub_lens))
        for i in range(10):
            out = outs[i]
            res_dict[roi][f'sub{i + 1:02d}'] = out
    pkl_file_path = 'mini_track.pkl'
    save_dict(res_dict, str(pkl_file_path))
    zipped_results = zipfile.ZipFile(str(output_dir.joinpath(Path(zip_file_name + '_mini_track.zip'))), 'w')
    zipped_results.write(pkl_file_path)
    zipped_results.close()
    os.remove(pkl_file_path)

