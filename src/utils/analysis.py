import torch
from torch import Tensor
import numpy as np
from tqdm.auto import tqdm
import itertools


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def my_ultimate_save_memory_corrcoef(x, chunk_size=4096):
    """
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.my_save_memory_corrcoef(x)
        >>> th_corr = my_save_memory_corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    num_samples = x.shape[0]
    size = x.size(1)

    for i in tqdm(range(size), desc='minus mean'):
        x[:, i] = x[:, i] - x[:, i].mean()

    # calculate covariance matrix of rows
    c = torch.zeros(num_samples, num_samples)
    for ci in tqdm(list(chunked_iterable(range(num_samples), chunk_size)), desc='chunck'):
        ci = torch.tensor(ci)
        for cj in list(chunked_iterable(range(ci.min(), num_samples), chunk_size)):
            cj = torch.tensor(cj)

            fill = x[ci] @ x[cj].t()
            c[ci.min():ci.max() + 1, cj.min():cj.max() + 1] = fill
            c[cj.min():cj.max() + 1, ci.min():ci.max() + 1] = fill.t()

    c = c / (size - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    #     print(stddev.shape)

    for i in tqdm(range(c.shape[0]), desc='normalize'):
        c[i] = c[i] / (stddev[i] * stddev)

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def my_save_memory_corrcoef(x, save_memory=True):
    """
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref: 
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.my_save_memory_corrcoef(x)
        >>> th_corr = my_save_memory_corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    size = x.size(1)
    mean_x = torch.mean(x, 1, keepdims=True)
    xm = x.sub(mean_x)
    if save_memory:
        del x
        del mean_x
    c = xm.mm(xm.t())
    if save_memory:
        del xm
    c = c / (size - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    #     print(stddev.shape)
    if not save_memory:
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
    else:
        for i in range(c.shape[0]):
            c[i] = c[i] / (stddev[i] * stddev)

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def my_corrcoef(x):
    """
    Mimics `np.corrcoef`
    Arguments
    ---------
    x : 2D torch.Tensor

    Returns
    -------
    c : torch.Tensor
        if x.size() = (5, 100), then return val will be of size (5,5)
    Numpy docs ref:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html
    Numpy code ref:
        https://github.com/numpy/numpy/blob/v1.12.0/numpy/lib/function_base.py#L2933-L3013
    Example:
        >>> x = np.random.randn(5,120)
        # result is a (5,5) matrix of correlations between rows
        >>> np_corr = np.my_corrcoef(x)
        >>> th_corr = my_corrcoef(torch.from_numpy(x))
        >>> np.allclose(np_corr, th_corr.numpy())
        # [out]: True
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    # probably not necessary but numpy does it
    c = torch.clamp(c, -1.0, 1.0)

    return c


def motion_correlation(cms: Tensor, fmris: Tensor):
    # shape: cms [1, 1000], fmris [n, 1000]
    num_voxels = fmris.shape[0]

    chunk_size = 1024
    start = 0
    end = start + chunk_size

    corrs = []
    while start <= num_voxels:
        end = start + chunk_size
        if end > num_voxels:
            end = num_voxels

        x = torch.cat([cms, fmris[start:end]], 0)
        corr = my_corrcoef(x)[0][1:].cpu()
        corrs.append(corr)

        start += chunk_size
    corrs = torch.cat(corrs)
    return corrs


def get_motion_correlation(cms: Tensor, fmris: Tensor):
    cms = cms.unsqueeze(0)
    assert cms.shape[0] == 1
    fmris = fmris.t()
    assert cms.shape[1] == fmris.shape[1]
    mc = motion_correlation(cms, fmris)
    return mc.cpu().numpy()


def get_internal_variance_motion_correlation(cms: Tensor, fmris: Tensor, n_samples=10, n_percent=0.5):
    # random sample videos n times
    # shape: cms [num_videos], fmris [num_videos, n]
    num_videos = fmris.shape[0]
    assert len(cms) == num_videos
    corrs = []
    for i in range(n_samples):
        np.random.seed(i)
        idxs = np.random.choice(np.arange(num_videos), int(num_videos * n_percent))
        idxs = torch.tensor(idxs)
        x = fmris[idxs, :]
        corr = motion_correlation(cms[idxs].unsqueeze(0), x.t())
        corrs.append(corr)
    corrs = torch.stack(corrs)
    return corrs.std(0)


def get_sequential_internal_variance_motion_correlation(cms: Tensor, fmris: Tensor, n_samples=10):
    # sequential divide
    # shape: cms [num_videos], fmris [num_videos, n]
    num_videos = fmris.shape[0]

    chunk_size = int(num_videos / n_samples)
    corrs = []
    for i in range(n_samples):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        x = fmris[start:end, :]
        corr = motion_correlation(cms[start:end].unsqueeze(0), x.t())
        corrs.append(corr)
    corrs = np.stack(corrs)
    return corrs.std(0)


def get_face_index(vid_idxs: np.array, fmris: Tensor, yes_face_vid_idxs: np.array):
    # shape: fmri  [num_videos, num_voxels]

    no_face_fmris = []
    yes_face_fmris = []
    for vid_idx in vid_idxs:
        vid_idx = vid_idx
        fmri = fmris[vid_idx]
        if vid_idx in yes_face_vid_idxs:
            yes_face_fmris.append(fmri)
        else:
            no_face_fmris.append(fmri)
    no_face_fmris = torch.stack(no_face_fmris)
    yes_face_fmris = torch.stack(yes_face_fmris)

    y_means = yes_face_fmris.mean(0, keepdim=True)
    n_means = no_face_fmris.mean(0, keepdim=True)
    all_fmri_maxs = fmris[vid_idxs].max(0, keepdim=True)[0]
    all_fmri_mins = fmris[vid_idxs].min(0, keepdim=True)[0]
    face_indexs = (y_means - n_means) / (all_fmri_maxs - all_fmri_mins)
    face_indexs = face_indexs[0]

    return face_indexs.cpu().numpy()


def get_sequential_internal_variance_face_index(fmris: Tensor, yes_face_vid_idxs, n_samples=10):
    # fmris [num_videos, n]
    num_videos = fmris.shape[0]

    chunk_size = int(num_videos / n_samples)
    corrs = []
    for i in range(n_samples):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        x = fmris[start:end, :]
        corr = get_face_index(np.arange(start, end), fmris, yes_face_vid_idxs)
        corrs.append(corr)
    corrs = np.stack(corrs)
    return corrs.std(0)
