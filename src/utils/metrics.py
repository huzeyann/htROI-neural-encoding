from torch import Tensor


def vectorized_correlation(x: Tensor, y: Tensor) -> Tensor:
    """

    :param x: Tensor shape [num_samples, num_voxels]
    :param y: Tensor shape [num_samples, num_voxels]
    :return: shape [num_voxels, ]
    """

    dim = 0
    centered_x = x - x.mean(dim, keepdims=True)
    centered_y = y - y.mean(dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim, keepdims=True) + 1e-8
    y_std = y.std(dim, keepdims=True) + 1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()
