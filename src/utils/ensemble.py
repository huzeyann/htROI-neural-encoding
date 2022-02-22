import torch
from scipy.optimize import differential_evolution
from torch import Tensor

from src.utils.metrics import vectorized_correlation


def optimize_val_correlation(vals: Tensor, y: Tensor, device='cpu', verbose=False, tol=0.01) -> Tensor:
    """
    vals: shape [N, num_voxels, num_models]
    y: shape [N, num_voxels]

    return: ws: shape [num_models, ]
    """
    with torch.no_grad():
        vals = vals.clone().to(device).float()
        y = y.clone().to(device).float()

        def correlation_evaluation_function(ws):
            ws = torch.tensor(ws).float().to(device)
            out = vals @ ws
            score = vectorized_correlation(out, y).mean().item()
            return -score

        # define range for input
        r_min, r_max = -1.0, 1.0
        # define the bounds on the search
        bounds = [[r_min, r_max] for _ in range(vals.shape[-1])]
        # perform the differential evolution search
        result = differential_evolution(correlation_evaluation_function, bounds, tol=tol, disp=verbose)
        score = -result['fun']

        ws = result['x'] # ensemble weight
        ws /= ws.sum()  # sum to 1
        ws = torch.tensor(ws).float()

        del vals
        del y
        torch.cuda.empty_cache()

    return ws
