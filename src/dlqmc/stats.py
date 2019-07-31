import numpy as np
import torch

from .utils import batch_eval


class GaussianKDEstimator:
    def __init__(self, xs, max_memory=1.0, weights=None, *, bw):
        assert len(xs.shape) == 2
        self._xs = xs
        self._weights = weights if weights is not None else None
        self._width = np.sqrt(2) * bw
        self._bs = int(2 ** 30 * max_memory) // (xs.nelement() * xs.element_size() or 1)

    def __call__(self, xs):
        assert len(xs.shape) == 2
        if len(xs) > self._bs:
            return batch_eval(self, xs.split(self._bs))
        kernel = ((xs[:, None] - self._xs) ** 2).sum(dim=-1) / self._width ** 2
        norm = 1 / (len(self._xs) * (np.sqrt(np.pi) * self._width) ** xs.shape[1])
        basis = torch.exp(-kernel)
        if self._weights is not None:
            basis = self._weights * basis
        return norm * basis.sum(dim=-1)


def outlier_mask(x, p, q, dim=None):
    x = x.detach()
    dim = dim if dim is not None else -1
    n = x.shape[dim]
    lb = x.kthvalue(int(p * n), dim=dim).values
    ub = x.kthvalue(int((1 - p) * n), dim=dim).values
    return (
        (x - (lb + ub).unsqueeze(dim) / 2).abs() > q * (ub - lb).unsqueeze(dim),
        (lb, ub),
    )


def clip_outliers(x, p, q):
    x = x.detach()
    n = len(x)
    lb = x.kthvalue(int(p * n)).values
    ub = x.kthvalue(int((1 - p) * n)).values
    mids = x[(x > lb) & (x < ub)]
    mean, std = mids.mean(), mids.std()
    return x.clamp(mean - q * std, mean + q * std)
