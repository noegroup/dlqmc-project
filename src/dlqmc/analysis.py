import numpy as np
import scipy.optimize
from uncertainties import unumpy as unp


def ewm(x, X, Y, alpha, thre=1e-10, with_std=False):
    if x is None:
        x = X
    deltas = -np.log(alpha) * (x[:, None] - X)
    mask = (0 <= deltas) & (deltas < -np.log(thre))
    ws = np.zeros_like(deltas)
    ws[mask] = np.exp(-deltas[mask])
    ws = ws / ws.sum(axis=-1)[:, None]
    mean = (ws * Y).sum(axis=-1)
    if not with_std:
        return mean
    std = np.sqrt((ws ** 2 * (mean[:, None] - Y) ** 2).sum(axis=-1))
    return unp.uarray(mean, std)


def infinite_training_limit(energy, start):
    step = np.arange(len(energy))
    step2 = np.arange(start, len(energy))
    E_ewm = ewm(
        step2, step, energy, (1 - 1 / (2 + step2 / 20))[:, None], with_std=True,
    )
    param = scipy.optimize.curve_fit(
        lambda x, Einf, slope: Einf + slope * x,
        1 / step2,
        unp.nominal_values(E_ewm),
        sigma=unp.std_devs(E_ewm),
        absolute_sigma=True,
    )
    param = unp.uarray(param[0], np.sqrt(np.diag(param[1])))
    return param[0], param[1], E_ewm
