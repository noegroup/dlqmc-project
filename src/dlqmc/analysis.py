import numpy as np
import scipy.optimize
from uncertainties import unumpy as unp

from deepqmc.ewm import ewm


def infinite_training_limit(energy, start):
    step = np.arange(len(energy))
    step2 = np.arange(start, len(energy))
    E_ewm = ewm(
        step2, step, energy, (1 - 1 / (2 + step2 / 20))[:, None], with_err=True,
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
