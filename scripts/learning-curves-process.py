from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
from itertools import product
from deepqmc.ewm import EWMAverage
from uncertainties import ufloat
from uncertainties import unumpy as unp

systems = ['H2', 'LiH', 'Be', 'B', 'Li2']
ansatzes = ['SD-SJ', 'SD-SJBF', 'MD-SJ', 'MD-SJBF']


def get_mean_err(energies):
    return energies.mean(), energies.mean(axis=0).std() / np.sqrt(energies.shape[1])


results = defaultdict(list)
with h5py.File(f'../data/raw/data_pub_small_systems.h5', 'r') as f:
    for system, ansatz in product(systems, ansatzes):
        E_mean = f[system][ansatz]['train'][...].mean(axis=1)
        ewm = EWMAverage(outlier_maxlen=3, outlier=3, decay_alpha=10)
        E_ewm = []
        for e in E_mean:
            ewm.update(e)
            E_ewm.append((ewm.mean.item().n, ewm.mean.item().s))
        E_ewm = unp.uarray(*zip(*E_ewm))
        for step, E_step in enumerate(E_ewm):
            results[system, ansatz, step] = pd.Series(
                {'energy': unp.nominal_values(E_step), 'err': unp.std_devs(E_step),}
            )
results = (
    pd.concat(results, names=['system', 'ansatz', 'step'])
    .unstack()
    .sort_index()
    .reset_index()
)
results.to_csv('../data/final/learning-curves.csv', index=False)
