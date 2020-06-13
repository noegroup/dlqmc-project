from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
from itertools import product

systems = ['H2', 'LiH', 'Be', 'B', 'Li2', 'C']
ansatzes = ['SD-SJ', 'SD-SJBF', 'MD-SJ', 'MD-SJBF']


def get_mean_err(energies):
    return energies.mean(), energies.mean(axis=0).std() / np.sqrt(energies.shape[1])


results = defaultdict(list)
with h5py.File(f'../data/raw/data_pub_small_systems.h5', 'r') as f:
    for system, ansatz in product(systems, ansatzes):
        results[system, ansatz] = (lambda x: pd.Series({'energy': x[0], 'err': x[1],}))(
            get_mean_err(f[system][ansatz]['evaluate'][...][:, :, 0])
        )
results = (
    pd.concat(results, names=['system', 'ansatz']).unstack().sort_index().reset_index()
)
results.to_csv('../data/final/small-systems.csv', index=False)
