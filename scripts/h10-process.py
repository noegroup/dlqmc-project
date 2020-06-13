from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
from itertools import product

dists = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6]
ansatzes = ['SD-SJ', 'SD-SJBF', 'MD-SJBF']


def get_mean_err(energies):
    return energies.mean(), energies.mean(axis=0).std() / np.sqrt(energies.shape[1])


results = defaultdict(list)
with h5py.File(f'../data/raw/data_pub_h10.h5', 'r') as f:
    for d, ansatz in product(dists, ansatzes):
        results[f'H10_d{d}', ansatz] = (
            lambda x: pd.Series({'energy': x[0], 'err': x[1],})
        )(get_mean_err(f[f'H10_d{d}'][ansatz]['evaluate'][...][:, :, 0]))
results = (
    pd.concat(results, names=['system', 'ansatz']).unstack().sort_index().reset_index()
)
results.to_csv('../data/final/h10.csv', index=False)
