from collections import defaultdict
import h5py
import numpy as np
import pandas as pd
from itertools import product

systems = ['Li2', 'Be2', 'B2', 'C2']
dets = [1, 3, 10, 30, 100]


def get_mean_err(energies):
    return energies.mean(), energies.mean(axis=0).std() / np.sqrt(energies.shape[1])


results = defaultdict(list)
with h5py.File(f'../data/raw/data_pub_diatomics.h5', 'r') as f:
    for system, d in product(systems, dets):
        results[system, d] = (lambda x: pd.Series({'energy': x[0], 'err': x[1],}))(
            get_mean_err(f[system][f'{d}det']['evaluate'][...][:, :, 0])
        )

results = (
    pd.concat(results, names=['system', 'ndet']).unstack().sort_index().reset_index()
)
results.to_csv('../data/final/diatomics.csv', index=False)
