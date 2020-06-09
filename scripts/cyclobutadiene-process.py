from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from uncertainties import ufloat

from deepqmc.ewm import EWMAverage


def filter_outliers(x, q=2):
    l, m, h = x.quantile(q=[0.25, 0.5, 0.75], axis=1).values
    x = x.where(np.abs(x.values - m[:, None]) < q * (h - l)[:, None])
    return x


def ewm_traj(x, **kwargs):
    ewm = EWMAverage(**kwargs)
    x_ewm = []
    for x in x:
        ewm.update(x)
        x_ewm.append(ewm.mean.item().n)
    return x_ewm


results = {}
for path in Path('data/raw/cyclobutadiene/fit').glob('*/*/*/fit.h5'):
    batch, idx, state = path.parts[-4:-1]
    batch, idx = int(batch.split('-')[1]), int(idx)
    with h5py.File(path, 'r', swmr=True) as f:
        E_loc = f['E_loc'][...]
    where_zero = (E_loc == 0).all(axis=-1).nonzero()[0]
    E_loc[where_zero] = np.nan
    results[batch, state, idx] = pd.Series(E_loc.mean(-1))
results = (
    pd.concat(results, names=['batch', 'state', 'idx', 'step'])
    .unstack('idx')
    .pipe(filter_outliers)
    .mean(axis=1)
    .groupby(['batch', 'state'])
    .apply(lambda x: pd.Series(ewm_traj(x), index=x.index))
    .to_frame('energy_ewm')
    .reset_index()
    .loc()[lambda x: x['step'] < 4500]
)

results.to_csv('data/final/cyclobutadiene-fit.csv', index=False)

results = defaultdict(list)
for path in Path('data/raw/cyclobutadiene/sample').glob('*/*/*/*/sample.h5'):
    idx_smpl, batch, idx, state = path.parts[-5:-1]
    idx_smpl, batch, idx = int(idx_smpl), int(batch.split('-')[1]), int(idx)
    with h5py.File(path, 'r', swmr=True) as f:
        enes = f['blocks/energy'][:, :, 0]
    results[batch, state, idx].append(enes)
for (batch, state, idx), enes in results.items():
    enes = np.concatenate(enes)
    ene = ufloat(enes.mean(), enes.mean(0).std() / np.sqrt(enes.shape[1]))
    results[batch, state, idx] = pd.Series(
        {'energy': ene.n, 'err': ene.s, 'n': len(enes)}
    )
results = (
    pd.concat(results, names=['batch', 'state', 'idx'])
    .unstack()
    .sort_index()
    .reset_index()
)

results.to_csv('data/final/cyclobutadiene-sample.csv', index=False)
