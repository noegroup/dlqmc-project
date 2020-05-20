import numpy as np
import pandas as pd
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


def log_clip(x, med, a, b):
    x = x - med
    a, b = a - med, b - med
    x = x.copy()
    n_clipped = (x < a).sum(), (x > b).sum()
    x[x < a] = a * (1 + np.log((1 + (x[x < a] / a) ** 2) / 2))
    x[x > b] = b * (1 + np.log((1 + (x[x > b] / b) ** 2) / 2))
    return med + x, n_clipped


def ewm_full(E_loc):
    traj = []
    percentiles = np.array(
        [
            50,
            50 - 68 / 2,
            50 + 68 / 2,
            50 - 95 / 2,
            50 + 95 / 2,
            50 - 99.7 / 2,
            50 + 99.7 / 2,
        ]
    )
    labels = (
        'med -1s +1s -2s +2s -3s +3s '
        'mean2 mean3 mean5 mean '
        'meanc2 meanc3 meanc5 meanc '
        'fmean2 fmean3 fmean5 fmean '
        'fmeanc2 fmeanc3 fmeanc5 fmeanc'
    ).split()
    blowup_detection = {}
    for i, E_loc_i in enumerate(E_loc):
        stat = np.empty(len(labels))
        stat_thre = np.empty_like(stat)
        a = np.empty_like(stat)
        stat[:7] = np.percentile(E_loc_i, percentiles)
        stat_thre[:7] = 3
        stat[7:11] = E_loc_i.mean()
        stat[15:19] = E_loc_i.mean()
        if i > 5:
            E_loc_i, _ = log_clip(E_loc_i, *ewm_mean[[0, 5, 6]])  # noqa: F821
        stat[11:15] = E_loc_i.mean()
        stat[19:23] = E_loc_i.mean()
        stat_thre[7:15] = [2, 3, 5, np.inf, 2, 3, 5, np.inf]
        stat_thre[15:23] = [2, 3, 5, np.inf, 2, 3, 5, np.inf]
        a[:15] = min(0.96, 1 - 1 / (2 + i / 10))
        a[15:23] = min(0.8, 1 - 1 / (2 + i / 10))
        if i == 0:
            ewm_mean = stat.copy()
            ewm_var = np.zeros_like(ewm_mean)
            ewm_err = np.zeros_like(ewm_mean)
            is_outlier = np.zeros_like(ewm_mean, dtype=bool)
            n_outlier = np.zeros_like(ewm_mean)
        else:
            if i > 5:
                is_outlier = np.abs(stat - ewm_mean) > stat_thre * np.sqrt(ewm_var)
            ewm_var_new = (1 - a) * (stat - ewm_mean) ** 2 + a * ewm_var
            ewm_mean_new = (1 - a) * stat + a * ewm_mean
            ewm_err_new = (1 - a) ** 2 * ewm_var + a ** 2 * ewm_err
            ewm_var = np.where(is_outlier, ewm_var, ewm_var_new)
            ewm_err = np.where(is_outlier, ewm_err, ewm_err_new)
            ewm_mean = np.where(is_outlier, ewm_mean, ewm_mean_new)
            n_outlier = np.where(is_outlier, n_outlier + 1, 0)
        blowup_candidate = is_outlier[1:7].sum() + is_outlier[8] >= 6
        if blowup_candidate:
            if not blowup_detection:
                blowup_detection = {
                    'step': i,
                    'start': ewm_mean[9],
                    'accum_delta': 0,
                }
            else:
                blowup_detection['step'] = i
        if blowup_detection and i - blowup_detection['step'] > 50:
            blowup_detection = {}
        if blowup_detection:
            blowup_detection['delta'] = (
                ewm_mean[8] - blowup_detection['start']
            ) / np.sqrt(ewm_var[8])
            blowup_detection['accum_delta'] += blowup_detection['delta']
        blowup = blowup_detection.get('delta', 0) > 0.5
        traj.append(
            {
                **dict(zip(labels, stat)),
                **dict(zip((f'ewmm_{l}' for l in labels), ewm_mean)),
                **dict(zip((f'ewms_{l}' for l in labels), np.sqrt(ewm_var))),
                **dict(zip((f'ewme_{l}' for l in labels), np.sqrt(ewm_err))),
                **dict(zip((f'out_{l}' for l in labels), is_outlier)),
                **dict(zip((f'nout_{l}' for l in labels), n_outlier)),
                'blowup_candidate': blowup_candidate,
                'blowup': blowup,
                'accum_delta': blowup_detection.get('accum_delta', 0),
                'delta': blowup_detection.get('delta', 0),
            }
        )
    traj = pd.DataFrame(traj).rename_axis('step').reset_index()
    return traj
