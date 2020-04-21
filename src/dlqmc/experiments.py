import os
from copy import deepcopy
from itertools import product
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tables
import toml
from uncertainties import ufloat

from deepqmc.utils import NestedDict


@click.command()
@click.pass_context
def all_systems(ctx):
    systems = [
        'H2',
        'B',
        'Be',
        'LiH',
        *({'name': 'Hn', 'n': 10, 'dist': d} for d in [1.2, 1.8, 3.6]),
    ]
    cass = {'H2': [2, 2], 'B': [4, 3], 'LiH': [4, 2], 'Hn': [6, 4], 'Be': [4, 2]}
    param_sets = ['SD-SJ', 'SD-SJBF', 'MD-SJ', 'MD-SJBF']
    for system, param_set in product(systems, param_sets):
        sys_name = system if isinstance(system, str) else system['name']
        sys_label = sys_name
        if sys_name == 'Hn':
            sys_label += f'-{system["dist"]}'
        path = ctx.obj['basedir'] / sys_label / param_set
        if path.exists():
            continue
        print(path)
        params = NestedDict()
        params['system'] = system
        if 'MD' in param_set:
            params['model_kwargs.cas'] = cass[sys_name]
        if 'BF' not in param_set:
            params['model_kwargs.omni_kwargs.with_backflow'] = False
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(params, encoder=toml.TomlEncoder()))


def collect_all_systems(basedir):
    results = []
    for path in Path(basedir).glob('**/blocks.h5'):
        with tables.open_file(path) as f:
            if 'blocks' not in f.root:
                continue
            ene = f.root['blocks'].col('energy')
            ene = ufloat(ene.mean(), ene.mean(0).std() / np.sqrt(ene.shape[-1]))
            system, ansatz = str(path).split('/')[-3:-1]
            results.append({'system': system, 'ansatz': ansatz, 'energy': ene})
    results = pd.DataFrame(results).set_index(['system', 'ansatz'])
    return results


@click.command()
@click.pass_context
def hyperparam_scan_co2(ctx):
    learning_rates = [0.3e-3, 1e-3, 3e-3]
    batch_sizes = [1000, 2000, 4000]
    epoch_sizes = [3, 5, 8]
    ns_decorrelate = [5, 10, 20]
    payload = product(learning_rates, batch_sizes, epoch_sizes, ns_decorrelate)
    for lr, bs, es, n_decorr in payload:
        label = f'lr-{lr}_bs-{bs}_es-{es}_decorr-{n_decorr}'
        path = ctx.obj['basedir'] / label
        if path.exists():
            continue
        print(path)
        params = NestedDict()
        params['system'] = 'CO2'
        params['train_kwargs.n_steps'] = 2000
        params['train_kwargs.learning_rate'] = lr
        params['train_kwargs.batch_size'] = bs
        params['train_kwargs.epoch_size'] = es
        params['train_kwargs.sampler_kwargs.n_decorrelate'] = n_decorr
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(params, encoder=toml.TomlEncoder()))


@click.command()
@click.argument('training', type=click.Path(exists=True))
@click.pass_context
def sampling(ctx, training):
    for param_path in Path(training).glob('**/param.toml'):
        train_path = param_path.parent
        label = train_path.relative_to(training)
        chkpts = [
            p.stem.split('-')[1] for p in (train_path / 'chkpts').glob('state-*.pt')
        ]
        if not chkpts:
            continue
        step = max(chkpts)
        path = ctx.obj['basedir'] / label
        print(path)
        params = toml.loads((train_path / 'param.toml').read_text())
        train_path = Path(os.path.relpath(train_path.resolve(), path.resolve()))
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(params, encoder=toml.TomlEncoder()))
        (path / 'state.pt').symlink_to(train_path / f'chkpts/state-{step}.pt')


@click.command()
@click.argument('param')
@click.pass_context
def cyclobutadiene(ctx, param):
    templ = toml.loads(Path(param).read_text())
    for label in ['ground', 'transition']:
        path = ctx.obj['basedir'] / label
        print(path)
        param = deepcopy(templ)
        param['system'] = f'dlqmc.systems:cyclobutadiene_{label}'
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(param))
