import os
import shutil
from itertools import chain, product
from pathlib import Path

import click
import numpy as np
import pandas as pd
import tables
import toml
from uncertainties import ufloat

from deepqmc.utils import NestedDict


@click.command()
@click.argument('param')
@click.pass_context
def custom(ctx, param):
    path = ctx.obj['basedir']
    print(path)
    path.mkdir(parents=True)
    param = Path(param)
    shutil.copy(param, path)
    if toml.loads(param.read_text()).get('hooks'):
        shutil.copy(param.parent / 'hooks.py', path)


@click.command()
@click.argument('script')
@click.pass_context
def script(ctx, script):
    path = ctx.obj['basedir']
    path.mkdir(parents=True)
    shutil.copy(script, path)
    print(path)


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
@click.argument('state', type=click.Path(exists=True), nargs=-1)
@click.option('--param')
@click.pass_context
def sampling_states(ctx, state, param):
    base = os.path.commonpath(state)
    if param:
        param = toml.loads(Path(param).read_text())
    for state_path in sorted(state):
        label = os.path.splitext(os.path.relpath(state_path, base))[0]
        train_path = Path(state_path).parents[1]
        params = NestedDict()
        path = ctx.obj['basedir'] / label
        print(path)
        params_train = toml.loads((train_path / 'param.toml').read_text())
        params.update(params_train)
        if param:
            params.update(param)
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(params, encoder=toml.TomlEncoder()))
        state_path = Path(os.path.relpath(Path(state_path).resolve(), path.resolve()))
        (path / 'state.pt').symlink_to(state_path)


@click.command()
@click.pass_context
def cyclobutadiene(ctx):
    for label in ['ground', 'transition']:
        path = ctx.obj['basedir'] / label
        print(path)
        param = NestedDict()
        param['system'] = f'dlqmc.systems:cyclobutadiene_{label}'
        param['model_kwargs.cas'] = [8, 4]
        param['model_kwargs.pauli_kwargs.conf_cutoff'] = 1e-8
        param['model_kwargs.pauli_kwargs.conf_limit'] = 10
        param['model_kwargs.pauli_kwargs.rc_scaling'] = 3.0
        param['model_kwargs.pauli_kwargs.cusp_alpha'] = 3.0
        param['model_kwargs.pauli_kwargs.use_sloglindet'] = 'training'
        param['model_kwargs.omni_kwargs.subnet_kwargs.n_layers_h'] = 2
        param['train_kwargs.n_steps'] = 5_000
        param['train_kwargs.batch_size'] = 1_000
        param['train_kwargs.epoch_size'] = 5
        param['train_kwargs.fit_kwargs.subbatch_size'] = 500
        param['train_kwargs.sampler_kwargs.n_decorrelate'] = 20
        param['train_kwargs.lr_scheduler_kwargs.CyclicLR.step_size_up'] = 375
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(param, encoder=toml.TomlEncoder()))


@click.command()
@click.pass_context
def boron(ctx):
    payload = chain(
        product(
            ['never', 'training', 'always'],
            [(None, 1), ([4, 3], 3), ([8, 3], 16), ([8, 3], 100)],
            [100, 10],
            [(None, {})],
        ),
        product(
            ['training'],
            [(None, 1), ([8, 3], 16)],
            [100, 10],
            [('lrx2', {'train_kwargs.lr_scheduler_kwargs.CyclicLR.max_lr': 0.02})],
        ),
        product(
            ['training'],
            [(None, 1), ([8, 3], 16)],
            [100, 10],
            [('lrx3', {'train_kwargs.lr_scheduler_kwargs.CyclicLR.max_lr': 0.03})],
        ),
        product(
            ['training'],
            [(None, 1), ([8, 3], 16)],
            [100, 10],
            [('qx2', {'train_kwargs.fit_kwargs.q': 10})],
        ),
    )
    for use_slgld, (cas, conf_lim), epoch_size, (extra_lbl, extra) in payload:
        label = f'{use_slgld}_ndet-{conf_lim}_epoch-{epoch_size}'
        if extra:
            label += f'_{extra_lbl}'
        path = ctx.obj['basedir'] / label
        if path.exists():
            continue
        print(path)
        param = NestedDict()
        param['system'] = 'B'
        param['train_kwargs.n_steps'] = 10_000
        param['train_kwargs.batch_size'] = 10_000
        param['train_kwargs.fit_kwargs.subbatch_size'] = 5_000
        param['train_kwargs.sampler_kwargs.n_discard'] = 10
        param['model_kwargs.pauli_kwargs.conf_cutoff'] = 1e-8
        param['model_kwargs.pauli_kwargs.rc_scaling'] = 3.0
        param['model_kwargs.pauli_kwargs.cusp_alpha'] = 3.0
        param['model_kwargs.omni_kwargs.subnet_kwargs.n_layers_h'] = 2
        param['model_kwargs.pauli_kwargs.conf_limit'] = conf_lim
        param['model_kwargs.pauli_kwargs.use_sloglindet'] = use_slgld
        param['model_kwargs.cas'] = cas
        param['train_kwargs.epoch_size'] = epoch_size
        for k, v in extra.items():
            param[k] = v
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(param, encoder=toml.TomlEncoder()))
