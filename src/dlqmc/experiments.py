import os
from itertools import product
from pathlib import Path

import click
import toml

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
        path = ctx.obj['base'] / sys_label / param_set
        if path.exists():
            continue
        params = NestedDict()
        params['system'] = system
        if 'MD' in param_set:
            params['pauli_kwargs.cas'] = cass[sys_name]
        if 'BF' not in param_set:
            params['pauli_kwargs.omni_kwargs.with_backflow'] = False
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
        path = ctx.obj['base'] / label
        params = toml.loads(train_path / 'param.toml')
        train_path = Path(os.path.relpath(train_path.resolve(), path.resolve()))
        path.mkdir(parents=True)
        (path / 'param.toml').write_text(toml.dumps(params, encoder=toml.TomlEncoder()))
        (path / 'state.pt').symlink_to(train_path / f'chkpts/state-{step}.pt')
