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
    for system in systems:
        sys_name = system if isinstance(system, str) else system['name']
        sys_label = sys_name
        if sys_name == 'Hn':
            sys_label += f'-{system["dist"]}'
        for param_set in param_sets:
            path = Path(f'{ctx.obj["path"]}/{sys_label}/{param_set}')
            if path.exists():
                continue
            params = NestedDict()
            params['system'] = system
            if 'MD' in param_set:
                params['pauli_kwargs.cas'] = cass[sys_name]
            if 'BF' not in param_set:
                params['pauli_kwargs.omni_kwargs.with_backflow'] = False
            path.mkdir(parents=True)
            (path / 'param.toml').write_text(
                toml.dumps(params, encoder=toml.TomlEncoder())
            )
