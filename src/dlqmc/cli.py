from pathlib import Path

import click
import torch

from deepqmc import evaluate, train

from . import wf_from_file
from .experiments import all_systems, sampling


@click.group()
def cli():
    pass


@cli.group()
@click.argument('path')
@click.pass_context
def prepare(ctx, path):
    ctx.ensure_object(dict)
    ctx.obj['basedir'] = Path(path)


for cmd in [all_systems, sampling]:
    prepare.add_command(cmd)


@cli.command('train')
@click.argument('path', type=click.Path(exists=True))
@click.option('--save-every', default=100, show_default=True)
@click.option('--cuda/--no-cuda', default=True)
def train_at(path, save_every, cuda):
    path = Path(path).resolve()
    state_file = path / 'state.pt'
    state = torch.load(state_file) if state_file.is_file() else None
    wf, params = wf_from_file(path / 'param.toml', state)
    if cuda:
        wf.cuda()
    train(
        wf,
        cwd=Path(path),
        state=state,
        save_every=save_every,
        **params.get('train_kwargs', {}),
    )


@cli.command('evaluate')
@click.argument('path', type=click.Path(exists=True))
@click.option('--cuda/--no-cuda', default=True)
def evaluate_at(path, cuda):
    path = Path(path).resolve()
    state = torch.load(path / 'state.pt', map_location=None if cuda else 'cpu')
    for _ in range(20):
        try:
            wf, params = wf_from_file(path / 'param.toml', state)
        except RuntimeError as exp:
            if 'size mismatch for conf_coeff.weight' not in exp.args[0]:
                raise
        else:
            break
    if cuda:
        wf.cuda()
    evaluate(wf, cwd=path, **params.get('evaluate_kwargs', {}))
