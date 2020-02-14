from pathlib import Path

import click
import torch

from deepqmc import train

from . import wf_from_file
from .experiments import all_systems


@click.group()
def cli():
    pass


@cli.command('train')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--state', type=click.Path(dir_okay=False))
@click.option('--save-every', default=100, show_default=True)
@click.option('--cuda/--no-cuda', default=True)
def train_from_file(path, state, save_every, cuda):
    state = torch.load(state) if state and Path(state).is_file() else None
    wf, train_kwargs = wf_from_file(path, state)
    if cuda:
        wf.cuda()
    train(
        wf, cwd=Path(path).parent, state=state, save_every=save_every, **train_kwargs,
    )


@cli.group()
@click.argument('path')
@click.pass_context
def prepare(ctx, path):
    ctx.ensure_object(dict)
    ctx.obj['path'] = path


for cmd in [all_systems]:
    prepare.add_command(cmd)
