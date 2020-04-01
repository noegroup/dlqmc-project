import logging
from pathlib import Path

import click
import tomlkit
import torch
from torch.utils.tensorboard import SummaryWriter

from deepqmc import evaluate, train
from deepqmc.errors import TrainingBlowup
from deepqmc.wf import PauliNet

from . import experiments, wf_from_file
from .defaults import DEEPQMC_MAPPING, collect_kwarg_defaults

log = logging.getLogger(__name__)


@click.group()
def cli():
    logging.basicConfig(style='{', format='{message}', datefmt='%H:%M:%S')
    logging.getLogger('deepqmc').setLevel(logging.DEBUG)


@cli.command()
def defaults():
    table = tomlkit.table()
    table['model_kwargs'] = collect_kwarg_defaults(PauliNet.from_hf, DEEPQMC_MAPPING)
    table['train_kwargs'] = collect_kwarg_defaults(train, DEEPQMC_MAPPING)
    table['evaluate_kwargs'] = collect_kwarg_defaults(evaluate, DEEPQMC_MAPPING)
    click.echo(tomlkit.dumps(table), nl=False)


@cli.command()
@click.argument('basedir', type=click.Path(exists=False))
@click.argument('HF', type=float)
@click.argument('exact', type=float)
@click.option('--fractions', default='0,90,99,100', type=str)
@click.option('--steps', '-n', default=2_000, type=int)
def draw_hlines(basedir, hf, exact, fractions, steps):
    basedir = Path(basedir)
    fractions = [float(x) / 100 for x in fractions.split(',')]
    for fraction in fractions:
        value = hf + fraction * (exact - hf)
        workdir = basedir / f'line-{value:.3f}'
        with SummaryWriter(log_dir=workdir, flush_secs=15, purge_step=0) as writer:
            for step in range(steps):
                writer.add_scalar('E_loc_loss/mean', value, step)
                writer.add_scalar('E_loc/mean', value, step)


@cli.group()
@click.argument('path')
@click.pass_context
def prepare(ctx, path):
    ctx.ensure_object(dict)
    ctx.obj['basedir'] = Path(path)


for attr in dir(experiments):
    obj = getattr(experiments, attr)
    if isinstance(obj, click.core.Command):
        prepare.add_command(obj)


@cli.command('train')
@click.argument('workdir', type=click.Path(exists=True))
@click.option('--save-every', default=100, show_default=True)
@click.option('--cuda/--no-cuda', default=True)
@click.option('--max-restarts', default=3, show_default=True)
@click.option('--min-rewind', default=30, show_default=True)
def train_at(workdir, save_every, cuda, max_restarts, min_rewind):
    workdir = Path(workdir).resolve()
    state_file = workdir / 'state.pt'
    if not state_file.is_file():
        state_file = None
    for attempt in range(max_restarts + 1):
        state = torch.load(state_file) if state_file else None
        wf, params = wf_from_file(workdir / 'param.toml', state)
        if cuda:
            wf.cuda()
        try:
            train(
                wf,
                workdir=workdir,
                state=state,
                save_every=save_every,
                **params.get('train_kwargs', {}),
            )
        except TrainingBlowup as e:
            if attempt == max_restarts:
                log.error(f'Detected blowup, maximum number of restarts reached')
                break
            for step, sf in reversed(e.chkpts):
                if step >= e.step - min_rewind:
                    continue
                state_file = sf
                log.warning(f'Detected blowup, restarting from step {step + 1}')
                break
            else:
                log.warning(f'Detected blowup, restarting from beginnig')
        else:
            break


@cli.command('evaluate')
@click.argument('workdir', type=click.Path(exists=True))
@click.option('--cuda/--no-cuda', default=True)
def evaluate_at(workdir, cuda):
    workdir = Path(workdir).resolve()
    state = torch.load(workdir / 'state.pt', map_location=None if cuda else 'cpu')
    for _ in range(20):
        try:
            wf, params = wf_from_file(workdir / 'param.toml', state)
        except RuntimeError as exp:
            if 'size mismatch for conf_coeff.weight' not in exp.args[0]:
                raise
        else:
            break
    if cuda:
        wf.cuda()
    evaluate(wf, workdir=workdir, **params.get('evaluate_kwargs', {}))
