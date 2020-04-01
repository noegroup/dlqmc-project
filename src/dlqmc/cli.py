from pathlib import Path

import click

from . import experiments


@click.group()
def cli():
    pass


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
