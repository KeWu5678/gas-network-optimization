from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

from . import experiment
from .config import ConfigError
from .gas.data import InvalidDataError


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # keep progress observable when stdout is redirected to a file
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    try:
        args.command(args)
    except (ConfigError, InvalidDataError, FileNotFoundError) as err:
        print('error: {}'.format(err), file=sys.stderr)
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='gasnetopt',
        description='Mixed-integer optimal control of energy networks: '
                    'each experiment is one TOML file in configs/.')
    commands = parser.add_subparsers(metavar='command', required=True)

    run = commands.add_parser(
        'run', help='solve an experiment, write results/<config name>/')
    run.add_argument('config', type=Path)
    run.set_defaults(command=_run)

    present = commands.add_parser(
        'present', help='redraw the figures of a finished run')
    present.add_argument('results_dir', type=Path)
    present.add_argument('--format', default='pdf', choices=['pdf', 'png'])
    present.set_defaults(command=_present)

    table = commands.add_parser(
        'table', help='print the method comparison of a run (Markdown)')
    table.add_argument('results_dir', type=Path)
    table.set_defaults(command=_table)

    topology = commands.add_parser(
        'topology', help='draw the network of a gas experiment')
    topology.add_argument('config', type=Path)
    topology.set_defaults(command=_topology)
    return parser


def _run(args: argparse.Namespace) -> None:
    experiment.run(args.config)


def _present(args: argparse.Namespace) -> None:
    paths = experiment.present(args.results_dir, args.format)
    print('saved {}'.format(', '.join(path.name for path in paths)))


def _table(args: argparse.Namespace) -> None:
    print(experiment.table(args.results_dir))


def _topology(args: argparse.Namespace) -> None:
    print('saved {}'.format(experiment.draw_network(args.config)))
