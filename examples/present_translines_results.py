'''
Present results of the transmission lines example computed with
examples/run_translines_adm.py (saves PNG/PDF figures; the former tikz export
via matplotlib2tikz was dropped since that package is deprecated). The
model is rebuilt from the config.toml saved next to the results.

Usage:
    uv run python examples/present_translines_results.py \
        results/translines [--format png]
'''

import argparse
from pathlib import Path

from gasnetopt.translines import experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_dir', type=Path,
                    help='output directory of a run (config.toml, '
                         'results.pkl)')
    ap.add_argument('--format', default='pdf', choices=['pdf', 'png'])
    args = ap.parse_args()
    paths = experiment.present(args.results_dir, args.format)
    print('Saved figures to {}: {}'.format(
        args.results_dir, ', '.join(path.name for path in paths)))


if __name__ == '__main__':
    main()
