'''
POC relaxation + Sum-Up Rounding + reoptimization for the transmission lines
example (replaces the former translines.py __main__).

Usage:
    uv run python examples/run_translines.py
        [configs/translines_coarse.toml] [--no-sur]
'''

import argparse

from gasnetopt import REPO_ROOT
from gasnetopt.translines import experiment


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config', nargs='?',
                    default=REPO_ROOT / 'configs' / 'translines_coarse.toml')
    ap.add_argument('--no-sur', action='store_true',
                    help='skip sum-up rounding and reoptimization')
    args = ap.parse_args()
    experiment.run(args.config, () if args.no_sur else ('sur',))


if __name__ == '__main__':
    main()
