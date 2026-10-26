'''
CIAP / penalty-ADM benchmark for the transmission lines example (replaces the
former pocadm.py __main__ / prepare_translines_results).

Runs the POC relaxation and then any of: SUR, COMB-CIAP (dwell-time MILP),
penalty ADM with/without CIAP. Results and the config are stored in the
output directory of the config for examples/present_translines_results.py.

Usage:
    uv run python examples/run_translines_adm.py
        [configs/translines_coarse.toml] [--methods sur comb adm adm_ciap]
'''

import argparse
import sys

from gasnetopt import REPO_ROOT
from gasnetopt.core.pipeline import METHODS
from gasnetopt.translines import experiment


def main() -> None:
    # keep progress observable when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config', nargs='?',
                    default=REPO_ROOT / 'configs' / 'translines_coarse.toml')
    ap.add_argument('--methods', nargs='*', choices=METHODS,
                    default=list(METHODS))
    args = ap.parse_args()
    experiment.run(args.config, tuple(args.methods))


if __name__ == '__main__':
    main()
