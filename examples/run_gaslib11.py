'''
Gas network prototype on GasLib-11 with TRR154 transient boundary data:
POC relaxation -> Sum-Up Rounding -> penalty ADM with dwell-time constraints
for the valve and the two compressor stations.

Usage:
    uv run python examples/run_gaslib11.py [configs/gaslib11_1h.toml]
        [--methods sur adm_ciap]
'''

import argparse
import sys

from gasnetopt import REPO_ROOT
from gasnetopt.core.pipeline import METHODS
from gasnetopt.gas import experiment


def main() -> None:
    # keep progress observable when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config', nargs='?',
                    default=REPO_ROOT / 'configs' / 'gaslib11_1h.toml')
    ap.add_argument('--methods', nargs='*', choices=METHODS,
                    default=list(experiment.DEFAULT_METHODS))
    args = ap.parse_args()
    experiment.run(args.config, tuple(args.methods))


if __name__ == '__main__':
    main()
