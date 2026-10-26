'''
Regenerate the method-comparison table (Markdown) from the results of a
run (examples/run_translines_adm.py or run_gaslib11.py). The README embeds
its output.

Usage:
    uv run python examples/make_results_table.py results/translines
'''

import argparse
from pathlib import Path

import numpy as np

from gasnetopt.results_io import load_results

METHODS = [
    ('poc', 'POC relaxation', 'no (fractional)'),
    ('sur', 'Sum-Up Rounding + reopt.', 'no'),
    ('comb', 'COMB-CIAP (dwell MILP)', 'yes'),
    ('adm', 'penalty ADM', 'yes'),
    ('adm_ciap', 'penalty ADM + CIAP', 'yes'),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results_dir', type=Path,
                    help='output directory of a run (results.pkl)')
    args = ap.parse_args()

    db = load_results(args.results_dir / 'results')
    r = db['r']

    print('| method | objective | switching events | dwell-feasible |')
    print('|---|---|---|---|')
    for key, label, feas in METHODS:
        if key not in db:
            continue
        solution = db[key]
        beta = np.asarray(solution.alpha)
        if np.abs(beta - np.round(beta)).max() < 1e-6:
            v_sw = np.round(beta).dot(r)
            switches = str(int(np.abs(np.diff(v_sw, axis=0)).sum()))
        else:
            switches = '-'
        print('| {} | {:.5g} | {} | {} |'.format(
            label, solution.objective, switches, feas))


if __name__ == '__main__':
    main()
