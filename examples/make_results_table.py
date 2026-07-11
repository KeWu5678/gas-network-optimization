'''
Regenerate the method-comparison table (Markdown) from a results pickle
written by examples/run_translines_adm.py. The README embeds its output.

Usage:
    uv run python examples/make_results_table.py \
        results/translines/translines_extended_tree_results
'''

import argparse

import numpy as np

from gasnetopt.results_io import load_results

METHODS = [
    ('poc', 'POC relaxation', 'no (fractional)'),
    ('sur', 'Sum-Up Rounding + reopt.', 'no'),
    ('comb', 'COMB-CIAP (dwell MILP)', 'yes'),
    ('adm_wo_ciap', 'penalty ADM', 'yes'),
    ('adm_with_ciap', 'penalty ADM + CIAP', 'yes'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', help='results pickle (without .pkl)')
    args = ap.parse_args()

    db = load_results(args.results)
    r = db['r']

    print('| method | objective | switching events | dwell-feasible |')
    print('|---|---|---|---|')
    for key, label, feas in METHODS:
        if key not in db:
            continue
        _, _, beta, _, _, obj, _, _ = db[key]
        beta = np.asarray(beta)
        if np.abs(beta - np.round(beta)).max() < 1e-6:
            v_sw = np.round(beta).dot(r)
            switches = str(int(np.abs(np.diff(v_sw, axis=0)).sum()))
        else:
            switches = '-'
        print('| {} | {:.3f} | {} | {} |'.format(label, obj, switches, feas))


if __name__ == '__main__':
    main()
