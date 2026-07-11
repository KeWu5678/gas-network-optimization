'''
Present results of the transmission lines example computed with
examples/run_translines_adm.py (saves PNG/PDF figures; the former tikz export
via matplotlib2tikz was dropped since that package is deprecated).

Usage:
    uv run python examples/present_translines_results.py \
        results/translines/translines_extended_tree_results [--show]
'''

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from gasnetopt.results_io import load_results
from gasnetopt.translines.model import extract_solution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', help='results pickle (without .pkl)')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--format', default='pdf', choices=['pdf', 'png'])
    args = ap.parse_args()

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    db = load_results(args.results)
    net_data = db['net']
    demand = db['demand']
    T = db['T']
    nt = db['nt']
    nx = db['nx']

    t = np.linspace(0, T, nt)
    _, A, producers, _, consumers, configs = net_data

    def net():
        return net_data

    # mode encoding from the model that produced the results (fallback for
    # pickles written before it was stored)
    r = db.get('r', np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))

    def step_fmt(v):
        return np.concatenate((v.flatten(), [np.nan]))

    keys = ['poc', 'sur', 'comb', 'adm_wo_ciap', 'adm_with_ciap']
    labels = ['POC', 'SUR', 'COMB-CIAP', 'ADM', 'ADM + CIAP']
    results, used_labels = [], []
    for key, label in zip(keys, labels):
        if key in db:
            results.append(db[key])
            used_labels.append(label)

    out_dir = Path(args.results).parent
    saved = []

    # discrete controls
    for ctrl in [0, 1]:
        plt.figure(ctrl + 1).clear()
        ax = plt.gca()
        for i, (result, label) in enumerate(zip(results, used_labels)):
            u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net,
                                                    nt, nx)
            v = alpha.dot(r)
            bl = 1.1 * (len(results) - 1 - i)
            p = ax.step(t, step_fmt(v[:, ctrl] + bl), where='post',
                        label=label)
            ax.fill_between(t, step_fmt(v[:, ctrl] + bl), y2=bl,
                            color=p[0].get_color(), step='post', alpha=0.1)
        ax.legend(loc='lower right')
        ax.set_ylim(-0.02, 1.1 * len(results) + 0.32)
        ax.set_xlim(0, T)
        ax.set_xlabel(r'time $t$')
        ax.yaxis.set_visible(False)
        saved.append('translines_discrete_control_{}'.format(ctrl + 1))
        plt.savefig(out_dir / (saved[-1] + '.' + args.format))

    # continuous controls
    for ctrl in [0, 1]:
        plt.figure(ctrl + 3).clear()
        ax = plt.gca()
        for result, label in zip(results, used_labels):
            u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net,
                                                    nt, nx)
            ax.step(t, step_fmt(u[:, ctrl]), label=label)
        ax.legend(loc='lower center')
        ax.set_xlim(0, T)
        ax.set_xlabel(r'time $t$')
        ax.set_ylabel(r'Continuous control $u_{}$'.format(ctrl + 1))
        saved.append('translines_continuous_control_{}'.format(ctrl + 1))
        plt.savefig(out_dir / (saved[-1] + '.' + args.format))

    # demand vs delivery
    n_consumer = len(consumers)
    fig, axes = plt.subplots(n_consumer, 1, num=5, clear=True,
                             figsize=(8, 2.0 * n_consumer))
    end_vertices = [j for (i, j) in A]
    for i, consumer in enumerate(consumers):
        line = end_vertices.index(consumer)
        axes[i].plot(t[:-1], demand[i, :], 'k--', label='demand')
        for result, label in zip(results, used_labels):
            u, alpha, xi_p, xi_m = extract_solution({'x': result[-2]}, net,
                                                    nt, nx)
            axes[i].plot(t, xi_p[line][-1, :], label=label)
        if i == 0:
            axes[i].legend(loc='lower center', fontsize=7)
        axes[i].set_xlim(0, T)
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel('Consumer {}'.format(i + 1))
    fig.tight_layout()
    saved.append('translines_demand_and_delivery')
    plt.savefig(out_dir / (saved[-1] + '.' + args.format))

    if args.show:
        plt.show()
    print('Saved figures to {}: {}'.format(out_dir, ', '.join(saved)))


if __name__ == '__main__':
    main()
