'''
POC relaxation + Sum-Up Rounding + reoptimization for the transmission lines
example (replaces the former translines.py __main__).

Usage:
    uv run python examples/run_translines.py [--net extended-tree] [--fine]
        [--no-sur] [--save-dir results/translines] [--show]
'''

import argparse
from pathlib import Path

import casadi as cas
import matplotlib
import numpy as np

from gasnetopt.ciap import sum_up_rounding
from gasnetopt.translines import model, networks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--net', choices=['extended-tree', 'subgrid'],
                    default='extended-tree')
    ap.add_argument('--fine', action='store_true',
                    help='dt = dx = 0.25 instead of 0.5')
    ap.add_argument('--no-sur', action='store_true',
                    help='skip sum-up rounding and reoptimization')
    ap.add_argument('--save-dir', default='results/translines')
    ap.add_argument('--show', action='store_true',
                    help='show figures interactively instead of saving')
    args = ap.parse_args()

    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    net = (networks.network_extended_tree if args.net == 'extended-tree'
           else networks.network_subgrid)
    T = 26.
    name = args.net.replace('-', '_')
    if args.fine:
        demand = model.load_demand('demand_{}.dat'.format(name))
        nt, nx = 4 * 26 + 1, 4
    else:
        demand = model.load_demand('demand_{}_coarse.dat'.format(name))
        nt, nx = 2 * 26 + 1, 2

    nlp, lbw, ubw, lbg, ubg, w0 = model.translines_nlp(
        net, demand, T, nt=nt, nx=nx)

    options = {'ipopt': {'tol': 1e-8}}
    solver = cas.nlpsol('solver', 'ipopt', nlp, options)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    status = solver.stats()['return_status']
    print('POC relaxation: {} (f = {:.6e})'.format(status, float(sol['f'])))

    u, alpha, xi_p, xi_m = model.extract_solution(sol, net, nt, nx)

    if not args.no_sur:
        t = np.linspace(0, T, nt)
        beta = sum_up_rounding(t, alpha)

        # reoptimize continuous controls for fixed integer variables
        for i, v in enumerate(np.reshape(beta, -1, order='F')):
            lbw[i] = v
            ubw[i] = v
        sol2 = solver(x0=sol['x'], lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        status = solver.stats()['return_status']
        print('SUR reoptimization: {} (f = {:.6e})'.format(
            status, float(sol2['f'])))
        u, alpha2, xi_p, xi_m = model.extract_solution(sol2, net, nt, nx)
    else:
        alpha2 = alpha

    model.plot_solution(net, u, alpha2, xi_p, xi_m, demand, T, nt)

    if args.show:
        plt.show()
    else:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        names = ['controls', 'multipliers', 'xi_plus', 'xi_minus', 'demand']
        for num, fname in enumerate(names, start=1):
            plt.figure(num).savefig(save_dir / '{}_{}.png'.format(name, fname),
                                    dpi=150)
        print('Figures saved to {}/'.format(save_dir))


if __name__ == '__main__':
    main()
