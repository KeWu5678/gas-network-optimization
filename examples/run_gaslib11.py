'''
Gas network prototype on GasLib-11 with TRR154 transient boundary data:
POC relaxation -> Sum-Up Rounding -> penalty ADM with dwell-time constraints
for the valve and the two compressor stations.

Usage:
    uv run python examples/run_gaslib11.py [--horizon 7200] [--nt 121]
        [--tau-min 900] [--method all] [--save-dir results/gaslib11]
'''

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from gasnetopt import DATA_DIR, adm, ciap
from gasnetopt.gas import gaslib_io
from gasnetopt.gas.ocmodel import GasOCModel


def plot_solution(s, title, fname, save_dir):
    import matplotlib.pyplot as plt
    n_sw = len(s['switch_ids'])
    n_sink = len(s['sink_ids'])
    fig, axes = plt.subplots(n_sw + n_sink + 1, 1,
                             figsize=(9, 2.0 * (n_sw + n_sink + 1)),
                             sharex=True, num=title, clear=True)
    t = s['t'] / 3600.
    tc = t[:-1]

    for j, sid in enumerate(s['switch_ids']):
        axes[j].step(tc, s['w_switch'][:, j], where='post')
        axes[j].set_ylabel(sid.split('_')[0])
        axes[j].set_ylim(-0.05, 1.05)
    for i, sid in enumerate(s['sink_ids']):
        ax = axes[n_sw + i]
        ax.plot(tc, s['demand_kg_s'][sid], 'k--', label='demand')
        ax.plot(tc, s['delivered_kg_s'][sid], 'b-', label='delivered')
        ax.set_ylabel(sid + ' [kg/s]')
        if i == 0:
            ax.legend(fontsize=8)
    ax = axes[-1]
    for nid, p in s['node_pressure_bar'].items():
        ax.plot(tc, p, label=nid, linewidth=1)
    ax.set_ylabel('p [bar]')
    ax.set_xlabel('time [h]')
    ax.legend(fontsize=6, ncol=4)
    fig.suptitle(title)
    fig.tight_layout()
    out = Path(save_dir) / fname
    fig.savefig(out, dpi=150)
    print('saved {}'.format(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--horizon', type=float, default=7200.,
                    help='time horizon [s]')
    ap.add_argument('--nt', type=int, default=121,
                    help='number of time grid points')
    ap.add_argument('--nx', type=int, default=2, help='cells per pipe')
    ap.add_argument('--tau-min', type=float, default=900.,
                    help='minimal dwell time [s]')
    ap.add_argument('--method', choices=['all', 'SUR', 'ADM'], default='all')
    ap.add_argument('--save-dir', default='results/gaslib11')
    ap.add_argument('--milp-backend', choices=['highs', 'gurobi'],
                    default=None)
    args = ap.parse_args()

    matplotlib.use('Agg')
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    g11 = DATA_DIR / 'gaslib' / 'GasLib-11'
    net = gaslib_io.parse_net(g11 / 'GasLib-11-v1-20211130.net')
    bc = gaslib_io.parse_bcd(g11 / 'GasLib-11-sinus-InputData.bcd')
    print('Network {}: {} nodes, {} pipes, {} valves, {} compressors'.format(
        net.name, len(net.nodes), len(net.pipes), len(net.valves),
        len(net.compressors)))

    model = GasOCModel(net, bc, T=args.horizon, nt=args.nt, nx=args.nx)
    print('NLP: {} variables, {} constraints, {} configurations '
          '({} switches)\n'.format(
              model.nw, len(model.lbg), model.n_confg, model.nv_ref))

    solver = model.create_NLP_solver(tol=1e-6)
    v_ref = np.array([[1] + [0] * (model.nv_ref - 1)] * (len(model.t) - 1))
    p = np.concatenate(([0.], v_ref.flatten()))

    print('=== POC relaxation ===')
    sol = solver(x0=model.w0, p=p, lbx=model.lbw, ubx=model.ubw,
                 lbg=model.lbg, ubg=model.ubg)
    ret = solver.stats()['return_status']
    assert ret in ['Solve_Succeeded', 'Solved_To_Acceptable_Level'], ret
    w_poc = sol['x'].full().flatten()
    obj = model.evaluate_objective(0., v_ref, w_poc)
    print('status {}, objective {:.6e}'.format(ret, obj))
    s = model.solution_dict(w_poc)
    plot_solution(s, 'GasLib-11 POC relaxation', 'poc_relaxed.png',
                  args.save_dir)

    y, u, alpha, v, nodes = model.extract(w_poc)
    tau = args.tau_min / model.scaling.T_ref

    if args.method in ('all', 'SUR'):
        print('\n=== Sum-Up Rounding + reoptimization ===')
        beta, _ = ciap.solve_ciap(model.t, alpha, strategy='SUR')
        y, u, beta, v, nodes, w_sur, _ = adm.resolve_with_fixed_controls(
            model, solver, w_poc.copy(), p, alpha_fix=beta)
        obj = model.evaluate_objective(0., v_ref, w_sur)
        print('objective {:.6e}'.format(obj))
        s = model.solution_dict(w_sur)
        plot_solution(s, 'GasLib-11 SUR', 'sur.png', args.save_dir)

    if args.method in ('all', 'ADM'):
        print('\n=== Penalty ADM with CIAP (tau_min = {:.0f} s) ==='.format(
            args.tau_min))
        y, u, beta, v_ref_out, nodes, obj, w_adm, timings = adm.miocp_adm(
            model, tau_min=tau, with_CIAP=True,
            milp_backend=args.milp_backend)
        print('timings: {}'.format(
            {k: round(v, 2) for k, v in timings.items()}))
        s = model.solution_dict(w_adm)
        plot_solution(s, 'GasLib-11 penalty ADM', 'adm.png', args.save_dir)

        # report switching structure
        t_sec = s['t'][:-1]
        for j, sid in enumerate(s['switch_ids']):
            x = np.round(s['w_switch'][:, j]).astype(int)
            n_switch = int(np.abs(np.diff(x)).sum())
            print('{}: {} switching events, state {:.0%} on'.format(
                sid, n_switch, x.mean()))
        err = max(np.abs(s['delivered_kg_s'][k] - s['demand_kg_s'][k]).max()
                  for k in s['sink_ids'])
        print('max delivery error: {:.4f} kg/s'.format(err))


if __name__ == '__main__':
    main()
