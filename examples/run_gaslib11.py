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

# CVD-safe categorical triple (blue / orange / purple), reused per group
GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#9467bd']


def plot_solution(s, title, fname, save_dir, obj=None):
    '''One figure per method: switch schedules, delivery mismatch, pressures.

    The delivery panel shows delivered - demand around a zero baseline (the
    quantity the objective penalizes) instead of two overlapping curves.'''
    import matplotlib.pyplot as plt
    n_sw = len(s['switch_ids'])
    fig, axes = plt.subplots(n_sw + 2, 1, figsize=(9, 2.0 * (n_sw + 2)),
                             sharex=True, num=title, clear=True)
    t = s['t'] / 3600.
    tc = t[:-1]

    # switch schedules (relaxed in [0, 1], binary after rounding/ADM)
    for j, sid in enumerate(s['switch_ids']):
        ax = axes[j]
        w = s['w_switch'][:, j]
        ax.step(tc, w, where='post', color=GROUP_COLORS[0], linewidth=2)
        ax.fill_between(tc, w, step='post', color=GROUP_COLORS[0],
                        alpha=0.12)
        ax.set_ylabel(sid.split('_')[0])
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0., 1.])

    # delivery mismatch per sink
    ax = axes[n_sw]
    err = 0.
    for i, sid in enumerate(s['sink_ids']):
        mis = s['delivered_kg_s'][sid] - s['demand_kg_s'][sid]
        err = max(err, float(np.abs(mis).max()))
        ax.plot(tc, mis, color=GROUP_COLORS[i % len(GROUP_COLORS)],
                linewidth=2, label=sid)
    ax.axhline(0., color='0.6', linewidth=1)
    ax.set_ylabel('delivered - demand\n[kg/s]')
    ax.legend(fontsize=8, ncol=len(s['sink_ids']))

    # node pressures: entries solid, exits dashed (color = member within
    # group), inner nodes muted gray
    ax = axes[-1]
    for nid, p in s['node_pressure_bar'].items():
        if nid in s['src_ids']:
            i = s['src_ids'].index(nid)
            ax.plot(tc, p, color=GROUP_COLORS[i % len(GROUP_COLORS)],
                    linewidth=1.8, label=nid)
        elif nid in s['sink_ids']:
            i = s['sink_ids'].index(nid)
            ax.plot(tc, p, '--', color=GROUP_COLORS[i % len(GROUP_COLORS)],
                    linewidth=1.8, label=nid)
        else:
            ax.plot(tc, p, color='0.75', linewidth=1)
    ax.plot([], [], color='0.75', linewidth=1, label='inner nodes')
    ax.set_ylabel('p [bar]')
    ax.set_xlabel('time [h]')
    ax.legend(fontsize=7, ncol=4)

    if obj is not None:
        title = '{}   (objective {:.2e}, max delivery error {:.3g} kg/s)' \
            .format(title, obj, err)
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

    findings = gaslib_io.validate(net, bc)
    for finding in findings:
        print('data validation: {}'.format(finding))
    assert not findings, 'aborting on invalid input data'

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
                  args.save_dir, obj=obj)

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
        plot_solution(s, 'GasLib-11 SUR', 'sur.png', args.save_dir, obj=obj)

    if args.method in ('all', 'ADM'):
        print('\n=== Penalty ADM with CIAP (tau_min = {:.0f} s) ==='.format(
            args.tau_min))
        y, u, beta, v_ref_out, nodes, obj, w_adm, timings = adm.miocp_adm(
            model, tau_min=tau, with_CIAP=True,
            milp_backend=args.milp_backend)
        print('timings: {}'.format(
            {k: round(v, 2) for k, v in timings.items()}))
        s = model.solution_dict(w_adm)
        plot_solution(s, 'GasLib-11 penalty ADM', 'adm.png', args.save_dir,
                      obj=obj)

        # report switching structure
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
