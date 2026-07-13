'''
CIAP / penalty-ADM benchmark for the transmission lines example (replaces the
former pocadm.py __main__ / prepare_translines_results).

Runs the POC relaxation and then any of: SUR, COMB-CIAP (dwell-time MILP),
penalty ADM with/without CIAP. Results are stored in a pickle file under
--save-dir for examples/present_translines_results.py.

Usage:
    uv run python examples/run_translines_adm.py --net "extended tree" \
        --method all --coarse
'''

import argparse
import sys

import numpy as np

from gasnetopt import adm, ciap, results_io
from gasnetopt.translines.model import TranslinesOCModel

METHODS = ['all', 'SUR', 'COMB CIAP', 'ADM', 'ADM without CIAP',
           'ADM with CIAP']


def main() -> None:
    # keep progress observable when stdout is redirected to a file
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--net', choices=['extended tree', 'subgrid'],
                    default='extended tree')
    ap.add_argument('--method', choices=METHODS, default='all')
    ap.add_argument('--coarse', action='store_true',
                    help='dt = dx = 0.5 instead of 0.25')
    ap.add_argument('--tau-min', type=float, default=1.0,
                    help='minimal dwell time')
    ap.add_argument('--save-dir', default='results/translines')
    ap.add_argument('--milp-backend', choices=['highs', 'gurobi'],
                    default=None)
    args = ap.parse_args()

    ocp_model = TranslinesOCModel(args.net, coarse=args.coarse)
    tau_min = args.tau_min
    print('Transmission lines example on {} net with tau_min={:9.3e}\n'
          .format(args.net, tau_min))

    out_file = '{}/translines_{}_results'.format(
        args.save_dir, '_'.join(args.net.split()))

    results_io.update_results(
        out_file, net_name=args.net, net=ocp_model.net(),
        demand=ocp_model.demand, T=ocp_model.T, tau_min=tau_min,
        nt=ocp_model.nt, nx=ocp_model.nx, r=ocp_model.r)

    timings = {'poc_nlp': 0.0, 'reopt_nlp': 0.0, 'comb': 0.0, 'sur': 0.0}
    solver = ocp_model.create_NLP_solver(tol=1e-8)
    v_ref = np.array([[1] + [0] * (ocp_model.nv_ref - 1)]
                     * (len(ocp_model.t) - 1))
    p = np.concatenate(([0], v_ref.flatten()))

    print('Partial Outer Convexification Relaxation:')
    sol = solver(x0=ocp_model.w0, p=p, lbx=ocp_model.lbw, ubx=ocp_model.ubw,
                 lbg=ocp_model.lbg, ubg=ocp_model.ubg)
    ret = solver.stats()['return_status']
    solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
    assert ret in solved, 'Solution of NLP failed'
    w_poc = sol['x'].full().flatten()
    y, u, alpha, v, nodes = ocp_model.extract(w_poc)
    obj_val = float(sol['f'])
    timings['poc_nlp'] = adm._nlp_wall_time(solver)
    results_io.update_results(
        out_file, poc=(y, u, alpha, v, nodes, obj_val, w_poc, dict(timings)))
    print('Final objective value: {:.10e}\n'.format(obj_val))
    print('Timings: {}\n'.format(timings))

    if args.method in ['all', 'SUR']:
        print('Sum-Up Rounding:')
        beta, timings['sur'] = ciap.solve_ciap(ocp_model.t, alpha,
                                               strategy='SUR')
        y, u, beta, v, nodes, w_sur, _ = adm.resolve_with_fixed_controls(
            ocp_model, solver, w_poc.copy(), p, alpha_fix=beta)
        obj_val = ocp_model.evaluate_objective(0., v_ref, w_sur)
        results_io.update_results(
            out_file,
            sur=(y, u, beta, v, nodes, obj_val, w_sur, dict(timings)))
        print('Final objective value: {:.10e}'.format(obj_val))
        print('Timings: {}\n'.format(timings))

    if args.method in ['all', 'COMB CIAP']:
        print('COMB CIAP:')
        beta, timings['sur'] = ciap.solve_ciap(
            ocp_model.t, alpha, strategy='COMB', tau_min=tau_min,
            backend=args.milp_backend)
        y, u, beta, v, nodes, w_comb, _ = adm.resolve_with_fixed_controls(
            ocp_model, solver, w_poc.copy(), p, alpha_fix=beta)
        obj_val = ocp_model.evaluate_objective(0., v_ref, w_comb)
        results_io.update_results(
            out_file,
            comb=(y, u, beta, v, nodes, obj_val, w_comb, dict(timings)))
        print('Final objective value: {:.10e}'.format(obj_val))
        print('Timings: {}\n'.format(timings))

    if args.method in ['all', 'ADM', 'ADM without CIAP']:
        print('ADM without CIAP:', '\n')
        adm_wo_ciap = adm.miocp_adm(ocp_model, tau_min=tau_min,
                                    with_CIAP=False,
                                    milp_backend=args.milp_backend)
        results_io.update_results(out_file, adm_wo_ciap=adm_wo_ciap)
        print('Timings: {}\n'.format(adm_wo_ciap[-1]))

    if args.method in ['all', 'ADM', 'ADM with CIAP']:
        print('ADM with CIAP:', '\n')
        adm_with_ciap = adm.miocp_adm(ocp_model, tau_min=tau_min,
                                      with_CIAP=True,
                                      milp_backend=args.milp_backend)
        results_io.update_results(out_file, adm_with_ciap=adm_with_ciap)
        print('Timings: {}\n'.format(adm_with_ciap[-1]))

    print('Results saved to {}.pkl'.format(out_file))


if __name__ == '__main__':
    main()
