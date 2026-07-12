'''
Penalty Alternating Direction Method for mixed-integer optimal control with
combinatorial constraints (dwell times) that prohibit chattering.

Implements Algorithm "penalty ADM" of

    Göttlich, Hante, Potschka, Schewe: Penalty alternating direction methods
    for mixed-integer optimal control with combinatorial constraints.
    Math. Program. 188 (2021) 599-619, doi:10.1007/s10107-021-01656-9.

Outer penalty loop over rho with the exact L1 coupling penalty, inner ADM
alternation between (a) the POC-relaxed NLP in (y, u, alpha) for fixed
combinatorial reference v_ref and (b) the combinatorial MILP tCOMB in v for
fixed relaxed controls, with the two termination criteria (i)/(ii) and a
final fix-and-reoptimize step for the continuous controls.

The OCP is accessed through the OCModel interface (collocation.OCModel):
extract/overwrite/evaluate_objective, time grid t, mode encoding r.
'''

from __future__ import annotations

import time

import casadi as cas
import numpy as np

from .ciap import _min_up_constraints, solve_ciap
from .collocation import OCModel
from .milp import MilpModel


def resolve_with_fixed_controls(
        ocp_model: OCModel, solver: cas.Function, w0: np.ndarray,
        p: np.ndarray, u_fix: np.ndarray | None = None,
        v_fix: np.ndarray | None = None,
        alpha_fix: np.ndarray | None = None) -> tuple:
    '''Solve problem with some controls fixed by bounds. Can be used to
    emulate a forward simulation.'''
    # adjust bounds of last POC mode to avoid linear dependence with SOS1
    # constraint
    if alpha_fix is not None:
        alpha_bound = alpha_fix.copy()
        alpha_bound[:, -1] = -1.  # < 0
    else:
        alpha_bound = None
    lbx = ocp_model.overwrite(np.array(ocp_model.lbw, dtype=float).copy(),
                              u=u_fix, v=v_fix, alpha=alpha_bound)
    if alpha_bound is not None:
        alpha_bound[:, -1] = +2.  # > 1
    ubx = ocp_model.overwrite(np.array(ocp_model.ubw, dtype=float).copy(),
                              u=u_fix, v=v_fix, alpha=alpha_bound)
    sol = solver(x0=w0, p=p, lbx=lbx, ubx=ubx,
                 lbg=ocp_model.lbg, ubg=ocp_model.ubg)
    ret = solver.stats()['return_status']
    solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
    if ret not in solved:
        print('Warning: reoptimization NLP returned "{}"'.format(ret))
    w = sol['x'].full().flatten()
    y, u, alpha, v, nodes = ocp_model.extract(w)
    wall_time = _nlp_wall_time(solver)
    return y, u, alpha, v, nodes, w, wall_time


def _nlp_wall_time(solver) -> float:
    stats = solver.stats()
    return sum(val for key, val in stats.items() if 't_wall' in key)


class TComb:
    '''MILP for the combinatorial constraints (tCOMB): project realized
    controls v onto the set of mode realizations r satisfying min-up (dwell
    time) constraints tau_min, in the weighted L1 distance.'''

    def __init__(self, t: np.ndarray, modes: int, tau_min: float,
                 backend: str | None = None) -> None:
        N = len(t)
        m = MilpModel('tCOMB')
        x = m.add_vars((N - 1, modes), binary=True)
        _min_up_constraints(m, x, t, modes, tau_min)
        # exactly one mode active per time step
        for k in range(N - 1):
            m.add_constr({int(x[k, i]): 1. for i in range(modes)},
                         lb=1., ub=1.)
        self.N = N
        self.modes = modes
        self.model = m
        self.x = x
        self.t = t
        self.backend = backend

    def solve(self, v: np.ndarray, r: np.ndarray,
              time_limit: float | None = None
              ) -> tuple[np.ndarray, float, float]:
        'Solve for given realized controls v, mode encodings r (modes x nv).'
        N, t, x = self.N, self.t, self.x
        hk = np.diff(t)[:, None]
        cost = hk * np.abs(v[:, None, :] - r[None, :, :]).sum(axis=2)
        self.model.set_objective(x.reshape(-1), cost.reshape(-1))
        sol, obj_val, wall_time = self.model.solve(backend=self.backend,
                                                   time_limit=time_limit)
        result = np.round(sol[x.reshape(-1)]).reshape(N - 1, self.modes)
        return result.dot(r), obj_val, wall_time


def _config_multipliers(v_ref: np.ndarray, r: np.ndarray) -> np.ndarray:
    '''Convert a switch-state schedule v_ref (N-1, nv) into binary
    configuration multipliers beta (N-1, modes) with beta[k, i] = 1 iff
    r[i, :] == v_ref[k, :]. The encoding r enumerates all configurations,
    so the match always exists.'''
    matches = np.all(v_ref[:, None, :] == r[None, :, :], axis=2)
    assert np.all(matches.sum(axis=1) == 1), \
        'v_ref contains a switch state not present in the encoding r'
    return matches.astype(float)


def miocp_adm(ocp_model: OCModel, tau_min: float = 0.01,
              with_CIAP: bool = True, online_plot=None,
              rho_values: np.ndarray | None = None,
              max_adm_iter: int = 100, epsilon: float = 1e-3,
              milp_backend: str | None = None,
              time_budget: float | None = None) -> tuple:
    '''Penalty ADM for mixed-integer optimal control problems with additional
    combinatorial constraints that couple over time.

    The POC relaxation is solved once up front; its solution initializes the
    combinatorial reference v_ref (tCOMB projection of the rounded schedule)
    and, if rho_values is None, scales the penalty schedule: the homotopy
    brackets the critical weight rho_crit = Psi_POC / (T * nv_ref) at which
    the L1 coupling penalty becomes comparable to the objective. A fixed
    schedule (the paper uses logspace(-3, 6)) can otherwise start with the
    penalty already dominant, locking the iteration onto the initial v_ref.

    time_budget [s]: soft wall-clock limit. The current dwell-feasible
    reference v_ref is a valid incumbent from the first iteration on; when
    the budget is exhausted, the iteration stops and the incumbent is
    returned with reoptimized continuous controls. The method never returns
    nothing, and the returned switching schedule is always dwell-feasible.

    Returns (y, u, beta, v_ref, nodes, obj_val, w, timings).'''

    t_start = time.time()

    def out_of_budget() -> bool:
        return (time_budget is not None
                and time.time() - t_start > time_budget)

    def remaining_budget() -> float | None:
        if time_budget is None:
            return None
        return max(1., time_budget - (time.time() - t_start))

    nlp_solver_name = 'ipopt'
    solver = ocp_model.create_NLP_solver(nlp_solver_name, tol=1e-8)
    v_ref = np.array([[1] + [0] * (ocp_model.nv_ref - 1)]
                     * (len(ocp_model.t) - 1))

    timings = {'poc_nlp': 0.0, 'reopt_nlp': 0.0, 'comb': 0.0, 'sur': 0.0}

    tcomb = TComb(ocp_model.t, ocp_model.nalpha, tau_min,
                  backend=milp_backend)

    # solve the POC relaxation (rho = 0) once
    w0 = np.array(ocp_model.w0, dtype=float).copy()
    p = np.concatenate(([0.], v_ref.flatten()))
    sol = solver(x0=w0, p=p, lbx=ocp_model.lbw, ubx=ocp_model.ubw,
                 lbg=ocp_model.lbg, ubg=ocp_model.ubg)
    if nlp_solver_name == 'ipopt':
        ret = solver.stats()['return_status']
        solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
        assert ret in solved, 'Solution of POC relaxation failed'
    w0 = sol['x'].full().flatten()
    y, u, alpha, v, nodes = ocp_model.extract(w0)
    timings['poc_nlp'] += _nlp_wall_time(solver)
    Psi_poc = ocp_model.evaluate_objective(0., v_ref, w0)

    if with_CIAP:
        beta, wall_t = solve_ciap(ocp_model.t, alpha, strategy='SUR')
        timings['sur'] += wall_t
    else:
        beta = alpha

    # data-driven initial combinatorial reference: dwell-feasible projection
    # of the POC schedule instead of the arbitrary constant configuration.
    # v_ref is the incumbent from here on: always binary, always
    # dwell-feasible.
    v_ref, _, wall_t = tcomb.solve(beta.dot(ocp_model.r), ocp_model.r,
                                   time_limit=remaining_budget())
    timings['comb'] += wall_t

    out_hdr = '{:>9} {:>9} {:>10} {:>11} {:>9} {:>9}'
    output = '{:9.2e} {:9.2e} {:>10s} {:>11s} {:>9s} {:>9s}'
    print(out_hdr.format('rho', 'Psi_l_l', 'Psi_l+1_l', 'Psi_l+1_l+1',
                         'L1 pen', 'termcond'))

    if rho_values is None:
        t_span = ocp_model.t[-1] - ocp_model.t[0]
        rho_crit = max(abs(Psi_poc), 1e-8) / (t_span * ocp_model.nv_ref)
        rho_values = rho_crit * np.logspace(-3, 6, num=10)

    # penalty loop
    for rho in rho_values:
        if out_of_budget():
            print('time budget exhausted, returning incumbent')
            break
        # forward simulation to obtain correct Psi_l_l
        p = np.concatenate(([rho], v_ref.flatten()))
        y, u, beta, v, nodes, w0, wall_t = resolve_with_fixed_controls(
            ocp_model, solver, w0.copy(), p, u_fix=u, v_fix=v,
            alpha_fix=beta)
        Psi_l_l = ocp_model.evaluate_objective(rho, v_ref, w0)
        timings['reopt_nlp'] += wall_t

        # ADM loop
        for iadm in range(max_adm_iter):
            if out_of_budget():
                break
            # solve POC relaxation for (y, u, alpha) at given v_ref
            p = np.concatenate(([rho], v_ref.flatten()))
            sol = solver(x0=w0, p=p, lbx=ocp_model.lbw, ubx=ocp_model.ubw,
                         lbg=ocp_model.lbg, ubg=ocp_model.ubg)
            if nlp_solver_name == 'ipopt':
                ret = solver.stats()['return_status']
                solved = ['Solve_Succeeded', 'Solved_To_Acceptable_Level']
                assert ret in solved, 'Solution of NLP failed'
            w0 = sol['x'].full().flatten()
            y, u, alpha, v, nodes = ocp_model.extract(w0)
            timings['poc_nlp'] += _nlp_wall_time(solver)

            if with_CIAP:
                # CIAP to obtain binary beta from alpha
                beta, wall_t = solve_ciap(ocp_model.t, alpha, strategy='SUR')
                timings['sur'] += wall_t
                # reoptimize continuous controls
                y, u, beta, v, nodes, w0, wall_t = \
                    resolve_with_fixed_controls(
                        ocp_model, solver, w0.copy(), p, v_fix=v,
                        alpha_fix=beta)
                timings['reopt_nlp'] += wall_t
            else:
                beta = alpha

            # objective Psi(u^{k,l+1}, v^{k,l+1}, vtilde^{k,l})
            Psi_lp_l = ocp_model.evaluate_objective(rho, v_ref, w0)

            # first termination criterion
            if Psi_lp_l >= Psi_l_l - epsilon:
                if online_plot is not None:
                    online_plot(ocp_model.t, rho, Psi_lp_l, y, alpha, beta,
                                v_ref)
                print(output.format(rho, Psi_l_l,
                                    '{:9.2e}'.format(Psi_lp_l), '', '',
                                    '(i)'))
                break

            # solve combinatorial constraints (tCOMB)
            prev_v_ref = v_ref.copy()
            v_ref, beta_deviation, wall_t = tcomb.solve(
                beta.dot(ocp_model.r), ocp_model.r,
                time_limit=remaining_budget())
            timings['comb'] += wall_t

            # objective Psi(u^{k,l+1}, v^{k,l+1}, vtilde^{k,l+1})
            Psi_lp_lp = ocp_model.evaluate_objective(rho, v_ref, w0)

            if online_plot is not None:
                online_plot(ocp_model.t, rho, Psi_lp_lp, y, alpha, beta,
                            v_ref)

            # second termination criterion
            if Psi_lp_lp >= Psi_lp_l - epsilon:
                if Psi_lp_lp > Psi_lp_l:
                    v_ref = prev_v_ref
                print(output.format(rho, Psi_l_l,
                                    '{:9.2e}'.format(Psi_lp_l),
                                    '{:9.2e}'.format(Psi_lp_lp),
                                    '{:9.2e}'.format(beta_deviation),
                                    '(ii)'))
                break

            Psi_l_l = Psi_lp_lp
            print(output.format(rho, Psi_l_l, '{:9.2e}'.format(Psi_lp_l),
                                '{:9.2e}'.format(Psi_lp_lp),
                                '{:9.2e}'.format(beta_deviation), ''))

        # stop increasing rho?
        if np.linalg.norm(alpha.dot(ocp_model.r) - v_ref, np.inf) < 1e-4:
            break

    # compute feasible point and reoptimize continuous controls. Fixing to
    # the multipliers of v_ref (rather than the last SUR rounding beta)
    # guarantees the returned schedule is exactly the dwell-feasible
    # incumbent, also when the loop was cut short by the time budget.
    beta = _config_multipliers(v_ref, ocp_model.r)
    y, u, alpha, v, nodes, w0, wall_t = resolve_with_fixed_controls(
        ocp_model, solver, w0.copy(), p, v_fix=v, alpha_fix=beta)
    timings['reopt_nlp'] += wall_t
    obj_val = ocp_model.evaluate_objective(0, v_ref, w0)
    print('\nFinal objective value: {:.10e}'.format(obj_val))

    if online_plot is not None:
        online_plot(ocp_model.t, rho, obj_val, y, alpha, beta, v_ref)

    return y, u, beta, v_ref, nodes, obj_val, w0, timings
