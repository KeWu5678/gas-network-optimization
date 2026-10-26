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

The OCP is accessed through the OCModel interface (ocmodel.OCModel).
'''

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .ciap import add_min_up_constraints, solve_ciap
from .milp import MilpModel
from .nlp import NlpSolveError, NlpSolver
from .ocmodel import OCModel


@dataclass(frozen=True)
class AdmSettings:
    rho_values: tuple[float, ...] | None = None  # None: scaled to Psi_POC
    max_iter: int = 100                 # ADM iterations per penalty weight
    epsilon: float = 1e-4               # termination, relative to |Psi_POC|
    nlp_tol: float = 1e-8
    convergence_tol: float = 1e-4       # stop raising rho once alpha r ~ v_ref
    feasibility_tol: float = 1e-6       # of the returned point
    time_budget: float | None = None    # wall clock [s]


@dataclass(frozen=True)
class Solution:
    alpha: np.ndarray            # fractional for the relaxation, else binary
    u: np.ndarray
    objective: float             # at rho = 0
    w: np.ndarray
    timings: dict[str, float]    # wall time [s] per component


def solve_relaxation(ocp_model: OCModel, solver: NlpSolver) -> Solution:
    sol = ocp_model.solve(solver, ocp_model.w0, ocp_model.parameters())
    alpha, u, _ = ocp_model.extract(sol.w)
    return Solution(alpha=alpha, u=u,
                    objective=ocp_model.evaluate_objective(sol.w), w=sol.w,
                    timings=_timings(poc_nlp=sol.wall_time))


def round_and_reoptimize(ocp_model: OCModel, solver: NlpSolver,
                         relaxed: Solution, strategy: str = 'SUR',
                         tau_min: float | None = None,
                         milp_backend: str | None = None) -> Solution:
    beta, ciap_time = solve_ciap(ocp_model.t, relaxed.alpha,
                                 strategy=strategy, tau_min=tau_min,
                                 backend=milp_backend)
    sol = ocp_model.solve(solver, relaxed.w, ocp_model.parameters(),
                          alpha_fix=beta)
    alpha, u, _ = ocp_model.extract(sol.w)
    timings = {**relaxed.timings, 'ciap': ciap_time,
               'reopt_nlp': sol.wall_time}
    return Solution(alpha=alpha, u=u,
                    objective=ocp_model.evaluate_objective(sol.w), w=sol.w,
                    timings=timings)


def miocp_adm(ocp_model: OCModel, tau_min: float, with_ciap: bool = True,
              settings: AdmSettings = AdmSettings(),
              milp_backend: str | None = None,
              online_plot: Callable[..., None] | None = None) -> Solution:
    '''Penalty ADM for mixed-integer optimal control problems with additional
    combinatorial constraints that couple over time.

    The POC relaxation is solved once up front; its solution initializes the
    combinatorial reference v_ref (tCOMB projection of the rounded schedule)
    and, if settings.rho_values is None, scales the penalty schedule: the
    homotopy brackets the critical weight rho_crit = Psi_POC / (T *
    n_switches) at which the L1 coupling penalty becomes comparable to the
    objective. A fixed schedule (the paper uses logspace(-3, 6)) can
    otherwise start with the penalty already dominant, locking the iteration
    onto the initial v_ref.

    settings.epsilon is a termination tolerance *relative* to |Psi_POC| (the
    criteria (i)/(ii) compare objectives whose magnitude is model-dependent;
    an absolute tolerance would stop the alternation before the first tCOMB
    update on models with small objectives).

    settings.time_budget [s]: soft wall-clock limit, checked between solver
    calls; NLP solves are additionally capped by an IPOPT wall-time limit of
    the full budget. The dwell-feasible reference v_ref is a valid incumbent
    from the first projection on; when the budget is exhausted, the
    iteration stops and the incumbent is returned with reoptimized
    continuous controls. The guarantee assumes the budget admits one POC
    solve, one projection MILP and one reoptimization; below that the
    method raises instead of fabricating a result.

    Returns the final Solution.'''
    budget = _TimeBudget(settings.time_budget)
    progress = _ProgressTable(budget)
    adm = _PenaltyADM(ocp_model, tau_min, with_ciap=with_ciap,
                      settings=settings, milp_backend=milp_backend,
                      budget=budget, progress=progress,
                      online_plot=online_plot)
    # termination tolerance relative to the objective magnitude
    eps_abs = settings.epsilon * max(abs(adm.psi_poc), 1e-12)
    progress.header()
    rho_values = (settings.rho_values if settings.rho_values is not None
                  else _default_rho_schedule(ocp_model, adm.psi_poc))

    for rho in rho_values:
        if budget.exhausted():
            print('time budget exhausted, returning incumbent', flush=True)
            break
        psi_l_l = adm.forward_simulate(rho)
        adm.alternate(psi_l_l, settings.max_iter, eps_abs)
        if adm.penalty_converged():
            break

    return adm.reoptimize_incumbent()


class _PenaltyADM:

    def __init__(self, ocp_model: OCModel, tau_min: float, with_ciap: bool,
                 settings: AdmSettings, milp_backend: str | None,
                 budget: _TimeBudget, progress: _ProgressTable,
                 online_plot: Callable[..., None] | None) -> None:
        self._model = ocp_model
        self._with_ciap = with_ciap
        self._settings = settings
        self._budget = budget
        self._progress = progress
        self._online_plot = online_plot
        self._solver = ocp_model.create_solver(
            tol=settings.nlp_tol, max_wall_time=budget.nlp_wall_limit)
        self._tcomb = TComb(ocp_model.t, ocp_model.r, tau_min,
                            backend=milp_backend)
        self.timings = _timings()
        self.rho = 0.
        self.p = ocp_model.parameters()

        poc = solve_relaxation(ocp_model, self._solver)
        self.timings['poc_nlp'] += poc.timings['poc_nlp']
        self.psi_poc = poc.objective
        self.w, self.u, self.alpha = poc.w, poc.u, poc.alpha
        self.beta = self._round(self.alpha)
        # data-driven initial combinatorial reference: dwell-feasible
        # projection of the POC schedule. v_ref is the incumbent from here
        # on: always binary, always dwell-feasible.
        self.v_ref, _ = self._project(self.beta)

    def forward_simulate(self, rho: float) -> float:
        # reoptimization with u and beta fixed, to obtain correct Psi_l_l
        self.rho = rho
        self.p = self._model.parameters(rho, self.v_ref)
        self._reoptimize(u_fix=self.u, alpha_fix=self.beta)
        return self._model.evaluate_objective(self.w, rho, self.v_ref)

    def alternate(self, psi_l_l: float, max_iter: int,
                  eps_abs: float) -> None:
        for _ in range(max_iter):
            if self._budget.exhausted():
                return
            psi_lp_l = self._relaxed_step()

            # first termination criterion
            if psi_lp_l >= psi_l_l - eps_abs:
                self._plot(psi_lp_l)
                self._progress.row(self.rho, psi_l_l, psi_lp_l,
                                   termination='(i)')
                return

            previous_v_ref = self.v_ref
            self.v_ref, l1_distance = self._project(self.beta)
            # objective Psi(u^{k,l+1}, v^{k,l+1}, vtilde^{k,l+1})
            psi_lp_lp = self._model.evaluate_objective(self.w, self.rho,
                                                       self.v_ref)
            self._plot(psi_lp_lp)

            # second termination criterion
            if psi_lp_lp >= psi_lp_l - eps_abs:
                if psi_lp_lp > psi_lp_l:
                    self.v_ref = previous_v_ref
                self._progress.row(self.rho, psi_l_l, psi_lp_l, psi_lp_lp,
                                   l1_distance, '(ii)')
                return

            psi_l_l = psi_lp_lp
            self._progress.row(self.rho, psi_l_l, psi_lp_l, psi_lp_lp,
                               l1_distance)

    def penalty_converged(self) -> bool:
        return bool(np.linalg.norm(self.alpha.dot(self._model.r)
                                   - self.v_ref, np.inf)
                    < self._settings.convergence_tol)

    def reoptimize_incumbent(self) -> Solution:
        # Fixing to the multipliers of v_ref (rather than the last SUR
        # rounding beta) guarantees the returned schedule is exactly the
        # dwell-feasible incumbent, also when the loop was cut short by the
        # time budget.
        beta = _config_multipliers(self.v_ref, self._model.r)
        lbx, ubx = self._model.bounds(alpha_fix=beta)
        try:
            sol = self._solver.solve(self.w, lbx, ubx, p=self.p)
        except NlpSolveError as err:
            raise RuntimeError(
                'Final reoptimization failed with status "{}"; no feasible '
                'point to return.'.format(err.solution.status)) from err
        self.timings['reopt_nlp'] += sol.wall_time
        _assert_feasible(self._model, sol.w, self.p, lbx, ubx,
                         tol=self._settings.feasibility_tol)
        objective = self._model.evaluate_objective(sol.w)
        print('\nFinal objective value: {:.10e}'.format(objective),
              flush=True)
        alpha, u, y = self._model.extract(sol.w)
        if self._online_plot is not None:
            self._online_plot(self._model.t, self.rho, objective, y, alpha,
                              beta, self.v_ref)
        return Solution(alpha=beta, u=u, objective=objective, w=sol.w,
                        timings=self.timings)

    def _relaxed_step(self) -> float:
        # solve POC relaxation for (y, u, alpha) at given v_ref
        self.p = self._model.parameters(self.rho, self.v_ref)
        sol = self._model.solve(self._solver, self.w, self.p)
        self.timings['poc_nlp'] += sol.wall_time
        self.w = sol.w
        self.alpha, self.u, _ = self._model.extract(sol.w)
        if self._with_ciap:
            self.beta = self._round(self.alpha)
            self._reoptimize(alpha_fix=self.beta)
        else:
            self.beta = self.alpha
        # objective Psi(u^{k,l+1}, v^{k,l+1}, vtilde^{k,l})
        return self._model.evaluate_objective(self.w, self.rho, self.v_ref)

    def _round(self, alpha: np.ndarray) -> np.ndarray:
        if not self._with_ciap:
            return alpha
        beta, wall_time = solve_ciap(self._model.t, alpha, strategy='SUR')
        self.timings['ciap'] += wall_time
        return beta

    def _project(self, beta: np.ndarray) -> tuple[np.ndarray, float]:
        v_ref, l1_distance, wall_time = self._tcomb.project(
            beta.dot(self._model.r), time_limit=self._budget.remaining())
        self.timings['tcomb'] += wall_time
        return v_ref, l1_distance

    def _reoptimize(self, u_fix: np.ndarray | None = None,
                    alpha_fix: np.ndarray | None = None) -> None:
        # An intermediate reoptimization that stops early (iteration or
        # wall-time limit) is not fatal: the iteration continues from its
        # last iterate, and only the final point must be feasible.
        try:
            sol = self._model.solve(self._solver, self.w, self.p,
                                    u_fix=u_fix, alpha_fix=alpha_fix)
        except NlpSolveError as err:
            print('Warning: reoptimization NLP returned "{}"'.format(
                err.solution.status))
            sol = err.solution
        self.timings['reopt_nlp'] += sol.wall_time
        self.w = sol.w
        fixed = self._model.extract(sol.w)
        self.u, self.beta = fixed.u, fixed.alpha

    def _plot(self, psi: float) -> None:
        if self._online_plot is not None:
            self._online_plot(self._model.t, self.rho, psi,
                              self._model.extract(self.w).y, self.alpha,
                              self.beta, self.v_ref)


class TComb:
    '''MILP for the combinatorial constraints (tCOMB): project realized
    controls v onto the set of mode realizations r satisfying min-up (dwell
    time) constraints tau_min, in the weighted L1 distance.'''

    def __init__(self, t: np.ndarray, r: np.ndarray, tau_min: float,
                 backend: str | None = None) -> None:
        n_steps, n_configs = len(t) - 1, r.shape[0]
        m = MilpModel('tCOMB')
        x = m.add_vars((n_steps, n_configs), binary=True)
        add_min_up_constraints(m, x, t, tau_min)
        # exactly one configuration active per time step
        for k in range(n_steps):
            m.add_constr({int(x[k, i]): 1. for i in range(n_configs)},
                         lb=1., ub=1.)
        self._t = t
        self._r = r
        self._model = m
        self._x = x
        self._backend = backend

    def project(self, v: np.ndarray, time_limit: float | None = None
                ) -> tuple[np.ndarray, float, float]:
        hk = np.diff(self._t)[:, None]
        cost = hk * np.abs(v[:, None, :] - self._r[None, :, :]).sum(axis=2)
        self._model.set_objective(self._x.reshape(-1), cost.reshape(-1))
        sol, l1_distance, wall_time = self._model.solve(
            backend=self._backend, time_limit=time_limit)
        configs = np.round(sol[self._x.reshape(-1)]).reshape(self._x.shape)
        return configs.dot(self._r), l1_distance, wall_time


def _default_rho_schedule(ocp_model: OCModel, psi_poc: float) -> np.ndarray:
    t_span = ocp_model.t[-1] - ocp_model.t[0]
    rho_crit = max(abs(psi_poc), 1e-8) / (t_span * ocp_model.n_switches)
    return rho_crit * np.logspace(-3, 6, num=10)


def _config_multipliers(v_ref: np.ndarray, r: np.ndarray) -> np.ndarray:
    '''Convert a switch-state schedule v_ref (N-1, nv) into binary
    configuration multipliers beta (N-1, modes) with beta[k, i] = 1 iff
    r[i, :] == v_ref[k, :]. The encoding r enumerates all configurations,
    so the match always exists.'''
    matches = np.all(v_ref[:, None, :] == r[None, :, :], axis=2)
    if not np.all(matches.sum(axis=1) == 1):
        raise ValueError(
            'v_ref contains a switch state not present in the encoding r')
    return matches.astype(float)


def _assert_feasible(ocp_model: OCModel, w: np.ndarray, p: np.ndarray,
                     lbx: np.ndarray, ubx: np.ndarray,
                     tol: float = 1e-6) -> None:
    '''Verify finiteness, variable bounds and NLP constraints.'''
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(p)):
        raise RuntimeError(
            'Final reoptimization returned non-finite values; no feasible '
            'point to return.')

    bound_viol = np.maximum(lbx - w, w - ubx).max()
    if bound_viol > tol:
        raise RuntimeError(
            'Final reoptimization violates variable bounds (max violation '
            '{:.3e} > {:.0e}); no feasible point to return.'.format(
                bound_viol, tol))

    viol = ocp_model.constraint_violation(w, p)
    if viol > tol:
        raise RuntimeError(
            'Final reoptimization is infeasible (max constraint violation '
            '{:.3e} > {:.0e}); no feasible point to return.'.format(
                viol, tol))


def _timings(**measured: float) -> dict[str, float]:
    return {'poc_nlp': 0., 'reopt_nlp': 0., 'ciap': 0., 'tcomb': 0.,
            **measured}


class _TimeBudget:

    def __init__(self, seconds: float | None) -> None:
        self._seconds = seconds
        self._start = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start

    def exhausted(self) -> bool:
        return self._seconds is not None and self.elapsed() > self._seconds

    def remaining(self) -> float | None:
        if self._seconds is None:
            return None
        return max(1., self._seconds - self.elapsed())

    @property
    def nlp_wall_limit(self) -> float | None:
        # each NLP solve gets at least 1 s (IPOPT rejects a zero wall limit),
        # consistent with the floor in remaining()
        if self._seconds is None:
            return None
        return max(self._seconds, 1.)


class _ProgressTable:
    # progress rows carry the cumulative wall time and are flushed so that
    # long runs remain observable through a redirected/piped stdout
    _HEADER = '{:>9} {:>9} {:>10} {:>11} {:>9} {:>9} {:>8}'
    _ROW = '{:9.2e} {:9.2e} {:>10s} {:>11s} {:>9s} {:>9s} {:>8s}'

    def __init__(self, budget: _TimeBudget) -> None:
        self._budget = budget

    def header(self) -> None:
        print(self._HEADER.format('rho', 'Psi_l_l', 'Psi_l+1_l',
                                  'Psi_l+1_l+1', 'L1 pen', 'termcond',
                                  'wall'), flush=True)

    def row(self, rho: float, psi_l_l: float, psi_lp_l: float | None = None,
            psi_lp_lp: float | None = None,
            l1_distance: float | None = None,
            termination: str = '') -> None:
        def cell(value: float | None) -> str:
            return '' if value is None else '{:9.2e}'.format(value)
        print(self._ROW.format(rho, psi_l_l, cell(psi_lp_l), cell(psi_lp_lp),
                               cell(l1_distance), termination,
                               '{:.0f}s'.format(self._budget.elapsed())),
              flush=True)
