'''
Combinatorial Integral Approximation Problem (CIAP): reconstruct binary
feasible convex multipliers beta from relaxed multipliers alpha.

Strategies (cf. Göttlich/Hante/Potschka/Schewe, Math. Program. 188 (2021)):
- 'SUR':     Sum-Up Rounding with SOS1 constraint (no MILP solve).
- 'MIP':     exact CIAP as MILP (max accumulated deviation objective).
- 'SUR+MIP': MIP warm-started with the SUR point (gurobi backend only).
- 'COMB':    MIP with additional min-up (dwell time) constraints tau_min.
'''

from __future__ import annotations

import time

import numpy as np

from .milp import MilpModel

CIAP_STRATEGIES = ('SUR', 'MIP', 'SUR+MIP', 'COMB')


def solve_ciap(t: np.ndarray, alpha: np.ndarray, strategy: str = 'SUR',
               tau_min: float | None = None, backend: str | None = None
               ) -> tuple[np.ndarray, float]:
    '''Solve the CIAP for relaxed multipliers alpha on grid t. Returns
    (beta, wall_time).'''
    if strategy not in CIAP_STRATEGIES:
        raise ValueError('Unknown CIAP strategy "{}"'.format(strategy))
    if strategy == 'COMB' and tau_min is None:
        raise ValueError('COMB requires tau_min')
    wall_time = 0.
    warm_start = None
    if strategy in ('SUR', 'SUR+MIP'):
        start = time.time()
        beta = sum_up_rounding(t, alpha)
        wall_time += time.time() - start
        warm_start = beta
    if strategy != 'SUR':
        beta, milp_time = ciap_milp(
            t, alpha, tau_min=tau_min if strategy == 'COMB' else None,
            warm_start=warm_start, backend=backend)
        wall_time += milp_time
    return beta, wall_time


def sum_up_rounding(t: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    '''Sum-Up Rounding with SOS1 constraint on grid t (len N). alpha has shape
    (N-1, n_configs). Returns binary beta of the same shape.'''
    dt = np.diff(t)
    N1, n_configs = alpha.shape
    if N1 != len(t) - 1:
        raise ValueError('alpha needs one row per grid interval')
    beta = np.zeros((N1, n_configs))
    accumulated_deviation = np.zeros(n_configs)
    unique = True
    for i in range(N1):
        accumulated_deviation += dt[i] * alpha[i, :]
        j = np.argmax(accumulated_deviation)
        if np.sum(accumulated_deviation == accumulated_deviation[j]) > 1:
            unique = False
        beta[i, j] = 1.
        accumulated_deviation[j] -= dt[i]
    if not unique:
        print('Warning: Sum-Up Rounding result not unique')
    return beta


def ciap_milp(t: np.ndarray, alpha: np.ndarray,
              tau_min: float | None = None,
              warm_start: np.ndarray | None = None,
              backend: str | None = None) -> tuple[np.ndarray, float]:
    N = len(t)
    dt = np.diff(t)
    n_configs = alpha.shape[1]

    # MILP formulation, equivalent to the original gurobi model:
    # min eta  s.t.  -eta <= delta + sum_{k'<k} dt[k'] (alpha-beta)[k',i] <= eta
    #                sum_i beta[k,i] = 1
    m = MilpModel('CIAP')
    x = m.add_vars((N - 1, n_configs), binary=True)
    delta = m.add_var(lb=-np.inf)
    eta = m.add_var(lb=-np.inf, obj=1.)

    for i in range(n_configs):
        for k in range(N):
            # delta - sum_{k'<k} dt[k'] beta[k',i] -+ eta <= / >=
            #     -sum_{k'<k} dt[k'] alpha[k',i]
            acc = float(np.dot(dt[:k], alpha[:k, i]))
            coeffs = {int(x[kk, i]): -dt[kk] for kk in range(k)}
            coeffs[delta] = 1.
            m.add_constr({**coeffs, eta: -1.}, ub=-acc)      # upper_max
            m.add_constr({**coeffs, eta: +1.}, lb=-acc)      # lower_max
    for k in range(N - 1):
        m.add_constr({int(x[k, i]): 1. for i in range(n_configs)},
                     lb=1., ub=1.)

    if tau_min is not None:
        add_min_up_constraints(m, x, t, tau_min)

    warm = None
    if warm_start is not None:
        warm = np.zeros(m.ncols)
        warm[x.reshape(-1)] = warm_start.reshape(-1)

    sol, _, wall_time = m.solve(backend=backend, warm_start=warm)
    beta = np.round(sol[x.reshape(-1)]).reshape(N - 1, n_configs)
    return beta, wall_time


def add_min_up_constraints(m: MilpModel, x: np.ndarray, t: np.ndarray,
                           tau_min: float) -> None:
    '''Add standard min-up (dwell time) constraints for binary variable array
    x of shape (N-1, n_configs) on grid t.'''
    N = len(t)
    for config in range(x.shape[1]):
        # first time step
        ell = 1
        while (t[ell] <= tau_min) and (ell < N - 1):
            # x[0] <= x[ell]
            m.add_constr({x[0, config]: 1., x[ell, config]: -1.}, ub=0.)
            ell += 1
        # all other time steps
        for k in range(1, N - 1):
            tk = t[k]
            ell = k + 1
            while (t[ell] <= tk + tau_min) and (ell < N - 1):
                # x[k] - x[k-1] <= x[ell]
                m.add_constr({x[k, config]: 1., x[k-1, config]: -1.,
                              x[ell, config]: -1.}, ub=0.)
                # x[k-1] - x[k] <= 1 - x[ell]
                m.add_constr({x[k-1, config]: 1., x[k, config]: -1.,
                              x[ell, config]: 1.}, ub=1.)
                ell += 1
