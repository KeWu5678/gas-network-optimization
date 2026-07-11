'''
Combinatorial Integral Approximation Problem (CIAP): reconstruct binary
feasible convex multipliers beta from relaxed multipliers alpha.

Strategies (cf. Göttlich/Hante/Potschka/Schewe, Math. Program. 188 (2021)):
- 'SUR':     Sum-Up Rounding with SOS1 constraint (no MILP solve).
- 'MIP':     exact CIAP as MILP (max accumulated deviation objective).
- 'SUR+MIP': MIP warm-started with the SUR point (gurobi backend only).
- 'COMB':    MIP with additional min-up (dwell time) constraints tau_min.
'''

import time

import numpy as np

from .milp import MilpModel


def sum_up_rounding(t, alpha):
    '''Sum-Up Rounding with SOS1 constraint on grid t (len N). alpha has shape
    (N-1, modes). Returns binary p of the same shape.'''
    dt = np.diff(t)
    N1, modes = alpha.shape
    assert N1 == len(t) - 1
    p = np.zeros((N1, modes))
    phat = np.zeros(modes)
    unique = True
    for i in range(N1):
        phat += dt[i] * alpha[i, :]
        j = np.argmax(phat)
        if np.sum(phat == phat[j]) > 1:
            unique = False
        p[i, j] = 1.
        phat[j] -= dt[i]
    if not unique:
        print('Warning: Sum-Up Rounding result not unique')
    return p


def _min_up_constraints(m, x, t, modes, tau_min):
    '''Add standard min-up (dwell time) constraints for binary variable array
    x of shape (N-1, modes) on grid t.'''
    N = len(t)
    for mode in range(modes):
        # first time step
        l = 1
        while (t[l] <= tau_min) and (l < N - 1):
            # x[0] <= x[l]
            m.add_constr({x[0, mode]: 1., x[l, mode]: -1.}, ub=0.)
            l += 1
        # all other time steps
        for k in range(1, N - 1):
            tk = t[k]
            l = k + 1
            while (t[l] <= tk + tau_min) and (l < N - 1):
                # x[k] - x[k-1] <= x[l]
                m.add_constr({x[k, mode]: 1., x[k-1, mode]: -1.,
                              x[l, mode]: -1.}, ub=0.)
                # x[k-1] - x[k] <= 1 - x[l]
                m.add_constr({x[k-1, mode]: 1., x[k, mode]: -1.,
                              x[l, mode]: 1.}, ub=1.)
                l += 1


def solve_ciap(t, alpha, strategy='SUR', tau_min=None, backend=None):
    '''Solve the CIAP for relaxed multipliers alpha on grid t. Returns
    (beta, wall_time).'''
    assert strategy in ['SUR', 'MIP', 'SUR+MIP', 'COMB'], \
        'Unknown CIAP strategy "{}"'.format(strategy)
    N = len(t)
    dt = np.diff(t)
    modes = alpha.shape[1]

    wall_time = 0.0
    p = None

    if strategy in ['SUR', 'SUR+MIP']:
        wall_time -= time.time()
        p = sum_up_rounding(t, alpha)
        wall_time += time.time()
        if strategy == 'SUR':
            return p, wall_time

    # MILP formulation, equivalent to the original gurobi model:
    # min eta  s.t.  -eta <= delta + sum_{k'<k} dt[k'] (alpha-beta)[k',i] <= eta
    #                sum_i beta[k,i] = 1
    m = MilpModel('CIAP')
    beta = m.add_vars((N - 1, modes), binary=True)
    delta = m.add_var(lb=-np.inf)
    eta = m.add_var(lb=-np.inf, obj=1.)

    for i in range(modes):
        for k in range(N):
            # delta - sum_{k'<k} dt[k'] beta[k',i] -+ eta <= / >=
            #     -sum_{k'<k} dt[k'] alpha[k',i]
            acc = float(np.dot(dt[:k], alpha[:k, i]))
            coeffs = {int(beta[kk, i]): -dt[kk] for kk in range(k)}
            coeffs[delta] = 1.
            m.add_constr({**coeffs, eta: -1.}, ub=-acc)      # upper_max
            m.add_constr({**coeffs, eta: +1.}, lb=-acc)      # lower_max
    for k in range(N - 1):
        m.add_constr({int(beta[k, i]): 1. for i in range(modes)},
                     lb=1., ub=1.)

    if strategy == 'COMB':
        assert tau_min is not None, 'COMB requires tau_min'
        _min_up_constraints(m, beta, t, modes, tau_min)

    warm = None
    if strategy == 'SUR+MIP':
        warm = np.zeros(m.ncols)
        warm[beta.reshape(-1)] = p.reshape(-1)

    x, _, wall = m.solve(backend=backend, warm_start=warm)
    wall_time += wall
    beta_sol = np.round(x[beta.reshape(-1)]).reshape(N - 1, modes)
    return beta_sol, wall_time
