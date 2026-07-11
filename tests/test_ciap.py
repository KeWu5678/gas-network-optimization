import numpy as np
import pytest

from gasnetopt.adm import TComb
from gasnetopt.ciap import solve_ciap, sum_up_rounding


def accumulated_deviation(t, alpha, beta):
    '''eta(beta) = min_delta max_{i,k} |delta + sum_{k''<k} dt (alpha-beta)|,
    the centered deviation the CIAP MILP minimizes (free offset delta shared
    over all modes; k = 0 contributes an empty sum, i.e. 0).'''
    dt = np.diff(t)[:, None]
    acc = np.cumsum(dt * (alpha - beta), axis=0)
    lo = min(acc.min(), 0.)
    hi = max(acc.max(), 0.)
    return (hi - lo) / 2.


@pytest.fixture
def relaxed_instance():
    rng = np.random.default_rng(42)
    N, modes = 9, 3
    t = np.linspace(0., 4., N)
    alpha = rng.random((N - 1, modes))
    alpha /= alpha.sum(axis=1, keepdims=True)
    return t, alpha


def test_sur_keeps_binary_input(relaxed_instance):
    t, _ = relaxed_instance
    alpha = np.zeros((len(t) - 1, 3))
    alpha[:, 1] = 1.
    p = sum_up_rounding(t, alpha)
    assert np.array_equal(p, alpha)


def test_sur_sos1(relaxed_instance):
    t, alpha = relaxed_instance
    p = sum_up_rounding(t, alpha)
    assert np.all(p.sum(axis=1) == 1)
    assert set(np.unique(p)) <= {0., 1.}


def test_mip_ciap_not_worse_than_sur(relaxed_instance):
    t, alpha = relaxed_instance
    p_sur, _ = solve_ciap(t, alpha, strategy='SUR')
    p_mip, _ = solve_ciap(t, alpha, strategy='MIP')
    assert np.all(p_mip.sum(axis=1) == 1)
    eta_sur = accumulated_deviation(t, alpha, p_sur)
    eta_mip = accumulated_deviation(t, alpha, p_mip)
    assert eta_mip <= eta_sur + 1e-9


def test_comb_respects_dwell_time(relaxed_instance):
    t, alpha = relaxed_instance
    tau_min = 1.5  # = 3 steps of dt = 0.5
    beta, _ = solve_ciap(t, alpha, strategy='COMB', tau_min=tau_min)
    assert np.all(beta.sum(axis=1) == 1)
    N = len(t)
    for mode in range(alpha.shape[1]):
        x = beta[:, mode]
        for k in range(1, N - 1):
            if x[k] > x[k - 1]:  # switched on at t[k]
                l = k + 1
                while l < N - 1 and t[l] <= t[k] + tau_min:
                    assert x[l] == 1, \
                        'dwell time violated for mode {}'.format(mode)
                    l += 1


def test_tcomb_returns_feasible_input_unchanged():
    N, modes = 9, 3
    t = np.linspace(0., 4., N)
    r = np.eye(modes)
    # piecewise constant with long plateaus -> feasible for tau_min = 1.0
    v = np.zeros((N - 1, modes))
    v[:4, 0] = 1.
    v[4:, 2] = 1.
    v_proj, obj_val, _ = TComb(t, modes, tau_min=1.0).solve(v, r)
    assert np.allclose(v_proj, v)
    assert obj_val <= 1e-9
