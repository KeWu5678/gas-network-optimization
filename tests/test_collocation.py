import casadi as cas
import numpy as np

from gasnetopt.collocation import direct_transcription, gauss_collocation


def test_gauss_collocation_partition_of_unity():
    for d in [1, 2, 3]:
        B, C, D = gauss_collocation(d)
        # continuity coefficients: Lagrange basis sums to 1 at tau = 1
        assert np.isclose(np.sum(D), 1.0)
        # quadrature weights integrate the constant 1 exactly
        assert np.isclose(np.sum(B), 1.0)


def test_direct_transcription_integrates_linear_ode():
    # ydot = -y, y(0) = 1  ->  y(1) = exp(-1)
    y = cas.MX.sym('y')
    prob = {
        'y': y, 'ydot': -y, 'L': 0, 'ny': 1, 'SOS1': [],
        'lb_ys': [1.], 'ub_ys': [1.],
    }
    t = np.linspace(0, 1, 11)
    nlp, lbw, ubw, lbg, ubg, w0, _ = direct_transcription(prob, 3, t)
    solver = cas.nlpsol('s', 'ipopt', nlp,
                        {'ipopt': {'print_level': 0}, 'print_time': False})
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    w = sol['x'].full().flatten()
    # final state is the last variable
    assert np.isclose(w[-1], np.exp(-1), atol=1e-6)
