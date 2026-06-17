import casadi as cas
import numpy as np

from gasnetopt.translines import model, networks


def test_poc_relaxation_solves_on_coarse_grid():
    net = networks.network_extended_tree
    demand = model.load_demand('demand_extended_tree_coarse.dat')
    T, nt, nx = 26., 2 * 26 + 1, 2
    nlp, lbw, ubw, lbg, ubg, w0 = model.translines_nlp(
        net, demand, T, nt=nt, nx=nx)
    solver = cas.nlpsol('solver', 'ipopt', nlp,
                        {'ipopt': {'tol': 1e-8, 'print_level': 0},
                         'print_time': False})
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    assert solver.stats()['return_status'] in [
        'Solve_Succeeded', 'Solved_To_Acceptable_Level']
    u, alpha, xi_p, xi_m = model.extract_solution(sol, net, nt, nx)
    # SOS1 satisfied by the relaxed multipliers
    assert np.allclose(alpha.sum(axis=1), 1., atol=1e-6)
    # objective finite and positive
    assert 0 < float(sol['f']) < np.inf


def test_ocmodel_extract_overwrite_roundtrip():
    m = model.TranslinesOCModel('extended tree', coarse=True)
    w = np.array(m.w0, dtype=float).copy()
    rng = np.random.default_rng(0)
    alpha = rng.random((m.nt - 1, m.n_confg))
    u = rng.random((m.nt - 1, m.n_ctrls))
    w = m.overwrite(w, u=u, alpha=alpha)
    y, u2, alpha2, _, _ = m.extract(w)
    assert np.allclose(u2, u)
    assert np.allclose(alpha2, alpha)
