import casadi as cas
import numpy as np
import pytest

from gasnetopt.core.nlp import SOLVED_STATUSES, NlpBuilder, NlpSolveError, NlpSolver


def test_builder_stores_matrices_column_major():
    b = NlpBuilder()
    a = b.add_variable(cas.MX.sym('a', 2, 3), np.arange(6.).reshape(2, 3),
                       0., 10.)
    b.add_variable(cas.MX.sym('c', 2), 7., -1., 1.)
    b.add_equality(a[:, 0] - 1.)
    b.add_upper_bound(a[0, 1])
    problem = b.build(cas.sumsqr(a))
    assert np.array_equal(problem.x0, [0., 3., 1., 4., 2., 5., 7., 7.])
    assert b.offsets == {'a': (0, 2, 3), 'c': (6, 2, 1)}
    assert np.array_equal(problem.lbg, [0., 0., -np.inf])
    assert np.array_equal(problem.ubg, [0., 0., 0.])


def test_builder_rejects_duplicate_names():
    b = NlpBuilder()
    b.add_variable(cas.MX.sym('a'), 0., 0., 1.)
    with pytest.raises(ValueError, match='duplicate'):
        b.add_variable(cas.MX.sym('a'), 0., 0., 1.)


def test_solver_returns_solution_or_raises():
    b = NlpBuilder()
    x = b.add_variable(cas.MX.sym('x'), 0., -10., 10.)
    b.add_constraint(x, 1., 1.)
    problem = b.build((x - 3.) ** 2)
    solver = NlpSolver(problem, tol=1e-10)

    sol = solver.solve(problem.x0, problem.lbx, problem.ubx)
    assert np.isclose(sol.w[0], 1.) and np.isclose(sol.f, 4.)
    assert sol.wall_time > 0.

    # x = 1 is outside the bounds [2, 10]: infeasible
    with pytest.raises(NlpSolveError) as err:
        solver.solve(problem.x0, np.array([2.]), problem.ubx)
    assert err.value.solution.status not in SOLVED_STATUSES
    assert err.value.solution.w.shape == (1,)
