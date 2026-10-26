import numpy as np
import pytest

from gasnetopt import DATA_DIR
from gasnetopt.core.nlp import NlpSolver
from gasnetopt.translines import model
from gasnetopt.translines.data import EXTENDED_TREE, NETWORKS, load_demand

COARSE_DEMAND = DATA_DIR / 'translines' / 'demand_extended_tree_coarse.dat'


@pytest.fixture(scope='module')
def coarse_model():
    return model.TranslinesOCModel(EXTENDED_TREE, load_demand(COARSE_DEMAND),
                                   model.TranslinesModelSettings('coarse'))


def test_poc_relaxation_solves_on_coarse_grid(coarse_model):
    nt, nx = model.GRIDS['coarse']
    problem = model.translines_nlp(EXTENDED_TREE, coarse_model.demand,
                                   model.HORIZON, nt=nt, nx=nx)
    sol = NlpSolver(problem, tol=1e-8).solve(problem.x0, problem.lbx,
                                             problem.ubx)
    # the model's NLP has the same variable layout as the bare one
    alpha = coarse_model.extract(sol.w).alpha
    # SOS1 satisfied by the relaxed multipliers
    assert np.allclose(alpha.sum(axis=1), 1., atol=1e-6)
    # objective finite and positive
    assert 0 < sol.f < np.inf


def test_ocmodel_extract_roundtrip(coarse_model):
    m = coarse_model
    rng = np.random.default_rng(0)
    alpha = rng.random((m.nt - 1, m.n_configs))
    u = rng.random((m.nt - 1, m.n_controls))
    w = m.with_variables(m.w0, u=u, alpha=alpha)
    alpha2, u2, _ = m.extract(w)
    assert np.allclose(u2, u)
    assert np.allclose(alpha2, alpha)


def test_solution_dict_splits_line_states(coarse_model):
    m = coarse_model
    s = m.solution_dict(np.arange(m.n_variables, dtype=float))
    assert len(s['xi_p']) == len(s['xi_m']) == len(EXTENDED_TREE.edges)
    assert s['xi_p'][0].shape == (m.nx, m.nt)
    # xi_m of a line follows its xi_p in the NLP vector
    assert s['xi_m'][0].flatten(order='F')[0] == \
        s['xi_p'][0].flatten(order='F')[-1] + 1


def test_demand_must_match_the_grid():
    with pytest.raises(ValueError, match='fine grid'):
        model.TranslinesOCModel(EXTENDED_TREE, load_demand(COARSE_DEMAND),
                                model.TranslinesModelSettings('fine'))


def test_switch_encoding_matches_configurations():
    # column j = switch j off; the order the stored results were made with
    legacy = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    for network in NETWORKS.values():
        assert np.array_equal(network.switch_encoding(), legacy)
