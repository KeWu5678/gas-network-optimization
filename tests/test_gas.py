import numpy as np
import pytest

from gasnetopt import DATA_DIR
from gasnetopt.core import adm
from gasnetopt.gas import data
from gasnetopt.gas.model import GasModelSettings, GasOCModel, switch_encoding

G11 = DATA_DIR / 'gaslib' / 'GasLib-11'
G40 = DATA_DIR / 'gaslib' / 'GasLib-40'
HALF_HOUR = GasModelSettings(horizon=1800., nt=31, nx=2)

# GasLib data is not redistributed with the repository (see README);
# skip the gas tests when it has not been placed under data/gaslib.
pytestmark = pytest.mark.skipif(
    not G11.exists(), reason='GasLib data not available under data/gaslib')


@pytest.fixture(scope='module')
def net():
    return data.parse_net(G11 / 'GasLib-11-v1-20211130.net')


@pytest.fixture(scope='module')
def bc():
    return data.parse_bcd(G11 / 'GasLib-11-sinus-InputData.bcd')


def test_parse_net_gaslib11(net):
    assert len(net.nodes) == 11
    assert len(net.pipes) == 8
    assert len(net.valves) == 1
    assert len(net.compressors) == 2
    assert {n.id for n in net.sources} == {'entry01', 'entry02', 'entry03'}
    assert {n.id for n in net.sinks} == {'exit01', 'exit02', 'exit03'}
    pipe = net.pipes[0]
    assert pipe.length == 55e3 and pipe.diameter == 0.5
    assert net.nodes['entry01'].pressure_max == 70e5
    # 50 * 1000 m^3/h * 0.785 kg/m^3 = 10.9 kg/s
    assert np.isclose(net.nodes['entry01'].flow_min, 50e3 * 0.785 / 3600.)


def test_parse_bcd_gaslib11(bc):
    assert bc.speed_of_sound == 340.
    assert bc.t_end == 86400.
    assert set(bc.pressure) == {'entry01', 'entry02', 'entry03'}
    assert set(bc.massflow) == {'exit01', 'exit02', 'exit03'}
    t = np.array([0., 3600.])
    assert np.allclose(bc.pressure_at('entry01', t), 53e5)
    assert np.all(bc.demand('exit01', t) > 0)   # withdrawals positive


def test_parse_state_gaslib40():
    state_file = G40 / 'GasLib-40-sinus_5000_60-initial.state'
    if not state_file.exists():
        pytest.skip('GasLib-40 state data not available')
    st = data.parse_state(state_file)
    assert len(st.pipe_profiles) == 39
    xs, qs, ps = st.pipe_profiles['pipe_1']
    assert xs[0] == 0. and len(xs) == len(qs) == len(ps)
    assert np.all(ps > 1e5)            # pressures in Pa


def test_switch_encoding():
    r = switch_encoding(3)
    assert r.shape == (8, 3)
    assert len({tuple(row) for row in r}) == 8


@pytest.fixture(scope='module')
def solved_model(net, bc):
    model = GasOCModel(net, bc, HALF_HOUR)
    poc = adm.solve_relaxation(model, model.create_solver(tol=1e-6))
    return model, poc.w


def test_gas_poc_solves_and_is_physical(solved_model):
    model, w = solved_model
    s = model.solution_dict(w)
    # demand tracking
    for sid in s['sink_ids']:
        err = np.abs(s['delivered_kg_s'][sid] - s['demand_kg_s'][sid]).max()
        assert err < 0.5, 'delivery error {} kg/s at {}'.format(err, sid)
    # pressure bounds from the .net file (40-70 bar)
    for nid, p_bar in s['node_pressure_bar'].items():
        assert p_bar.min() > 39.9 and p_bar.max() < 70.1
    # SOS1
    assert np.allclose(s['alpha'].sum(axis=1), 1., atol=1e-6)


def test_gas_constraints_hold_at_solution(solved_model):
    model, w = solved_model
    # verify with the exact NLP constraint values (boundary flux variables
    # equal cell values only up to O(dx), so a check on the extracted
    # arrays would be approximate)
    assert model.constraint_violation(w, model.parameters()) < 1e-6


def test_gas_extract_overwrite_roundtrip(net, bc):
    model = GasOCModel(net, bc, HALF_HOUR)
    rng = np.random.default_rng(1)
    alpha = rng.random((len(model.t) - 1, model.n_configs))
    u = rng.random((len(model.t) - 1, model.n_controls))
    w = model.with_variables(model.w0, u=u, alpha=alpha)
    alpha2, u2, y = model.extract(w)
    assert np.allclose(u2, u)
    assert np.allclose(alpha2, alpha)
    assert np.array_equal(y, model.extract(model.w0).y)
