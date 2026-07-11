import casadi as cas
import numpy as np
import pytest

from gasnetopt import DATA_DIR
from gasnetopt.gas import gaslib_io
from gasnetopt.gas.model import switch_encoding
from gasnetopt.gas.ocmodel import GasOCModel

G11 = DATA_DIR / 'gaslib' / 'GasLib-11'
G40 = DATA_DIR / 'gaslib' / 'GasLib-40'

# GasLib data is not redistributed with the repository (see README);
# skip the gas tests when it has not been placed under data/gaslib.
pytestmark = pytest.mark.skipif(
    not G11.exists(), reason='GasLib data not available under data/gaslib')


@pytest.fixture(scope='module')
def net():
    return gaslib_io.parse_net(G11 / 'GasLib-11-v1-20211130.net')


@pytest.fixture(scope='module')
def bc():
    return gaslib_io.parse_bcd(G11 / 'GasLib-11-sinus-InputData.bcd')


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
    st = gaslib_io.parse_state(state_file)
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
    model = GasOCModel(net, bc, T=1800., nt=31, nx=2)
    solver = model.create_NLP_solver(tol=1e-6)
    v_ref = np.array([[1] + [0] * (model.nv_ref - 1)]
                     * (len(model.t) - 1))
    p = np.concatenate(([0.], v_ref.flatten()))
    sol = solver(x0=model.w0, p=p, lbx=model.lbw, ubx=model.ubw,
                 lbg=model.lbg, ubg=model.ubg)
    assert solver.stats()['return_status'] in [
        'Solve_Succeeded', 'Solved_To_Acceptable_Level']
    return model, sol['x'].full().flatten()


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


def test_gas_mass_balance_at_innodes(solved_model):
    'Mass balance residual at inner nodes must vanish.'
    model, w = solved_model
    d = model.data
    sca = model.scaling
    s = model.solution_dict(w)
    # innode N01: pipe02 leaves, CS01 enters, valve V01 leaves
    m_pipe02 = s['pipe_m']['pipe02_N01_N02']
    area = model.net.pipes[1].area
    q_out = m_pipe02[0, :-1] * area   # rough check: flux into first cell
    q_cs01 = s['flows_kg_s']['CS01_entry03_N01']
    q_v01 = s['flows_kg_s']['V01_N01_N03']
    # boundary flux variable equals cell value only up to O(dx); use the
    # exact constraint instead: residual of net_inflow was enforced == 0,
    # so verify with the NLP constraint values
    g_fun = cas.Function('g', [model.nlp['x']], [model.nlp['g']])
    gval = np.array(g_fun(w)).flatten()
    lbg = np.array(model.lbg)
    ubg = np.array(model.ubg)
    eq = lbg == ubg
    assert np.abs(gval[eq] - lbg[eq]).max() < 1e-6


def test_gas_extract_overwrite_roundtrip(net, bc):
    model = GasOCModel(net, bc, T=1800., nt=31, nx=2)
    w = np.array(model.w0, dtype=float).copy()
    rng = np.random.default_rng(1)
    alpha = rng.random((model.nt - 1, model.n_confg))
    u = rng.random((model.nt - 1, model.n_src))
    w = model.overwrite(w, u=u, alpha=alpha)
    y, u2, alpha2, _, _ = model.extract(w)
    assert np.allclose(u2, u)
    assert np.allclose(alpha2, alpha)
