'''Gas code path on the synthetic Tiny-3 fixture (runs in CI without
GasLib/TRR154 data): parsing, data validation, model build, POC solve, and
the penalty-ADM time-budget fallback.'''

from pathlib import Path

import numpy as np

from gasnetopt import adm
from gasnetopt.gas import gaslib_io
from gasnetopt.gas.ocmodel import GasOCModel

FIXTURES = Path(__file__).parent / 'fixtures'


def tiny():
    net = gaslib_io.parse_net(FIXTURES / 'tiny.net')
    bc = gaslib_io.parse_bcd(FIXTURES / 'tiny.bcd')
    return net, bc


def test_parse_tiny_fixture():
    net, bc = tiny()
    assert len(net.nodes) == 3
    assert len(net.pipes) == 1 and len(net.compressors) == 1
    assert net.pipes[0].length == 10e3
    assert net.nodes['src'].pressure_max == 70e5
    # 300 * 1000 m^3/h * 0.785 kg/m^3 / 3600 s
    assert np.isclose(net.nodes['src'].flow_max, 300e3 * 0.785 / 3600.)
    assert bc.speed_of_sound == 340.
    assert np.allclose(bc.demand('snk', np.array([0., 3600.])), [20., 25.])


def test_validate_clean_fixture():
    net, bc = tiny()
    assert gaslib_io.validate(net, bc) == []


def test_validate_diagnoses_dirty_data():
    net, bc = tiny()
    net.compressors[0].pressure_out_max = np.inf   # missing bound -> big-M
    del bc.massflow['snk']                         # sink without demand data
    findings = gaslib_io.validate(net, bc)
    assert any('pressureOutMax' in f for f in findings)
    assert any('snk' in f and 'massflow' in f for f in findings)


def test_validate_detects_disconnected_node():
    net, bc = tiny()
    net.nodes['orphan'] = gaslib_io.Node(
        id='orphan', type='innode', pressure_min=40e5, pressure_max=70e5)
    findings = gaslib_io.validate(net, bc)
    assert any('orphan' in f and 'not connected' in f for f in findings)


def test_poc_solves_on_fixture():
    net, bc = tiny()
    model = GasOCModel(net, bc, T=600., nt=42, nx=2)
    solver = model.create_NLP_solver(tol=1e-6)
    v_ref = np.zeros((len(model.t) - 1, model.nv_ref))
    p = np.concatenate(([0.], v_ref.flatten()))
    sol = solver(x0=model.w0, p=p, lbx=model.lbw, ubx=model.ubw,
                 lbg=model.lbg, ubg=model.ubg)
    assert solver.stats()['return_status'] in [
        'Solve_Succeeded', 'Solved_To_Acceptable_Level']
    w = sol['x'].full().flatten()
    s = model.solution_dict(w)
    assert np.allclose(s['alpha'].sum(axis=1), 1., atol=1e-6)   # SOS1
    err = np.abs(s['delivered_kg_s']['snk'] - s['demand_kg_s']['snk']).max()
    assert err < 0.5


def test_adm_time_budget_returns_dwell_feasible_incumbent():
    'Exhausted budget must still return a binary dwell-feasible schedule.'
    net, bc = tiny()
    model = GasOCModel(net, bc, T=600., nt=42, nx=2)
    tau = 120. / model.scaling.T_ref
    y, u, beta, v_ref, nodes, obj, w, timings = adm.miocp_adm(
        model, tau_min=tau, with_CIAP=True, time_budget=0.)
    # binary multipliers with SOS1
    assert set(np.unique(beta)) <= {0., 1.}
    assert np.all(beta.sum(axis=1) == 1)
    # returned switch schedule respects the dwell time (interior blocks)
    v = v_ref[:, 0]
    t = model.t
    for k in range(1, len(v)):
        if v[k] != v[k - 1]:                       # switch at t[k]
            run = 1
            while k + run < len(v) and v[k + run] == v[k]:
                run += 1
            if k + run < len(v):                   # interior block only
                assert t[k + run] - t[k] >= tau - 1e-9
    assert np.isfinite(obj)
