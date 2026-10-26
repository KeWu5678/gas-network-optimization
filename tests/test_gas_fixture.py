'''Gas code path on the synthetic Tiny-3 fixture (runs in CI without
GasLib/TRR154 data): parsing, data validation, model build, POC solve, and
the penalty-ADM time-budget fallback.'''

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gasnetopt.core import adm
from gasnetopt.core.adm import AdmSettings
from gasnetopt.gas import data
from gasnetopt.gas.model import GasModelSettings, GasOCModel

FIXTURES = Path(__file__).parent / 'fixtures'
TEN_MINUTES = GasModelSettings(horizon=600., nt=42, nx=2)


def tiny():
    net = data.parse_net(FIXTURES / 'tiny.net')
    bc = data.parse_bcd(FIXTURES / 'tiny.bcd')
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
    data.validate(net, bc)


def findings(net, bc):
    with pytest.raises(data.InvalidDataError) as err:
        data.validate(net, bc)
    return err.value.findings


def test_validate_diagnoses_dirty_data():
    net, bc = tiny()
    net.compressors[0].pressure_out_max = np.inf   # missing bound -> big-M
    del bc.massflow['snk']                         # sink without demand data
    found = findings(net, bc)
    assert any('pressureOutMax' in f for f in found)
    assert any('snk' in f and 'massflow' in f for f in found)


def test_validate_uses_simultaneous_peak():
    'Sinks peaking at different times must not be summed as simultaneous.'
    net, bc = tiny()
    net.nodes['snk2'] = data.Node(
        id='snk2', type='sink', pressure_min=40e5, pressure_max=70e5)
    net.pipes.append(data.Pipe(
        id='pipe2', from_node='mid', to_node='snk2', length=10e3,
        diameter=0.5, roughness=5e-5))
    # per-sink maxima sum to 120 kg/s > capacity (~65.4 kg/s), but the
    # peaks are staggered: simultaneous total never exceeds 65 kg/s
    t = np.array([0., 1800., 3600.])
    bc.massflow['snk'] = (t, np.array([-60., -5., -5.]))
    bc.massflow['snk2'] = (t, np.array([-5., -5., -60.]))
    data.validate(net, bc)


def test_validate_detects_disconnected_node():
    net, bc = tiny()
    net.nodes['orphan'] = data.Node(
        id='orphan', type='innode', pressure_min=40e5, pressure_max=70e5)
    assert any('orphan' in f and 'not connected' in f
               for f in findings(net, bc))


def test_model_build_rejects_missing_big_m_bound():
    net, bc = tiny()
    net.valves.append(data.Valve(id='v', from_node='src', to_node='mid'))
    with pytest.raises(data.InvalidDataError, match='flowMax missing'):
        GasOCModel(net, bc, TEN_MINUTES)


def test_poc_solves_on_fixture():
    net, bc = tiny()
    model = GasOCModel(net, bc, TEN_MINUTES)
    poc = adm.solve_relaxation(model, model.create_solver(tol=1e-6))
    s = model.solution_dict(poc.w)
    assert np.allclose(s['alpha'].sum(axis=1), 1., atol=1e-6)   # SOS1
    err = np.abs(s['delivered_kg_s']['snk'] - s['demand_kg_s']['snk']).max()
    assert err < 0.5


def test_compressor_envelope_applies_in_bypass():
    net, bc = tiny()
    net.compressors[0].pressure_in_min = 55e5
    net.compressors[0].pressure_out_max = 60e5
    model = GasOCModel(
        net, bc, GasModelSettings(horizon=600., nt=42, nx=2, p_init=56e5))
    solver = model.create_solver(tol=1e-7)
    beta = np.zeros((len(model.t) - 1, model.n_configs))
    beta[:, 0] = 1.  # compressor bypassed
    sol = model.solve(solver, model.w0, model.parameters(), alpha_fix=beta)
    solution = model.solution_dict(sol.w)
    for nid in ('src', 'mid'):
        pressure = solution['node_pressure_bar'][nid]
        assert pressure.min() >= 55. - 1e-5
        assert pressure.max() <= 60. + 1e-5


def test_final_feasibility_checks_finiteness_bounds_and_constraints():
    def model(violation):
        return SimpleNamespace(constraint_violation=lambda w, p: violation)
    lbx, ubx, p = np.array([0.]), np.array([1.]), np.array([])

    adm._assert_feasible(model(0.), np.array([0.5]), p, lbx, ubx)
    with pytest.raises(RuntimeError, match='non-finite'):
        adm._assert_feasible(model(0.), np.array([np.nan]), p, lbx, ubx)
    with pytest.raises(RuntimeError, match='variable bounds'):
        adm._assert_feasible(model(0.), np.array([2.]), p, lbx, ubx)
    with pytest.raises(RuntimeError, match='infeasible'):
        adm._assert_feasible(model(np.inf), np.array([0.5]), p, lbx, ubx)


def test_adm_time_budget_returns_dwell_feasible_incumbent():
    'Exhausted budget must still return a binary dwell-feasible schedule.'
    net, bc = tiny()
    model = GasOCModel(net, bc, TEN_MINUTES)
    tau = model.to_model_time(120.)
    result = adm.miocp_adm(model, tau, with_ciap=True,
                           settings=AdmSettings(time_budget=0.))
    assert_binary_dwell_feasible(model, result.alpha, tau)
    assert np.isfinite(result.objective)


def test_adm_returns_dwell_feasible_schedule():
    # the schedule itself differs between platforms (IPOPT/MUMPS rounding in
    # the macOS and Linux builds), so check what the method guarantees
    net, bc = tiny()
    model = GasOCModel(net, bc, TEN_MINUTES)
    tau = model.to_model_time(120.)
    result = adm.miocp_adm(model, tau, with_ciap=True)
    assert_binary_dwell_feasible(model, result.alpha, tau)
    assert 0. <= result.objective < np.inf
    assert set(result.timings) == {'poc_nlp', 'reopt_nlp', 'ciap', 'tcomb'}


def assert_binary_dwell_feasible(model, alpha, tau):
    # binary multipliers with SOS1
    assert set(np.unique(alpha)) <= {0., 1.}
    assert np.all(alpha.sum(axis=1) == 1)
    # every switch keeps its state at least tau (interior blocks only; the
    # end of the horizon may cut a block short)
    t = model.t
    for v in alpha.dot(model.r).T:
        for k in range(1, len(v)):
            if v[k] != v[k - 1]:                   # switch at t[k]
                run = 1
                while k + run < len(v) and v[k + run] == v[k]:
                    run += 1
                if k + run < len(v):
                    assert t[k + run] - t[k] >= tau - 1e-9
