'''Gas code path on the synthetic Tiny-3 fixture (runs in CI without
GasLib/TRR154 data): parsing, data validation, model build, POC solve, and
the penalty-ADM time-budget fallback.'''

from pathlib import Path
from types import SimpleNamespace

import casadi as cas
import numpy as np
import pytest

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


def test_validate_uses_simultaneous_peak():
    'Sinks peaking at different times must not be summed as simultaneous.'
    net, bc = tiny()
    net.nodes['snk2'] = gaslib_io.Node(
        id='snk2', type='sink', pressure_min=40e5, pressure_max=70e5)
    net.pipes.append(gaslib_io.Pipe(
        id='pipe2', from_node='mid', to_node='snk2', length=10e3,
        diameter=0.5, roughness=5e-5))
    # per-sink maxima sum to 120 kg/s > capacity (~65.4 kg/s), but the
    # peaks are staggered: simultaneous total never exceeds 65 kg/s
    t = np.array([0., 1800., 3600.])
    bc.massflow['snk'] = (t, np.array([-60., -5., -5.]))
    bc.massflow['snk2'] = (t, np.array([-5., -5., -60.]))
    assert gaslib_io.validate(net, bc) == []


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


def test_compressor_envelope_applies_in_bypass():
    net, bc = tiny()
    net.compressors[0].pressure_in_min = 55e5
    net.compressors[0].pressure_out_max = 60e5
    model = GasOCModel(net, bc, T=600., nt=42, nx=2, p_init=56e5)
    solver = model.create_NLP_solver(tol=1e-7)
    v_ref = np.zeros((len(model.t) - 1, model.nv_ref))
    p = np.concatenate(([0.], v_ref.flatten()))
    beta = np.zeros((len(model.t) - 1, model.nalpha))
    beta[:, 0] = 1.  # compressor bypassed
    *_, w, _ = adm.resolve_with_fixed_controls(
        model, solver, model.w0.copy(), p, alpha_fix=beta)
    assert solver.stats()['return_status'] in [
        'Solve_Succeeded', 'Solved_To_Acceptable_Level']
    solution = model.solution_dict(w)
    for nid in ('src', 'mid'):
        pressure = solution['node_pressure_bar'][nid]
        assert pressure.min() >= 55. - 1e-5
        assert pressure.max() <= 60. + 1e-5


def test_final_feasibility_checks_status_finiteness_and_bounds():
    x = cas.MX.sym('x')
    model = SimpleNamespace(
        nlp={'x': x, 'g': x - x}, lbg=[0.], ubg=[0.])
    lbx, ubx, p = np.array([0.]), np.array([1.]), np.array([])

    with pytest.raises(RuntimeError, match='status'):
        adm._assert_feasible(
            model, np.array([0.5]), p, lbx, ubx, 'Maximum_Iterations_Exceeded')
    with pytest.raises(RuntimeError, match='non-finite'):
        adm._assert_feasible(
            model, np.array([np.nan]), p, lbx, ubx, 'Solve_Succeeded')
    with pytest.raises(RuntimeError, match='variable bounds'):
        adm._assert_feasible(
            model, np.array([2.]), p, lbx, ubx, 'Solve_Succeeded')


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
