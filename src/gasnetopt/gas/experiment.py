from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config, results_io
from ..config import MilpSettings, NlpSettings, Output, Switching
from ..core import pipeline
from ..core.adm import AdmSettings, Solution
from . import data, plotting
from .model import GasModelSettings, GasOCModel

DEFAULT_METHODS = ('sur', 'adm_ciap')


@dataclass(frozen=True)
class GasData:
    network: Path                   # GasLib .net
    boundary: Path                  # TRR154 .bcd


@dataclass(frozen=True)
class GasExperiment:
    data: GasData
    model: GasModelSettings
    switching: Switching            # tau_min [s]
    nlp: NlpSettings
    adm: AdmSettings
    milp: MilpSettings
    output: Output


def run(config_path: str | Path,
        methods: tuple[str, ...] = DEFAULT_METHODS) -> dict[str, Solution]:
    cfg = config.load(config_path, GasExperiment)
    config.copy_to_output(config_path, cfg.output.dir)
    net, bc = load_data(cfg.data)
    model = GasOCModel(net, bc, cfg.model)
    print('NLP: {} variables, {} constraints, {} configurations '
          '({} switches)'.format(model.n_variables, model.n_constraints,
                                 model.n_configs, model.n_switches))

    results = cfg.output.dir / 'results'
    results_io.update_results(results, r=model.r,
                              tau_min=cfg.switching.tau_min)
    solutions = {}
    for method, solution in pipeline.run_methods(
            model, methods, cfg.switching.tau_min / model.scaling.T_ref,
            nlp_tol=cfg.nlp.tol, adm_settings=cfg.adm,
            milp_backend=cfg.milp.backend):
        results_io.update_results(results, **{method: solution})
        _report(net.name, model, method, solution, cfg.output.dir)
        solutions[method] = solution
    return solutions


def load_data(paths: GasData) -> tuple[data.GasNetwork,
                                       data.BoundaryConditions]:
    net = data.parse_net(paths.network)
    bc = data.parse_bcd(paths.boundary)
    print('Network {}: {} nodes, {} pipes, {} valves, {} compressors'.format(
        net.name, len(net.nodes), len(net.pipes), len(net.valves),
        len(net.compressors)))
    findings = data.validate(net, bc)
    if findings:
        raise ValueError('invalid input data:\n  ' + '\n  '.join(findings))
    return net, bc


def _report(network_name: str, model: GasOCModel, method: str,
            solution: Solution, out_dir: Path) -> None:
    print('objective {:.10e}'.format(solution.objective))
    print('timings: {}'.format(
        {k: round(v, 2) for k, v in solution.timings.items()}))
    s = model.solution_dict(solution.w)
    if method != 'poc':
        for j, sid in enumerate(s['switch_ids']):
            state = np.round(s['switch_state'][:, j]).astype(int)
            print('{}: {} switching events, state {:.0%} on'.format(
                sid, int(np.abs(np.diff(state)).sum()), state.mean()))
    error = max(np.abs(s['delivered_kg_s'][k] - s['demand_kg_s'][k]).max()
                for k in s['sink_ids'])
    print('max delivery error: {:.4f} kg/s'.format(error))
    path = out_dir / '{}.png'.format(method)
    plotting.plot_solution(s, '{} {}'.format(network_name,
                                             pipeline.LABELS[method]),
                           path, objective=solution.objective)
    print('saved {}'.format(path.name))
