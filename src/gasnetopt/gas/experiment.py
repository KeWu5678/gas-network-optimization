from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Experiment
from ..core import pipeline
from ..core.adm import Solution
from . import data, plotting
from .model import GasModelSettings, GasOCModel


@dataclass(frozen=True)
class GasData:
    network: Path                   # GasLib .net
    boundary: Path                  # TRR154 .bcd


@dataclass(frozen=True)
class GasExperiment(Experiment):    # [switching] tau_min in seconds
    data: GasData
    model: GasModelSettings


def build_model(cfg: GasExperiment) -> GasOCModel:
    net = data.parse_net(cfg.data.network)
    bc = data.parse_bcd(cfg.data.boundary)
    print('Network {}: {} nodes, {} pipes, {} valves, {} compressors'.format(
        net.name, len(net.nodes), len(net.pipes), len(net.valves),
        len(net.compressors)))
    data.validate(net, bc)
    return GasOCModel(net, bc, cfg.model)


def report(model: GasOCModel, method: str, solution: Solution,
           out_dir: Path) -> None:
    s = model.solution_dict(solution.w)
    if method != 'poc':
        events = pipeline.switching_events(s['switch_state'])
        share_on = np.round(s['switch_state']).mean(axis=0)
        for sid, n_events, on in zip(s['switch_ids'], events, share_on):
            print('{}: {} switching events, state {:.0%} on'.format(
                sid, n_events, on))
    print('max delivery error: {:.4f} kg/s'.format(
        s['max_delivery_error_kg_s']))
    path = _plot(model, method, s, solution.objective,
                 out_dir / '{}.png'.format(method))
    print('saved {}'.format(path.name))


def present(model: GasOCModel, solutions: dict[str, Solution],
            out_dir: Path, fmt: str) -> list[Path]:
    return [_plot(model, method, model.solution_dict(solution.w),
                  solution.objective, out_dir / '{}.{}'.format(method, fmt))
            for method, solution in solutions.items()]


def draw_network(cfg: GasExperiment, out_dir: Path) -> Path:
    path = out_dir / 'topology.png'
    plotting.plot_topology(data.parse_net(cfg.data.network), path)
    return path


def _plot(model: GasOCModel, method: str, s: dict, objective: float,
          path: Path) -> Path:
    title = '{} {}'.format(model.network.name,
                           pipeline.METHOD_INFO[method].label)
    plotting.plot_solution(s, title, path, objective=objective)
    return path
