from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config, results_io
from ..config import MilpSettings, NlpSettings, Output, Switching
from ..core import pipeline
from ..core.adm import AdmSettings, Solution
from . import data, plotting
from .model import TranslinesModelSettings, TranslinesOCModel


@dataclass(frozen=True)
class TranslinesData:
    network: str                    # built-in network, see data.NETWORKS
    demand: Path                    # rows: consumers, columns: time steps

    def __post_init__(self) -> None:
        if self.network not in data.NETWORKS:
            raise ValueError('network "{}" is not one of {}'.format(
                self.network, ', '.join(data.NETWORKS)))


@dataclass(frozen=True)
class TranslinesExperiment:
    data: TranslinesData
    model: TranslinesModelSettings
    switching: Switching            # tau_min in model time units
    nlp: NlpSettings
    adm: AdmSettings
    milp: MilpSettings
    output: Output


def run(config_path: str | Path,
        methods: tuple[str, ...] = pipeline.METHODS) -> dict[str, Solution]:
    cfg = config.load(config_path, TranslinesExperiment)
    config.copy_to_output(config_path, cfg.output.dir)
    model = build_model(cfg)
    print('Transmission lines example on {} net with tau_min={:9.3e}'.format(
        model.network.name, cfg.switching.tau_min))

    results = cfg.output.dir / 'results'
    results_io.update_results(results, r=model.r,
                              tau_min=cfg.switching.tau_min)
    solutions = {}
    for method, solution in pipeline.run_methods(
            model, methods, cfg.switching.tau_min, nlp_tol=cfg.nlp.tol,
            adm_settings=cfg.adm, milp_backend=cfg.milp.backend):
        results_io.update_results(results, **{method: solution})
        _report(model, method, solution, cfg.output.dir)
        solutions[method] = solution
    return solutions


def build_model(cfg: TranslinesExperiment) -> TranslinesOCModel:
    return TranslinesOCModel(data.NETWORKS[cfg.data.network],
                             data.load_demand(cfg.data.demand), cfg.model)


def present(results_dir: Path, fmt: str = 'pdf') -> list[Path]:
    cfg = config.load(results_dir / 'config.toml', TranslinesExperiment)
    model = build_model(cfg)
    db = results_io.load_results(results_dir / 'results')
    solutions = {method: model.solution_dict(db[method].w)
                 for method in ('poc',) + pipeline.METHODS if method in db}
    return plotting.plot_comparison(model.network, solutions, results_dir,
                                    fmt)


def _report(model: TranslinesOCModel, method: str, solution: Solution,
            out_dir: Path) -> None:
    print('objective {:.10e}'.format(solution.objective))
    print('timings: {}'.format(solution.timings))
    figures = plotting.plot_solution(model.network,
                                     model.solution_dict(solution.w),
                                     out_dir, prefix=method)
    print('saved {}'.format(', '.join(path.name for path in figures)))
