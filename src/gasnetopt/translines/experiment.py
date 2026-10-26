from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Experiment
from ..core.adm import Solution
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
class TranslinesExperiment(Experiment):  # [switching] tau_min in model time
    data: TranslinesData
    model: TranslinesModelSettings


def build_model(cfg: TranslinesExperiment) -> TranslinesOCModel:
    return TranslinesOCModel(data.NETWORKS[cfg.data.network],
                             data.load_demand(cfg.data.demand), cfg.model)


def report(model: TranslinesOCModel, method: str, solution: Solution,
           out_dir: Path) -> None:
    figures = plotting.plot_solution(model.network,
                                     model.solution_dict(solution.w),
                                     out_dir, prefix=method)
    print('saved {}'.format(', '.join(path.name for path in figures)))


def present(model: TranslinesOCModel, solutions: dict[str, Solution],
            out_dir: Path, fmt: str) -> list[Path]:
    return plotting.plot_comparison(
        model.network,
        {method: model.solution_dict(solution.w)
         for method, solution in solutions.items()},
        out_dir, fmt)
