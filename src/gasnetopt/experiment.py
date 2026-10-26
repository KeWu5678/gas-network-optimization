from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from . import REPO_ROOT, config, results_io
from .config import ConfigError, Experiment
from .core import pipeline
from .core.adm import Solution
from .core.ocmodel import OCModel
from .gas import experiment as gas
from .translines import experiment as translines

# a run of configs/<name>.toml writes to results/<name>/
RESULTS_DIR = REPO_ROOT / 'results'


@dataclass(frozen=True)
class Application:
    schema: type[Experiment]
    build_model: Callable[[Any], OCModel]
    report: Callable[[Any, str, Solution, Path], None]
    present: Callable[[Any, dict[str, Solution], Path, str], list[Path]]
    draw_network: Callable[[Any, Path], Path] | None = None


APPLICATIONS = {
    'gas': Application(gas.GasExperiment, gas.build_model, gas.report,
                       gas.present, gas.draw_network),
    'translines': Application(translines.TranslinesExperiment,
                              translines.build_model, translines.report,
                              translines.present),
}


def run(config_path: str | Path) -> dict[str, Solution]:
    config_path = Path(config_path)
    application, cfg = load(config_path)
    out_dir = output_dir(config_path)
    config.copy_to_output(config_path, out_dir)
    with _logged(out_dir / 'run.log'):
        return _solve_and_record(application, cfg, out_dir)


def present(results_dir: str | Path, fmt: str = 'pdf') -> list[Path]:
    results_dir = Path(results_dir)
    application, cfg = load(results_dir / 'config.toml')
    solutions = _solutions(results_io.load_results(results_dir / 'results'))
    return application.present(application.build_model(cfg), solutions,
                               results_dir, fmt)


def table(results_dir: str | Path) -> str:
    db = results_io.load_results(Path(results_dir) / 'results')
    return pipeline.results_table(_solutions(db), db['r'])


def draw_network(config_path: str | Path) -> Path:
    application, cfg = load(config_path)
    if application.draw_network is None:
        raise ConfigError('{}: the {} application has no network '
                          'drawing'.format(config_path, cfg.application))
    return application.draw_network(cfg, output_dir(config_path))


def load(config_path: str | Path) -> tuple[Application, Experiment]:
    try:
        entries = config.read(config_path)
        name = entries.get('application')
        if name not in APPLICATIONS:
            raise ConfigError('application must be one of {}, got {!r}'
                              .format(', '.join(APPLICATIONS), name))
        application = APPLICATIONS[name]
        return application, config.build(application.schema, entries)
    except ConfigError as err:
        raise ConfigError('{}: {}'.format(config_path, err)) from err


def output_dir(config_path: str | Path) -> Path:
    return RESULTS_DIR / Path(config_path).stem


def _solve_and_record(application: Application, cfg: Experiment,
                      out_dir: Path) -> dict[str, Solution]:
    model = application.build_model(cfg)
    print('NLP: {} variables, {} constraints, {} configurations '
          '({} switches)'.format(model.n_variables, model.n_constraints,
                                 model.n_configs, model.n_switches))
    results = out_dir / 'results'
    results_io.save_results(results, {'r': model.r,
                                      'tau_min': cfg.switching.tau_min})
    solutions = {}
    for method, solution in pipeline.run_methods(
            model, cfg.methods, model.to_model_time(cfg.switching.tau_min),
            nlp_tol=cfg.nlp.tol, adm_settings=cfg.adm,
            milp_backend=cfg.milp.backend):
        results_io.update_results(results, **{method: solution})
        print('objective {:.10e}'.format(solution.objective))
        print('timings: {}'.format(
            {k: round(v, 2) for k, v in solution.timings.items()}))
        application.report(model, method, solution, out_dir)
        solutions[method] = solution
    return solutions


def _solutions(db: dict[str, Any]) -> dict[str, Solution]:
    return {method: db[method] for method in ('poc',) + pipeline.METHODS
            if method in db}


@contextlib.contextmanager
def _logged(path: Path) -> Iterator[None]:
    # progress goes to the terminal and to run.log in the output directory
    with open(path, 'w') as log, \
            contextlib.redirect_stdout(_Tee(sys.stdout, log)):
        yield


class _Tee(io.TextIOBase):

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, text: str) -> int:
        for stream in self._streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()
