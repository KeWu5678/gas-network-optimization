from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .adm import (
    AdmSettings,
    Solution,
    miocp_adm,
    round_and_reoptimize,
    solve_relaxation,
)
from .ocmodel import OCModel

# the POC relaxation always runs first; these methods start from it
METHODS = ('sur', 'comb', 'adm', 'adm_ciap')


@dataclass(frozen=True)
class MethodInfo:
    label: str              # headings and tables
    abbreviation: str       # plot legends
    dwell_feasible: bool    # schedule satisfies tau_min by construction


METHOD_INFO = {
    'poc': MethodInfo('POC relaxation', 'POC', False),
    'sur': MethodInfo('Sum-Up Rounding + reoptimization', 'SUR', False),
    'comb': MethodInfo('COMB-CIAP (dwell-time MILP) + reoptimization',
                       'COMB-CIAP', True),
    'adm': MethodInfo('penalty ADM', 'ADM', True),
    'adm_ciap': MethodInfo('penalty ADM + CIAP', 'ADM + CIAP', True),
}


def run_methods(ocp_model: OCModel, methods: tuple[str, ...], tau_min: float,
                nlp_tol: float, adm_settings: AdmSettings,
                milp_backend: str) -> Iterator[tuple[str, Solution]]:
    check_methods(methods)
    solver = ocp_model.create_solver(tol=nlp_tol)

    _announce('poc')
    poc = solve_relaxation(ocp_model, solver)
    yield 'poc', poc
    if 'sur' in methods:
        _announce('sur')
        yield 'sur', round_and_reoptimize(ocp_model, solver, poc, 'SUR')
    if 'comb' in methods:
        _announce('comb')
        yield 'comb', round_and_reoptimize(ocp_model, solver, poc, 'COMB',
                                           tau_min=tau_min,
                                           milp_backend=milp_backend)
    if 'adm' in methods:
        _announce('adm')
        yield 'adm', miocp_adm(ocp_model, tau_min, with_ciap=False,
                               settings=adm_settings,
                               milp_backend=milp_backend)
    if 'adm_ciap' in methods:
        _announce('adm_ciap')
        yield 'adm_ciap', miocp_adm(ocp_model, tau_min, with_ciap=True,
                                    settings=adm_settings,
                                    milp_backend=milp_backend)


def check_methods(methods: tuple[str, ...]) -> None:
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError('unknown method(s) {}; choose from {}'.format(
            ', '.join(unknown), ', '.join(METHODS)))


def results_table(solutions: dict[str, Solution], r: np.ndarray) -> str:
    rows = ['| method | objective | switching events | dwell-feasible |',
            '|---|---|---|---|']
    for method, solution in solutions.items():
        info = METHOD_INFO[method]
        alpha = np.asarray(solution.alpha)
        if np.abs(alpha - np.round(alpha)).max() < 1e-6:
            events = str(switching_events(np.round(alpha).dot(r)).sum())
        else:
            events = '-'
        rows.append('| {} | {:.5g} | {} | {} |'.format(
            info.label, solution.objective, events,
            'yes' if info.dwell_feasible else 'no'))
    return '\n'.join(rows)


def switching_events(switch_state: np.ndarray) -> np.ndarray:
    # per switch: changes of the rounded state between consecutive intervals
    return np.abs(np.diff(np.round(switch_state), axis=0)).sum(
        axis=0).astype(int)


def _announce(method: str) -> None:
    print('\n=== {} ==='.format(METHOD_INFO[method].label), flush=True)
