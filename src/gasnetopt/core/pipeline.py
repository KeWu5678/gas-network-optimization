from __future__ import annotations

from typing import Iterator

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

LABELS = {
    'poc': 'POC relaxation',
    'sur': 'Sum-Up Rounding + reoptimization',
    'comb': 'COMB-CIAP (dwell-time MILP) + reoptimization',
    'adm': 'penalty ADM',
    'adm_ciap': 'penalty ADM + CIAP',
}


def run_methods(ocp_model: OCModel, methods: tuple[str, ...], tau_min: float,
                nlp_tol: float, adm_settings: AdmSettings,
                milp_backend: str) -> Iterator[tuple[str, Solution]]:
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError('unknown method(s) {}; choose from {}'.format(
            ', '.join(unknown), ', '.join(METHODS)))
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


def _announce(method: str) -> None:
    print('\n=== {} ==='.format(LABELS[method]), flush=True)
