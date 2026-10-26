from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as cas
import numpy as np

SOLVED_STATUSES = ('Solve_Succeeded', 'Solved_To_Acceptable_Level')


class NlpBuilder:
    # variables and constraints in storage order; matrix variables are
    # stored column-major (CasADi's vec)

    def __init__(self) -> None:
        self._x: list[cas.MX] = []
        self._x0: list[float] = []
        self._lbx: list[float] = []
        self._ubx: list[float] = []
        self._g: list[cas.MX] = []
        self._lbg: list[float] = []
        self._ubg: list[float] = []
        self._offsets: dict[str, tuple[int, int, int]] = {}
        self._n = 0

    @property
    def offsets(self) -> dict[str, tuple[int, int, int]]:
        # variable name -> (position in x, rows, cols)
        return dict(self._offsets)

    def add_variable(self, sym: cas.MX, x0, lb, ub) -> cas.MX:
        n = sym.numel()
        if sym.name() in self._offsets:
            raise ValueError('duplicate variable name "{}"'.format(sym.name()))
        self._offsets[sym.name()] = (self._n, sym.size1(), sym.size2())
        self._n += n
        self._x.append(cas.reshape(sym, -1, 1))
        self._x0.extend(_column_major(x0, n))
        self._lbx.extend(_column_major(lb, n))
        self._ubx.extend(_column_major(ub, n))
        return sym

    def add_constraint(self, expr: cas.MX, lb, ub) -> None:
        n = expr.numel()
        self._g.append(expr)
        self._lbg.extend(_column_major(lb, n))
        self._ubg.extend(_column_major(ub, n))

    def add_equality(self, expr: cas.MX) -> None:
        self.add_constraint(expr, 0., 0.)

    def add_upper_bound(self, expr: cas.MX) -> None:
        self.add_constraint(expr, -np.inf, 0.)

    def build(self, f: cas.MX) -> NlpProblem:
        return NlpProblem(
            f=f, x=cas.vertcat(*self._x), g=cas.vertcat(*self._g),
            lbx=np.array(self._lbx), ubx=np.array(self._ubx),
            lbg=np.array(self._lbg), ubg=np.array(self._ubg),
            x0=np.array(self._x0))


@dataclass(frozen=True)
class NlpProblem:
    f: cas.MX
    x: cas.MX
    g: cas.MX
    lbx: np.ndarray
    ubx: np.ndarray
    lbg: np.ndarray
    ubg: np.ndarray
    x0: np.ndarray
    p: cas.MX | None = None


class NlpSolver:

    def __init__(self, problem: NlpProblem, tol: float,
                 max_wall_time: float | None = None) -> None:
        ipopt: dict = {'tol': tol, 'print_level': 0}
        if max_wall_time is not None:
            ipopt['max_wall_time'] = max_wall_time
        nlp = {'f': problem.f, 'x': problem.x, 'g': problem.g}
        if problem.p is not None:
            nlp['p'] = problem.p
        self._solver = cas.nlpsol('solver', 'ipopt', nlp,
                                  {'print_time': False, 'ipopt': ipopt})
        self._lbg = problem.lbg
        self._ubg = problem.ubg

    def solve(self, x0: np.ndarray, lbx: np.ndarray, ubx: np.ndarray,
              p: np.ndarray | None = None) -> NlpSolution:
        args = {'x0': x0, 'lbx': lbx, 'ubx': ubx,
                'lbg': self._lbg, 'ubg': self._ubg}
        if p is not None:
            args['p'] = p
        start = time.perf_counter()
        sol = self._solver(**args)
        wall_time = time.perf_counter() - start
        solution = NlpSolution(
            w=sol['x'].full().flatten(), f=float(sol['f']),
            status=self._solver.stats()['return_status'],
            wall_time=wall_time)
        if solution.status not in SOLVED_STATUSES:
            raise NlpSolveError(solution)
        return solution


@dataclass(frozen=True)
class NlpSolution:
    w: np.ndarray
    f: float
    status: str
    wall_time: float


class NlpSolveError(RuntimeError):

    def __init__(self, solution: NlpSolution) -> None:
        super().__init__('IPOPT returned "{}"'.format(solution.status))
        self.solution = solution


def _column_major(values, n: int) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return [float(arr)] * n
    if arr.size != n:
        raise ValueError('expected {} values, got {}'.format(n, arr.size))
    return arr.flatten(order='F').tolist()
