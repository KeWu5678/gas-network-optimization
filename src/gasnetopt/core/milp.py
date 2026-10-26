'''
Thin solver-agnostic MILP layer.

The combinatorial subproblems of the CIAP/ADM algorithms (CIAP-MILP, dwell-time
COMB and tCOMB) are small MILPs. They are formulated once against this builder
and solved with one of two backends:

- 'highs' (default): HiGHS via scipy.optimize.milp, no extra dependency.
- 'gurobi': gurobipy, if installed (optional extra ``gasnetopt[gurobi]``).

The backend is selected per call.
'''

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

BACKENDS = ('highs', 'gurobi')


class MilpModel:
    '''Incrementally built MILP: min c'x s.t. lb_r <= A x <= ub_r, bounds,
    integrality. Supports re-solving with updated objective coefficients
    (used by tCOMB inside the ADM loop).'''

    def __init__(self, name: str = '') -> None:
        self.name = name
        self._obj: list[float] = []
        self._lb: list[float] = []
        self._ub: list[float] = []
        self._binary: list[bool] = []
        # constraint rows: (dict col -> coeff, row lb, row ub)
        self._rows: list[tuple[dict[int, float], float, float]] = []
        self._A = None  # cached sparse constraint matrix

    @property
    def ncols(self) -> int:
        return len(self._obj)

    def add_var(self, lb: float = 0., ub: float = np.inf,
                obj: float = 0., binary: bool = False) -> int:
        'Add one variable, return its column index.'
        if binary:
            lb, ub = 0., 1.
        self._obj.append(obj)
        self._lb.append(lb)
        self._ub.append(ub)
        self._binary.append(binary)
        self._A = None
        return self.ncols - 1

    def add_vars(self, shape: int | tuple[int, ...], lb: float = 0.,
                 ub: float = np.inf, obj: float = 0.,
                 binary: bool = False) -> np.ndarray:
        'Add an array of variables, return array of column indices.'
        n = int(np.prod(shape))
        idx = np.array([self.add_var(lb, ub, obj, binary) for _ in range(n)])
        return idx.reshape(shape)

    def add_constr(self, coeffs: dict[int, float],
                   lb: float = -np.inf, ub: float = np.inf) -> None:
        '''Add a row given as dict {col: coeff} with row bounds. Equality:
        lb == ub.'''
        self._rows.append((dict(coeffs), lb, ub))
        self._A = None

    def set_objective(self, idx: np.ndarray, coeffs: np.ndarray) -> None:
        'Overwrite objective coefficients for columns idx (array-like).'
        obj = np.asarray(self._obj, dtype=float)
        obj[np.asarray(idx).reshape(-1)] = np.asarray(coeffs).reshape(-1)
        self._obj = obj.tolist()

    def _matrix(self) -> sparse.csr_matrix:
        if self._A is None:
            data, ri, ci = [], [], []
            for r, (coeffs, _, _) in enumerate(self._rows):
                for c, v in coeffs.items():
                    ri.append(r)
                    ci.append(int(c))
                    data.append(float(v))
            self._A = sparse.csr_matrix(
                (data, (ri, ci)), shape=(len(self._rows), self.ncols))
        return self._A

    def solve(self, backend: str | None = None,
              warm_start: np.ndarray | None = None,
              time_limit: float | None = None
              ) -> tuple[np.ndarray, float, float]:
        '''Solve, return (x, obj_val, wall_time). warm_start is only used by
        the gurobi backend (HiGHS via scipy has no warm-start interface).
        With a time_limit [s], the best incumbent found within the limit is
        returned (with a warning) instead of failing.'''
        backend = backend or 'highs'
        if backend == 'highs':
            if warm_start is not None:
                print('Warning: MILP "{}": the HiGHS backend ignores the warm '
                      'start'.format(self.name))
            return self._solve_highs(time_limit)
        if backend == 'gurobi':
            return self._solve_gurobi(warm_start, time_limit)
        raise ValueError('Unknown MILP backend "{}"'.format(backend))

    def _solve_highs(self, time_limit: float | None = None
                     ) -> tuple[np.ndarray, float, float]:
        from scipy.optimize import Bounds, LinearConstraint, milp
        A = self._matrix()
        row_lb = np.array([r[1] for r in self._rows])
        row_ub = np.array([r[2] for r in self._rows])
        options = {} if time_limit is None else {'time_limit': time_limit}
        t0 = time.time()
        res = milp(
            c=np.asarray(self._obj),
            constraints=LinearConstraint(A, row_lb, row_ub),
            bounds=Bounds(np.asarray(self._lb), np.asarray(self._ub)),
            integrality=np.asarray(self._binary, dtype=int),
            options=options,
        )
        wall = time.time() - t0
        if not res.success and res.status == 1 and res.x is not None:
            # hit the time/iteration limit with a feasible incumbent
            print('Warning: MILP "{}" returned incumbent at time limit'
                  .format(self.name))
            return res.x, res.fun, wall
        if not res.success:
            raise RuntimeError('MILP "{}" failed: {}'.format(
                self.name, res.message))
        return res.x, res.fun, wall

    def _solve_gurobi(self, warm_start: np.ndarray | None = None,
                      time_limit: float | None = None
                      ) -> tuple[np.ndarray, float, float]:
        import gurobipy as grb
        m = grb.Model(self.name)
        m.setParam('LogToConsole', 0)
        m.setParam('Threads', 1)      # determinism over speed
        if time_limit is not None:
            m.setParam('TimeLimit', time_limit)
        n = self.ncols
        vtypes = [grb.GRB.BINARY if b else grb.GRB.CONTINUOUS
                  for b in self._binary]
        x = m.addVars(range(n), lb=self._lb, ub=self._ub, obj=self._obj)
        for i in range(n):
            x[i].VType = vtypes[i]
            if warm_start is not None and self._binary[i]:
                x[i].Start = warm_start[i]
        for coeffs, lb, ub in self._rows:
            expr = grb.quicksum(v * x[c] for c, v in coeffs.items())
            if lb == ub:
                m.addConstr(expr == lb)
            else:
                if np.isfinite(ub):
                    m.addConstr(expr <= ub)
                if np.isfinite(lb):
                    m.addConstr(expr >= lb)
        m.optimize()
        if m.status == grb.GRB.Status.TIME_LIMIT and m.SolCount > 0:
            print('Warning: MILP "{}" returned incumbent at time limit'
                  .format(self.name))
        elif m.status != grb.GRB.Status.OPTIMAL:
            raise RuntimeError('MILP "{}" failed (gurobi status {})'.format(
                self.name, m.status))
        sol = np.array([x[i].X for i in range(n)])
        return sol, m.ObjVal, m.Runtime
