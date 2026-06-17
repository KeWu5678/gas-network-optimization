'''
Thin solver-agnostic MILP layer.

The combinatorial subproblems of the CIAP/ADM algorithms (CIAP-MILP, dwell-time
COMB and tCOMB) are small MILPs. They are formulated once against this builder
and solved with one of two backends:

- 'highs' (default): HiGHS via scipy.optimize.milp, no extra dependency.
- 'gurobi': gurobipy, if installed (optional extra ``gasnetopt[gurobi]``).

The backend can be selected per call or globally via the environment variable
``GASNETOPT_MILP_BACKEND``.
'''

import os
import time

import numpy as np
from scipy import sparse


def default_backend():
    return os.environ.get('GASNETOPT_MILP_BACKEND', 'highs')


class MilpModel:
    '''Incrementally built MILP: min c'x s.t. lb_r <= A x <= ub_r, bounds,
    integrality. Supports re-solving with updated objective coefficients
    (used by tCOMB inside the ADM loop).'''

    def __init__(self, name=''):
        self.name = name
        self.obj = []
        self.lb = []
        self.ub = []
        self.binary = []
        # constraint rows: (dict col -> coeff, row lb, row ub)
        self.rows = []
        self._A = None  # cached sparse constraint matrix

    @property
    def ncols(self):
        return len(self.obj)

    def add_var(self, lb=0., ub=np.inf, obj=0., binary=False):
        'Add one variable, return its column index.'
        if binary:
            lb, ub = 0., 1.
        self.obj.append(obj)
        self.lb.append(lb)
        self.ub.append(ub)
        self.binary.append(binary)
        self._A = None
        return self.ncols - 1

    def add_vars(self, shape, lb=0., ub=np.inf, obj=0., binary=False):
        'Add an array of variables, return array of column indices.'
        n = int(np.prod(shape))
        idx = np.array([self.add_var(lb, ub, obj, binary) for _ in range(n)])
        return idx.reshape(shape)

    def add_constr(self, coeffs, lb=-np.inf, ub=np.inf):
        '''Add a row given as dict {col: coeff} with row bounds. Equality:
        lb == ub.'''
        self.rows.append((dict(coeffs), lb, ub))
        self._A = None

    def set_objective(self, idx, coeffs):
        'Overwrite objective coefficients for columns idx (array-like).'
        obj = np.asarray(self.obj, dtype=float)
        obj[np.asarray(idx).reshape(-1)] = np.asarray(coeffs).reshape(-1)
        self.obj = obj.tolist()

    def _matrix(self):
        if self._A is None:
            data, ri, ci = [], [], []
            for r, (coeffs, _, _) in enumerate(self.rows):
                for c, v in coeffs.items():
                    ri.append(r)
                    ci.append(int(c))
                    data.append(float(v))
            self._A = sparse.csr_matrix(
                (data, (ri, ci)), shape=(len(self.rows), self.ncols))
        return self._A

    def solve(self, backend=None, warm_start=None):
        '''Solve, return (x, obj_val, wall_time). warm_start is only used by
        the gurobi backend (HiGHS via scipy has no warm-start interface).'''
        backend = backend or default_backend()
        if backend == 'highs':
            return self._solve_highs()
        if backend == 'gurobi':
            return self._solve_gurobi(warm_start)
        raise ValueError('Unknown MILP backend "{}"'.format(backend))

    def _solve_highs(self):
        from scipy.optimize import milp, LinearConstraint, Bounds
        A = self._matrix()
        row_lb = np.array([r[1] for r in self.rows])
        row_ub = np.array([r[2] for r in self.rows])
        t0 = time.time()
        res = milp(
            c=np.asarray(self.obj),
            constraints=LinearConstraint(A, row_lb, row_ub),
            bounds=Bounds(np.asarray(self.lb), np.asarray(self.ub)),
            integrality=np.asarray(self.binary, dtype=int),
        )
        wall = time.time() - t0
        assert res.success, 'MILP "{}" failed: {}'.format(
            self.name, res.message)
        return res.x, res.fun, wall

    def _solve_gurobi(self, warm_start=None):
        import gurobipy as grb
        m = grb.Model(self.name)
        m.setParam('LogToConsole', 0)
        m.setParam('Threads', 1)
        n = self.ncols
        vtypes = [grb.GRB.BINARY if b else grb.GRB.CONTINUOUS
                  for b in self.binary]
        x = m.addVars(n, lb=self.lb, ub=self.ub, obj=self.obj)
        for i in range(n):
            x[i].VType = vtypes[i]
            if warm_start is not None and self.binary[i]:
                x[i].Start = warm_start[i]
        for coeffs, lb, ub in self.rows:
            expr = grb.quicksum(v * x[c] for c, v in coeffs.items())
            if lb == ub:
                m.addConstr(expr == lb)
            else:
                if np.isfinite(ub):
                    m.addConstr(expr <= ub)
                if np.isfinite(lb):
                    m.addConstr(expr >= lb)
        m.optimize()
        assert m.status == grb.GRB.Status.OPTIMAL, \
            'MILP "{}" failed (gurobi status {})'.format(self.name, m.status)
        sol = np.array([x[i].X for i in range(n)])
        return sol, m.ObjVal, m.Runtime
