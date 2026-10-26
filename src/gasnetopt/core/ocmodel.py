from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import casadi as cas
import numpy as np

from .nlp import NlpProblem, NlpSolution, NlpSolver


class OCModel:

    def __init__(self, problem: NlpProblem, t: np.ndarray, r: np.ndarray,
                 n_controls: int, objective_scale: float = 1.,
                 time_unit: float = 1.) -> None:
        # t: model time grid, r: configuration encoding (n_configs,
        # n_switches), time_unit: application time per model time unit
        self._t = t
        self._r = r
        self._n_controls = n_controls
        self._time_unit = time_unit
        rho = cas.MX.sym('rho')
        V_ref = cas.MX.sym('V_ref', self.n_switches, len(t) - 1)
        p = cas.vertcat(rho, cas.vec(V_ref))
        alpha = cas.reshape(problem.x[self._blocks()[0]], len(t) - 1,
                            self.n_configs)
        f = objective_scale * problem.f \
            + rho * l1_coupling_penalty(alpha, V_ref, r, t)
        self._problem = replace(problem, f=f, p=p)
        self._objective = cas.Function('objective', [problem.x, p], [f])
        self._constraints = cas.Function('g', [problem.x, p], [problem.g])

    @property
    def t(self) -> np.ndarray:
        return self._t

    @property
    def r(self) -> np.ndarray:
        return self._r

    @property
    def n_configs(self) -> int:
        return self.r.shape[0]

    @property
    def n_switches(self) -> int:
        return self.r.shape[1]

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def n_variables(self) -> int:
        return self._problem.x.numel()

    @property
    def n_constraints(self) -> int:
        return self._problem.g.numel()

    @property
    def w0(self) -> np.ndarray:
        return self._problem.x0.copy()

    def to_model_time(self, duration: float) -> float:
        return duration / self._time_unit

    def create_solver(self, tol: float,
                      max_wall_time: float | None = None) -> NlpSolver:
        return NlpSolver(self._problem, tol=tol, max_wall_time=max_wall_time)

    def parameters(self, rho: float = 0.,
                   v_ref: np.ndarray | None = None) -> np.ndarray:
        # p = (rho, vec(V_ref)) with V_ref = v_ref^T: the column-major vec of
        # V_ref is the row-major flattening of v_ref. The reference is
        # immaterial at rho = 0.
        if v_ref is None:
            v_ref = np.zeros((len(self.t) - 1, self.n_switches))
        return np.concatenate(([rho], v_ref.flatten()))

    def solve(self, solver: NlpSolver, w0: np.ndarray, p: np.ndarray,
              u_fix: np.ndarray | None = None,
              alpha_fix: np.ndarray | None = None) -> NlpSolution:
        lbx, ubx = self.bounds(u_fix=u_fix, alpha_fix=alpha_fix)
        return solver.solve(w0, lbx, ubx, p=p)

    def bounds(self, u_fix: np.ndarray | None = None,
               alpha_fix: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
        alpha_lb = alpha_ub = None
        if alpha_fix is not None:
            # leave the last configuration multiplier free: the SOS1
            # constraint determines it, and fixing it too would make the
            # active constraints linearly dependent
            alpha_lb = np.array(alpha_fix, dtype=float)
            alpha_lb[:, -1] = -1.
            alpha_ub = np.array(alpha_fix, dtype=float)
            alpha_ub[:, -1] = 2.
        lbx = self.with_variables(self._problem.lbx, alpha=alpha_lb, u=u_fix)
        ubx = self.with_variables(self._problem.ubx, alpha=alpha_ub, u=u_fix)
        return lbx, ubx

    def with_variables(self, w: np.ndarray, alpha: np.ndarray | None = None,
                       u: np.ndarray | None = None,
                       y: np.ndarray | None = None) -> np.ndarray:
        w = np.array(w, dtype=float)
        alpha_block, u_block, y_block = self._blocks()
        if alpha is not None:
            w[alpha_block] = np.asarray(alpha).flatten(order='F')
        if u is not None:
            w[u_block] = np.asarray(u).flatten(order='F')
        if y is not None:
            w[y_block] = y
        return w

    def extract(self, w: np.ndarray) -> Variables:
        alpha_block, u_block, y_block = self._blocks()
        n_steps = len(self.t) - 1
        return Variables(
            alpha=np.reshape(w[alpha_block], (n_steps, self.n_configs),
                             order='F'),
            u=np.reshape(w[u_block], (n_steps, self.n_controls), order='F'),
            y=w[y_block])

    def evaluate_objective(self, w: np.ndarray, rho: float = 0.,
                           v_ref: np.ndarray | None = None) -> float:
        return float(self._objective(w, self.parameters(rho, v_ref)))

    def constraint_violation(self, w: np.ndarray, p: np.ndarray) -> float:
        g = np.array(self._constraints(w, p)).flatten()
        if not np.all(np.isfinite(g)):
            return np.inf
        return float(np.maximum(self._problem.lbg - g,
                                g - self._problem.ubg).max())

    def _blocks(self) -> tuple[slice, slice, slice]:
        # w = [vec(alpha) | vec(u) | y], alpha (nt-1, n_configs) and
        # u (nt-1, n_controls) column-major, y all remaining variables
        n_alpha = (len(self.t) - 1) * self.n_configs
        n_u = (len(self.t) - 1) * self.n_controls
        return (slice(0, n_alpha), slice(n_alpha, n_alpha + n_u),
                slice(n_alpha + n_u, None))


def l1_coupling_penalty(alpha: cas.MX, V_ref: cas.MX, r: np.ndarray,
                        t: np.ndarray) -> cas.MX:
    # sum_k h_k sum_i alpha_{k,i} sum_j |r_{i,j} - V_ref_{j,k}| with explicit
    # cases for the signs in the absolute value (r binary, V_ref in [0, 1])
    if not np.isin(r, (0, 1)).all():
        raise ValueError('configuration encoding r is not binary')
    penalty = 0
    for k, h in enumerate(np.diff(t)):
        integrand = 0
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if r[i, j] == 0:  # -> |x| = +x >= 0
                    integrand += V_ref[j, k] * alpha[k, i]
                else:  # -> |x| = -x >= 0
                    integrand += (1 - V_ref[j, k]) * alpha[k, i]
        penalty = penalty + h * integrand
    return penalty


class Variables(NamedTuple):
    alpha: np.ndarray
    u: np.ndarray
    y: np.ndarray
