'''
Discretized NLP for the transmission lines example with partial outer
convexification (POC).

[1] Göttlich, Potschka, Teuber: A partial outer convexification approach to
    control transmission lines. Comput. Optim. Appl. (2019).

Originally Copyright 2018 Andreas Potschka, Claus Teuber (GPL-3.0-or-later).
'''

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import casadi as cas
import numpy as np
from scipy.sparse import dok_matrix

from ..core.nlp import NlpBuilder, NlpProblem
from ..core.ocmodel import OCModel
from .data import TranslinesNetwork

HORIZON = 26.
# (nt, nx) with dt = dx = 0.5 (coarse) or 0.25 (fine)
GRIDS = {'coarse': (2 * 26 + 1, 2), 'fine': (4 * 26 + 1, 4)}


@dataclass(frozen=True)
class TranslinesModelSettings:
    grid: str = 'coarse'

    def __post_init__(self) -> None:
        if self.grid not in GRIDS:
            raise ValueError('grid "{}" is not one of {}'.format(
                self.grid, ', '.join(GRIDS)))


class TranslinesOCModel(OCModel):
    '''Optimal control model interface for the transmission lines example,
    augmented with the outer-convexified L1 penalty term for the penalty ADM
    (parametric in rho and the combinatorial reference V_ref).'''

    def __init__(self, network: TranslinesNetwork, demand: np.ndarray,
                 settings: TranslinesModelSettings = TranslinesModelSettings()
                 ) -> None:
        nt, nx = GRIDS[settings.grid]
        if demand.shape != (len(network.consumers), nt - 1):
            raise ValueError(
                'demand has shape {}, the {} grid needs {} (consumers x '
                'time steps)'.format(demand.shape, settings.grid,
                                     (len(network.consumers), nt - 1)))
        self._network = network
        self._demand = demand
        self._nx = nx
        problem = translines_nlp(network, demand, HORIZON, nt=nt, nx=nx)
        super().__init__(problem, t=np.linspace(0, HORIZON, nt),
                         r=network.switch_encoding(),
                         n_controls=len(network.producers),
                         objective_scale=1e-3)

    @property
    def network(self) -> TranslinesNetwork:
        return self._network

    @property
    def demand(self) -> np.ndarray:
        return self._demand

    def solution_dict(self, w: np.ndarray) -> dict:
        alpha, u, y = self.extract(np.asarray(w).flatten())
        nt, nx = len(self.t), self._nx
        # y holds xi_p, xi_m of every line in turn, each (nx, nt) column-major
        xi = [np.reshape(y[k * nx * nt:(k + 1) * nx * nt], (nx, nt),
                         order='F')
              for k in range(2 * len(self._network.edges))]
        return {
            't': self.t,
            'alpha': alpha,
            'switch_state': alpha.dot(self.r),
            'u': u,
            'xi_p': xi[0::2],
            'xi_m': xi[1::2],
            'demand': self._demand,
        }


def translines_nlp(network: TranslinesNetwork, demand: np.ndarray,
                   T: float, nt: int, nx: int, lr: float = 1., L: float = 1.,
                   C: float = 1., R: float = 1e-3,
                   G: float = 2e-3) -> NlpProblem:
    'Compose NLP from network and other data'
    dx = lr / nx
    dt = T / (nt - 1)

    if dt > sqrt(L * C) * dx:
        raise ValueError('Violation of CFL condition')

    b11 = 0.5 * (R / L + G / C)
    b21 = 0.5 * (R / L - G / C)
    b12 = b21
    b22 = b11

    lambda_p = 1. / sqrt(L * C)
    lambda_m = -lambda_p

    A = network.edges
    producers = network.producers
    consumers = network.consumers
    n_edges = len(A)
    n_controls = len(producers)
    n_configs = len(network.configurations)

    delta_in = network.in_edges()
    delta_out = network.out_edges()
    D_p, D_m = [], []
    for off in network.configurations:
        D_p += [equal_distribution_matrix(delta_out, off)]
        D_m += [equal_distribution_matrix(delta_in, off)]

    b = NlpBuilder()

    # convex configuration multipliers (partial outer convexification)
    alpha = b.add_variable(cas.MX.sym('alpha', nt - 1, n_configs), .5, 0., 2.)
    # SOS1 constraint for alpha
    b.add_constraint(cas.mtimes(alpha, cas.DM.ones(n_configs)), 1., 1.)

    # inflow control
    u = b.add_variable(cas.MX.sym('u', nt - 1, n_controls), 0., 0.,
                       np.tile(network.control_bounds, (nt - 1, 1)))

    # characteristic variables. Initial condition xi = 0: this bound pattern
    # of the original implementation fixes the entries k % nt == 0 of the
    # column-major (nx, nt) vector, which is the t = 0 column only for
    # nx = 1 (for nx > 1, some initial cells stay free and some later
    # entries are fixed to 0). Kept unchanged so that results stay
    # reproducible.
    lb_initial = np.array(([0.] + [-cas.inf] * (nt - 1)) * nx)
    ub_initial = np.array(([0.] + [+cas.inf] * (nt - 1)) * nx)
    xi_p, xi_m = [], []
    for line in range(n_edges):
        xi_p += [b.add_variable(
            cas.MX.sym('xi_p_{}_{}'.format(line, A[line]), nx, nt),
            0., lb_initial, ub_initial)]
        xi_m += [b.add_variable(
            cas.MX.sym('xi_m_{}_{}'.format(line, A[line]), nx, nt),
            0., lb_initial, ub_initial)]

    # dynamics on network edges
    for line in range(n_edges):
        start_vertex, end_vertex = A[line]

        # dynamic equations (upwind discretization)
        for t in range(nt - 1):
            # xi_p variables
            if start_vertex in producers:
                idx = producers.index(start_vertex)
                theta = u[t, idx]
            else:
                theta = sum(alpha[t, conf] * D_p[conf][line, j]
                            * xi_p[j][-1, t]
                            for j in delta_in[line]
                            for conf in range(n_configs))
            theta = cas.vertcat(theta, xi_p[line][:-1, t])
            adv = -lambda_p * (xi_p[line][:, t] - theta) / dx
            src = -b11 * xi_p[line][:, t] - b12 * xi_m[line][:, t]
            b.add_equality(xi_p[line][:, t + 1] - xi_p[line][:, t]
                           - dt * (adv + src))
            # xi_m variables
            if end_vertex in consumers:
                theta = 0
            else:
                theta = sum(alpha[t, conf] * D_m[conf][line, j]
                            * xi_m[j][0, t]
                            for j in delta_out[line]
                            for conf in range(n_configs))
            theta = cas.vertcat(xi_m[line][1:, t], theta)
            adv = lambda_m * (xi_m[line][:, t] - theta) / dx
            src = -b21 * xi_p[line][:, t] - b22 * xi_m[line][:, t]
            b.add_equality(xi_m[line][:, t + 1] - xi_m[line][:, t]
                           - dt * (adv + src))

    # objective
    J = 0
    end_vertices = [j for (i, j) in A]
    for i, consumer in enumerate(consumers):
        lines = [line for line, end_node in enumerate(end_vertices)
                 if end_node == consumer]
        # box rule (consistent with [1] up to a scaling with dt)
        for t in range(nt - 1):
            J = J + 0.5 * dt * (sum(xi_p[line][-1, t] for line in lines)
                                - demand[i, t]) ** 2

    return b.build(J)


def equal_distribution_matrix(delta: list[list[int]],
                              off: tuple[int, ...]) -> dok_matrix:
    '''Create equal distribution matrix from list of incoming/outgoing edges
    delta and configuration off'''
    n_edges = len(delta)
    D = dok_matrix((n_edges, n_edges))
    for i in range(n_edges):
        incident = [j for j in delta[i] if j not in off]
        n_incident = len(incident)
        entry = 1. / n_incident if n_incident > 0 else 0
        for j in incident:
            D[j, i] = entry
    return D
