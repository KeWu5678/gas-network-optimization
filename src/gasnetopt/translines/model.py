'''
Discretized NLP for the transmission lines example with partial outer
convexification (POC).

[1] Göttlich, Potschka, Teuber: A partial outer convexification approach to
    control transmission lines. Comput. Optim. Appl. (2019).

Originally Copyright 2018 Andreas Potschka, Claus Teuber (GPL-3.0-or-later).
'''

from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path
from typing import Callable

import casadi as cas
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import dok_matrix

from .. import DATA_DIR, collocation
from . import networks
from .networks import in_edges, out_edges


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


def translines_nlp(net: Callable[[], networks.NetworkData],
                   demand: np.ndarray, T: float = 15., lr: float = 1.,
                   L: float = 1., C: float = 1., R: float = 1e-3,
                   G: float = 2e-3, nx: int = 10,
                   nt: int = 151) -> tuple:
    'Compose NLP from network and other data'
    dx = lr / nx
    dt = T / (nt - 1)

    assert dt <= sqrt(L * C) * dx, 'Violation of CFL condition'

    b11 = 0.5 * (R / L + G / C)
    b21 = 0.5 * (R / L - G / C)
    b12 = b21
    b22 = b11

    lambda_p = 1. / sqrt(L * C)
    lambda_m = -lambda_p

    # get network data
    V, A, producers, ctrl_bounds, consumers, configs = net()
    n_edges = len(A)
    n_ctrls = len(producers)
    n_confg = len(configs)

    delta_in = in_edges(A)
    delta_out = out_edges(A)
    D_p, D_m = [], []
    for off in configs:
        D_p += [equal_distribution_matrix(delta_out, off)]
        D_m += [equal_distribution_matrix(delta_in, off)]

    # start with empty NLP
    w = []
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g = []
    lbg = []
    ubg = []

    # convex configuration multipliers (partial outer convexification)
    alpha = cas.MX.sym('alpha', nt - 1, n_confg)
    w += [cas.reshape(alpha, -1, 1)]
    w0 += [.5] * (nt - 1) * n_confg
    lbw += [0.] * (nt - 1) * n_confg
    ubw += [2.] * (nt - 1) * n_confg

    # SOS1 constraint for alpha
    g += [cas.mtimes(alpha, cas.DM.ones(n_confg))]
    lbg += [1.] * (nt - 1)
    ubg += [1.] * (nt - 1)

    # inflow control
    u = cas.MX.sym('u', nt - 1, n_ctrls)
    w += [cas.reshape(u, -1, 1)]
    for i in range(n_ctrls):
        w0 += [0.] * (nt - 1)
        lbw += [0.] * (nt - 1)
        ubw += [ctrl_bounds[i]] * (nt - 1)

    # characteristic variables
    xi_p, xi_m = [], []
    for line in range(n_edges):
        xi_p += [cas.MX.sym('xi_p_{}_{}'.format(line, A[line]), nx, nt)]
        xi_m += [cas.MX.sym('xi_m_{}_{}'.format(line, A[line]), nx, nt)]
        w += [cas.reshape(xi_p[line], -1, 1), cas.reshape(xi_m[line], -1, 1)]
        w0 += [0.] * (2 * nx * nt)
        lbw += ([0.] + [-cas.inf] * (nt - 1)) * nx * 2
        ubw += ([0.] + [+cas.inf] * (nt - 1)) * nx * 2

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
                            for conf in range(n_confg))
            theta = cas.vertcat(theta, xi_p[line][:-1, t])
            adv = -lambda_p * (xi_p[line][:, t] - theta) / dx
            src = -b11 * xi_p[line][:, t] - b12 * xi_m[line][:, t]
            g += [xi_p[line][:, t + 1] - xi_p[line][:, t] - dt * (adv + src)]
            lbg += [0.] * nx
            ubg += [0.] * nx
            # xi_m variables
            if end_vertex in consumers:
                theta = 0
            else:
                theta = sum(alpha[t, conf] * D_m[conf][line, j]
                            * xi_m[j][0, t]
                            for j in delta_out[line]
                            for conf in range(n_confg))
            theta = cas.vertcat(xi_m[line][1:, t], theta)
            adv = lambda_m * (xi_m[line][:, t] - theta) / dx
            src = -b21 * xi_p[line][:, t] - b22 * xi_m[line][:, t]
            g += [xi_m[line][:, t + 1] - xi_m[line][:, t] - dt * (adv + src)]
            lbg += [0.] * nx
            ubg += [0.] * nx

    # objective
    end_vertices = [j for (i, j) in A]
    for i, consumer in enumerate(consumers):
        lines = [line for line, end_node in enumerate(end_vertices)
                 if end_node == consumer]
        # box rule (consistent with [1] up to a scaling with dt)
        for t in range(nt - 1):
            J = J + 0.5 * dt * (sum(xi_p[line][-1, t] for line in lines)
                                - demand[i, t]) ** 2

    nlp = {}
    nlp['f'] = J
    nlp['x'] = cas.vertcat(*w)
    nlp['g'] = cas.vertcat(*g)

    return nlp, lbw, ubw, lbg, ubg, w0


def extract_solution(sol: dict, net: Callable[[], networks.NetworkData],
                     nt: int, nx: int) -> tuple:
    'Extract NumPy arrays from CasADi NLP solution'
    _, A, producers, _, _, configs = net()
    n_ctrls = len(producers)
    n_confg = len(configs)
    offset = 0
    alpha = np.array(cas.reshape(sol['x'][offset:offset + (nt-1)*n_confg],
                                 nt - 1, n_confg))
    offset += (nt - 1) * n_confg
    u = np.array(cas.reshape(sol['x'][offset:offset + (nt-1)*n_ctrls],
                             nt - 1, n_ctrls))
    offset += (nt - 1) * n_ctrls
    xi_p, xi_m = [], []
    for line in range(len(A)):
        xi_p += [np.array(cas.reshape(sol['x'][offset:offset + nt*nx],
                                      nx, nt))]
        offset += nt * nx
        xi_m += [np.array(cas.reshape(sol['x'][offset:offset + nt*nx],
                                      nx, nt))]
        offset += nt * nx
    return u, alpha, xi_p, xi_m


def load_demand(name: str,
                data_dir: str | Path | None = None) -> np.ndarray:
    'Load a demand file (e.g. "demand_extended_tree_coarse.dat").'
    data_dir = DATA_DIR / 'translines' if data_dir is None else Path(data_dir)
    path = data_dir / name
    if not path.exists():
        raise FileNotFoundError(
            'Demand file {} not found. Only the extended-tree demand data '
            'ships with the repository; for other networks provide the file '
            'yourself (rows = consumers, columns = time steps).'.format(path))
    return np.loadtxt(path)


def plot_solution(net: Callable[[], networks.NetworkData],
                  u: np.ndarray, alpha: np.ndarray, xi_p: list,
                  xi_m: list, demand: np.ndarray, T: float,
                  nt: int) -> None:
    'Plot extracted solution'
    _, A, _, _, consumers, _ = net()
    n_ctrls = u.shape[1]
    n_confg = alpha.shape[1]
    n_consumer = demand.shape[0]
    t = np.linspace(0, T, nt)

    fig, axs = plt.subplots(n_ctrls, 1, num=1, clear=True,
                            figsize=(10, 2.2 * n_ctrls), squeeze=False)
    axes = axs.ravel()
    for i in range(n_ctrls):
        axes[i].step(t, np.concatenate(([np.nan], u[:, i])))
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'control $u_{}$'.format(i))
    fig.tight_layout()

    fig, axs = plt.subplots(n_confg + 1, 1, num=2, clear=True,
                            figsize=(10, 2.5 * (n_confg + 1)), squeeze=False)
    axes = axs.ravel()
    for i in range(n_confg):
        axes[i].step(t, np.concatenate(([np.nan], alpha[:, i])), linewidth=3)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'multiplier $\alpha_{}$'.format(i))
    axes[-1].step(t, np.concatenate(([np.nan],
                                     alpha.dot(np.arange(n_confg)))),
                  'r', linewidth=3)
    axes[-1].set_xlabel(r'time $t$')
    axes[-1].set_ylabel(r'configuration')
    axes[-1].set_ylim(-0.02, n_confg - 0.98)
    fig.tight_layout()

    n_edges = len(xi_p)
    n = int(ceil(sqrt(n_edges)))
    m = n - 1 if n * (n - 1) >= n_edges else n

    fig, axs = plt.subplots(m, n, num=3, clear=True,
                            figsize=(3.2 * n, 2.8 * m), squeeze=False)
    axes = axs.ravel()
    for i in range(n_edges):
        axes[i].pcolormesh(xi_p[i], cmap='jet')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'space $x$')
        axes[i].set_title(r'$\xi_+$ on edge {}'.format(i))
    for i in range(n_edges, len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()

    fig, axs = plt.subplots(m, n, num=4, clear=True,
                            figsize=(3.2 * n, 2.8 * m), squeeze=False)
    axes = axs.ravel()
    for i in range(n_edges):
        axes[i].pcolormesh(xi_m[i], cmap='jet')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'space $x$')
        axes[i].set_title(r'$\xi_-$ on edge {}'.format(i))
    for i in range(n_edges, len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()

    fig, axs = plt.subplots(n_consumer, 1, num=5, clear=True,
                            figsize=(10, 2.2 * n_consumer), squeeze=False)
    axes = axs.ravel()
    end_vertices = [j for (i, j) in A]
    for i, consumer in enumerate(consumers):
        line = end_vertices.index(consumer)
        axes[i].plot(t[:-1], demand[i, :], 'r-', label='demand')
        axes[i].plot(t, xi_p[line][-1, :], 'b-', label='delivery')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'vertex {}'.format(end_vertices[line]))
        if i == 0:
            axes[i].legend(fontsize=8)
    fig.tight_layout()


class TranslinesOCModel(collocation.OCModel):
    '''Optimal control model interface for the transmission lines example,
    augmented with the outer-convexified L1 penalty term for the penalty ADM
    (parametric in rho and the combinatorial reference V_ref).'''

    def __init__(self, network_name: str, coarse: bool = True,
                 data_dir: str | Path | None = None) -> None:
        if network_name == 'extended tree':
            self.net = networks.network_extended_tree
        elif network_name == 'subgrid':
            self.net = networks.network_subgrid
        else:
            raise Exception('Unknown network name "{}"'.format(network_name))
        T = 26.
        suffix = '_coarse' if coarse else ''
        fname = 'demand_{}{}.dat'.format('_'.join(network_name.split()),
                                         suffix)
        self.demand = load_demand(fname, data_dir)
        if coarse:  # dt = dx = 0.5
            nt = 2 * 26 + 1
            nx = 2
        else:  # dt = dx = 0.25
            nt = 4 * 26 + 1
            nx = 4
        self.T = T
        self.t = np.linspace(0, T, nt)
        self.nt = nt
        self.nx = nx
        d = translines_nlp(self.net, self.demand, T, nt=nt, nx=nx)
        self.nlp, self.lbw, self.ubw, self.lbg, self.ubg, w0 = d
        self.w0 = np.array(w0)
        _, self.A, self.producers, _, _, self.configs = self.net()
        self.n_ctrls = len(self.producers)
        self.n_confg = len(self.configs)
        self.nv_ref = 2
        self.nalpha = 4
        # mode encodings: config index -> binary states of the two switches
        self.r = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        assert self.n_confg == self.nalpha, 'Implement case nalpha != 4'

        # augment problem with outer-convexified L1 penalization term
        self.add_l1_penalty(obj_sca=1e-3)

    def extract(self, w: np.ndarray) -> tuple:
        'Extract and return (y, u, alpha, v, nodes) from NLP variable w.'
        offset = 0
        alpha = np.reshape(w[offset:offset + (self.nt - 1) * self.n_confg],
                           (self.nt - 1, self.n_confg), order='F')
        offset += (self.nt - 1) * self.n_confg
        u = np.reshape(w[offset:offset + (self.nt - 1) * self.n_ctrls],
                       (self.nt - 1, self.n_ctrls), order='F')
        offset += (self.nt - 1) * self.n_ctrls
        y = w[offset:offset + 2 * len(self.A) * self.nt * self.nx]
        return y, u, alpha, None, None

    def overwrite(self, w: np.ndarray, y=None, u=None, alpha=None,
                  v=None, nodes=None) -> np.ndarray:
        '''Overwrite given components (y, u, alpha, v, nodes) in NLP variable
        w. Returns overwritten w.'''
        offset = 0
        if v is not None or nodes is not None:
            raise Exception('Cannot overwrite unknown variables')
        if alpha is not None:
            w[offset:offset + (self.nt - 1) * self.n_confg] = \
                alpha.flatten(order='F')
        offset += (self.nt - 1) * self.n_confg
        if u is not None:
            w[offset:offset + (self.nt - 1) * self.n_ctrls] = \
                u.flatten(order='F')
        offset += (self.nt - 1) * self.n_ctrls
        if y is not None:
            w[offset:offset + 2 * len(self.A) * self.nt * self.nx] = y
        return w
