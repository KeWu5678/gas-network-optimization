from __future__ import annotations

from math import ceil, sqrt
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from ..core.pipeline import METHOD_INFO
from .data import TranslinesNetwork


def plot_solution(network: TranslinesNetwork, s: dict, out_dir: Path,
                  prefix: str) -> list[Path]:
    'Plot extracted solution'
    figures = {
        'controls': _controls(s),
        'multipliers': _multipliers(s),
        'xi_plus': _line_states(s['xi_p'], r'$\xi_+$'),
        'xi_minus': _line_states(s['xi_m'], r'$\xi_-$'),
        'demand': _demand_and_delivery(network, s),
    }
    paths = []
    for name, fig in figures.items():
        paths.append(out_dir / '{}_{}.png'.format(prefix, name))
        fig.savefig(paths[-1], dpi=150)
    return paths


def plot_comparison(network: TranslinesNetwork, solutions: dict[str, dict],
                    out_dir: Path, fmt: str = 'pdf') -> list[Path]:
    figures = {}
    for ctrl in [0, 1]:
        figures['translines_discrete_control_{}'.format(ctrl + 1)] = \
            _compare_switch_states(solutions, ctrl)
    for ctrl in [0, 1]:
        figures['translines_continuous_control_{}'.format(ctrl + 1)] = \
            _compare_controls(solutions, ctrl)
    figures['translines_demand_and_delivery'] = _compare_deliveries(
        network, solutions)
    paths = []
    for name, fig in figures.items():
        paths.append(out_dir / '{}.{}'.format(name, fmt))
        fig.savefig(paths[-1])
    return paths


def _controls(s: dict) -> Figure:
    t, u = s['t'], s['u']
    n_controls = u.shape[1]
    fig = Figure(figsize=(10, 2.2 * n_controls))
    axes = fig.subplots(n_controls, 1, squeeze=False).ravel()
    for i in range(n_controls):
        axes[i].step(t, np.concatenate(([np.nan], u[:, i])))
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'control $u_{}$'.format(i))
    fig.tight_layout()
    return fig


def _multipliers(s: dict) -> Figure:
    t, alpha = s['t'], s['alpha']
    n_configs = alpha.shape[1]
    fig = Figure(figsize=(10, 2.5 * (n_configs + 1)))
    axes = fig.subplots(n_configs + 1, 1, squeeze=False).ravel()
    for i in range(n_configs):
        axes[i].step(t, np.concatenate(([np.nan], alpha[:, i])), linewidth=3)
        axes[i].set_ylim(-0.02, 1.02)
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'multiplier $\alpha_{}$'.format(i))
    axes[-1].step(t, np.concatenate(([np.nan],
                                     alpha.dot(np.arange(n_configs)))),
                  'r', linewidth=3)
    axes[-1].set_xlabel(r'time $t$')
    axes[-1].set_ylabel(r'configuration')
    axes[-1].set_ylim(-0.02, n_configs - 0.98)
    fig.tight_layout()
    return fig


def _line_states(xi: list, symbol: str) -> Figure:
    n_edges = len(xi)
    n = int(ceil(sqrt(n_edges)))
    m = n - 1 if n * (n - 1) >= n_edges else n
    fig = Figure(figsize=(3.2 * n, 2.8 * m))
    axes = fig.subplots(m, n, squeeze=False).ravel()
    for i in range(n_edges):
        axes[i].pcolormesh(xi[i], cmap='jet')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'space $x$')
        axes[i].set_title(r'{} on edge {}'.format(symbol, i))
    for i in range(n_edges, len(axes)):
        axes[i].set_visible(False)
    fig.tight_layout()
    return fig


def _demand_and_delivery(network: TranslinesNetwork, s: dict) -> Figure:
    t, demand = s['t'], s['demand']
    fig = Figure(figsize=(10, 2.2 * demand.shape[0]))
    axes = fig.subplots(demand.shape[0], 1, squeeze=False).ravel()
    end_vertices = [j for (i, j) in network.edges]
    for i, consumer in enumerate(network.consumers):
        line = end_vertices.index(consumer)
        axes[i].plot(t[:-1], demand[i, :], 'r-', label='demand')
        axes[i].plot(t, s['xi_p'][line][-1, :], 'b-', label='delivery')
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel(r'vertex {}'.format(end_vertices[line]))
        if i == 0:
            axes[i].legend(fontsize=8)
    fig.tight_layout()
    return fig


def _compare_switch_states(solutions: dict[str, dict], ctrl: int) -> Figure:
    fig = Figure()
    ax = fig.subplots()
    for i, (method, s) in enumerate(solutions.items()):
        t = s['t']
        baseline = 1.1 * (len(solutions) - 1 - i)
        state = _step(s['switch_state'][:, ctrl] + baseline)
        p = ax.step(t, state, where='post',
                    label=METHOD_INFO[method].abbreviation)
        ax.fill_between(t, state, y2=baseline, color=p[0].get_color(),
                        step='post', alpha=0.1)
    ax.legend(loc='lower right')
    ax.set_ylim(-0.02, 1.1 * len(solutions) + 0.32)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(r'time $t$')
    ax.yaxis.set_visible(False)
    return fig


def _compare_controls(solutions: dict[str, dict], ctrl: int) -> Figure:
    fig = Figure()
    ax = fig.subplots()
    for method, s in solutions.items():
        ax.step(s['t'], _step(s['u'][:, ctrl]),
                label=METHOD_INFO[method].abbreviation)
    ax.legend(loc='lower center')
    ax.set_xlim(0, s['t'][-1])
    ax.set_xlabel(r'time $t$')
    ax.set_ylabel(r'Continuous control $u_{}$'.format(ctrl + 1))
    return fig


def _compare_deliveries(network: TranslinesNetwork,
                        solutions: dict[str, dict]) -> Figure:
    n_consumer = len(network.consumers)
    fig = Figure(figsize=(8, 2.0 * n_consumer))
    axes = fig.subplots(n_consumer, 1)
    end_vertices = [j for (i, j) in network.edges]
    any_solution = next(iter(solutions.values()))
    t, demand = any_solution['t'], any_solution['demand']
    for i, consumer in enumerate(network.consumers):
        line = end_vertices.index(consumer)
        axes[i].plot(t[:-1], demand[i, :], 'k--', label='demand')
        for method, s in solutions.items():
            axes[i].plot(t, s['xi_p'][line][-1, :],
                         label=METHOD_INFO[method].abbreviation)
        if i == 0:
            axes[i].legend(loc='lower center', fontsize=7)
        axes[i].set_xlim(0, t[-1])
        axes[i].set_xlabel(r'time $t$')
        axes[i].set_ylabel('Consumer {}'.format(i + 1))
    fig.tight_layout()
    return fig


def _step(v: np.ndarray) -> np.ndarray:
    return np.concatenate((v.flatten(), [np.nan]))
