from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from .data import GasNetwork

# CVD-safe categorical triple (blue / orange / purple), reused per group
GROUP_COLORS = ['#1f77b4', '#ff7f0e', '#9467bd']


def plot_solution(s: dict, title: str, path: Path,
                  objective: float | None = None) -> None:
    '''One figure per method: switch schedules, delivery mismatch, pressures.

    The delivery panel shows delivered - demand around a zero baseline (the
    quantity the objective penalizes) instead of two overlapping curves.'''
    n_sw = len(s['switch_ids'])
    fig = Figure(figsize=(9, 2.0 * (n_sw + 2)))
    axes = fig.subplots(n_sw + 2, 1, sharex=True)
    t = s['t'] / 3600.
    tc = t[:-1]

    # switch schedules (relaxed in [0, 1], binary after rounding/ADM)
    for j, sid in enumerate(s['switch_ids']):
        ax = axes[j]
        state = s['switch_state'][:, j]
        ax.step(tc, state, where='post', color=GROUP_COLORS[0], linewidth=2)
        ax.fill_between(tc, state, step='post', color=GROUP_COLORS[0],
                        alpha=0.12)
        ax.set_ylabel(sid.split('_')[0])
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0., 1.])

    # delivery mismatch per sink
    ax = axes[n_sw]
    err = 0.
    for i, sid in enumerate(s['sink_ids']):
        mis = s['delivered_kg_s'][sid] - s['demand_kg_s'][sid]
        err = max(err, float(np.abs(mis).max()))
        ax.plot(tc, mis, color=GROUP_COLORS[i % len(GROUP_COLORS)],
                linewidth=2, label=sid)
    ax.axhline(0., color='0.6', linewidth=1)
    ax.set_ylabel('delivered - demand\n[kg/s]')
    ax.legend(fontsize=8, ncol=len(s['sink_ids']))

    # node pressures: entries solid, exits dashed (color = member within
    # group), inner nodes muted gray
    ax = axes[-1]
    for nid, p in s['node_pressure_bar'].items():
        if nid in s['source_ids']:
            i = s['source_ids'].index(nid)
            ax.plot(tc, p, color=GROUP_COLORS[i % len(GROUP_COLORS)],
                    linewidth=1.8, label=nid)
        elif nid in s['sink_ids']:
            i = s['sink_ids'].index(nid)
            ax.plot(tc, p, '--', color=GROUP_COLORS[i % len(GROUP_COLORS)],
                    linewidth=1.8, label=nid)
        else:
            ax.plot(tc, p, color='0.75', linewidth=1)
    ax.plot([], [], color='0.75', linewidth=1, label='inner nodes')
    ax.set_ylabel('p [bar]')
    ax.set_xlabel('time [h]')
    ax.legend(fontsize=7, ncol=4)

    if objective is not None:
        title = '{}   (objective {:.2e}, max delivery error {:.3g} kg/s)' \
            .format(title, objective, err)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


NODE_STYLE = {
    # node type -> (color, marker, size)
    'source': ('#1f77b4', '^', 140),
    'sink': ('#ff7f0e', 'v', 140),
    'innode': ('#6a6a6a', 'o', 60),
}


def plot_topology(net: GasNetwork, path: Path) -> None:
    pos = {n.id: (n.x, n.y) for n in net.nodes.values()}

    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()

    def draw_edge(e, **kwargs):
        (x0, y0), (x1, y1) = pos[e.from_node], pos[e.to_node]
        ax.plot([x0, x1], [y0, y1], **kwargs)
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    for pipe in net.pipes:
        draw_edge(pipe, color='0.55', linewidth=2, zorder=1)
    for valve in net.valves:
        xm, ym = draw_edge(valve, color='#9467bd', linewidth=3, zorder=2)
        ax.annotate('valve\n' + valve.id.split('_')[0], (xm, ym),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color='#9467bd')
    for comp in net.compressors:
        xm, ym = draw_edge(comp, color='#d62728', linewidth=3, zorder=2)
        ax.annotate('compressor\n' + comp.id.split('_')[0], (xm, ym),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color='#d62728')

    for ntype, (color, marker, size) in NODE_STYLE.items():
        nodes = [n for n in net.nodes.values() if n.type == ntype]
        ax.scatter([n.x for n in nodes], [n.y for n in nodes],
                   c=color, marker=marker, s=size, zorder=3,
                   label='{} ({})'.format(ntype, len(nodes)))
        for n in nodes:
            ax.annotate(n.id, (n.x, n.y), textcoords='offset points',
                        xytext=(6, -12), fontsize=7, color='0.3')

    ax.set_title('{}: {} nodes, {} pipes, {} switchable elements'.format(
        net.name, len(net.nodes), len(net.pipes), len(net.switches)))
    ax.legend(loc='best', fontsize=8)
    ax.set_axis_off()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
