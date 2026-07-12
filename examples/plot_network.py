'''
Draw a GasLib network as a graph: pipes as edges, switchable elements
(valves, compressor stations) highlighted, sources/sinks/inner nodes as
distinct markers at the coordinates from the .net file.

Usage:
    uv run python examples/plot_network.py \
        data/gaslib/GasLib-11/GasLib-11-v1-20211130.net -o assets/topo.png
'''

import argparse
from pathlib import Path

import matplotlib

from gasnetopt.gas import gaslib_io

NODE_STYLE = {
    # node type -> (color, marker, size)
    'source': ('#1f77b4', '^', 140),
    'sink': ('#ff7f0e', 'v', 140),
    'innode': ('#6a6a6a', 'o', 60),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('netfile', help='GasLib .net file')
    ap.add_argument('-o', '--out', default='results/topology.png')
    args = ap.parse_args()

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    net = gaslib_io.parse_net(args.netfile)
    pos = {n.id: (n.x, n.y) for n in net.nodes.values()}

    fig, ax = plt.subplots(figsize=(7, 5))

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

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print('saved {}'.format(out))


if __name__ == '__main__':
    main()
