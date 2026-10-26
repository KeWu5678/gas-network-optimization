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

from gasnetopt.gas import data, plotting


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('netfile', help='GasLib .net file')
    ap.add_argument('-o', '--out', type=Path,
                    default=Path('results/topology.png'))
    args = ap.parse_args()
    plotting.plot_topology(data.parse_net(args.netfile), args.out)
    print('saved {}'.format(args.out))


if __name__ == '__main__':
    main()
