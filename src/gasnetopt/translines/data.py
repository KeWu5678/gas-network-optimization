'''
Network data for the transmission lines example.
'''

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

Edges = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TranslinesNetwork:
    name: str
    vertices: tuple[int, ...]
    edges: Edges
    producers: tuple[int, ...]
    control_bounds: tuple[float, ...]             # per producer
    consumers: tuple[int, ...]
    configurations: tuple[tuple[int, ...], ...]   # switched-off edges
    switches: tuple[tuple[int, ...], ...]         # edges of each switch

    def switch_encoding(self) -> np.ndarray:
        r = np.array([[int(set(group) <= set(off)) for group in self.switches]
                      for off in self.configurations])
        for off, states in zip(self.configurations, r):
            removed = {e for group, state in zip(self.switches, states)
                       if state for e in group}
            if removed != set(off):
                raise ValueError('configuration {} is not a union of switch '
                                 'groups {}'.format(off, self.switches))
        return r

    def in_edges(self) -> list[list[int]]:
        'Compute lists of incoming edges per edge'
        return [[k for k, (_, end) in enumerate(self.edges) if end == start]
                for start, _ in self.edges]

    def out_edges(self) -> list[list[int]]:
        'Compute lists of outgoing edges per edge'
        return [[k for k, (start, _) in enumerate(self.edges) if start == end]
                for _, end in self.edges]


EXTENDED_TREE = TranslinesNetwork(
    name='extended-tree',
    vertices=tuple(range(11)),
    edges=((0, 2), (1, 4), (2, 5), (2, 3), (2, 4), (4, 5), (3, 6), (3, 7),
           (4, 8), (4, 9), (5, 10)),
    producers=(0, 1),
    control_bounds=(120., 80.),
    consumers=(6, 7, 10, 8, 9),
    configurations=((), (4,), (5,), (4, 5)),
    switches=((5,), (4,)))

SUBGRID = TranslinesNetwork(
    name='subgrid',
    vertices=tuple(range(14)),
    edges=((0, 2), (1, 3), (3, 5), (2, 4), (2, 8), (4, 6), (5, 6), (6, 7),
           (8, 7), (2, 9), (8, 10), (3, 11), (7, 12), (7, 13)),
    producers=(0, 1),
    control_bounds=(120., 80.),
    consumers=(9, 10, 11, 12, 13),
    configurations=((), (2, 3, 7), (8,), (2, 3, 7, 8)),
    switches=((8,), (2, 3, 7)))

NETWORKS = {net.name: net for net in (EXTENDED_TREE, SUBGRID)}


def load_demand(path: Path) -> np.ndarray:
    'Load a demand file (rows = consumers, columns = time steps).'
    if not path.exists():
        raise FileNotFoundError(
            'Demand file {} not found. Only the extended-tree demand data '
            'ships with the repository; for other networks provide the file '
            'yourself (rows = consumers, columns = time steps).'.format(path))
    return np.loadtxt(path)
