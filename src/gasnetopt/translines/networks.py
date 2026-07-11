'''
Network data for the transmission lines example.

A network is given as a tuple
    (V, A, producers, bounds, consumers, configs)
with vertices V, directed edges A, producer node indices with control upper
bounds, consumer node indices and switching configurations (each given as a
tuple of switched-off edge indices).
'''

from __future__ import annotations

Edges = tuple[tuple[int, int], ...]
NetworkData = tuple[tuple[int, ...], Edges, tuple[int, ...],
                    tuple[float, ...], tuple[int, ...],
                    tuple[tuple[int, ...], ...]]


def network_extended_tree() -> NetworkData:
    'Extended tree network data'
    V = tuple(range(11))
    A = ((0, 2), (1, 4), (2, 5), (2, 3), (2, 4), (4, 5), (3, 6), (3, 7),
         (4, 8), (4, 9), (5, 10))
    producers = (0, 1)
    bounds = (120., 80.)
    consumers = (6, 7, 10, 8, 9)
    configs = ((), (4,), (5,), (4, 5))
    return V, A, producers, bounds, consumers, configs


def network_subgrid() -> NetworkData:
    'Subgrid network data'
    V = tuple(range(14))
    A = ((0, 2), (1, 3), (3, 5), (2, 4), (2, 8), (4, 6), (5, 6), (6, 7),
         (8, 7), (2, 9), (8, 10), (3, 11), (7, 12), (7, 13))
    producers = (0, 1)
    bounds = (120., 80.)
    consumers = (9, 10, 11, 12, 13)
    configs = ((), (2, 3, 7), (8,), (2, 3, 7, 8))
    return V, A, producers, bounds, consumers, configs


def in_edges(A: Edges) -> list[list[int]]:
    'Compute lists of incoming edges per edge'
    delta: list[list[int]] = [[] for _ in A]
    for line, (i, j) in enumerate(A):
        delta[line] += [k for k, (lo, m) in enumerate(A) if m == i]
    return delta


def out_edges(A: Edges) -> list[list[int]]:
    'Compute lists of outgoing edges per edge'
    delta: list[list[int]] = [[] for _ in A]
    for line, (i, j) in enumerate(A):
        delta[line] += [k for k, (lo, m) in enumerate(A) if lo == j]
    return delta
