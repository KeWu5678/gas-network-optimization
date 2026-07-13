'''
Generic plotting helpers for POC/CIAP results.
'''

from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_relaxed_vs_binary_config_index(
    t_grid: np.ndarray,
    alpha_relax: np.ndarray,
    beta: np.ndarray,
    *,
    n_confg: Optional[int] = None,
    configs: Optional[Iterable[object]] = None,
    figsize: Tuple[float, float] = (10, 4),
    show: bool = True,
):
    '''
    Plot relaxed vs. binary configuration index over time.

    t_grid: 1D time grid (length nt).
    alpha_relax: relaxed multipliers (shape (nt-1, n_confg)).
    beta: binary multipliers (shape (nt-1, n_confg)).
    n_confg / configs: pass one of the two to fix the number of modes.
    '''
    if n_confg is None:
        if configs is None:
            raise ValueError('Provide either `n_confg` or `configs`.')
        n_confg = len(list(configs))

    idx = np.arange(n_confg)
    relaxed_index = np.concatenate(([np.nan], alpha_relax.dot(idx)))
    binary_index = np.concatenate(([np.nan], beta.dot(idx)))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].step(t_grid, relaxed_index, linewidth=2)
    axes[0].set_ylabel('config')
    axes[0].set_title(r'Relaxed $\alpha$ (POC) -- weighted config index')
    axes[0].set_ylim(-0.1, n_confg - 0.9)

    axes[1].step(t_grid, binary_index, 'r', linewidth=2)
    axes[1].set_ylabel('config')
    axes[1].set_title(r'Binary $\beta$ (CIAP) -- integer config index')
    axes[1].set_ylim(-0.1, n_confg - 0.9)
    axes[1].set_xlabel(r'time $t$')

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes
