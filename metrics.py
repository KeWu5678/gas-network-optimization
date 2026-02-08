"""
Metrics / plotting helpers for the transmission-lines notebooks.

This module is imported from the notebook by adding `p-adm/t-lines` to `sys.path`.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


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
    """
    Plot relaxed vs. binary configuration index over time.

    Parameters
    ----------
    t_grid:
        1D time grid (length nt).
    alpha_relax:
        Relaxed multipliers from Step 1 (typically shape (nt-1, n_confg)).
    beta:
        Binary multipliers from Step 2 (typically shape (nt-1, n_confg)).
    n_confg / configs:
        Either pass `n_confg` directly, or pass `configs` and it will be inferred
        as `len(configs)`.
    figsize:
        Figure size passed to `plt.subplots`.
    show:
        If True, call `plt.show()` at the end.

    Returns
    -------
    (fig, axes):
        Matplotlib figure and axes (2 rows).
    """
    if n_confg is None:
        if configs is None:
            raise ValueError("Provide either `n_confg` or `configs`.")
        n_confg = len(list(configs))

    idx = np.arange(n_confg)
    relaxed_index = np.concatenate(([np.nan], alpha_relax.dot(idx)))
    binary_index = np.concatenate(([np.nan], beta.dot(idx)))

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].step(t_grid, relaxed_index, linewidth=2)
    axes[0].set_ylabel("config")
    axes[0].set_title(r"Relaxed $\bar\omega$ (Step 1) -- weighted config index")
    axes[0].set_ylim(-0.1, n_confg - 0.9)

    axes[1].step(t_grid, binary_index, "r", linewidth=2)
    axes[1].set_ylabel("config")
    axes[1].set_title(r"Binary $\beta$ (Step 2) -- integer config index")
    axes[1].set_ylim(-0.1, n_confg - 0.9)
    axes[1].set_xlabel(r"time $t$")

    plt.tight_layout()
    if show:
        plt.show()

    return fig, axes
