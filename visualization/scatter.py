from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_particles(
    ax: plt.Axes,
    particles: torch.Tensor,
    color: str = "tab:blue",
    size: float = 5,
    alpha: float = 0.7,
    label: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """
    Draw a set of 2D particles.

    Parameters
    ----------
    ax
        Matplotlib axes.

    particles
        Tensor of shape (N,2).

    color
        Marker color.

    size
        Marker size.

    alpha
        Marker transparency.

    label
        Legend label.

    xlim
        X-axis limits.

    ylim
        Y-axis limits.

    Returns
    -------
    matplotlib.collections.PathCollection
        Scatter artist.
    """

    if isinstance(particles, torch.Tensor):
        particles = particles.detach().cpu().numpy()

    scatter = ax.scatter(
        particles[:, 0],
        particles[:, 1],
        s=size,
        c=color,
        alpha=alpha,
        edgecolors="none",
        label=label,
    )

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_aspect("equal")

    return scatter