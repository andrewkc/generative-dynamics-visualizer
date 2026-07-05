from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_density(
    ax: plt.Axes,
    particles: torch.Tensor,
    bins: int = 100,
    cmap: str = "viridis",
):
    """
    Plot a 2D density using a histogram.
    """

    if isinstance(particles, torch.Tensor):
        particles = particles.detach().cpu().numpy()

    hist = ax.hist2d(
        particles[:, 0],
        particles[:, 1],
        bins=bins,
        cmap=cmap,
    )

    return hist