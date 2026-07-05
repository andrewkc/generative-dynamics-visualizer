from __future__ import annotations

import matplotlib.pyplot as plt


def create_figure(
    title: str | None = None,
    figsize: tuple[int, int] = (6, 6),
    xlim: tuple[float, float] = (-3, 3),
    ylim: tuple[float, float] = (-3, 3),
    grid: bool = True,
):
    """
    Create a standard figure for 2D visualizations.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
    )

    ax.set_xlim(*xlim)

    ax.set_ylim(*ylim)

    ax.set_aspect("equal")

    if grid:
        ax.grid(True)

    if title is not None:
        ax.set_title(title)

    return fig, ax