from __future__ import annotations

import matplotlib.pyplot as plt
import torch


def plot_trajectories(
    ax: plt.Axes,
    trajectory,
    color="black",
    alpha=0.25,
    linewidth=0.5,
):
    """
    Draw particle trajectories.

    trajectory:
        list[Tensor(N,2)]
    """

    for i in range(trajectory[0].shape[0]):

        xs = []

        ys = []

        for state in trajectory:

            xs.append(state[i, 0].item())

            ys.append(state[i, 1].item())

        ax.plot(
            xs,
            ys,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )