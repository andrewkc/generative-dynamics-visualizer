from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_vector_field(
    ax: plt.Axes,
    X,
    Y,
    U,
    V,
    color="tab:orange",
):
    """
    Plot a vector field.
    """

    quiver = ax.quiver(
        X,
        Y,
        U,
        V,
        color=color,
    )

    return quiver
