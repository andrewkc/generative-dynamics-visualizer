from __future__ import annotations

import numpy as np
import torch

from visualization.animation import ParticleAnimation


class DensityAnimation(ParticleAnimation):
    """
    Evolution of the particle density during the forward process.

    Displays the density as a 2D histogram.
    """

    def __init__(
        self,
        trajectory,
        bins=120,
        interval=40,
        xlim=(-6, 6),
        ylim=(-6, 6),
    ):

        self.trajectory = trajectory

        self.bins = bins

        self.xmin, self.xmax = xlim

        self.ymin, self.ymax = ylim

        super().__init__(
            nrows=1,
            ncols=1,
            figsize=(7, 7),
            xlim=xlim,
            ylim=ylim,
            interval=interval,
            point_size=0,
            titles=[
                "Density Evolution",
            ],
        )

        #
        # Remove scatter.
        #

        self.scatters[0].remove()

        self.scatters = []

        #
        # Initial density.
        #

        H = self.compute_density(
            self.trajectory[0],
        )

        self.image = self.axes[0].imshow(
            H,
            origin="lower",
            extent=(
                self.xmin,
                self.xmax,
                self.ymin,
                self.ymax,
            ),
            cmap="viridis",
            interpolation="nearest",
            animated=True,
        )

        self.fig.colorbar(
            self.image,
            ax=self.axes[0],
            shrink=0.9,
        )

    # -----------------------------------------------------

    def compute_density(
        self,
        particles,
    ):

        if isinstance(
            particles,
            torch.Tensor,
        ):

            particles = particles.detach().cpu().numpy()

        H, _, _ = np.histogram2d(
            particles[:, 0],
            particles[:, 1],
            bins=self.bins,
            range=[
                [
                    self.xmin,
                    self.xmax,
                ],
                [
                    self.ymin,
                    self.ymax,
                ],
            ],
        )

        return H.T

    # -----------------------------------------------------

    def get_frame(
        self,
        frame,
    ):

        return []

    # -----------------------------------------------------

    def get_title(
        self,
        frame,
    ):

        return "Forward Density Evolution"

    # -----------------------------------------------------

    def _update(
        self,
        frame,
    ):

        H = self.compute_density(
            self.trajectory[frame],
        )

        self.image.set_data(
            H,
        )

        self.fig.suptitle(
            self.get_title(frame),
            fontsize=15,
        )

        self.frame_text.set_text(f"Frame {frame+1}/{self.n_frames}")

        t = frame / (self.n_frames - 1)

        self.time_text.set_text(f"t = {t:.3f}")

        return [
            self.image,
            self.frame_text,
            self.time_text,
        ]
