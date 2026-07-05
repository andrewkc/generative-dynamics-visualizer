from __future__ import annotations

import numpy as np
import torch

from visualization.animation import ParticleAnimation


class ScoreFieldAnimation(ParticleAnimation):
    """
    Animated visualization of the learned score field.
    """

    def __init__(
        self,
        score_model,
        interval=40,
        grid_size=25,
        xlim=(-6, 6),
        ylim=(-6, 6),
    ):

        self.score_model = score_model

        self.grid_size = grid_size

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
                "Learned Score Field",
            ],
        )

        #
        # Remove scatter
        #

        self.scatters[0].remove()

        self.scatters = []

        #
        # Grid
        #

        xs = np.linspace(
            self.xmin,
            self.xmax,
            grid_size,
        )

        ys = np.linspace(
            self.ymin,
            self.ymax,
            grid_size,
        )

        X, Y = np.meshgrid(
            xs,
            ys,
        )

        self.X = X

        self.Y = Y

        self.grid = torch.tensor(
            np.stack(
                [
                    X.ravel(),
                    Y.ravel(),
                ],
                axis=1,
            ),
            dtype=torch.float32,
        )

        #
        # Initial quiver
        #

        self.quiver = self.axes[0].quiver(
            X,
            Y,
            np.zeros_like(X),
            np.zeros_like(Y),
            color="tab:orange",
        )

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

        return "Learned Score Field"

    # -----------------------------------------------------

    @torch.no_grad()
    def _update(
        self,
        frame,
    ):

        t = frame / (self.n_frames - 1)

        t_tensor = torch.full(
            (self.grid.shape[0],),
            t,
        )

        score = self.score_model(
            self.grid,
            t_tensor,
        )

        score = score.cpu().numpy()

        U = score[:, 0].reshape(
            self.X.shape,
        )

        V = score[:, 1].reshape(
            self.Y.shape,
        )

        self.quiver.set_UVC(
            U,
            V,
        )

        self.fig.suptitle(
            self.get_title(frame),
            fontsize=15,
        )

        self.frame_text.set_text(f"Frame {frame+1}/{self.n_frames}")

        self.time_text.set_text(f"t = {t:.3f}")

        return [
            self.quiver,
            self.frame_text,
            self.time_text,
        ]
