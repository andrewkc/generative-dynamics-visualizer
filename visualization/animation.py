from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


class ParticleAnimation:
    """
    Generic particle animation.

    Supports:

    - one subplot
    - multiple subplots
    - one scatter per subplot
    - mp4
    - gif

    Child classes only need to implement

        get_frame(frame)

    which must return a list of particle arrays
    (one per subplot).
    """

    def __init__(
        self,
        nrows=1,
        ncols=1,
        figsize=(6, 6),
        xlim=(-3, 3),
        ylim=(-3, 3),
        interval=30,
        titles=None,
        point_size=5,
    ):

        self.interval = interval
        self.nrows = nrows
        self.ncols = ncols

        self.fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
        )

        #
        # Normalize axes to a flat list
        #

        if isinstance(axes, np.ndarray):

            self.axes = axes.ravel().tolist()

        else:

            self.axes = [axes]

        #
        # Create scatters
        #

        self.scatters = []

        for i, ax in enumerate(self.axes):

            ax.set_xlim(*xlim)

            ax.set_ylim(*ylim)

            ax.set_aspect("equal")

            ax.grid(True)

            if titles is not None:

                ax.set_title(
                    titles[i],
                )

            scatter = ax.scatter(
                [],
                [],
                s=point_size,
            )

            self.scatters.append(
                scatter,
            )

        #
        # Information shown above the figure
        #

        self.frame_text = self.fig.text(
            0.02,
            0.97,
            "",
            fontsize=11,
        )

        self.time_text = self.fig.text(
            0.82,
            0.97,
            "",
            fontsize=11,
        )

        self.n_frames = None

    # --------------------------------------------------

    def get_frame(
        self,
        frame: int,
    ):
        """
        Must return

        [
            particles_subplot_1,
            particles_subplot_2,
            ...
        ]
        """

        raise NotImplementedError

    # --------------------------------------------------

    def get_title(
        self,
        frame,
    ):

        return None

    # --------------------------------------------------

    def _update(
        self,
        frame,
    ):

        particle_sets = self.get_frame(
            frame,
        )

        artists = []

        for scatter, particles in zip(
            self.scatters,
            particle_sets,
        ):

            if hasattr(
                particles,
                "cpu",
            ):

                particles = particles.detach().cpu().numpy()

            scatter.set_offsets(
                particles,
            )

            artists.append(
                scatter,
            )

        #
        # Global title
        #

        title = self.get_title(
            frame,
        )

        if title is not None:

            self.fig.suptitle(
                title,
                fontsize=15,
            )

        #
        # Frame counter
        #

        self.frame_text.set_text(f"Frame {frame+1}/{self.n_frames}")

        #
        # Continuous time
        #

        t = frame / (self.n_frames - 1)

        self.time_text.set_text(f"t = {t:.3f}")

        artists.append(
            self.frame_text,
        )

        artists.append(
            self.time_text,
        )

        return artists

    # --------------------------------------------------

    def animate(
        self,
        n_frames,
        repeat=False,
    ):

        self.n_frames = n_frames

        animation = FuncAnimation(
            self.fig,
            self._update,
            frames=n_frames,
            interval=self.interval,
            repeat=repeat,
            blit=False,
        )

        return animation

    # --------------------------------------------------

    def save_mp4(
        self,
        animation,
        filename,
        fps=30,
    ):

        filename = Path(
            filename,
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        animation.save(
            filename,
            writer="ffmpeg",
            fps=fps,
        )

    # --------------------------------------------------

    def save_gif(
        self,
        animation,
        filename,
        fps=30,
    ):

        filename = Path(
            filename,
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        animation.save(
            filename,
            writer="pillow",
            fps=fps,
        )

    # --------------------------------------------------

    def save_png(
        self,
        filename,
    ):

        filename = Path(
            filename,
        )

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.fig.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

    # --------------------------------------------------

    def show(
        self,
    ):

        plt.show()
