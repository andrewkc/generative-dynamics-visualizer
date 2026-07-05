from __future__ import annotations

import torch

from visualization.animation import ParticleAnimation


class ForwardTrajectoriesAnimation(ParticleAnimation):
    """
    Animate the forward diffusion process together with
    particle trajectories.
    """

    def __init__(
        self,
        trajectory,
        max_particles=200,
        interval=40,
        xlim=(-6, 6),
        ylim=(-6, 6),
    ):

        self.trajectory = trajectory

        self.max_particles = min(
            max_particles,
            trajectory[0].shape[0],
        )

        super().__init__(
            nrows=1,
            ncols=1,
            figsize=(7, 7),
            xlim=xlim,
            ylim=ylim,
            interval=interval,
            point_size=8,
            titles=[
                "Forward Trajectories",
            ],
        )

        #
        # Create trajectory lines
        #

        self.lines = []

        ax = self.axes[0]

        for _ in range(self.max_particles):

            (line,) = ax.plot(
                [],
                [],
                color="black",
                alpha=0.25,
                linewidth=0.6,
            )

            self.lines.append(
                line,
            )

    # --------------------------------------------------

    def get_frame(
        self,
        frame,
    ):

        return [
            self.trajectory[frame],
        ]

    # --------------------------------------------------

    def get_title(
        self,
        frame,
    ):

        return "Forward Diffusion Trajectories"

    # --------------------------------------------------

    def _update(
        self,
        frame,
    ):

        particles = self.trajectory[frame]

        if isinstance(
            particles,
            torch.Tensor,
        ):

            particles_np = particles.detach().cpu().numpy()

        else:

            particles_np = particles

        #
        # Update scatter
        #

        self.scatters[0].set_offsets(
            particles_np,
        )

        #
        # Update trajectory lines
        #

        for idx in range(self.max_particles):

            xs = []

            ys = []

            for k in range(frame + 1):

                p = self.trajectory[k][idx]

                xs.append(p[0].item())

                ys.append(p[1].item())

            self.lines[idx].set_data(
                xs,
                ys,
            )

        #
        # Global title
        #

        self.fig.suptitle(
            self.get_title(frame),
            fontsize=15,
        )

        self.frame_text.set_text(f"Frame {frame+1}/{self.n_frames}")

        t = frame / (self.n_frames - 1)

        self.time_text.set_text(f"t = {t:.3f}")

        artists = []

        artists.extend(self.scatters)

        artists.extend(self.lines)

        artists.append(self.frame_text)

        artists.append(self.time_text)

        return artists
