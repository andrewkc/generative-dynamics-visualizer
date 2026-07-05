from __future__ import annotations

import torch

from visualization.animation import ParticleAnimation


class StepsComparisonAnimation(ParticleAnimation):
    """
    Compare different integration step counts.

    Each subplot corresponds to a different
    number of solver steps.
    """

    def __init__(
        self,
        trajectories,
        step_values,
        interval=40,
        xlim=(-6, 6),
        ylim=(-6, 6),
    ):

        self.trajectories = trajectories

        self.step_values = step_values

        self.max_frames = max(len(t) for t in trajectories)

        super().__init__(
            nrows=1,
            ncols=len(step_values),
            figsize=(4 * len(step_values), 4),
            xlim=xlim,
            ylim=ylim,
            interval=interval,
            point_size=4,
            titles=[f"{n} steps" for n in step_values],
        )

    # --------------------------------------------------

    def get_frame(
        self,
        frame,
    ):

        particle_sets = []

        for trajectory in self.trajectories:

            #
            # Synchronize animations
            #

            idx = int(frame * (len(trajectory) - 1) / (self.max_frames - 1))

            particle_sets.append(trajectory[idx])

        return particle_sets

    # --------------------------------------------------

    def get_title(
        self,
        frame,
    ):

        return "Effect of Integration Steps"

    # --------------------------------------------------

    def animate(
        self,
        repeat=False,
    ):

        return super().animate(
            n_frames=self.max_frames,
            repeat=repeat,
        )
