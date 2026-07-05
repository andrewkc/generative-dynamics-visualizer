from __future__ import annotations
from visualization.animation import ParticleAnimation


class ForwardComparisonAnimation(ParticleAnimation):
    """
    Comparison of the forward diffusion processes

        VP
        VE
        sub-VP

    using the same initial particles.
    """

    def __init__(
        self,
        vp_trajectory,
        ve_trajectory,
        subvp_trajectory,
        interval=40,
    ):

        self.vp = vp_trajectory
        self.ve = ve_trajectory
        self.subvp = subvp_trajectory

        assert len(self.vp) == len(self.ve)
        assert len(self.vp) == len(self.subvp)

        super().__init__(
            nrows=1,
            ncols=3,
            figsize=(15, 5),
            xlim=(-6, 6),
            ylim=(-6, 6),
            interval=interval,
            point_size=3,
            titles=[
                "Variance Preserving (VP)",
                "Variance Exploding (VE)",
                "sub-VP",
            ],
        )

    # --------------------------------------------------

    def get_frame(
        self,
        frame,
    ):

        return [
            self.vp[frame],
            self.ve[frame],
            self.subvp[frame],
        ]

    # --------------------------------------------------

    def get_title(
        self,
        frame,
    ):

        return "Forward Diffusion Processes"
