from __future__ import annotations

import argparse
from pathlib import Path

from datasets import get_dataset

from diffusion import (
    get_sde,
    ScoreModel,
)

from models import (
    DiffusionModel,
    FlowModel,
)

from samplers import (
    ReverseSDESampler,
    ProbabilityFlowSampler,
    FlowODESampler,
)

from utils import load_checkpoint

from visualization.animations import (
    ForwardComparisonAnimation,
    DensityAnimation,
    ForwardTrajectoriesAnimation,
    ReverseSDEAnimation,
    ProbabilityFlowAnimation,
    FlowMatchingAnimation,
    ScoreFieldAnimation,
    StepsComparisonAnimation,
)

# ============================================================
# Helpers
# ============================================================


def build_model_from_checkpoint(checkpoint):

    config = checkpoint["config"]

    model_type = config["model"]

    if model_type == "diffusion":

        model = DiffusionModel(
            hidden_dim=config.get(
                "hidden_dim",
                256,
            ),
            time_embedding_dim=config.get(
                "time_embedding_dim",
                64,
            ),
            prediction_type=config.get(
                "prediction",
                "epsilon",
            ),
        )

    elif model_type == "flow":

        model = FlowModel(
            hidden_dim=config.get(
                "hidden_dim",
                256,
            ),
            time_embedding_dim=config.get(
                "time_embedding_dim",
                64,
            ),
        )

    else:

        raise ValueError(f"Unknown model '{model_type}'.")

    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    return model


# ============================================================


def save_animation(
    animator,
    animation,
    output_dir,
    filename,
    fps,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    animator.save_mp4(
        animation,
        output_dir / f"{filename}.mp4",
        fps=fps,
    )

    animator.save_gif(
        animation,
        output_dir / f"{filename}.gif",
        fps=fps,
    )

    animator.save_png(
        output_dir / f"{filename}.png",
    )


# ============================================================
# Parser
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--animation",
    required=True,
    choices=[
        "forward",
        "density",
        "forward_trajectories",
        "reverse_sde",
        "probability_flow",
        "flow_matching",
        "score_field",
        "steps_comparison",
    ],
)

parser.add_argument(
    "--checkpoint",
    default=None,
)

parser.add_argument(
    "--dataset",
    default="two_moons",
)

parser.add_argument(
    "--samples",
    type=int,
    default=5000,
)

parser.add_argument(
    "--steps",
    type=int,
    default=250,
)

parser.add_argument(
    "--fps",
    type=int,
    default=30,
)

parser.add_argument(
    "--output_dir",
    default="outputs/animations",
)

parser.add_argument(
    "--sampler",
    choices=[
        "reverse_sde",
        "probability_flow",
        "flow",
    ],
    default="reverse_sde",
)

args = parser.parse_args()

output_dir = Path(
    args.output_dir,
)


# ============================================================
# Forward animations
# ============================================================

if args.animation in [
    "forward",
    "density",
    "forward_trajectories",
]:

    dataset = get_dataset(
        args.dataset,
    )

    x0 = dataset.sample(
        args.samples,
    )

    vp = get_sde(
        "vp",
    )

    trajectory = vp.sample_forward_trajectory(
        x0,
        steps=args.steps,
    )

    if args.animation == "forward":

        ve = get_sde(
            "ve",
        )

        subvp = get_sde(
            "subvp",
        )

        trajectory_ve = ve.sample_forward_trajectory(
            x0,
            steps=args.steps,
        )

        trajectory_subvp = subvp.sample_forward_trajectory(
            x0,
            steps=args.steps,
        )

        animator = ForwardComparisonAnimation(
            trajectory,
            trajectory_ve,
            trajectory_subvp,
        )

        animation = animator.animate(
            len(
                trajectory,
            )
        )

        save_animation(
            animator,
            animation,
            output_dir,
            "forward_comparison",
            args.fps,
        )

        animator.show()

    elif args.animation == "density":

        animator = DensityAnimation(
            trajectory,
        )

        animation = animator.animate(
            len(
                trajectory,
            )
        )

        save_animation(
            animator,
            animation,
            output_dir,
            "density",
            args.fps,
        )

        animator.show()

    elif args.animation == "forward_trajectories":

        animator = ForwardTrajectoriesAnimation(
            trajectory,
        )

        animation = animator.animate(
            len(
                trajectory,
            )
        )

        save_animation(
            animator,
            animation,
            output_dir,
            "forward_trajectories",
            args.fps,
        )

        animator.show()
# ============================================================
# Reverse-time SDE
# ============================================================

elif args.animation == "reverse_sde":

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required.")

    checkpoint = load_checkpoint(
        args.checkpoint,
    )

    model = build_model_from_checkpoint(
        checkpoint,
    )

    config = checkpoint["config"]

    sde = get_sde(
        config["sde"],
    )

    sampler = ReverseSDESampler(
        model=model,
        sde=sde,
    )

    _, trajectory = sampler.sample(
        n_samples=args.samples,
        steps=args.steps,
        return_trajectory=True,
    )

    animator = ReverseSDEAnimation(
        trajectory,
    )

    animation = animator.animate(
        len(trajectory),
    )

    save_animation(
        animator,
        animation,
        output_dir,
        "reverse_sde",
        args.fps,
    )

    animator.show()


# ============================================================
# Probability Flow ODE
# ============================================================

elif args.animation == "probability_flow":

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required.")

    checkpoint = load_checkpoint(
        args.checkpoint,
    )

    model = build_model_from_checkpoint(
        checkpoint,
    )

    config = checkpoint["config"]

    sde = get_sde(
        config["sde"],
    )

    sampler = ProbabilityFlowSampler(
        model=model,
        sde=sde,
    )

    _, trajectory = sampler.sample(
        n_samples=args.samples,
        steps=args.steps,
        return_trajectory=True,
    )

    animator = ProbabilityFlowAnimation(
        trajectory,
    )

    animation = animator.animate(
        len(trajectory),
    )

    save_animation(
        animator,
        animation,
        output_dir,
        "probability_flow",
        args.fps,
    )

    animator.show()


# ============================================================
# Flow Matching
# ============================================================

elif args.animation == "flow_matching":

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required.")

    checkpoint = load_checkpoint(
        args.checkpoint,
    )

    model = build_model_from_checkpoint(
        checkpoint,
    )

    sampler = FlowODESampler(
        model=model,
    )

    _, trajectory = sampler.sample(
        n_samples=args.samples,
        steps=args.steps,
        return_trajectory=True,
    )

    animator = FlowMatchingAnimation(
        trajectory,
    )

    animation = animator.animate(
        len(trajectory),
    )

    save_animation(
        animator,
        animation,
        output_dir,
        "flow_matching",
        args.fps,
    )

    animator.show()

# ============================================================
# Score Field
# ============================================================

elif args.animation == "score_field":

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required.")

    checkpoint = load_checkpoint(
        args.checkpoint,
    )

    config = checkpoint["config"]

    if config["model"] != "diffusion":
        raise ValueError("Score field is only available for diffusion models.")

    model = build_model_from_checkpoint(
        checkpoint,
    )

    sde = get_sde(
        config["sde"],
    )

    score_model = ScoreModel(
        model=model,
        sde=sde,
    )

    animator = ScoreFieldAnimation(
        score_model,
    )

    animation = animator.animate(
        n_frames=args.steps,
    )

    save_animation(
        animator,
        animation,
        output_dir,
        "score_field",
        args.fps,
    )

    animator.show()


# ============================================================
# Steps Comparison
# ============================================================

elif args.animation == "steps_comparison":

    if args.checkpoint is None:
        raise ValueError("--checkpoint is required.")

    checkpoint = load_checkpoint(
        args.checkpoint,
    )

    config = checkpoint["config"]

    model = build_model_from_checkpoint(
        checkpoint,
    )

    step_values = [
        10,
        25,
        50,
        100,
        250,
    ]

    trajectories = []

    if args.sampler == "reverse_sde":

        sde = get_sde(
            config["sde"],
        )

        sampler = ReverseSDESampler(
            model=model,
            sde=sde,
        )

    elif args.sampler == "probability_flow":

        sde = get_sde(
            config["sde"],
        )

        sampler = ProbabilityFlowSampler(
            model=model,
            sde=sde,
        )

    elif args.sampler == "flow":

        sampler = FlowODESampler(
            model=model,
        )

    else:

        raise ValueError(f"Unknown sampler '{args.sampler}'.")

    for n_steps in step_values:

        _, trajectory = sampler.sample(
            n_samples=args.samples,
            steps=n_steps,
            return_trajectory=True,
        )

        trajectories.append(
            trajectory,
        )

    animator = StepsComparisonAnimation(
        trajectories,
        step_values,
    )

    animation = animator.animate()

    save_animation(
        animator,
        animation,
        output_dir,
        f"steps_comparison_{args.sampler}",
        args.fps,
    )

    animator.show()


# ============================================================
# Unknown animation
# ============================================================

else:

    raise RuntimeError(f"Unknown animation '{args.animation}'.")


# 1. Forward VP vs VE vs subVP
# python animate.py --animation forward --dataset two_moons --samples 5000 --steps 250
# python animate.py --animation forward --dataset eight_gaussians --samples 10000 --steps 250

# 2. Density Evolution
# python animate.py --animation density --dataset two_moons --samples 5000 --steps 250
    
# 3. Forward Trajectories
# python animate.py --animation forward_trajectories --dataset two_moons --samples 5000 --steps 250
    
# 4. Reverse-time SDE
# python animate.py --animation reverse_sde --checkpoint checkpoints/vp_epsilon.pt --samples 5000 --steps 250

# 5. Probability Flow ODE
# python animate.py --animation probability_flow --checkpoint checkpoints/vp_epsilon.pt --samples 5000 --steps 250

# 6. Flow Matching
# python animate.py --animation flow_matching --checkpoint checkpoints/flow.pt --samples 5000 --steps 250
    
# 7. Score Field
# python animate.py --animation score_field --checkpoint checkpoints/vp_epsilon.pt --steps 200
    
# 8. Steps Comparison usando Reverse SDE
# python animate.py --animation steps_comparison --sampler probability_flow --checkpoint checkpoints/vp_epsilon.pt --samples 5000
    
# 10. Steps Comparison usando Flow Matching
# python animate.py --animation steps_comparison --sampler flow --checkpoint checkpoints/flow.pt --samples 5000