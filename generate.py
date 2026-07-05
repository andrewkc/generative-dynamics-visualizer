from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion import get_sde

from models import (
    DiffusionModel,
    FlowModel,
)

from samplers import (
    ReverseSDESampler,
    ProbabilityFlowSampler,
    FlowODESampler,
)

from utils import (
    get_device,
    load_checkpoint,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Checkpoint path.",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="Number of generated samples.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=250,
        help="Integration steps.",
    )
    
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=[
            "reverse_sde",
            "probability_flow_ode",
            "flow_ode",
        ],
        help="Override sampler stored in checkpoint.",
    )

    args = parser.parse_args()

    #
    # Device
    #

    device = get_device()

    #
    # Load checkpoint
    #

    checkpoint = load_checkpoint(
        args.checkpoint,
        map_location=device,
    )

    config = checkpoint["config"]

    #
    # Build model
    #

    if config["model"] == "diffusion":

        model = DiffusionModel(
            hidden_dim=config["hidden_dim"],
            time_embedding_dim=config["time_embedding_dim"],
            prediction_type=config["prediction_type"],
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        model.to(device)

        sde = get_sde(
            config["sde"],
        )

        sampler_name = args.sampler

        if sampler_name is None:

            sampler_name = config.get(
                "sampler",
                "reverse_sde",
            )

        if sampler_name == "reverse_sde":

            sampler = ReverseSDESampler(
                model=model,
                sde=sde,
                device=device,
            )

        elif sampler_name == "probability_flow_ode":

            sampler = ProbabilityFlowSampler(
                model=model,
                sde=sde,
                device=device,
            )

        else:

            raise ValueError(f"Unknown sampler '{sampler_name}'")

    elif config["model"] == "flow_matching":
        model = FlowModel(
            hidden_dim=config["hidden_dim"],
            time_embedding_dim=config["time_embedding_dim"],
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(device)

        sampler_name = args.sampler

        if sampler_name is None:

            sampler_name = config.get(
                "sampler",
                "flow_ode",
            )

        if sampler_name != "flow_ode":

            raise ValueError(
                f"Flow Matching only supports flow_ode. Got '{sampler_name}'."
            )

        sampler = FlowODESampler(
            model=model,
            device=device,
        )

    else:

        raise ValueError(f"Unknown model '{config['model']}'")

    # Generate

    samples = sampler.sample(
        n_samples=args.samples,
        steps=args.steps,
    )

    print()
    print(f"Model      : {config['model']}")
    print(f"Sampler    : {sampler_name}")
    print(f"Samples    : {args.samples}")
    print(f"Steps      : {args.steps}")
    print(f"Device     : {device}")
    
    # Save
    output_dir = Path("outputs")

    output_dir.mkdir(
        exist_ok=True,
        parents=True,
    )

    torch.save(
        samples,
        output_dir / f"samples-{config['model']}-{sampler_name}.pt",
    )

    np.save(
        output_dir / f"samples-{config['model']}-{sampler_name}.pt",
        samples.cpu().numpy(),
    )

    print()

    print(f"Generated {len(samples)} samples.")
    print(f"Saved to {output_dir}/samples-{config['model']}-{sampler_name}.pt")


if __name__ == "__main__":

    main()


# python generate.py --checkpoint checkpoints/vp_epsilon.pt --samples 20000 --steps 500

# python generate.py --checkpoint checkpoints/vp_epsilon.pt --sampler reverse_sde
# python generate.py --checkpoint checkpoints/vp_epsilon.pt --sampler probability_flow_ode
# python generate.py --checkpoint checkpoints/flow_matching.pt --sampler flow_ode