from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam

from datasets import get_dataset

from diffusion import get_sde

from models import (
    DiffusionModel,
    FlowModel,
)

from trainers import (
    DiffusionTrainer,
    FlowTrainer,
)

from utils import (
    get_device,
    load_config,
    save_checkpoint,
    set_seed,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to YAML configuration file.",
    )

    args = parser.parse_args()

    #
    # Load configuration
    #

    config = load_config(
        args.config,
    )

    #
    # Seed
    #

    set_seed(
        config["seed"],
    )

    #
    # Device
    #

    if config["device"] == "auto":
        device = get_device()
    else:
        device = torch.device(
            config["device"],
        )

    print(f"Device : {device}")

    #
    # Dataset
    #

    dataset = get_dataset(
        config["dataset"],
    )

    #
    # Model
    #

    if config["model"] == "diffusion":

        model = DiffusionModel(
            hidden_dim=config["hidden_dim"],
            time_embedding_dim=config["time_embedding_dim"],
            prediction_type=config["prediction_type"],
        )

        sde = get_sde(
            config["sde"],
        )

        optimizer = Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )

        trainer = DiffusionTrainer(
            model=model,
            sde=sde,
            optimizer=optimizer,
            device=device,
            prediction_type=config["prediction_type"],
        )

    elif config["model"] == "flow_matching":

        model = FlowModel(
            hidden_dim=config["hidden_dim"],
            time_embedding_dim=config["time_embedding_dim"],
        )

        optimizer = Adam(
            model.parameters(),
            lr=config["learning_rate"],
        )

        trainer = FlowTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
        )

    else:

        raise ValueError(f"Unknown model '{config['model']}'")

    #
    # Training
    #

    print()

    print("Training...")

    for epoch in range(config["epochs"]):

        loss = trainer.train_epoch(
            dataset=dataset,
            steps_per_epoch=config["steps_per_epoch"],
            batch_size=config["batch_size"],
        )

        print(f"Epoch [{epoch+1}/{config['epochs']}] " f"Loss: {loss:.6f}")

    #
    # Save checkpoint
    #

    checkpoint_dir = Path(
        config["checkpoint_dir"],
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = checkpoint_dir / config["checkpoint_name"]

    save_checkpoint(
        path=str(checkpoint_path),
        model=model,
        optimizer=optimizer,
        epoch=config["epochs"],
        config=config,
    )

    print()

    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":

    main()


# python train.py --config configs/diffusion/vp_epsilon.yaml
# python train.py --config configs/diffusion/vp_velocity.yaml
# python train.py --config configs/diffusion/ve_epsilon.yaml
# python train.py --config configs/diffusion/subvp_epsilon.yaml
# python train.py --config configs/flow_matching/flow_matching.yaml