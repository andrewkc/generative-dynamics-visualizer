from __future__ import annotations

import torch


def save_checkpoint(
    path: str,
    model,
    config: dict,
    optimizer=None,
    epoch=None,
):
    """
    Save a training checkpoint.
    """

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    torch.save(
        checkpoint,
        path,
    )


def load_checkpoint(
    path: str,
    map_location="cpu",
):
    """
    Load a checkpoint from disk.

    Returns
    -------
    checkpoint : dict
    """

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    return checkpoint
