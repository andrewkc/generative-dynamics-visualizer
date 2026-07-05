from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file.
    """

    path = Path(path)

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config