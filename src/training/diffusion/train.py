from pathlib import Path
import sys
import torch

import wandb

sys.path.insert(0, str(Path.cwd()))

from src.training.train import train  # noqa: E402
from src.data.generate_datasets import generate_diffusion_dataset  # noqa: E402


TRAIN_PARAMETERS = [0.1, 0.5, 1]
VALIDATION_PARAMETERS = [0.8]
TEST_PARAMETERS = [0.6]


if __name__ == "__main__":
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    #torch.set_default_dtype(torch.float64)

    config = wandb.config

    print(config)

    train_dataset = generate_diffusion_dataset(TRAIN_PARAMETERS)
    test_dataset = generate_diffusion_dataset(TEST_PARAMETERS)
    validation_dataset = generate_diffusion_dataset(VALIDATION_PARAMETERS)

    train(
        train_dataset,
        test_dataset,
        validation_dataset,
        Path("models/diffusion"),
        config
    )
