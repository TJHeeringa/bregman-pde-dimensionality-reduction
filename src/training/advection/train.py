from pathlib import Path
import sys

import torch
import wandb

sys.path.insert(0, str(Path.cwd()))

from src.training.train import train  # noqa: E402
from src.data.generate_datasets import generate_advection_dataset  # noqa: E402


TRAIN_PARAMETERS = torch.tensor([0.6, 0.9, 1.2])
VALIDATION_PARAMETERS = torch.tensor([0.75])
TEST_PARAMETERS = torch.tensor([1.05])


if __name__ == "__main__":
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    torch.set_default_dtype(torch.float64)

    config = wandb.config

    print(config)

    train_dataset = generate_advection_dataset(TRAIN_PARAMETERS)
    test_dataset = generate_advection_dataset(TEST_PARAMETERS)
    validation_dataset = generate_advection_dataset(VALIDATION_PARAMETERS)

    train(train_dataset, test_dataset, validation_dataset, Path("models/advection"), config)
