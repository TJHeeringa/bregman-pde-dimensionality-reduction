from pathlib import Path
import sys

import torch.utils.data
import wandb

sys.path.insert(0, str(Path.cwd()))

from src.training.train import train  # noqa: E402
from src.data.generate_datasets import generate_reaction_diffusion_dataset  # noqa: E402


if __name__ == "__main__":
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
    torch.set_default_dtype(torch.float64)

    config = wandb.config

    print(config)

    dataset = generate_reaction_diffusion_dataset()

    subsampled_dataset = torch.utils.data.Subset(dataset, torch.arange(start=5000, end=50_000, step=36))

    train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
        subsampled_dataset, [750, 250, 250], generator=torch.Generator().manual_seed(42)
    )

    train(train_dataset, test_dataset, validation_dataset, Path("models/reaction_diffusion"), config)
