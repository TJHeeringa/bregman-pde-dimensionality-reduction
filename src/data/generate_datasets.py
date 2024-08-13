from pathlib import Path

import torch

from src.data.databases.diffusion_1d_database import (
    Diffusion1DDatabase,
    Diffusion1DParamSet,
)
from src.data.databases.advection_1d_database import (
    Advection1DAnalyticDatabase,
    Advection1DParamSet,
)
from src.data.databases.pde_database import Config
from src.data.databases.reaction_diffusion_2d_database import (
    ReactionDiffusion2DParamSet,
    ReactionDiffusion2DDatabase,
)
from src.data.initial_conditions.gaussian import (
    Gaussian,
    GaussianParamSet,
)
from src.data.initial_conditions.spiral import SpiralParamSet, Spiral


BASE_DIFFUSION_CONFIG = Config(
    Nt=5001,
    T=1,
    pde_params=None,
    initial_condition_params=GaussianParamSet(mu=torch.tensor([0]), sigma=torch.tensor([0.02])),
    xmin=-1,
    xmax=1,
    Nx=101,
)

BASE_ADVECTION_CONFIG = Config(
    Nt=200,
    T=1,
    pde_params=None,
    initial_condition_params=GaussianParamSet(mu=torch.tensor([0.2]), sigma=torch.tensor([1e-3])),
    xmin=0,
    xmax=2,
    Nx=256,
)

BASE_REACTION_DIFFUSION_CONFIG = Config(
    Nt=50_000,
    T=50,
    pde_params=None,
    initial_condition_params=SpiralParamSet(),
    xmin=-10,
    xmax=10,
    Nx=100,
    ymin=-10,
    ymax=10,
    Ny=100,
)


def generate_diffusion_dataset(parameters, base_storage_path=Path("data/diffusion"), return_stacked=False):
    databases = []
    for diff in parameters:
        config_ = BASE_DIFFUSION_CONFIG
        config_.pde_params = Diffusion1DParamSet(diffusion=diff)

        database = Diffusion1DDatabase(
            initial_condition=Gaussian,
            config=config_,
            storage_path=base_storage_path / str(diff),
        )
        database.generate()

        databases.append(database)

    # The [0:-1:20] is there to subsample every 20th point
    stacked_datasets = torch.vstack([database.data[0:-1:20, :] for database in databases])
    if return_stacked:
        return stacked_datasets

    return torch.utils.data.TensorDataset(stacked_datasets)


def generate_advection_dataset(parameters, base_storage_path=Path("data/advection"), return_stacked=False):
    databases = []
    for adv in parameters:
        config_ = BASE_ADVECTION_CONFIG
        config_.pde_params = Advection1DParamSet(advection=adv)

        database = Advection1DAnalyticDatabase(
            initial_condition=Gaussian,
            config=config_,
            storage_path=base_storage_path / str(adv),
        )
        database.generate(force=True)

        databases.append(database)

    stacked_datasets = torch.vstack([database.data for database in databases])
    if return_stacked:
        return stacked_datasets

    return torch.utils.data.TensorDataset(stacked_datasets)


def generate_reaction_diffusion_dataset(
    base_storage_path=Path("data/reaction-diffusion"),
    return_database=False,
    u_only=True,
):
    config_ = BASE_REACTION_DIFFUSION_CONFIG
    config_.pde_params = ReactionDiffusion2DParamSet(
        diffusion_u=torch.tensor(0.1),
        diffusion_v=torch.tensor(0.1),
        coupling=torch.tensor(1),
    )

    database = ReactionDiffusion2DDatabase(
        initial_condition=Spiral,
        config=config_,
        storage_path=base_storage_path / str(1),
    )
    database.generate()

    if return_database:
        return database

    if u_only:
        return torch.utils.data.TensorDataset(database.data[:, 0, :, :].view(database.config.Nt, -1))

    return torch.utils.data.TensorDataset(database.data.view(database.config.Nt, -1))
