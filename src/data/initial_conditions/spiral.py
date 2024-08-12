from dataclasses import dataclass

import torch


@dataclass
class SpiralParamSet:
    """Parameters for the Spiral class."""

    pass


class Spiral:
    def __init__(self, param_set: SpiralParamSet):
        pass

    def __call__(self, meshgrid_x, meshgrid_y):
        angles = torch.atan2(meshgrid_y, meshgrid_x)
        radii = torch.sqrt(meshgrid_x**2 + meshgrid_y**2)
        u_0 = torch.tanh(radii * torch.cos(angles - radii))
        v_0 = torch.tanh(radii * torch.sin(angles - radii))
        return torch.stack([u_0, v_0])
