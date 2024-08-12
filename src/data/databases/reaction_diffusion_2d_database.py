from dataclasses import dataclass

import torch

from .pde_database import PdeDatabase


@dataclass
class ReactionDiffusion2DParamSet:
    """Parameters for the ReactionDiffusion2D class.

    :param diffusion_u: diffusion for u
    :param diffusion_u: diffusion for y
    :param coupling: coupling strength between u and v
    """

    diffusion_u: torch.Tensor
    diffusion_v: torch.Tensor
    coupling: torch.Tensor


class ReactionDiffusion2DDatabase(PdeDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_axis = torch.linspace(self.config.xmin, self.config.xmax, self.config.Nx)
        self.y_axis = torch.linspace(self.config.ymin, self.config.ymax, self.config.Ny)
        self.meshgrid_x, self.meshgrid_y = torch.meshgrid(self.x_axis, self.y_axis)

    def initialize(self):
        self.data = torch.zeros((self.config.Nt, 2, self.config.Nx, self.config.Ny))
        self.data[0, :] = self.initial_condition(self.config.initial_condition_params)(
            self.meshgrid_x, self.meshgrid_y
        )

    def _laplacian2d_b(self, f):
        lap = torch.zeros_like(f)

        lap[1:-1, 1:-1] = (
            f[0:-2, 1:-1] - 2 * f[1:-1, 1:-1] + f[2:, 1:-1]
        ) / self.config.dx**2 + (
            f[1:-1, 0:-2] - 2 * f[1:-1, 1:-1] + f[1:-1, 2:]
        ) / self.config.dy**2

        # Neumann boundary conditions
        lap[0, :] = lap[1, :]
        lap[-1, :] = lap[-2, :]
        lap[:, 0] = lap[:, 1]
        lap[:, -1] = lap[:, -2]

        return lap

    def _laplacian2d(self, f):
        laplacian_f_y = (
            torch.diff(
                f,
                n=2,
                dim=0,
                # prepend=torch.zeros_like(f[0, :]).reshape(1, -1),
                # append=torch.zeros_like(f[-1, :]).reshape(1, -1)
                prepend=f[0, :].reshape(1, -1),
                append=f[-1, :].reshape(1, -1),
            )
            / (self.config.dy**2)
        )
        laplacian_f_x = (
            torch.diff(
                f,
                n=2,
                dim=1,
                # prepend=torch.zeros_like(f[:, 0]).reshape(-1, 1),
                # append=torch.zeros_like(f[:, -1]).reshape(-1, 1)
                prepend=f[:, 0].reshape(-1, 1),
                append=f[:, -1].reshape(-1, 1),
            )
            / (self.config.dx**2)
        )
        return laplacian_f_x + laplacian_f_y

    def forward(self, n: int, t: float):
        """
        Args:
            n: current iteration
            t: current time

        Returns:
        """
        u = self.data[n - 1, 0]
        v = self.data[n - 1, 1]

        # Precompute repeated terms
        u2v2 = u**2 + v**2

        # Update u and v using the given equations
        u_new = u + self.dt * (
            self.config.pde_params.diffusion_u * self._laplacian2d(u)
            + (1 - u2v2) * u
            + self.config.pde_params.coupling * u2v2 * v
        )
        v_new = v + self.dt * (
            self.config.pde_params.diffusion_v * self._laplacian2d(v)
            - self.config.pde_params.coupling * u2v2 * u
            + (1 - u2v2) * v
        )

        # Update the data for the next time step
        self.data[n, 0] = u_new
        self.data[n, 1] = v_new
