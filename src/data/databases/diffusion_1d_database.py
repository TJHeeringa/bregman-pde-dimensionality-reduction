from dataclasses import dataclass

import torch

from .pde_database import PdeDatabase


@dataclass
class Diffusion1DParamSet:
    diffusion: float


class Diffusion1DDatabase(PdeDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.config.pde_params.__class__ is self.parameter_set_class()
        self.domain = torch.linspace(self.config.xmin, self.config.xmax, self.config.Nx)
        self.diffusion = self.config.pde_params.diffusion
        self.c = self.config.pde_params.diffusion * self.dt / self.config.dx**2
        self.c_tilde = 1 - 2 * self.c
        if self.c >= 0.5:
            print(
                f"Warning: c value is {self.c}, which is bigger than 0.5. "
                f"This means the solution will be unstable."
            )

    def parameter_set(self):
        return Diffusion1DParamSet(diffusion=self.diffusion)

    @staticmethod
    def parameter_set_class():
        return Diffusion1DParamSet

    def initialize(self):
        self.data = torch.zeros((self.config.Nt, self.config.Nx))
        self.data[0, :] = self.initial_condition(self.config.initial_condition_params)(
            self.domain
        )

    def forward(self, n: int, t: float):
        """
        Args:
            n: current iteration
            t: current time

        Returns:
        """
        self.data[n, 1:-1] = self.c_tilde * self.data[n - 1, 1:-1] + self.c * (
            self.data[n - 1, 0:-2] + self.data[n - 1, 2:]
        )
