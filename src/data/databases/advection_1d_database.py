import copy
from dataclasses import dataclass

import torch

from .pde_database import PdeDatabase


@dataclass
class Advection1DParamSet:
    advection: torch.Tensor


class Advection1DDatabase(PdeDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = torch.linspace(
            self.config.xmin,
            self.config.xmax - self.config.dx,
            self.config.Nx,
        )
        self.cfl = torch.abs(self.config.pde_params.advection) * self.config.dt / self.config.dx
        self.direction = torch.sgn(self.config.pde_params.advection)
        self.a = 1 - self.direction * self.cfl**2
        self.b = self.direction * self.cfl / 2 + self.cfl**2 / 2
        self.c = -self.direction * self.cfl / 2 + self.cfl**2 / 2

    def initialize(self):
        self.data = torch.zeros((self.config.Nt, self.config.Nx))
        self.data[0, :] = self.initial_condition(self.config.initial_condition_params)(self.domain)

    def forward(self, n: int, t: float):
        """
        Args:
            n: current iteration
            t: current time

        Returns:
        """
        f_n = self.data[n - 1, :]
        f_plus = torch.roll(f_n, -1)
        f_minus = torch.roll(f_n, 1)
        self.data[n, :] = self.a * f_n + self.b * f_plus + self.c * f_minus


class Advection1DAnalyticDatabase(PdeDatabase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain = torch.linspace(
            self.config.xmin,
            self.config.xmax - self.config.dx,
            self.config.Nx,
        )
        self.shift_per_time = self.config.pde_params.advection * self.config.dt / self.config.dx

    def initialize(self):
        self.data = torch.zeros((self.config.Nt, self.config.Nx))
        self.data[0, :] = self.initial_condition(self.config.initial_condition_params)(self.domain)

    def forward(self, n: int, t: float):
        """
        Args:
            n: current iteration
            t: current time

        Returns:
        """
        wave_params = copy.deepcopy(self.config.initial_condition_params)
        if n <= 10:
            print(wave_params, self.config.initial_condition_params, n * self.config.pde_params.advection * self.config.dt)
        wave_params.mu += n * self.config.pde_params.advection * self.config.dt
        if n <= 10:
            print(wave_params)
        self.data[n, :] = self.initial_condition(wave_params)(self.domain)
