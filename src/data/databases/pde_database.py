from typing import Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import json

from os import PathLike

import torch


class TorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class Config:
    T: float
    Nt: int
    dt: float = field(init=False)
    pde_params: Any
    initial_condition_params: Any
    Nd: int = field(init=False)
    xmin: float
    xmax: float
    Nx: int
    dx: float = field(init=False)
    ymin: float | None = None
    ymax: float | None = None
    Ny: int | None = None
    dy: float = field(init=False, default=None)

    def __post_init__(self):
        assert self.Nt > 1
        self.dt = self.T / (self.Nt - 1)

        assert self.xmin <= self.xmax
        assert self.Nx > 1
        self.dx = (self.xmax - self.xmin) / (self.Nx - 1)
        self.Nd = 1

        if self.ymin is not None or self.ymax is not None or self.Ny is not None:
            assert (
                self.ymin is not None and self.ymax is not None and self.Ny is not None
            )
            assert self.ymin <= self.ymax
            assert self.Ny > 1
            self.dy = (self.ymax - self.ymin) / (self.Ny - 1)
            self.Nd = 2

    def to_json(self) -> str:
        return json.dumps(
            asdict(
                self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None}
            ),
            cls=TorchEncoder,
            indent=4,
        )

    @classmethod
    def from_json(cls, json_):
        return cls(
            **{
                k: v
                for (k, v) in json.loads(json_).items()
                if k not in ["Nd", "dt", "dx", "dy"]
            }
        )


class PdeDatabase:
    def __init__(
        self, initial_condition, config: Config, storage_path: str | PathLike[str]
    ):
        self.initial_condition = initial_condition
        self._config = config
        self.storage_path = storage_path
        self.data = None
        self.dt = self.config.T / (self.config.Nt - 1)
        self.time = torch.linspace(0, self.config.T, self.config.Nt)

    def parameter_set(self):
        pass

    @staticmethod
    def parameter_set_class():
        pass

    @classmethod
    def load(cls, initial_condition, storage_path: str | PathLike[str]):
        storage_folder = Path(storage_path)
        dataset = PdeDatabase(
            initial_condition=initial_condition,
            config=Config.from_json((storage_folder / "config.json").read_text()),
            storage_path=storage_path,
        )
        dataset.data = torch.load(storage_folder / "data.pt")
        dataset.time = torch.load(storage_folder / "time.pt")
        return dataset

    def save(self):
        storage_folder = Path(self.storage_path)
        storage_folder.mkdir(parents=True, exist_ok=True)

        (storage_folder / "config.json").write_text(self.config.to_json())
        torch.save(self.data, storage_folder / "data.pt")
        torch.save(self.time, storage_folder / "time.pt")

    def generate(self, force=False, save_post_generation=True):
        if not force and self.has_been_generated_before():
            self.data = torch.load(Path(self.storage_path) / "data.pt")
            return

        self.initialize()

        for n, t in enumerate(self.time[1:]):
            self.forward(n + 1, t)

        if save_post_generation:
            self.save()

    def has_been_generated_before(self) -> (bool, bool | None):
        config_file = Path(self.storage_path) / "config.json"
        if not config_file.exists():
            return False
        config = Config.from_json(config_file.read_text())
        return config == self.config

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_config: Config):
        self._config = new_config

    def __len__(self):
        return len(self.data)

    def initialize(self):
        pass

    def forward(self, iteration, current_time):
        pass
