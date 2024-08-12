import math

import torch


def init_linear(module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
        torch.nn.init.uniform_(module.bias, 0, 1 / math.sqrt(fan_in))
