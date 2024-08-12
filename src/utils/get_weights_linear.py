import torch


def get_weights_linear(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            yield m.weight
        else:
            continue
