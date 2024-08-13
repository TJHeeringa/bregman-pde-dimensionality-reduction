import torch


def get_bias(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
            if not (m.bias is None):
                yield m.bias
        else:
            continue
