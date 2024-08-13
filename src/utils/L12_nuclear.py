import typing

from bregman import L12, Nuclear, Null

from .get_bias import get_bias

if typing.TYPE_CHECKING:
    from bregman import AutoEncoder


def L12_nuclear(model: "AutoEncoder", rc1, rc2=None):
    if rc2 is None:
        rc2 = rc1

    preset = [
        {"params": get_bias(model), "reg": Null()},
        {"params": model.encoder[-1].weight, "reg": Nuclear(rc=rc2)},
    ]

    for i in range(len(model.encoder_layers) - 2):
        preset.append({"params": model.encoder[2 * i].weight, "reg": L12(rc=rc1)})

    for i in range(len(model.decoder_layers) - 1):
        preset.append({"params": model.decoder[2 * i].weight, "reg": L12(rc=rc1)})

    return preset
