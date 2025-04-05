from typing import Dict

from torch import optim


def config_torch_optimizer(model, config: Dict = None):
    optimizer_name = config.get("name", "Adam")
    optimizer_params = config.get("params", {"lr": 0.01})

    # Dynamically get the optimizer class by name from torch.optim
    optimizer_class = getattr(optim, optimizer_name)

    return optimizer_class(model.parameters(), **optimizer_params)
