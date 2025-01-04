import torch
import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return next(module.buffers()).device


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def melt_model(model):
    for param in model.parameters():
        param.requires_grad = True
