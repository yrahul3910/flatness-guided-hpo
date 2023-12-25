from dataclasses import dataclass
from typing import Literal
import random


@dataclass
class Config:
    depth: int
    width: int
    batch_size: int
    alpha: float
    learning_rate_init: float
    

config_space = {
    "depth": (1, 4),
    "width": [16, 32, 64, 128, 256, 512, 1024],
    "batch_size": [4, 8, 16, 32, 64, 128, 256],
    "alpha": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.],
    "learning_rate_init": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.]
}

def get_random_hyperparams(options: dict) -> Config:
    """
    Get hyperparameters from options.
    """
    hyperparams = {}
    for key, value in options.items():
        if isinstance(value, list):
            hyperparams[key] = random.choice(value)
        elif isinstance(value, tuple):
            if isinstance(value[0], int):
                hyperparams[key] = random.randint(value[0], value[1])
            else:
                hyperparams[key] = random.uniform(value[0], value[1])
    return Config(**hyperparams)


def get_many_random_hyperparams(options: dict, n: int) -> list:
    """
    Get n hyperparameters from options.
    """
    hyperparams = []
    for _ in range(n):
        hyperparams.append(get_random_hyperparams(options))
    return hyperparams