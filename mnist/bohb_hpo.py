from bohb import BOHB
import bohb.configspace as bohb_space
from copy import deepcopy
import numpy as np

from src.util import get_data, get_model
from common import hpo_space, eval


if __name__ == '__main__':
    data = get_data()

    model_fn = lambda config: get_model(deepcopy(data), config, 10)
    # Convert the space to bohb format
    space = []
    for key, val in hpo_space.items():
        if isinstance(val, list):
            space.append(bohb_space.CategoricalHyperparameter(key, val))
        elif isinstance(val, tuple):
            if isinstance(val[0], int):
                space.append(bohb_space.IntegerUniformHyperparameter(key, *val))
            else:
                space.append(bohb_space.UniformHyperparameter(key, *val))
        else:
            raise ValueError(f"Key {key} must be a list or tuple")

    opt = BOHB(configspace=bohb_space.ConfigurationSpace(space), evaluate=eval, min_budget=1, max_budget=30)
    logs = opt.optimize()
    config = logs.best["hyperparameter"].to_dict()
    score = eval(config)

    print(f'[main] Accuracy: {score}')
