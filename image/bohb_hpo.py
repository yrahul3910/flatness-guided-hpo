from copy import deepcopy

import bohb.configspace as bohb_space
from bohb import BOHB
from common import eval, hpo_space
from src.util import get_data, get_model

if __name__ == "__main__":
    data = get_data()

    def model_fn(config):
        return get_model(deepcopy(data), config, 10)
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
            msg = f"Key {key} must be a list or tuple"
            raise ValueError(msg)

    opt = BOHB(configspace=bohb_space.ConfigurationSpace(space), evaluate=eval, min_budget=1, max_budget=30)
    logs = opt.optimize()
    config = logs.best["hyperparameter"].to_dict()
    score = eval(config)

