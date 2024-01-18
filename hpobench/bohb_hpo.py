from typing import Callable

from bohb import BOHB
import bohb.configspace as bohb_space

from util.data import Dataset
from util.experiment import run_experiment


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    # Convert the space to bohb format
    space = []
    for key, val in config_space.items():
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
    return score


if __name__ == '__main__':
    run_experiment(opt_fn)
