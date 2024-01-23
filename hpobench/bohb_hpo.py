from typing import Callable

from bohb import BOHB
import bohb.configspace as bohb_space

from util.config import Config
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
    
    def eval_fn(config: dict):
        # Importantly, eval returns something to *maximize*
        return [-x for x in eval(Config(**config))]
    
    def eval_fn_auc(config: dict, args=None):
        return eval_fn(config)[-1]

    opt = BOHB(configspace=bohb_space.ConfigurationSpace(space), evaluate=eval_fn_auc, min_budget=1, max_budget=30)
    logs = opt.optimize()
    config = logs.best["hyperparameter"].to_dict()
    score = eval(Config(**config))

    print(f'[main] Accuracy: {score}')
    return score


if __name__ == '__main__':
    run_experiment(opt_fn)
