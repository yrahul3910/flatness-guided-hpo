from collections.abc import Callable

from hyperopt import fmin, hp, tpe
from util.config import Config
from util.data import Dataset
from util.experiment import run_experiment


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    space = {}
    for key, val in config_space.items():
        if isinstance(val, list):
            space[key] = hp.choice(key, val)
        elif isinstance(val, tuple):
            if isinstance(val[0], int):
                space[key] = hp.randint(key, *val)
            else:
                space[key] = hp.uniform(key, *val)
        else:
            msg = "Space must be a list or tuple"
            raise ValueError(msg)

    def eval_fn(config: dict):
        # Importantly, eval returns something to *maximize*
        return [-x for x in eval(Config(**config))]

    def eval_fn_auc(config: dict):
        return eval_fn(config)[-1]

    best = fmin(eval_fn_auc, space, algo=tpe.suggest, max_evals=30)
    for key, val in config_space.items():
        if isinstance(val, list):
            best[key] = val[best[key]]
    return eval_fn(best)



if __name__ == "__main__":
    run_experiment(opt_fn)
