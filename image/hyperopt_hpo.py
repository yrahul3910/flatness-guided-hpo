from copy import deepcopy

from common import eval, hpo_space
from hyperopt import fmin, hp, tpe
from src.util import get_data, get_model

if __name__ == "__main__":
    data = get_data()

    def model_fn(config):
        return get_model(deepcopy(data), config, 10)
    # Convert the space to bohb format
    space = {}
    for key, val in hpo_space.items():
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

    best = fmin(eval, space, algo=tpe.suggest, max_evals=30)
    for key, val in hpo_space.items():
        if isinstance(val, list):
            best[key] = val[best[key]]
    score = eval(best)

