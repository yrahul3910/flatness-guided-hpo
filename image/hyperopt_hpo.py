from hyperopt import hp, fmin, tpe
from copy import deepcopy

from src.util import get_data, get_model
from common import hpo_space, eval


if __name__ == '__main__':
    data = get_data()

    model_fn = lambda config: get_model(deepcopy(data), config, 10)
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
            raise ValueError("Space must be a list or tuple")

    best = fmin(eval, space, algo=tpe.suggest, max_evals=30)
    for key, val in hpo_space.items():
        if isinstance(val, list):
            best[key] = val[best[key]]
    score = eval(best)

    print(f'[main] Accuracy: {score}')
