import traceback

from hyperopt import fmin, hp, tpe

from image.src.config import hpo_space
from image.src.data import Dataset
from image.src.util import get_data, run_experiment

if __name__ == "__main__":
    DATASET = "cifar10"
    N_CLASSES = 10

    data: Dataset = get_data(DATASET)

    def evaluate_hyperopt(config):
        try:
            acc = float(run_experiment(data, config, N_CLASSES, DATASET))
            # fmin minimizes the objective
            return 1.0 - acc
        except Exception:
            traceback.print_exc()
            return 100.0

    # Convert the space to hyperopt format
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

    best = fmin(evaluate_hyperopt, space, algo=tpe.suggest, max_evals=30)
    for key, val in hpo_space.items():
        if isinstance(val, list):
            best[key] = val[best[key]]
    
    try:
        score = float(run_experiment(data, best, N_CLASSES, DATASET))
        print(f"Best config: {best}")
        print(f"Best score: {score}")
    except Exception:
        traceback.print_exc()

