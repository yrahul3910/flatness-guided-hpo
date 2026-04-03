import traceback

import bohb.configspace as bohb_space
from bohb import BOHB

from image.src.config import hpo_space
from image.src.data import Dataset
from image.src.util import get_data, run_experiment

if __name__ == "__main__":
    DATASET = "cifar10"
    N_CLASSES = 10

    data: Dataset = get_data(DATASET)

    def evaluate_bohb(config, budget=100, *args, **kwargs):
        try:
            acc = float(run_experiment(data, config, N_CLASSES, DATASET, epochs=int(budget)))
            # BOHB minimizes, so we return negative accuracy or 1.0 - acc
            return 1.0 - acc
        except Exception:
            traceback.print_exc()
            return 100.0

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

    opt = BOHB(
        configspace=bohb_space.ConfigurationSpace(space),
        evaluate=evaluate_bohb,
        min_budget=1,
        max_budget=30,
    )
    logs = opt.optimize()
    config = logs.best["hyperparameter"].to_dict()
    
    try:
        score = float(run_experiment(data, config, N_CLASSES, DATASET))
        print(f"Best config: {config}")
        print(f"Best score: {score}")
    except Exception:
        traceback.print_exc()
