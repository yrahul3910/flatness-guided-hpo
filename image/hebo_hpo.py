import traceback

import numpy as np
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from image.src.config import hpo_space
from image.src.data import Dataset
from image.src.util import get_data, run_experiment

if __name__ == "__main__":
    DATASET = "cifar10"
    N_CLASSES = 10

    data: Dataset = get_data(DATASET)

    # Convert hpo_space to HEBO DesignSpace format
    config_space = []
    for key, val in hpo_space.items():
        config = {}
        config["name"] = key
        if isinstance(val, list):
            config["type"] = "cat"
            config["categories"] = val
        elif isinstance(val, tuple):
            if isinstance(val[0], int):
                config["type"] = "int"
            else:
                config["type"] = "num"
            config["lb"] = val[0]
            config["ub"] = val[1]
        else:
            msg = f"Key {key} must be a list or tuple"
            raise ValueError(msg)

        config_space.append(config)

    config_space_parsed = DesignSpace().parse(config_space)
    hebo = HEBO(config_space_parsed)

    def evaluate_hebo(config):
        try:
            acc = float(run_experiment(data, config, N_CLASSES, DATASET))
            # HEBO minimizes, so we return 1.0 - acc
            return 1.0 - acc
        except Exception:
            traceback.print_exc()
            return 100.0

    # Use HEBO to perform hyper-parameter optimization for 30 iterations.
    scores = []
    for _ in range(5):
        configs = hebo.suggest(n_suggestions=6)
        
        y = []
        for _, config in configs.iterrows():
            cfg_dict = config.to_dict()
            y.append(evaluate_hebo(cfg_dict))
            
        hebo.observe(configs, np.array(y).reshape(-1, 1))

    scores.append(1.0 - hebo.best_y)
    print(f"Best config: {hebo.best_x}")
    print(f"Best score: {scores[-1]}")
