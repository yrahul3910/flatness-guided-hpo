import traceback

import numpy as np
from turbo import Turbo1

from image.src.config import hpo_space
from image.src.data import Dataset
from image.src.util import get_data, run_experiment

if __name__ == "__main__":
    DATASET = "cifar10"
    N_CLASSES = 10

    data: Dataset = get_data(DATASET)

    def eval_wrapper(config):
        cfg = {}
        j = 0  # index that we access config
        for _i, (key, val) in enumerate(hpo_space.items()):
            if isinstance(val, tuple):
                if isinstance(val[0], float):
                    cfg[key] = config[j]
                else:
                    cfg[key] = round(config[j])
                j += 1
            else:
                cfg[key] = np.random.choice(val)

        try:
            acc = float(run_experiment(data, cfg, N_CLASSES, DATASET))
            # Turbo1 minimizes f
            return 1.0 - acc
        except Exception:
            traceback.print_exc()
            return 100.0


    lb = []
    ub = []
    for val in hpo_space.values():
        if isinstance(val, tuple):
            lb.append(val[0])
            ub.append(val[1])

    turbo1 = Turbo1(
        f=eval_wrapper,
        lb=np.array(lb),
        ub=np.array(ub),
        n_init=2 * len(lb) + 1,
        max_evals=30,
        verbose=True,
    )
    turbo1.optimize()
