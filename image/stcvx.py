import gc
import json
import os
import random
import traceback
from copy import deepcopy

from image.src.config import Config, hpo_space
from image.src.data import Dataset
from image.src.util import (
    get_convexity,
    get_data,
    get_many_random_hyperparams,
    run_experiment,
)

file_number = os.getenv("SLURM_JOB_ID") or random.randint(1, 10000)

DATASET = "cifar10"
N_CLASSES = 10

data_orig: Dataset = get_data(DATASET)

# Run actual experiment
best_betas: list[float] = []
best_configs: list[Config] = []
keep_configs = 5
num_configs = 30

configs: list[Config] = get_many_random_hyperparams(hpo_space, num_configs)

for _i, config in enumerate(configs):
    try:
        data = deepcopy(data_orig)
        convexity = get_convexity(data, config, dataset=DATASET)

        if len(best_betas) < keep_configs or convexity < max(best_betas):
            best_betas.append(convexity)
            best_configs.append(config)

            best_betas, best_configs = zip(
                *sorted(
                    zip(best_betas, best_configs, strict=False), key=lambda x: x[0]
                ),
                strict=False,
            )
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])

        _ = gc.collect()
    except KeyboardInterrupt:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()

for beta, config in zip(best_betas, best_configs, strict=False):
    data: Dataset = deepcopy(data_orig)
    acc = float(run_experiment(data, config, N_CLASSES, DATASET))
    print(json.dumps({"beta": float(beta), **config, "accuracy": acc}))  # noqa: T201
