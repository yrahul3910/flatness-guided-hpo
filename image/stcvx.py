import gc
import os
import random
import traceback
from copy import deepcopy

from common import hpo_space
from src.util import get_convexity, get_data, get_many_random_hyperparams

file_number = os.getenv("SLURM_JOB_ID") or random.randint(1, 10000)

data_orig = get_data("svhn")

# Run actual experiment
best_betas = []
best_configs = []
keep_configs = 5
num_configs = 30

configs = get_many_random_hyperparams(hpo_space, num_configs)

for _i, config in enumerate(configs):
    try:
        data = deepcopy(data_orig)
        convexity = get_convexity(data, config, dataset="svhn")

        if len(best_betas) < keep_configs or convexity < max(best_betas):
            best_betas.append(convexity)
            best_configs.append(config)

            best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs, strict=False), key=lambda x: x[0]), strict=False)
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])

        gc.collect()
    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()

for _beta, config in zip(best_betas, best_configs, strict=False):
    data = deepcopy(data_orig)
