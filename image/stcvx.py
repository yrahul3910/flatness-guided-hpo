import os
import traceback
import random
import gc

from copy import deepcopy
from src.util import get_many_random_hyperparams, get_convexity, run_experiment, get_data
from common import hpo_space


file_number = os.getenv('SLURM_JOB_ID') or random.randint(1, 10000)

data_orig = get_data('svhn')

# Run actual experiment
best_betas = []
best_configs = []
keep_configs = 5
num_configs = 30

configs = get_many_random_hyperparams(hpo_space, num_configs)

for i, config in enumerate(configs):
    try:
        data = deepcopy(data_orig)
        print(f'[main] ({i}/{num_configs}) Computing convexity for config:', config)
        convexity = get_convexity(data, config, dataset='svhn')

        if len(best_betas) < keep_configs or convexity < max(best_betas):
            best_betas.append(convexity)
            best_configs.append(config)

            best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs), key=lambda x: x[0]))
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])

        gc.collect()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f'Error, skipping config: {e}')
        traceback.print_exc()
    
for beta, config in zip(best_betas, best_configs):
    data = deepcopy(data_orig)
    print(f'Config: {config}\mu: {beta}')
    print('[main] Accuracy:', run_experiment(data, config, 10))
