import json
import os
import random
import traceback

import numpy as np

from image.src.config import Config, hpo_space
from image.src.data import Dataset
from image.src.util import (
    get_convexity,
    get_data,
    get_random_hyperparams,
    run_experiment,
)

# Identify this run for persistence
file_number = os.getenv("SLURM_JOB_ID") or random.randint(1, 10000)
results_log = f"stcvx_search_{file_number}.csv"

DATASET = "cifar100"
N_CLASSES = 100

data: Dataset = get_data(DATASET)

# Run HPO search
best_betas: list[float] = []
best_configs: list[Config] = []
keep_configs = 10
num_configs = 100

# Write CSV header
with open(results_log, "w") as f:
    f.write("valid_count,mu," + ",".join(hpo_space.keys()) + "\n")

valid_count = 0
while valid_count < num_configs:
    try:
        config = get_random_hyperparams(hpo_space)
        convexity = get_convexity(data, config, n_class=N_CLASSES, dataset=DATASET)

        if not np.isfinite(convexity) or convexity <= 0:
            continue

        valid_count += 1
        print(f"Config {valid_count}/{num_configs}: mu={convexity:.4f}")

        # Log all explored configs
        with open(results_log, "a") as f:
            f.write(f"{valid_count},{convexity}," + ",".join(str(config.get(k, "")) for k in hpo_space.keys()) + "\n")

        if len(best_betas) < keep_configs or (convexity < max(best_betas) and convexity > 0.1):
            best_betas.append(convexity)
            best_configs.append(config)

            # Keep top K
            sorted_pairs = sorted(zip(best_betas, best_configs, strict=False), key=lambda x: x[0])
            best_betas, best_configs = zip(*sorted_pairs, strict=False)
            best_betas = list(best_betas[:keep_configs])
            best_configs = list(best_configs[:keep_configs])

    except KeyboardInterrupt:
        raise
    except Exception:
        traceback.print_exc()
        continue

print(f"\nEvaluating top {keep_configs} configurations...")
for beta, config in zip(best_betas, best_configs, strict=False):
    try:
        acc = float(run_experiment(data, config, N_CLASSES, DATASET))
        # This specific print format is REQUIRED by image/parsers/parse.py
        print(f"Accuracy: {acc}")
        print(json.dumps({"beta": float(beta), **config, "accuracy": acc}))
    except Exception:
        traceback.print_exc()

