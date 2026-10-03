"""Run STCVX with validation selection and one held-out test evaluation."""

import os
import traceback

import numpy as np
from keras.utils import set_random_seed

from image.src.config import Config, hpo_space
from image.src.experiment import SearchRun
from image.src.util import get_convexity, get_random_hyperparams

if __name__ == "__main__":
    with SearchRun("stcvx") as run:
        best_configs: list[tuple[float, Config]] = []
        keep_configs = 10
        num_configs = 100
        min_convexity = 0.1
        max_attempts = int(os.getenv("FGHPO_MAX_PROBE_ATTEMPTS", "1000"))
        valid_count = 0

        for attempt in range(1, max_attempts + 1):
            probe_seed = run.seed + attempt
            set_random_seed(probe_seed)
            config = get_random_hyperparams(hpo_space)
            set_random_seed(probe_seed)

            try:
                convexity = get_convexity(run.data, config, n_class=run.n_classes, dataset=run.dataset)
            except (ValueError, RuntimeError) as error:
                run.record_proxy(config, float("nan"), seed=probe_seed, error=repr(error))
                traceback.print_exc()
                continue

            run.record_proxy(config, convexity, seed=probe_seed)
            if not np.isfinite(convexity) or convexity <= min_convexity:
                continue

            valid_count += 1
            best_configs.append((convexity, config))
            best_configs.sort(key=lambda pair: pair[0])
            del best_configs[keep_configs:]
            if valid_count == num_configs:
                break

        if valid_count < num_configs:
            msg = f"Only {valid_count}/{num_configs} eligible probes completed within {max_attempts} attempts"
            raise RuntimeError(msg)

        for _, config in best_configs:
            try:
                run.evaluate(config)
            except (ValueError, RuntimeError):
                traceback.print_exc()

        run.finish()
