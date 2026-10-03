"""Run Hyperopt with validation selection and one held-out test evaluation."""

import traceback

import numpy as np
from hyperopt import fmin, hp, tpe

from image.src.config import Config, hpo_space
from image.src.experiment import SearchRun

if __name__ == "__main__":
    with SearchRun("hyperopt") as run:
        space = {}

        def evaluate_hyperopt(config: Config) -> float:
            try:
                return 1.0 - run.evaluate(config)
            except (ValueError, RuntimeError):
                traceback.print_exc()
                return 100.0

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
                raise TypeError(msg)

        fmin(evaluate_hyperopt, space, algo=tpe.suggest, max_evals=30, rstate=np.random.default_rng(run.seed))
        run.finish()
