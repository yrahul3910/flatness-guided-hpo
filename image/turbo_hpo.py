"""Run TuRBO with validation selection and one held-out test evaluation.

Integer ranges are rounded; listed choices use equal-width ordinal bins in a continuous box. This adapter is not a
categorical TuRBO kernel.
"""

import traceback

import numpy as np
import torch
from turbo import Turbo1

from image.src.config import Config, hpo_space
from image.src.experiment import SearchRun

if __name__ == "__main__":
    with SearchRun("turbo") as run:
        torch.manual_seed(run.seed)

        def eval_wrapper(config: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
            cfg = {}

            for j, (key, val) in enumerate(hpo_space.items()):
                if isinstance(val, tuple):
                    if isinstance(val[0], float):
                        cfg[key] = float(config[j])
                    else:
                        cfg[key] = round(config[j])
                elif isinstance(val, list):
                    cfg[key] = val[min(int(config[j]), len(val) - 1)]
                else:
                    msg = f"Key {key} must be a list or tuple"
                    raise TypeError(msg)

            # SearchRun reseeds NumPy for model training; preserve TuRBO's proposal stream across objective calls.
            optimizer_rng_state = np.random.get_state()  # noqa: NPY002 - upstream TuRBO uses NumPy's legacy global RNG.

            try:
                return 1.0 - run.evaluate(Config(**cfg))
            except (ValueError, RuntimeError):
                traceback.print_exc()
                return 100.0
            finally:
                np.random.set_state(optimizer_rng_state)  # noqa: NPY002 - restore the upstream optimizer's global RNG.

        lb = []
        ub = []

        for val in hpo_space.values():
            if isinstance(val, tuple):
                lb.append(val[0])
                ub.append(val[1])
            elif isinstance(val, list):
                lb.append(0)
                ub.append(len(val))

        turbo1 = Turbo1(
            f=eval_wrapper,
            lb=np.array(lb),
            ub=np.array(ub),
            n_init=2 * len(lb) + 1,
            max_evals=30,
            batch_size=1,
            verbose=True,
        )
        turbo1.optimize()
        run.finish()
