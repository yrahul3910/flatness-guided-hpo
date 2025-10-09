import traceback
from collections.abc import Callable

from util.config import get_many_random_hyperparams
from util.data import Dataset
from util.experiment import run_experiment
from util.stcvx import get_convexity


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    keep_configs = 10
    best_betas = []
    best_configs = []

    configs = get_many_random_hyperparams(config_space, 50)

    for config in configs:
        try:
            convexity = get_convexity(data, config)

            if (len(best_betas) < keep_configs or convexity < max(best_betas)) and convexity > 0:
                best_betas.append(convexity)
                best_configs.append(config)

                best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs, strict=False), key=lambda x: x[0]), strict=False)
                best_betas = list(best_betas[:keep_configs])
                best_configs = list(best_configs[:keep_configs])
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()

    scores = []
    for _beta, config in zip(best_betas, best_configs, strict=False):

        score = eval(config)
        scores.append(score)

    return max(scores)


if __name__ == "__main__":
    run_experiment(opt_fn)
