from collections.abc import Callable

from util.config import get_many_random_hyperparams
from util.data import Dataset
from util.experiment import run_experiment


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    N_RANDOM = 30
    configs = get_many_random_hyperparams(config_space, N_RANDOM)

    scores = []
    for config in configs:
        try:
            scores.append(eval(config))
        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    return max(scores)


if __name__ == "__main__":
    run_experiment(opt_fn)
