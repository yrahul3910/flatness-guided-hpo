from typing import Callable

from util.data import Dataset
from util.experiment import run_experiment
from util.config import get_many_random_hyperparams


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    N_RANDOM = 30
    configs = get_many_random_hyperparams(config_space, N_RANDOM)

    scores = []
    for config in configs:
        try:
            print(f'Config: {config}')
            scores.append(eval(config))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'[main] Error, skipping config: {e}')
            continue

    print(scores)
    print('[main] Accuracy:', max(scores))
    return max(scores)


if __name__ == '__main__':
    run_experiment(opt_fn)
