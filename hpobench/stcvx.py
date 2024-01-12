from typing import Callable

from util.config import get_many_random_hyperparams
from util.data import Dataset
from util.stcvx import get_convexity
from util.experiment import run_experiment


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    keep_configs = 10
    best_betas = []
    best_configs = []

    configs = get_many_random_hyperparams(config_space, 50)

    for config in configs:
        try:
            convexity = get_convexity(data, config)

            if len(best_betas) < keep_configs or convexity < max(best_betas):
                best_betas.append(convexity)
                best_configs.append(config)

                best_betas, best_configs = zip(*sorted(zip(best_betas, best_configs), key=lambda x: x[0]))
                best_betas = list(best_betas[:keep_configs])
                best_configs = list(best_configs[:keep_configs])
        except KeyboardInterrupt:
            raise
        except:  # noqa: E722
            print('Error, skipping config')
    
    scores = []
    for beta, config in zip(best_betas, best_configs):
        print(f'Config: {config} | mu: {beta}')

        score = eval(config)
        scores.append(score)
    
    return max(scores)


if __name__ == '__main__':
    run_experiment(opt_fn)
