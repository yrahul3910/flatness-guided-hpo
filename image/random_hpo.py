from dataclasses import asdict

from src.util import get_many_random_hyperparams, get_data
from common import hpo_space, eval


if __name__ == '__main__':
    N_RANDOM = 30
    configs = get_many_random_hyperparams(hpo_space, N_RANDOM)

    scores = []
    for config in configs:
        try:
            print(f'Config: {config}')
            scores.append(eval(config))
        except:
            continue

    print(scores)
    print('[main] Accuracy:', max(scores))
