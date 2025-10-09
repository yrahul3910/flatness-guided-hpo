
from common import eval, hpo_space
from src.util import get_many_random_hyperparams

if __name__ == "__main__":
    N_RANDOM = 30
    configs = get_many_random_hyperparams(hpo_space, N_RANDOM)

    scores = []
    for config in configs:
        try:
            scores.append(eval(config))
        except:
            continue

