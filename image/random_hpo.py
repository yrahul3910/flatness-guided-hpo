from image.src.config import hpo_space
from image.src.data import Dataset
from image.src.util import get_data, get_many_random_hyperparams, run_experiment

if __name__ == "__main__":
    DATASET = "cifar10"
    N_CLASSES = 10
    N_RANDOM = 50

    configs = get_many_random_hyperparams(hpo_space, N_RANDOM)
    data: Dataset = get_data(DATASET)

    scores = []
    for config in configs:
        try:
            scores.append(float(run_experiment(data, config, N_CLASSES, DATASET)))
        except:
            continue

    print(scores)
