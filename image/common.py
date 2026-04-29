import gc
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from image.src.config import Config
from image.src.util import get_data, run_experiment


def evaluate(config: Config, dataset: str = "mnist", n_class: int = 10) -> float:
    data = get_data(dataset)

    try:
        score = run_experiment(data, config, n_class, dataset)
        _ = gc.collect()
    except ValueError:
        return 100.0

    return score
