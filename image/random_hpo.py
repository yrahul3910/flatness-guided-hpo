"""Run random search with validation selection and one held-out test evaluation."""

import traceback

from image.src.config import hpo_space
from image.src.experiment import SearchRun
from image.src.util import get_many_random_hyperparams

if __name__ == "__main__":
    with SearchRun("random") as run:
        configs = get_many_random_hyperparams(hpo_space, 30)

        for config in configs:
            try:
                run.evaluate(config)
            except (ValueError, RuntimeError):
                traceback.print_exc()

        run.finish()
