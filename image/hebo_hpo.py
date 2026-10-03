"""Run HEBO with validation selection and one held-out test evaluation."""

import traceback

import numpy as np
import torch
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from image.src.config import Config, hpo_space
from image.src.experiment import SearchRun

if __name__ == "__main__":
    with SearchRun("hebo") as run:
        torch.manual_seed(run.seed)
        config_space = []

        for key, val in hpo_space.items():
            config = {}
            config["name"] = key

            if isinstance(val, list):
                config["type"] = "cat"
                config["categories"] = val
            elif isinstance(val, tuple):
                if isinstance(val[0], int):
                    config["type"] = "int"
                else:
                    config["type"] = "num"

                config["lb"] = val[0]
                config["ub"] = val[1]
            else:
                msg = f"Key {key} must be a list or tuple"
                raise TypeError(msg)

            config_space.append(config)

        config_space_parsed = DesignSpace().parse(config_space)
        hebo = HEBO(config_space_parsed, scramble_seed=run.seed)

        def evaluate_hebo(config: Config) -> float:
            try:
                return 1.0 - run.evaluate(config)
            except (ValueError, RuntimeError):
                traceback.print_exc()
                return 100.0

        for _ in range(5):
            configs = hebo.suggest(n_suggestions=6)
            y = []

            for _, config in configs.iterrows():
                y.append(evaluate_hebo(config.to_dict()))

            hebo.observe(configs, np.array(y).reshape(-1, 1))

        run.finish()
