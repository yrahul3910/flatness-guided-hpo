from collections.abc import Callable

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from util.config import Config
from util.data import Dataset
from util.experiment import run_experiment


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]) -> None:
    # Convert hpo_space to HEBO DesignSpace format
    cs = []
    for key, val in config_space.items():
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
            raise ValueError(msg)

        cs.append(config)

    config_space = DesignSpace().parse(cs)
    hebo = HEBO(config_space)

    def eval_fn(config: dict):
        # Importantly, eval returns something to *maximize*
        return [-x for x in eval(Config(**config))]

    def eval_fn_auc(config: dict, args=None):
        return eval_fn(config)[-1]

    # Use HEBO to perform hyper-parameter optimization for 30 iterations
    scores = []
    for _ in range(20):
        for _ in range(5):
            configs = hebo.suggest(n_suggestions=6)
            hebo.observe(configs, [eval_fn_auc(config) for _, config in configs.iterrows()])

        scores.append(hebo.best_y)



if __name__ == "__main__":
    run_experiment(opt_fn)
