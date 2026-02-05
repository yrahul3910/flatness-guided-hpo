from common import evaluate, hpo_space
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

# Convert hpo_space to HEBO DesignSpace format
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
        raise ValueError(msg)

    config_space.append(config)


config_space = DesignSpace().parse(config_space)
hebo = HEBO(config_space)

# Use HEBO to perform hyper-parameter optimization for 30 iterations.
scores = []
for _ in range(5):
    configs = hebo.suggest(n_suggestions=6)
    hebo.observe(configs, [evaluate(config) for _, config in configs.iterrows()])

scores.append(hebo.best_y)
