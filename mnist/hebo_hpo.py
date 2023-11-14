from hebo.optimizers.hebo import HEBO
from hebo.design_space.design_space import DesignSpace

from common import hpo_space, eval


# Convert hpo_space to HEBO DesignSpace format
config_space = []
for key, val in hpo_space.items():
    config = {}
    config['name'] = key
    if isinstance(val, list):
        config['type'] = 'cat'
        config['categories'] = val
    elif isinstance(val, tuple):
        if isinstance(val[0], int):
            config['type'] = 'int'
        else:
            config['type'] = 'float'
        config['lb'] = val[0]
        config['ub'] = val[1]
    else:
        raise ValueError(f"Key {key} must be a list or tuple")

    config_space.append(config)


config_space = DesignSpace().parse(config_space)
hebo = HEBO(config_space)

for _ in range(5):
    configs = hebo.suggest(n_suggestions=6)
    hebo.observe(configs, [eval(config) for _, config in configs.iterrows()])

print("[main] Accuracy:", hebo.best_y)
