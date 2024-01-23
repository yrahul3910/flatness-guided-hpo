from turbo import Turbo1
import numpy as np

from typing import Callable

from util.config import Config
from util.data import Dataset
from util.experiment import run_experiment


def opt_fn(data: Dataset, config_space: dict, eval: Callable[[dict], float]):
    lb = []
    ub = []
    for key, val in config_space.items():
        if isinstance(val, tuple):
            lb.append(val[0])
            ub.append(val[1])
    
    def eval_wrapper(config):
        def eval_fn(config: dict):
            # Importantly, eval returns something to *maximize*
            return [-x for x in eval(Config(**config))]
        
        def eval_fn_auc(config: dict):
            return eval_fn(config)[-1]

        cfg = {}
        j = 0  # index that we access config
        for i, (key, val) in enumerate(config_space.items()):
            if isinstance(val, tuple):
                if isinstance(val[0], float):
                    cfg[key] = config[j]
                else:
                    cfg[key] = round(config[j])
                j += 1
            else:
                cfg[key] = np.random.choice(val)

        try:
            return eval_fn_auc(cfg)
        except:  # noqa: E722
            return 10.

    turbo1 = Turbo1(
        f=eval_wrapper,
        lb=np.array(lb),
        ub=np.array(ub),
        n_init=2*len(lb)+1,
        max_evals=30,
        verbose=True
    )
    turbo1.optimize()
    print('Score:', -min(turbo1.fX))

    return min(turbo1.fX)


if __name__ == '__main__':
    run_experiment(opt_fn)
