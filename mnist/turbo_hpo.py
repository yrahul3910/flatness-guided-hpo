from turbo import Turbo1
import numpy as np
from common import hpo_space, eval


def eval_wrapper(config):
    cfg = {}
    j = 0  # index that we access config
    for i, (key, val) in enumerate(hpo_space.items()):
        if isinstance(val, tuple):
            if isinstance(val[0], float):
                cfg[key] = config[j]
            else:
                cfg[key] = round(config[j])
            j += 1
        else:
            cfg[key] = np.random.choice(val)

    try:
        return 100. - eval(cfg)
    except:
        return 100.


lb = []
ub = []
for key, val in hpo_space.items():
    if isinstance(val, tuple):
        lb.append(val[0])
        ub.append(val[1])

turbo1 = Turbo1(
    f=eval_wrapper,
    lb=np.array(lb),
    ub=np.array(ub),
    n_init=2*len(lb)+1,
    max_evals=30,
    verbose=True
)
turbo1.optimize()
print('Score:', 100. - min(turbo1.fX))
