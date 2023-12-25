from dataclasses import dataclass

import numpy as np


@dataclass
class Dataset:
    x_train: np.array
    y_train: np.array
    x_test: np.array
    y_test: np.array
