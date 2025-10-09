import time
from collections.abc import Callable

import numpy as np
import openml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, Normalizer, OneHotEncoder

from util.config import config_space
from util.data import Dataset
from util.eval import eval


def run_experiment(opt_fn: Callable[[Dataset, dict, Callable[[dict], float]], None]) -> None:
    #task_ids = [10101, 53, 146818, 146821, 9952, 146822, 31, 3917]
    task_ids = [9952]

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        n_repeats, n_folds, _ = task.get_split_dimensions()

        repeat_perfs = []

        for i in range(n_repeats):
            fold_perfs = []
            for j in range(n_folds):
                train_idx, test_idx = task.get_train_test_split_indices(repeat=i, fold=j)
                X, y = task.get_X_and_y(dataset_format="dataframe")
                X_train = X.iloc[train_idx]
                y_train = np.array(y.iloc[train_idx])
                X_test = X.iloc[test_idx]
                y_test = np.array(y.iloc[test_idx])

                # This changes for each dataset: see the OpenML task analysis page.
                pipeline = Pipeline([
                    ("hotencoding", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ("scaler", Normalizer()),
                    #('variance_threshold', VarianceThreshold()),
                ])
                X_train = pipeline.fit_transform(X_train)
                X_test = pipeline.transform(X_test)

                X_train = np.array(X_train)
                X_test = np.array(X_test)

                if y_train.dtype == "object":
                    binarizer = LabelBinarizer()
                    y_train = binarizer.fit_transform(y_train)
                    y_test = binarizer.transform(y_test)

                if len(y_train.shape) == 2 and y_train.shape[1] == 1:
                    y_train = y_train.flatten()
                    y_test = y_test.flatten()

                y_train = y_train.astype(np.float32)
                y_test = y_test.astype(np.float32)

                data = Dataset(X_train, y_train, X_test, y_test)


                time.time()

                def evaluator(config, args=None):
                    return eval(config, data)

                score = opt_fn(data, config_space, evaluator)
                time.time()

                fold_perfs.append(score)

            repeat_perfs.append(np.mean(fold_perfs, axis=0))

