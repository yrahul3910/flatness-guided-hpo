import time

from typing import Callable

from util.config import config_space
from util.data import Dataset
from util.eval import eval

from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder, Normalizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import numpy as np
import openml


def run_experiment(opt_fn: Callable[[Dataset, dict, Callable[[dict], float]], None]):
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
                    ('hotencoding', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ('scaler', Normalizer()),
                    #('variance_threshold', VarianceThreshold()),
                ])
                X_train = pipeline.fit_transform(X_train)
                X_test = pipeline.transform(X_test)

                X_train = np.array(X_train)
                X_test = np.array(X_test)

                if y_train.dtype == 'object':
                    binarizer = LabelBinarizer()
                    y_train = binarizer.fit_transform(y_train)
                    y_test = binarizer.transform(y_test)
                
                if len(y_train.shape) == 2 and y_train.shape[1] == 1:
                    y_train = y_train.flatten()
                    y_test = y_test.flatten()
                
                y_train = y_train.astype(np.float32)
                y_test = y_test.astype(np.float32)
                
                data = Dataset(X_train, y_train, X_test, y_test)

                print(
                    f"Repeat #{i}, fold #{j}: X_train.shape: {X_train.shape}, "
                    f"y_train.shape {y_train.shape}, X_test.shape {X_test.shape}, y_test.shape {y_test.shape}"
                )

                start = time.time()

                def evaluator(config):
                    return eval(config, data)
                
                score = opt_fn(data, config_space, evaluator)
                end = time.time()

                fold_perfs.append(score)
                print(f"Time taken: {end - start:.2f}s")
            
            repeat_perfs.append(np.mean(fold_perfs, axis=0))
        
        print(f"Task {task_id}: {repeat_perfs}")
        print(f"Median: {np.median(repeat_perfs, axis=0)}")
