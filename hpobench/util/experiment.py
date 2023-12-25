import time

from typing import Callable

from util.config import config_space
from util.data import Dataset
from util.eval import eval

from sklearn.preprocessing import LabelBinarizer, StandardScaler 
import numpy as np
import openml


def run_experiment(opt_fn: Callable[[Dataset, dict, Callable[[dict], float]], None]):
    #task_ids = [10101, 53, 146818, 146821, 9952, 146822, 31, 3917]
    task_ids = [53]

    for task_id in task_ids:
        task = openml.tasks.get_task(task_id)
        n_repeats, n_folds, _ = task.get_split_dimensions()

        repeat_perfs = []
        
        for i in range(n_repeats):
            fold_perfs = []
            for j in range(n_folds):
                train_idx, test_idx = task.get_train_test_split_indices(repeat=i, fold=j)
                X, y = task.get_X_and_y(dataset_format="dataframe")
                X_train = np.array(X.iloc[train_idx])
                y_train = y.iloc[train_idx]
                X_test = np.array(X.iloc[test_idx])
                y_test = y.iloc[test_idx]

                binarizer = LabelBinarizer()
                y_train = binarizer.fit_transform(y_train)
                y_test = binarizer.transform(y_test)

                transform = StandardScaler()
                X_train = transform.fit_transform(X_train)
                X_test = transform.transform(X_test)

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
            
            repeat_perfs.append(np.mean(fold_perfs))
        
        print(f"Task {task_id}: {repeat_perfs}")
