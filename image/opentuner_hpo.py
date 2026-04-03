import traceback

import opentuner
from opentuner import (
    ConfigurationManipulator,
    EnumParameter,
    FloatParameter,
    IntegerParameter,
    MeasurementInterface,
    Result,
)

from image.src.config import hpo_space
from image.src.data import Dataset
from image.src.util import get_data, run_experiment

DATASET = "cifar10"
N_CLASSES = 10
data: Dataset = get_data(DATASET)

scores = []
total = 0

class MyTuner(MeasurementInterface):
    def manipulator(self):
        manipulator = ConfigurationManipulator()

        for key, val in hpo_space.items():
            if isinstance(val, list):
                manipulator.add_parameter(EnumParameter(key, val))
            elif isinstance(val, tuple):
                if isinstance(val[0], int):
                    manipulator.add_parameter(IntegerParameter(key, val[0], val[1]))
                else:
                    manipulator.add_parameter(FloatParameter(key, val[0], val[1]))

        return manipulator

    def run(self, desired_result, input, limit):
        global total, scores
        cfg = desired_result.configuration.data

        # No idea how to stop this thing, so this is how we're doing it
        total += 1
        if total == 31:
            total = 0
            msg = "Limit exceeded."
            raise AssertionError(msg)

        try:
            acc = float(run_experiment(data, cfg, N_CLASSES, DATASET))
            # OpenTuner minimizes time
            result = Result(time=1.0 - acc)
            score = acc
        except Exception:
            traceback.print_exc()
            result = Result(time=100.0)
            score = 0.0

        scores.append(score)

        return result


if __name__ == "__main__":
    argparser = opentuner.default_argparser()

    all_scores = []
    for _ in range(20):
        try:
            scores = []
            MyTuner.main(argparser.parse_args())

            all_scores.append(max(scores))
            print(f"Max score in this run: {all_scores[-1]}")
        except AssertionError:
            continue
