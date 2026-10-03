"""Exercise TuRBO's image-space adapter without installing TuRBO or training models."""

import runpy
import sys
import types
import unittest
from functools import partial
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from image.src import experiment
from image.src.config import Config, hpo_space


class ImageTurboTests(unittest.TestCase):
    def test_discrete_coordinates_are_repeatable_and_preserve_python_types(self) -> None:
        """Decode every listed choice, box boundary, and repeated point using the entrypoint's objective.

        The optimizer is replaced so its constructor and callback can be checked without importing TuRBO or training.
        """
        turbo = types.ModuleType("turbo")
        optimizer = MagicMock()
        turbo.__dict__["Turbo1"] = optimizer

        with patch.dict(sys.modules, {"turbo": turbo}), patch.object(experiment, "SearchRun") as search_run:
            run = search_run.return_value.__enter__.return_value
            run.seed = 31
            run.evaluate.return_value = 0.75
            _ = runpy.run_module("image.turbo_hpo", run_name="__main__")
            arguments = optimizer.call_args.kwargs
            objective = arguments["f"]
            lower, upper = arguments["lb"], arguments["ub"]

            np.testing.assert_equal(torch.initial_seed(), 31)
            np.testing.assert_equal(lower, [32, 2, 0, 2, 0, 1e-4, 1e-5])
            np.testing.assert_equal(upper, [128, 6, 2, 5, 5, 1e-2, 1e-3])
            np.testing.assert_equal((arguments["n_init"], arguments["max_evals"], arguments["batch_size"]), (15, 30, 1))
            run.finish.assert_called_once_with()

            for point in (lower, upper, (lower + upper) / 2):
                np.testing.assert_equal(objective(point), 0.25)
                first = run.evaluate.call_args.args[0]
                np.testing.assert_equal(objective(point), 0.25)
                np.testing.assert_equal(run.evaluate.call_args.args[0], first)

                for key in ("n_filters", "kernel_size", "n_blocks", "n_units"):
                    np.testing.assert_equal(type(first[key]) is int, desired=True)

            for coordinate, key in ((2, "padding"), (4, "n_units")):
                for index, expected in enumerate(hpo_space[key]):
                    for fraction in (0.0, 0.25, 0.75):
                        point = (lower + upper) / 2
                        point[coordinate] = index + fraction
                        objective(point)
                        np.testing.assert_equal(run.evaluate.call_args.args[0][key], expected)

    def test_objective_restores_numpy_rng_after_success_and_failure(self) -> None:
        """Model reseeding and random draws must not advance the optimizer's NumPy stream.

        Both successful evaluations and handled objective failures must restore the incoming state.
        """
        turbo = types.ModuleType("turbo")
        optimizer = MagicMock()
        turbo.__dict__["Turbo1"] = optimizer

        with patch.dict(sys.modules, {"turbo": turbo}), patch.object(experiment, "SearchRun") as search_run:
            run = search_run.return_value.__enter__.return_value
            run.seed = 31
            _ = runpy.run_module("image.turbo_hpo", run_name="__main__")
            arguments = optimizer.call_args.kwargs
            objective = arguments["f"]
            point = (arguments["lb"] + arguments["ub"]) / 2

            def evaluate(_config: Config, *, fail: bool) -> float:
                np.random.seed(999)  # noqa: NPY002 - reproduce SearchRun's global NumPy reseeding.

                if fail:
                    message = "controlled objective failure"
                    raise RuntimeError(message)

                return 0.75

            for failure in (False, True):
                run.evaluate.side_effect = partial(evaluate, fail=failure)
                expected = np.random.RandomState(73).get_state()
                np.random.set_state(expected)  # noqa: NPY002 - initialize the legacy RNG used by upstream TuRBO.

                with patch("traceback.print_exc"):
                    score = objective(point)

                np.testing.assert_equal(score, 100.0 if failure else 0.25)
                actual = np.random.get_state()  # noqa: NPY002 - verify restoration of upstream TuRBO's legacy RNG.
                np.testing.assert_equal(actual, expected)


if __name__ == "__main__":
    _ = unittest.main()
