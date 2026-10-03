"""Exercise the unit-coordinate BOHB adapter and real installed KDE proposals without neural training."""

import json
import runpy
import tempfile
import unittest
from functools import partial
from pathlib import Path
from typing import TypedDict
from unittest.mock import MagicMock, patch

import bohb.configspace as bohb_space
import numpy as np
from bohb import BOHB
from bohb.bohb import KDEMultivariate
from numpy.typing import NDArray

from image.src.config import Config
from image.src.matched_contract import decode


class KDERun(TypedDict):
    seed: int
    considered: list[list[float]]
    selected: dict[str, float]


class KDEEvidence(TypedDict):
    observations: list[list[float]]
    vartypes: str
    good_bandwidth: list[float]
    bad_bandwidth: list[float]
    random_percent: float
    n_samples: int
    runs: list[KDERun]


class ImageBohbTests(unittest.TestCase):
    callback_evidence: str | None = None
    kde_evidence: str | None = None

    def entrypoint(self, seed: int, directory: Path) -> tuple[BOHB, MagicMock]:
        """Construct the actual entrypoint's optimizer while disabling optimization and model training."""
        directory.mkdir()

        with (
            patch("image.src.experiment.SearchRun", autospec=True) as search_run,
            patch("bohb.BOHB.optimize", autospec=True),
            patch("bohb.configspace.ConfigurationSpace", wraps=bohb_space.ConfigurationSpace) as configspace,
        ):
            run = search_run.return_value.__enter__.return_value
            run.seed = seed
            run.directory = directory
            namespace = runpy.run_path(str(Path(__file__).parents[1] / "image/bohb_hpo.py"), run_name="__main__")
            search_run.assert_called_once_with("bohb-unit-coordinate")
            np.testing.assert_equal(configspace.call_args.kwargs.get("seed"), seed)
            run.finish.assert_called_once_with()

        return namespace["opt"], run

    def test_run_seed_controls_configuration_proposals(self) -> None:
        """Replay initial proposals for one seed and verify variation for a second seed."""
        proposals = []

        with tempfile.TemporaryDirectory() as directory:
            for index, seed in enumerate((31, 31, 47)):
                optimizer, run = self.entrypoint(seed, Path(directory) / str(index))
                proposals.append([optimizer.get_sample().to_dict() for _ in range(16)])
                run.evaluate.assert_not_called()

        np.testing.assert_equal(proposals[0], proposals[1])
        np.testing.assert_equal(proposals[0] != proposals[2], desired=True)

    def test_shared_decoder_budgets_rng_and_failure_ledger(self) -> None:
        """Check the adapter-to-contract boundary and retained success/failure identity.

        The shared decoder's exhaustive bin/geometry cases have their own tests. This check exercises actual objective
        callbacks, raw-to-mapped budgets, native scalar types, and restoration after a model reseeds NumPy.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run"
            optimizer, run = self.entrypoint(607, path)

            def evaluate(_config: Config, *, fail: bool, **_kwargs: int | bool) -> float:
                np.random.seed(999)  # noqa: NPY002 - reproduce SearchRun's global model seed reset.

                if fail:
                    message = "controlled model failure"
                    raise RuntimeError(message)

                return 0.75

            for index, (coordinate, budget, failure) in enumerate(
                ((0.0, 1, False), (0.5, 99, False), (1.0, 100, True))
            ):
                run.evaluate.side_effect = partial(evaluate, fail=failure)
                expected_rng = np.random.RandomState(73).get_state()
                np.random.set_state(expected_rng)  # noqa: NPY002 - initialize the installed optimizer's stream.

                with patch("traceback.print_exc"):
                    loss = optimizer.evaluate({f"x{i}": coordinate for i in range(7)}, budget)

                np.testing.assert_equal(loss, 100.0 if failure else 0.25)
                actual_rng = np.random.get_state()  # noqa: NPY002 - check restoration of the installed optimizer's stream.
                np.testing.assert_equal(actual_rng, expected_rng)
                actual = run.evaluate.call_args
                np.testing.assert_equal(actual.args[0], decode([coordinate] * 7))
                np.testing.assert_equal(actual.kwargs, {"epochs": 1 if index == 0 else 100, "select": index != 0})

                for key in ("n_filters", "kernel_size", "n_blocks", "n_units"):
                    np.testing.assert_equal(type(actual.args[0][key]) is int, desired=True)

            events = [json.loads(line) for line in (path / "bohb-proposals.jsonl").read_text().splitlines()]
            np.testing.assert_equal([row["event"] for row in events], ["start", "end"] * 3)
            np.testing.assert_equal([row["call_id"] for row in events], [1, 1, 2, 2, 3, 3])
            np.testing.assert_equal([row["raw_budget"] for row in events[::2]], [1, 99, 100])
            np.testing.assert_equal(events[-1]["error"], "RuntimeError('controlled model failure')")
            self.callback_evidence = json.dumps(
                {"metadata": json.loads((path / "adapter.json").read_text()), "events": events},
                indent=2,
                allow_nan=False,
            )

    def test_actual_kde_proposals_stay_in_unit_coordinates(self) -> None:
        """Exercise the installed KDE branch with nondegenerate fixed observations and record all candidates.

        Two synthetic KDE populations are diagnostic fixtures, not trained search observations. Three seeded calls
        check replay and variation while each call evaluates the installed 64-candidate density-ratio rule.
        """
        with tempfile.TemporaryDirectory() as directory:
            optimizer, run = self.entrypoint(607, Path(directory) / "run")
            parameters = optimizer.configspace.hyperparameters
            np.testing.assert_equal([parameter.name for parameter in parameters], [f"x{i}" for i in range(7)])
            np.testing.assert_equal(parameters[2].choices, [0.25, 0.75])
            np.testing.assert_equal(parameters[4].choices, [0.1, 0.3, 0.5, 0.7, 0.9])

            rows = np.random.default_rng(601).uniform(0.05, 0.95, size=(48, 7))
            rows[:, 2] = np.arange(48) % 2
            rows[:, 4] = np.arange(48) % 5
            observations = [
                bohb_space.Configuration(
                    [
                        parameter.new(int(value) if j in (2, 4) else value)
                        for j, (parameter, value) in enumerate(zip(parameters, row, strict=True))
                    ]
                )
                for row in rows
            ]

            optimizer.kde_good = KDEMultivariate(observations[:24])
            optimizer.kde_bad = KDEMultivariate(observations[24:])
            good_bw, bad_bw = optimizer.kde_good.bw, optimizer.kde_bad.bw

            if good_bw is None or bad_bw is None:
                message = "The actual KDE fits must produce bandwidth arrays."
                raise AssertionError(message)

            optimizer.kde_good.bw = np.clip(good_bw, optimizer.min_bandwidth, None)
            optimizer.kde_bad.bw = np.clip(bad_bw, optimizer.min_bandwidth, None)
            np.testing.assert_equal(observations[0].kde_vartypes, "ccococc")
            optimizer.random_percent = 0
            pdf = optimizer.kde_bad.pdf
            attempts = []

            def record_pdf(values: list[float]) -> NDArray[np.float64]:
                attempts.append([float(value) for value in values])
                return pdf(values)

            evidence = KDEEvidence(
                observations=rows.tolist(),
                vartypes=observations[0].kde_vartypes,
                good_bandwidth=optimizer.kde_good.bw.tolist(),
                bad_bandwidth=optimizer.kde_bad.bw.tolist(),
                random_percent=optimizer.random_percent,
                n_samples=optimizer.n_samples,
                runs=[],
            )

            for seed in (607, 607, 619):
                np.random.seed(seed)  # noqa: NPY002 - the actual BOHB/SciPy KDE sampler uses global NumPy.
                attempts.clear()

                with patch.object(optimizer.kde_bad, "pdf", side_effect=record_pdf):
                    selected = optimizer.get_sample().to_dict()

                evidence["runs"].append({"seed": seed, "considered": list(attempts), "selected": selected})
                self.kde_evidence = json.dumps(evidence, indent=2, allow_nan=False)
                matrix = np.array(attempts)
                np.testing.assert_equal(matrix.shape, (64, 7))
                continuous = matrix[:, [0, 1, 3, 5, 6]]

                np.testing.assert_equal(((continuous >= 0) & (continuous <= 1)).all(), desired=True)
                np.testing.assert_equal((np.ptp(continuous, axis=0) > 0).all(), desired=True)
                np.testing.assert_equal(set(matrix[:, 2]), {0, 1})
                np.testing.assert_equal(set(matrix[:, 4]), {0, 1, 2, 3, 4})

                _ = decode([selected[f"x{i}"] for i in range(7)])

            run.evaluate.assert_not_called()
            runs = evidence["runs"]
            np.testing.assert_equal(runs[0], runs[1])
            np.testing.assert_equal(runs[0]["considered"] != runs[2]["considered"], desired=True)


if __name__ == "__main__":
    _ = unittest.main()
