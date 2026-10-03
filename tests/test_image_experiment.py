"""Exercise validation-only search and its saved-model test boundary with real Keras training."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from image.src.config import Config
from image.src.data import Dataset
from image.src.experiment import SearchRun


class ImageExperimentTests(unittest.TestCase):
    def test_validation_search_records_failures_and_tests_only_once(self) -> None:
        """Train without test arrays, retain a validation winner, then permit one final test.

        Failed and low-budget candidates remain in the ledger. A fresh run must not overwrite its evidence directory.
        """
        rng = np.random.default_rng(31)
        images = rng.normal(size=(40, 28, 28, 1)).astype(np.float32)
        labels = np.eye(10, dtype=np.float32)[np.arange(40) % 10]
        data = Dataset(images, labels, images[:8], labels[:8])
        del data.x_test
        del data.y_test
        config = Config(
            n_filters=4, kernel_size=3, padding="same", n_blocks=1, n_units=32, learning_rate=0.001, weight_decay=0.0001
        )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            environment = {"DATASET": "mnist", "FGHPO_SEED": "31", "FGHPO_OUTPUT_DIR": str(output)}

            with patch.dict(os.environ, environment), patch("image.src.experiment.get_data", return_value=data):
                with SearchRun("regression-test") as run:
                    invalid = config.copy()
                    invalid["n_blocks"] = 10

                    with np.testing.assert_raises(ValueError):
                        run.evaluate(invalid, epochs=1)

                    run.evaluate(config, epochs=1, select=False)
                    np.testing.assert_equal(run.checkpoint.exists(), desired=False)
                    validation_score = run.evaluate(config, epochs=1)
                    np.testing.assert_equal(run.checkpoint.exists(), desired=True)
                    np.testing.assert_equal(0.0 <= validation_score <= 1.0, desired=True)
                    data.x_test, data.y_test = images[:8], labels[:8]
                    test_score = run.finish()
                    np.testing.assert_equal(0.0 <= test_score <= 1.0, desired=True)

                    with np.testing.assert_raises(RuntimeError):
                        run.finish()

                    with np.testing.assert_raises(RuntimeError):
                        run.evaluate(config, epochs=1)

                with np.testing.assert_raises(FileExistsError):
                    SearchRun("regression-test")

            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            trials = [event for event in events if event["event"] == "trial"]
            np.testing.assert_equal([event["seed"] for event in trials], [32, 33, 34])
            np.testing.assert_equal("error" in trials[0], desired=True)
            np.testing.assert_equal("incumbent" in trials[1], desired=False)
            np.testing.assert_equal(sum(event["event"] == "final_test" for event in events), 1)
            np.testing.assert_equal(events[-1]["finished"], desired=True)
            np.testing.assert_equal((output / "source/image/src/util.py").exists(), desired=True)


if __name__ == "__main__":
    unittest.main()
