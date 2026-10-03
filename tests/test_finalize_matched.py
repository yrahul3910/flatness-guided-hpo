"""Check sealing and evaluation access using temporary records and mocked inference only."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import keras
import numpy as np

from image.finalize_matched import METHODS, finalize_suite, seal_suite


class FinalizeMatchedTests(unittest.TestCase):
    def test_suite_gate_and_unique_evaluation(self) -> None:
        """Exercise valid shared evaluation and a consumed attempt with a corrupted archive."""
        for corrupt_archive in (False, True):
            self._exercise(corrupt_archive=corrupt_archive)

    def _exercise(self, *, corrupt_archive: bool) -> None:
        """Reject incomplete or modified decisions before any evaluation archive opens.

        Mock inference returns one error in two rows. Four arms share that single evaluation and a second attempt fails.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assignment = root / "assignment.json"
            evaluation = root / "evaluation.npz"
            _ = evaluation.write_bytes(b"mock evaluation archive")
            assigned = {
                "runs": [{"dataset": "cifar10", "repetition_id": 0}],
                "evaluation_archives": {
                    "cifar10": {"path": str(evaluation), "sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest()}
                },
            }
            run = root / "run"
            phase = run / "final/c000000"
            phase.mkdir(parents=True)
            checkpoint = phase / "model.keras"
            _ = checkpoint.write_bytes(b"mock model")

            arm = {
                "status": "complete",
                "selected": ["c000000", "c000001"],
                "incumbent": "c000000",
                "validation_accuracy": 0.7,
                "validation_ce": 0.8,
                "checkpoint": str(checkpoint),
            }
            result = {
                **{key: arm[key] for key in ("status", "validation_accuracy", "validation_ce", "checkpoint")},
                "metadata": {
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "weights_sha256": "mock weights",
                },
            }
            records = {
                "arms": dict.fromkeys(METHODS, arm),
                "run-end": {"status": "incomplete"},
                "manifest": {"dataset": "cifar10", "repetition_id": 0},
                "settings": {"shortlist_size": 2},
                "shortlists": {method: {"selected": arm["selected"]} for method in METHODS},
                "metadata": {},
            }

            for name, record in records.items():
                _ = (run / f"{name}.json").write_text(json.dumps(record))

            _ = (phase / "result.json").write_text(json.dumps(result))
            _ = (run / "probes.jsonl").write_text("{}\n")
            assigned["runs"] = [
                {
                    "dataset": "cifar10",
                    "repetition_id": 0,
                    **{
                        f"{name}_sha256": hashlib.sha256(
                            json.dumps(records[name], sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest()
                        for name in ("manifest", "settings")
                    },
                }
            ]
            _ = assignment.write_text(json.dumps(assigned))
            other = run / "final/c000001"
            other.mkdir()
            _ = (other / "model.keras").write_bytes(checkpoint.read_bytes())
            other_result = {
                **result,
                "checkpoint": str(other / "model.keras"),
                "validation_accuracy": 0.6,
                "validation_ce": 0.1,
            }
            _ = (other / "result.json").write_text(json.dumps(other_result))

            _ = (run / "run-end.json").write_text('{"status":"complete"}')

            self._reject_changed_records(
                run,
                assignment,
                [
                    (run / "run-end.json", '{"status":"incomplete"}'),
                    (assignment, json.dumps({**assigned, "runs": [{"dataset": "cifar10", "repetition_id": 1}]})),
                    (phase / "result.json", json.dumps({**result, "status": "incomplete"})),
                    (run / "manifest.json", json.dumps({**records["manifest"], "changed": True})),
                    (run / "settings.json", json.dumps({**records["settings"], "changed": True})),
                    (
                        run / "arms.json",
                        json.dumps(
                            {
                                method: {
                                    **arm,
                                    "incumbent": "c000001",
                                    "checkpoint": other_result["checkpoint"],
                                    "validation_accuracy": 0.6,
                                    "validation_ce": 0.1,
                                }
                                for method in METHODS
                            }
                        ),
                    ),
                ],
            )

            lock = seal_suite([run], root / "seal", assignment)
            model = MagicMock(spec=keras.Model)
            model.predict.return_value = np.eye(10, dtype=np.float32)[[0, 0]]

            with (
                patch("image.finalize_matched.np.load") as load,
                patch("image.finalize_matched.keras.models.load_model", return_value=model) as restore,
                patch("image.finalize_matched.weights_hash", return_value="mock weights"),
            ):
                load.return_value.__enter__.return_value = {
                    "evaluation_x": np.zeros((2, 32, 32, 3), dtype=np.float32),
                    "evaluation_y": np.eye(10, dtype=np.float32)[:2],
                }

                def open_archive(_path: Path, *, allow_pickle: bool) -> MagicMock:
                    np.testing.assert_equal(allow_pickle, desired=False)
                    np.testing.assert_equal(
                        [(directory / "evaluation-attempt.json").exists() for directory in (lock.parent, run)],
                        [True, True],
                    )

                    return load.return_value

                load.side_effect = open_archive
                _ = checkpoint.write_bytes(b"changed")

                with np.testing.assert_raises(ValueError):
                    finalize_suite(lock, {"cifar10": root / "evaluation.npz"})

                load.assert_not_called()
                _ = checkpoint.write_bytes(b"mock model")

                if corrupt_archive:
                    self._reject_archive(lock, evaluation, load)
                    return

                finalize_suite(lock, {"cifar10": root / "evaluation.npz"})
                np.testing.assert_equal((load.call_count, restore.call_count), (1, 1))
                rows = json.loads((lock.parent / "classification.json").read_text())
                np.testing.assert_equal([row["errors"] for row in rows], [1] * 4)

                with np.testing.assert_raises(FileExistsError):
                    finalize_suite(lock, {"cifar10": root / "evaluation.npz"})

                np.testing.assert_equal(load.call_count, 1)

    @staticmethod
    def _reject_changed_records(run: Path, assignment: Path, invalid_records: list[tuple[Path, str]]) -> None:
        for path, invalid in invalid_records:
            original = path.read_text()
            _ = path.write_text(invalid)

            with np.testing.assert_raises(ValueError):
                _ = seal_suite([run], run.parent / "seal", assignment)

            _ = path.write_text(original)

    @staticmethod
    def _reject_archive(lock: Path, evaluation: Path, load: MagicMock) -> None:
        _ = evaluation.write_bytes(b"changed evaluation archive")

        for error in (ValueError, FileExistsError):
            with np.testing.assert_raises(error):
                finalize_suite(lock, {"cifar10": evaluation})

        load.assert_not_called()
