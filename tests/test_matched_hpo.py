"""Check matched-run accounting and selection without additional model training."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from image.matched_hpo import MatchedRun, RunSettings, augmentation_seeds
from image.src.matched_contract import decode, make_manifest
from image.src.matched_training import PhaseResult, TrainingArrays


class MatchedRunTests(unittest.TestCase):
    def test_shared_probes_union_reuse_and_accuracy_selection(self) -> None:
        """All methods share probes and union fits, while CE restoration and accuracy selection remain distinct."""
        images = np.zeros((6, 32, 32, 3), dtype=np.float32)
        labels = np.eye(10, dtype=np.float32)[:6]
        data = TrainingArrays(images, labels, images[:3], labels[:3])
        manifest = make_manifest("cifar10", 0, 3)
        probes = [
            PhaseResult(status="complete", rms=rms, head_trace=ce, validation_ce=ce, epochs=1, optimizer_steps=1)
            for rms, ce in [(0.3, 0.1), (0.1, 0.3), (0.2, 0.2)]
        ]
        finals = [
            PhaseResult(
                status="complete", validation_ce=ce, validation_accuracy=accuracy, checkpoint=Path(f"c{i}.keras")
            )
            for i, (ce, accuracy) in enumerate([(0.1, 0.7), (0.2, 0.8), (0.9, 0.9)])
        ]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run = MatchedRun(data, manifest, RunSettings(shortlist_size=2, probe_size=5), output)

            with patch("image.matched_hpo.fit_phase", side_effect=[*probes, *finals]) as fit:
                run.run()

            np.testing.assert_equal(fit.call_count, 6)
            np.testing.assert_equal([call.args[2] for call in fit.call_args_list], ["probe"] * 3 + ["final"] * 3)
            np.testing.assert_equal(len((output / "probes.jsonl").read_text().splitlines()), 3)
            np.testing.assert_equal((output / "probe-order.npy").exists(), desired=True)
            arms = json.loads((output / "arms.json").read_text())
            np.testing.assert_equal(arms["rms"]["incumbent"], "c000002")
            np.testing.assert_equal(arms["ce"]["incumbent"], "c000002")
            np.testing.assert_equal(arms["ce"]["validation_ce"], 0.9)
            np.testing.assert_equal(set(arms), {"rms", "ce", "trace", "random"})
            np.testing.assert_equal(arms["trace"]["incumbent"], "c000002")
            np.testing.assert_equal(arms["rms"]["logical_optimizer_steps"], 3)

            for index, call in enumerate(fit.call_args_list[3:]):
                candidate = manifest["candidates"][index]
                np.testing.assert_equal(call.args[3].initialization, candidate["seeds"]["final/init"])
                np.testing.assert_equal(
                    call.args[3].augmentation, augmentation_seeds(candidate["seeds"]["final/augmentation"])
                )

            end = json.loads((output / "run-end.json").read_text())
            np.testing.assert_equal((end["status"], end["test_evaluations"]), ("complete", 0))

            with np.testing.assert_raises(RuntimeError):
                run.run()

            with np.testing.assert_raises(FileExistsError):
                MatchedRun(data, manifest, RunSettings(), output)

    def test_failed_and_incomplete_work_is_retained_without_refill(self) -> None:
        """Failed probes remain in raw records; incomplete pools and final failures never cause replacement fits."""
        images = np.zeros((3, 32, 32, 3), dtype=np.float32)
        labels = np.eye(10, dtype=np.float32)[:3]
        data = TrainingArrays(images, labels, images, labels)
        manifest = make_manifest("cifar10", 0, 2)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            failed = MatchedRun(data, manifest, RunSettings(shortlist_size=1), root / "failed")

            saved = root / "failed/probe/c000000/result.json"
            saved.parent.mkdir(parents=True)
            _ = saved.write_text(json.dumps({"epochs": 1, "optimizer_steps": 2}))

            with (
                patch("image.matched_hpo.fit_phase", side_effect=ValueError("fixture probe failure")) as fit,
                np.testing.assert_raises(ValueError),
            ):
                failed.run()

            np.testing.assert_equal(fit.call_count, 2)
            np.testing.assert_equal(failed.results[("probe", "c000000")].optimizer_steps, 2)
            np.testing.assert_equal(
                json.loads((root / "failed/run-end.json").read_text())["physical_optimizer_steps"], 2
            )
            np.testing.assert_equal(len((root / "failed/probes.jsonl").read_text().splitlines()), 2)
            np.testing.assert_equal(json.loads((root / "failed/run-end.json").read_text())["status"], "failed")
            incomplete = MatchedRun(data, manifest, RunSettings(shortlist_size=1), root / "incomplete")

            with patch("image.matched_hpo.fit_phase", return_value=PhaseResult(status="incomplete")) as fit:
                incomplete.run()

            np.testing.assert_equal(fit.call_count, 2)
            np.testing.assert_equal((root / "incomplete/shortlists.json").exists(), desired=False)
            np.testing.assert_equal(json.loads((root / "incomplete/run-end.json").read_text())["status"], "incomplete")
            degraded = MatchedRun(data, manifest, RunSettings(shortlist_size=2), root / "degraded")
            probe = PhaseResult(status="complete", rms=0.2, head_trace=0.0, validation_ce=0.5)
            final = PhaseResult(
                status="complete", validation_accuracy=0.7, validation_ce=0.8, checkpoint=Path("ok.keras")
            )

            with patch(
                "image.matched_hpo.fit_phase", side_effect=[probe, probe, RuntimeError("failed final"), final]
            ) as fit:
                degraded.run()

            np.testing.assert_equal(fit.call_count, 4)
            arms = json.loads((root / "degraded/arms.json").read_text())
            np.testing.assert_equal([arm["status"] for arm in arms.values()], ["degraded"] * 4)
            np.testing.assert_equal([arm["incumbent"] for arm in arms.values()], ["c000001"] * 4)

    def test_external_fixture_and_dataset_identity_are_explicit(self) -> None:
        """Reject changed generated proposals and class mismatches before any output or model allocation."""
        images = np.zeros((3, 32, 32, 3), dtype=np.float32)
        labels = np.eye(10, dtype=np.float32)[:3]
        data = TrainingArrays(images, labels, images, labels)
        manifest = make_manifest("cifar10", 0, 2)
        manifest["candidates"][0]["coordinates"] = [0.0] * 7
        manifest["candidates"][0]["config"] = decode([0.0] * 7)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fixture"

            with np.testing.assert_raises(ValueError):
                MatchedRun(data, manifest, RunSettings(), output)

            _ = MatchedRun(data, manifest, replace(RunSettings(), proposal_mode="fixed_fixture"), output)

            with np.testing.assert_raises(ValueError):
                MatchedRun(data, make_manifest("cifar100", 0, 2), RunSettings(), Path(temporary) / "mismatch")


if __name__ == "__main__":
    _ = unittest.main()
