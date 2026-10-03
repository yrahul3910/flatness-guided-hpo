"""Validate paired CIFAR lifecycle plumbing on bounded synthetic data, not benchmark evidence."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import keras
import numpy as np

from image.src.config import Config
from image.src.matched_training import (
    FitSettings,
    FloatArray,
    PhaseSeeds,
    TrainingArrays,
    _probe_score,
    fit_phase,
    head_trace_terms,
    weights_hash,
)


class MatchedTrainingTests(unittest.TestCase):
    def test_exact_trace_matches_head_hessian_and_weighted_tail(self) -> None:
        """Check the numeric trace and original score with synthetic arrays and mocked inference, without models."""
        rng = np.random.default_rng(967)
        hidden = rng.normal(size=(65, 3)).astype(np.float32)
        images = np.arange(65, dtype=np.float32)[:, None]

        for classes in (10, 100):
            kernel = rng.normal(scale=0.1, size=(3, classes)).astype(np.float32)
            bias = np.zeros(classes, dtype=np.float32)
            logits = hidden.astype(np.float64) @ kernel.astype(np.float64)
            exponentials = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
            predicted = probabilities.astype(np.float32)
            hessian = sum(
                np.kron(np.diag(p) - np.outer(p, p), np.outer(h, h))
                for p, h in zip(probabilities, hidden.astype(np.float64), strict=True)
            ) / len(hidden)
            model = Mock()
            model.layers = [Mock(), Mock()]
            model.layers[-1].get_weights.return_value = (kernel, bias)

            def infer(rows: FloatArray, *, training: bool, probabilities: FloatArray = predicted) -> tuple[Mock, Mock]:
                np.testing.assert_equal(training, desired=False)
                ids = rows[:, 0].astype(int)

                return Mock(numpy=lambda: probabilities[ids]), Mock(numpy=lambda: hidden[ids])

            with patch("image.src.matched_training.keras.Model", return_value=infer):
                rms, trace, batches = _probe_score(model, images, 64)
                _, repartitioned, _ = _probe_score(model, images, 10)

            np.testing.assert_allclose([trace, repartitioned], np.trace(hessian), rtol=1e-12, atol=1e-12)
            recorded = json.loads(json.dumps(batches, allow_nan=False))
            np.testing.assert_equal([batch["count"] for batch in recorded], [64, 1])
            original = max(
                float(
                    np.sqrt(np.mean(predicted[start : start + 64] ** 2))
                    * np.sqrt(np.mean(hidden[start : start + 64] ** 2))
                    / np.sqrt(np.mean(kernel**2))
                )
                for start in (0, 64)
            )
            np.testing.assert_equal(rms, original)
            zero = head_trace_terms(np.full_like(predicted, 1 / classes), np.zeros_like(hidden), kernel, bias)
            np.testing.assert_equal(zero.terms, np.zeros(65))

    def test_probe_final_probe_replay_and_saved_metrics(self) -> None:
        """Exercise real augmentation, tails, cold initialization and checkpoint restoration in eight updates.

        Two final epochs check callback plumbing, not patience-25 stopping. Evidence stays on disk when the audit sets
        MATCHED_BACKEND_EVIDENCE_DIR; ordinary test runs use a temporary directory.
        """
        rng = np.random.default_rng(20261002)
        images = rng.normal(size=(193, 32, 32, 3)).astype(np.float32)
        labels = np.eye(10, dtype=np.float32)[np.arange(193) % 10]
        full = TrainingArrays(images[:128], labels[:128], images[128:], labels[128:])
        probe = TrainingArrays(images[:65], labels[:65], images[128:], labels[128:])
        config = Config(
            n_filters=32,
            kernel_size=2,
            padding="same",
            n_blocks=2,
            n_units=32,
            learning_rate=0.001,
            weight_decay=0.0001,
        )
        probe_seeds, final_seeds = PhaseSeeds(707, 709, (719, 727)), PhaseSeeds(733, 739, (743, 751))
        settings = FitSettings(max_epochs=2, time_limit_seconds=90)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(os.environ.get("MATCHED_BACKEND_EVIDENCE_DIR", temporary))
            first = fit_phase(probe, config, "probe", probe_seeds, settings, output / "probe-first")
            final = fit_phase(full, config, "final", final_seeds, settings, output / "final")
            replay = fit_phase(probe, config, "probe", probe_seeds, settings, output / "probe-replay")

            np.testing.assert_equal([item.status for item in (first, final, replay)], ["complete"] * 3)
            np.testing.assert_equal([item.optimizer_steps for item in (first, final, replay)], [2, 4, 2])
            np.testing.assert_equal(
                [first.rms, first.validation_ce, first.head_trace],
                [replay.rms, replay.validation_ce, replay.head_trace],
            )
            np.testing.assert_equal(first.metadata["weights_sha256"], replay.metadata["weights_sha256"])
            np.testing.assert_equal(first.metadata["augmentation_state"], replay.metadata["augmentation_state"])
            np.testing.assert_equal(first.metadata["initial_augmentation_state"], [[719, 0], [727, 0]])
            np.testing.assert_equal(
                first.metadata["initial_weights_sha256"] == final.metadata["initial_weights_sha256"], desired=False
            )
            np.testing.assert_array_equal(
                np.load(output / "probe-first/shuffle-1.npy"), np.load(output / "probe-replay/shuffle-1.npy")
            )

            record = json.loads((output / "probe-first/result.json").read_text())
            np.testing.assert_equal([batch["count"] for batch in record["metadata"]["score_batches"]], [64, 1])

            for result in (first, final, replay):
                if result.checkpoint is None or result.validation_ce is None:
                    self.fail("Completed phase did not retain a checkpoint.")

                loaded = keras.models.load_model(result.checkpoint)

                if not isinstance(loaded, keras.Model):
                    self.fail("Checkpoint did not load a Keras model.")

                np.testing.assert_equal(weights_hash(loaded), result.metadata["weights_sha256"])
                loaded.reset_metrics()
                _ = loaded.test_on_batch(full.validation_x[:64], full.validation_y[:64], return_dict=True)
                metrics = loaded.test_on_batch(full.validation_x[64:], full.validation_y[64:], return_dict=True)
                np.testing.assert_allclose(metrics["loss"], result.validation_ce, rtol=1e-6, atol=1e-6)
                np.testing.assert_equal(metrics["accuracy"], result.validation_accuracy)
                np.testing.assert_equal(int(loaded.optimizer.iterations.numpy()), result.optimizer_steps)

            if final.validation_ce is None:
                self.fail("Completed final phase did not report validation CE.")

            history = [json.loads(line) for line in (output / "final/history.jsonl").read_text().splitlines()]
            np.testing.assert_allclose(final.validation_ce, min(row["validation_ce"] for row in history), atol=1e-6)
            np.testing.assert_equal(final.metadata["decay_steps"], 4)
            np.testing.assert_equal(final.stop_epoch, None)

            with np.testing.assert_raises(FileExistsError):
                _ = fit_phase(probe, config, "probe", probe_seeds, settings, output / "probe-first")

    def test_validation_failure_preserves_strict_result(self) -> None:
        """Retain malformed-input failure without model creation or updates.

        Invalid labels must produce a durable failed status and null metrics before the model builder is called.
        """
        images = np.zeros((1, 32, 32, 3), dtype=np.float32)
        labels = np.zeros((1, 10), dtype=np.float32)
        data = TrainingArrays(images, labels, images, labels)
        config = Config(
            n_filters=32,
            kernel_size=2,
            padding="same",
            n_blocks=2,
            n_units=32,
            learning_rate=0.001,
            weight_decay=0.0001,
        )

        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("image.src.matched_training.util.get_cifar10_model") as build,
        ):
            directory = Path(temporary) / "failure"

            with np.testing.assert_raises(ValueError):
                _ = fit_phase(data, config, "probe", PhaseSeeds(707, 709, (719, 727)), FitSettings(), directory)

            build.assert_not_called()
            result = json.loads((directory / "result.json").read_text())
            np.testing.assert_equal(result["status"], "failed")
            np.testing.assert_equal(result["optimizer_steps"], 0)
            np.testing.assert_equal(result["validation_ce"], None)


if __name__ == "__main__":
    _ = unittest.main()
