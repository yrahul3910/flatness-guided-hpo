"""Run a frozen four-arm KMNIST screen, then evaluate locked selected checkpoints.

The protocol hash is required before any prepared data array is read. Evaluation arrays are read only after every
arm decision has been written and hashed; upstream preparation owns the frozen fit, tuning, and evaluation splits.
"""

import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from image.src import matched_training, util
from image.src.config import Config
from image.src.matched_training import _metrics, array_hash, weights_hash
from image.src.util import get_mnist_model

ARMS = ("original_rms", "head_trace", "early_tuning_ce", "random")
PROBABILITY_TOLERANCE = 1e-5
TRACE_TOLERANCE = 1e-10
LEGACY_THRESHOLD = 0.1
BATCH_SIZE = 64


def write_json(path: Path, value: object) -> None:
    _ = path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed(block: int, candidate: str, phase: str, role: str) -> int:
    payload = ["fghpo-four-arm-v1", "kmnist", block, candidate, phase, role]
    encoded = json.dumps(payload, separators=(",", ":")).encode()

    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big") & 0x7FFFFFFF


def load_arrays(dataset: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(dataset) as archive:
        images = archive[f"{split}_x"]
        labels = archive[f"{split}_y"]

    if (
        images.dtype != np.float32
        or labels.dtype != np.float32
        or len(images) != len(labels)
        or not len(images)
        or images.shape[1:] != (28, 28, 1)
        or labels.shape[1:] != (10,)
        or not np.isfinite(images).all()
        or not np.isfinite(labels).all()
        or not np.isin(labels, [0, 1]).all()
        or not np.all(labels.sum(axis=1) == 1)
    ):
        msg = "Expected finite, preprocessed 28x28x1 float32 images and ten-class one-hot labels."
        raise ValueError(msg)

    return images, labels


def score_arrays(
    probability: np.ndarray, hidden: np.ndarray, kernel: np.ndarray, bias: np.ndarray
) -> tuple[dict, dict]:
    """Return the original float32 batch maximum and sample-weighted float64 head trace.

    The head trace uses softmax reconstructed from the saved inference-mode activations and affine output layer. It
    perturbs kernel entries in ordinary Euclidean coordinates with biases and features fixed; it changes under feature
    rescaling. The RMS baseline keeps its original batch statistic. Nonfinite scores fail common eligibility.
    """
    weight_rms = float(np.sqrt(np.mean(kernel**2)))
    batches = []

    for start in range(0, len(hidden), 64):
        p_rms = float(np.sqrt(np.mean(probability[start : start + 64] ** 2)))
        h_rms = float(np.sqrt(np.mean(hidden[start : start + 64] ** 2)))

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = float(np.divide(np.float32(p_rms) * np.float32(h_rms), np.float32(weight_rms)))

        batches.append(
            {
                "start": start,
                "count": len(hidden[start : start + 64]),
                "p_rms": p_rms,
                "h_rms": h_rms,
                "ratio": ratio if math.isfinite(ratio) else None,
            }
        )

    logits = hidden.astype(np.float64) @ kernel.astype(np.float64) + bias.astype(np.float64)
    exponential = np.exp(logits - logits.max(axis=1, keepdims=True))
    p64 = exponential / exponential.sum(axis=1, keepdims=True)
    hidden_square = np.sum(hidden.astype(np.float64) ** 2, axis=1)
    probability_square = np.sum(p64**2, axis=1)
    traces = (1 - probability_square) * hidden_square
    reconstruction = float(np.max(np.abs(p64 - probability)))

    if reconstruction > PROBABILITY_TOLERANCE or not np.isfinite(traces).all() or np.any(traces < -TRACE_TOLERANCE):
        msg = "Probe probabilities or trace components are invalid."
        raise ValueError(msg)

    finite_ratios = [batch["ratio"] for batch in batches if batch["ratio"] is not None]
    original = max(finite_ratios, default=-math.inf)
    metrics = {
        "original_rms": original if original > 0 else None,
        "head_trace": float(traces.mean()),
        "weight_rms": weight_rms,
        "probability_reconstruction_max_error": reconstruction,
        "legacy_threshold_eligible": original > LEGACY_THRESHOLD,
        "batches": batches,
    }

    return metrics, {
        "hidden_squared_norm": hidden_square,
        "probability_squared_norm": probability_square,
        "head_trace_per_example": traces,
        "reconstructed_probabilities": p64,
    }


def shortlists(probes: dict, block: int, size: int) -> dict:
    pool = sorted(
        candidate
        for candidate, result in probes.items()
        if result.get("status") == "complete"
        and result.get("original_rms") is not None
        and result["original_rms"] > 0
        and math.isfinite(result["original_rms"])
        and math.isfinite(result["head_trace"])
        and result["head_trace"] >= 0
        and math.isfinite(result["tuning_ce"])
    )

    if len(pool) < size:
        msg = "The shared eligible population is smaller than the fixed shortlist size."
        raise ValueError(msg)

    chosen = {arm: sorted(pool, key=lambda candidate: (probes[candidate][arm], candidate))[:size] for arm in ARMS[:-1]}
    random.Random(seed(block, "shared", "selection", "random")).shuffle(pool)
    chosen["random"] = pool[:size]

    return chosen


@dataclass
class Pilot:
    directory: Path
    protocol: dict
    fit_x: np.ndarray
    fit_y: np.ndarray
    tuning_x: np.ndarray
    tuning_y: np.ndarray
    started: float = field(default_factory=time.perf_counter)

    def record(self, event: dict) -> None:
        """Append strict JSON events in execution order."""
        with (self.directory / "events.jsonl").open("a") as stream:
            _ = stream.write(
                json.dumps({**event, "elapsed_seconds": time.perf_counter() - self.started}, allow_nan=False) + "\n"
            )

    def check_time(self) -> None:
        """Abort before more work when the cooperative budget is exhausted."""
        if time.perf_counter() - self.started >= self.protocol["max_seconds"]:
            msg = "Cooperative run time limit reached; no holdout evaluation is allowed."
            raise TimeoutError(msg)

    def train_epochs(self, model: keras.Model, block: int, phase: str, destination: Path) -> dict:
        """Run complete epochs and the installed CE callback, preserving shuffle and callback histories.

        Time is checked before each optimizer step. Any interruption aborts the complete comparison. Final weights
        are restored only after the declared stopping condition, while optimizer state remains at its last step.
        """
        maximum = 1 if phase == "probe" else self.protocol["max_epochs"]
        rng = np.random.default_rng(seed(block, "shared", phase, "shuffle"))
        callback = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.protocol["patience"], restore_best_weights=True
        )
        callback.set_model(model)
        callback.set_params({"epochs": maximum, "steps": math.ceil(len(self.fit_x) / 64), "verbose": 0})
        callback.on_train_begin()
        model.stop_training = False
        histories = []

        for epoch in range(maximum):
            permutation = rng.permutation(len(self.fit_x))
            np.save(destination / f"shuffle-{epoch + 1}.npy", permutation)
            model.reset_metrics()

            for start in range(0, len(permutation), 64):
                self.check_time()
                batch = permutation[start : start + 64]
                _ = model.train_on_batch(self.fit_x[batch], self.fit_y[batch])

            if not all(np.isfinite(weight).all() for weight in model.get_weights()):
                msg = "Training weights became nonfinite."
                raise ValueError(msg)

            ce, accuracy = _metrics(model, self.tuning_x, self.tuning_y, 64)

            if phase == "final":
                callback.on_epoch_end(epoch, {"val_loss": ce})

            row = {
                "epoch": epoch + 1,
                "tuning_ce": ce,
                "tuning_accuracy": accuracy,
                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                "shuffle_sha256": array_hash(permutation),
                "best_epoch": callback.best_epoch + 1 if phase == "final" else None,
                "callback_wait": callback.wait,
                "callback_stop": bool(model.stop_training),
            }
            histories.append(row)

            with (destination / "history.jsonl").open("a") as stream:
                _ = stream.write(json.dumps(row, allow_nan=False) + "\n")

            if model.stop_training:
                break

        if phase == "final":
            callback.on_train_end()

        ce, accuracy = _metrics(model, self.tuning_x, self.tuning_y, 64)

        return {
            "epochs": len(histories),
            "optimizer_steps": int(model.optimizer.iterations.numpy()),
            "best_epoch": callback.best_epoch + 1 if phase == "final" else None,
            "restored": phase == "final",
            "tuning_ce": ce,
            "tuning_accuracy": accuracy,
            "early_tuning_ce": ce,
            "terminal_optimizer_state": True,
            "shuffle_rng_state": rng.bit_generator.state,
        }

    def fit(self, block: int, candidate: str, config: Config, phase: str) -> dict:
        """Build a fresh candidate-keyed model and retain failed or complete phases without overwriting.

        Recoverable training failures consume their candidate slot. Checkpoint or logging failures abort the run;
        partial models never qualify for selection. A hard external kill may leave only the append-only history.
        """
        started = time.perf_counter()
        self.check_time()
        destination = self.directory / f"block{block}" / phase / candidate
        destination.mkdir(parents=True, exist_ok=False)
        keras.backend.clear_session()

        initialization = seed(block, candidate, phase, "initialization")
        keras.utils.set_random_seed(initialization)
        steps = self.protocol["max_epochs"] * math.ceil(len(self.fit_x) / 64) if phase == "final" else None
        model = get_mnist_model(config, decay_steps=steps)
        model.build((None, *self.fit_x.shape[1:]))

        result = {
            "block": block,
            "candidate": candidate,
            "phase": phase,
            "config": config,
            "status": "failed",
            "initialization_seed": initialization,
            "shuffle_seed": seed(block, "shared", phase, "shuffle"),
            "augmentation": None,
            "initial_weights_sha256": weights_hash(model),
        }
        self.record({"event": "phase_start", **result})
        primary_error = None

        try:
            result.update(self.train_epochs(model, block, phase, destination))

            if phase == "probe":
                order = np.random.default_rng(seed(block, "shared", "probe", "subset")).permutation(len(self.fit_x))
                np.save(destination / "measurement-order.npy", order)
                inference = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])
                probability, hidden = inference(self.fit_x[order], training=False)
                kernel, bias = model.layers[-1].get_weights()
                metrics, components = score_arrays(probability.numpy(), hidden.numpy(), kernel, bias)
                result.update(metrics)
                np.savez_compressed(
                    destination / "score-components.npz",
                    probability=probability.numpy(),
                    hidden=hidden.numpy(),
                    kernel=kernel,
                    bias=bias,
                    **components,
                )

            result["status"] = "complete"
        except BaseException as error:
            primary_error = error
            result["error"] = repr(error)

            if not isinstance(error, (ValueError, RuntimeError, tf.errors.OpError)):
                raise
        finally:
            self.save_phase(model, destination, result, started, primary_error)

        return result

    def save_phase(
        self, model: keras.Model, destination: Path, result: dict, started: float, primary_error: BaseException | None
    ) -> None:
        """Retain checkpoints and strict records while preserving any primary training exception."""
        try:
            result["optimizer_steps"] = int(model.optimizer.iterations.numpy())
            checkpoint = destination / f"model-{result['status']}.keras"
            model.save(checkpoint)
            result.update(
                {
                    "checkpoint": str(checkpoint.relative_to(self.directory)),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "weights_sha256": weights_hash(model),
                    "seconds": time.perf_counter() - started,
                }
            )
            write_json(destination / "result.json", result)
            self.record({"event": "phase_end", **result})
        except Exception as error:
            if primary_error is not None:
                sys.stderr.write(f"Artifact save failed: {error!r}; primary error: {primary_error!r}\n")
                raise primary_error from error

            raise

    def lock_decisions(self) -> dict:
        """Complete the shared probes, retrain each shortlist union, and lock every arm before evaluation.

        Invalid probe rows stay in the raw population and are excluded from the common pool. Final failures consume
        slots with no replacement. Any incomplete selected finalist aborts the whole comparison before evaluation.
        """
        probes, lists, finals, decisions = {}, {}, {}, {}

        for block in self.protocol["blocks"]:
            probes[block] = {
                candidate: self.fit(block, candidate, config, "probe")
                for candidate, config in self.protocol["configurations"].items()
            }
            lists[block] = shortlists(probes[block], block, self.protocol["shortlist_size"])

        write_json(self.directory / "probes.json", probes)
        write_json(self.directory / "shortlists.json", lists)

        for block in self.protocol["blocks"]:
            union = sorted({candidate for shortlist in lists[block].values() for candidate in shortlist})
            finals[block] = {
                candidate: self.fit(block, candidate, self.protocol["configurations"][candidate], "final")
                for candidate in union
            }
            decisions[block] = {}

            for arm, shortlist in lists[block].items():
                complete = [candidate for candidate in shortlist if finals[block][candidate]["status"] == "complete"]

                if len(complete) != len(shortlist):
                    msg = f"Incomplete shortlist for block {block}, arm {arm}."
                    raise RuntimeError(msg)

                incumbent = min(
                    complete,
                    key=lambda candidate: (
                        -finals[block][candidate]["tuning_accuracy"],
                        finals[block][candidate]["tuning_ce"],
                        candidate,
                    ),
                )
                decisions[block][arm] = {
                    "shortlist": shortlist,
                    "incumbent": incumbent,
                    "result": finals[block][incumbent],
                    "logical_probe_seconds": sum(row["seconds"] for row in probes[block].values()),
                    "logical_probe_epochs": sum(row.get("epochs", 0) for row in probes[block].values()),
                    "logical_final_epochs": sum(finals[block][candidate].get("epochs", 0) for candidate in shortlist),
                    "logical_probe_optimizer_steps": sum(
                        row.get("optimizer_steps", 0) for row in probes[block].values()
                    ),
                    "logical_final_optimizer_steps": sum(
                        finals[block][candidate].get("optimizer_steps", 0) for candidate in shortlist
                    ),
                    "logical_final_seconds": sum(finals[block][candidate]["seconds"] for candidate in shortlist),
                }

        write_json(
            self.directory / "costs.json",
            {
                "physical_probe_phases": sum(len(rows) for rows in probes.values()),
                "physical_final_phases": sum(len(rows) for rows in finals.values()),
                "physical_probe_seconds": sum(row["seconds"] for rows in probes.values() for row in rows.values()),
                "physical_final_seconds": sum(row["seconds"] for rows in finals.values() for row in rows.values()),
                "logical_final_references": sum(
                    len(shortlist) for arms in lists.values() for shortlist in arms.values()
                ),
                "epochs": sum(
                    row.get("epochs", 0)
                    for panel in (probes, finals)
                    for rows in panel.values()
                    for row in rows.values()
                ),
                "optimizer_steps": sum(
                    row.get("optimizer_steps", 0)
                    for panel in (probes, finals)
                    for rows in panel.values()
                    for row in rows.values()
                ),
            },
        )
        write_json(self.directory / "finals.json", finals)
        write_json(self.directory / "decisions.json", decisions)
        digest = hashlib.sha256((self.directory / "decisions.json").read_bytes()).hexdigest()
        write_json(
            self.directory / "decision-lock.json",
            {"sha256": digest, "locked_utc": datetime.now(UTC).isoformat(), "evaluation_rows_loaded": False},
        )
        self.record({"event": "decisions_locked", "sha256": digest})

        return decisions

    def evaluate(self) -> None:
        """Evaluate only incumbents recovered from the hash-verified decision file.

        Each distinct selected checkpoint is loaded and checked once. Evaluation arrays are accessed only after the
        decision lock passes. A changed checkpoint, nonfinite prediction, or time limit aborts the evaluation panel.
        """
        self.check_time()
        locked = json.loads((self.directory / "decision-lock.json").read_text())

        if hashlib.sha256((self.directory / "decisions.json").read_bytes()).hexdigest() != locked["sha256"]:
            msg = "Locked decisions changed before evaluation."
            raise RuntimeError(msg)

        decisions = json.loads((self.directory / "decisions.json").read_text())
        images, labels = load_arrays(Path(self.protocol["dataset_path"]), "evaluation")
        self.record(
            {
                "event": "evaluation_rows_loaded",
                "images_sha256": array_hash(images),
                "labels_sha256": array_hash(labels),
                "decision_sha256": locked["sha256"],
            }
        )
        evaluated = {}

        for block, arms in decisions.items():
            for arm, decision in arms.items():
                self.check_time()
                checkpoint = decision["result"]["checkpoint"]

                if checkpoint not in evaluated:
                    path = self.directory / checkpoint

                    if hashlib.sha256(path.read_bytes()).hexdigest() != decision["result"]["checkpoint_sha256"]:
                        msg = "Selected checkpoint changed before evaluation."
                        raise RuntimeError(msg)

                    model = keras.models.load_model(path)

                    if not isinstance(model, keras.Model):
                        msg = "Selected checkpoint did not contain a Keras model."
                        raise TypeError(msg)

                    predictions = np.concatenate(
                        [
                            model(images[start : start + 64], training=False).numpy()
                            for start in range(0, len(images), 64)
                        ]
                    )

                    if (
                        not np.isfinite(predictions).all()
                        or weights_hash(model) != decision["result"]["weights_sha256"]
                    ):
                        msg = "Selected checkpoint weights or probabilities failed validation."
                        raise ValueError(msg)

                    predicted = predictions.argmax(axis=1)
                    errors = int(np.count_nonzero(predicted != labels.argmax(axis=1)))
                    evaluated[checkpoint] = {
                        "errors": errors,
                        "count": len(labels),
                        "error_rate": errors / len(labels),
                        "accuracy": 1 - errors / len(labels),
                    }
                    np.savez_compressed(
                        path.parent / "evaluation-predictions.npz",
                        probabilities=predictions,
                        predictions=predicted,
                        labels=labels.argmax(axis=1),
                    )

                self.record(
                    {
                        "event": "arm_evaluation",
                        "block": block,
                        "arm": arm,
                        "incumbent": decision["incumbent"],
                        **evaluated[checkpoint],
                    }
                )

        write_json(self.directory / "evaluation.json", evaluated)


def main() -> None:
    """Verify the frozen protocol and source set, preserve provenance, and execute once.

    The command is `run_pilot.py PROTOCOL_PATH SHA256 OUTPUT_DIRECTORY`. A pre-existing output directory is an error.
    Failures retain raw phase records and mark the run incomplete; no partial arm panel is interpreted as a result.
    """
    protocol_path, expected, output = sys.argv[1:]
    protocol_path, directory = Path(protocol_path), Path(output)

    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != expected:
        msg = "Protocol hash mismatch."
        raise ValueError(msg)

    protocol = json.loads(protocol_path.read_text())

    for item in protocol["sources"].values():
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            msg = f"Source hash mismatch: {item['path']}"
            raise ValueError(msg)

    frozen_paths = {Path(item["path"]).resolve() for item in protocol["sources"].values()}

    if (
        not {Path(__file__).resolve(), Path(util.__file__).resolve(), Path(matched_training.__file__).resolve()}
        <= frozen_paths
    ):
        msg = "Runner or imported model/metric helper is outside the frozen source set."
        raise ValueError(msg)

    if protocol["batch_size"] != BATCH_SIZE:
        msg = "This reviewed runner fixes batch size at 64, including the original score batches."
        raise ValueError(msg)

    dataset = Path(protocol["dataset_path"])

    if hashlib.sha256(dataset.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        msg = "Dataset hash mismatch."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(protocol_path, directory / "protocol.json")

    for logical, item in protocol["sources"].items():
        destination = directory / "sources" / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["path"], destination)

    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    fit_x, fit_y = load_arrays(dataset, "fit")
    tuning_x, tuning_y = load_arrays(dataset, "tuning")
    write_json(
        directory / "metadata.json",
        {
            "protocol_sha256": expected,
            "started_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "command": sys.argv,
            "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy")},
            "fit_images_sha256": array_hash(fit_x),
            "fit_labels_sha256": array_hash(fit_y),
            "tuning_images_sha256": array_hash(tuning_x),
            "tuning_labels_sha256": array_hash(tuning_y),
            "augmentation": None,
        },
    )

    pilot = Pilot(directory, protocol, fit_x, fit_y, tuning_x, tuning_y)
    complete = False

    try:
        _ = pilot.lock_decisions()
        pilot.evaluate()
        complete = True
    finally:
        write_json(
            directory / "summary.json",
            {
                "complete": complete,
                "seconds": time.perf_counter() - pilot.started,
                "status": "exploratory",
                "primary_metric": "heldout_top1_error_count",
            },
        )


if __name__ == "__main__":
    main()
