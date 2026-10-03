"""Run the fixed, prospective learning-rate mechanism diagnostic after review.

The historical preprocessing includes the internal validation rows. This is a mechanism reproduction condition, not
unbiased validation or HPO evidence. No official test arrays are opened. Existing execution artifacts are preserved.
"""

# This standalone evidence script is not an importable package.

# ruff: noqa: INP001

import hashlib
import json
import math
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray
from scipy.special import expit, logsumexp, xlogy
from sklearn.preprocessing import LabelBinarizer, Normalizer, OneHotEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DATA = HERE.parent / "data-recovery"
OUTPUT = HERE / "execution"
TASKS = (53, 10101, 146822)
SEEDS = (211, 223)
RATES = (1.0, 0.001)
EPOCHS = 30
BATCH_SIZE = 128
WIDTH = 64
DEPTH = 3
SECONDS_LIMIT = 300
type Array = NDArray[np.float32]
type Observer = Callable[..., tuple[tf.Tensor, tf.Tensor]]


@dataclass(frozen=True)
class Candidate:
    task_id: int
    seed: int
    learning_rate: float

    @property
    def name(self) -> str:
        """Identify this prescribed dataset, initialization and learning-rate combination."""
        return f"task{self.task_id}-seed{self.seed}-lr{self.learning_rate:g}"


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)

    if isinstance(value, Mapping):
        return {key: json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def record(payload: Mapping[str, object]) -> None:
    with (OUTPUT / "events.jsonl").open("a") as stream:
        _ = stream.write(json.dumps(json_safe(payload), allow_nan=False) + "\n")


def array_hash(arrays: Sequence[NDArray[np.generic]]) -> str:
    digest = hashlib.sha256()

    for array in arrays:
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())

    return digest.hexdigest()


def checkpoint(model: keras.Model, candidate: Candidate, label: str) -> bool:
    path = OUTPUT / f"{candidate.name}-{label}.keras"

    try:
        model.save(path)
        record(
            {
                "event": "checkpoint",
                **asdict(candidate),
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "weights_sha256": array_hash(model.get_weights()),
            }
        )
    # Serialization errors from Keras and filesystem failures both need evidence at this batch boundary.
    except Exception as error:  # noqa: BLE001
        record(
            {
                "event": "checkpoint_failure",
                **asdict(candidate),
                "file": path.name,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )
        return False
    else:
        return True


def prepare(task_id: int, expected_hash: str) -> tuple[Array, Array, int]:
    path = DATA / f"task-{task_id}-train-repeat0-fold0.npz"

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        message = f"Input hash changed for task {task_id}; review the frozen protocol"
        raise ValueError(message)

    with np.load(path, allow_pickle=False) as archive:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        features = Normalizer().fit_transform(encoder.fit_transform(archive["X"]))
        binarizer = LabelBinarizer()
        labels = binarizer.fit_transform(archive["y"]).astype(np.float32)
        split = int(len(features) * 0.8)
        np.savez_compressed(
            OUTPUT / f"task{task_id}-inner-split.npz",
            train_row_indices=archive["row_indices"][:split],
            validation_row_indices=archive["row_indices"][split:],
        )
        record(
            {
                "event": "dataset",
                "task_id": task_id,
                "input_sha256": expected_hash,
                "feature_shape": list(features.shape),
                "classes": binarizer.classes_.tolist(),
                "columns": archive["column_names"].tolist(),
                "train_rows": split,
                "validation_rows": len(features) - split,
                "features_sha256": array_hash([features]),
                "labels_sha256": array_hash([labels]),
                "row_order_sha256": array_hash([archive["row_indices"]]),
                "encoder_categories": [category.tolist() for category in encoder.categories_],
            }
        )

    return features, labels, split


def build(
    candidate: Candidate, dimensions: int, classes: int
) -> tuple[keras.Model, Observer, keras.layers.Dense, keras.optimizers.Adam]:
    keras.backend.clear_session()
    keras.utils.set_random_seed(candidate.seed)
    inputs = keras.Input(shape=(dimensions,))
    hidden = inputs

    for _ in range(DEPTH):
        hidden = keras.layers.Dense(WIDTH, activation="relu")(hidden)

    binary = classes == 1
    head = keras.layers.Dense(classes, name="head")
    logits = head(hidden)
    outputs = keras.layers.Activation("sigmoid" if binary else "softmax")(logits)
    model = keras.Model(inputs, outputs)
    observer = tf.function(keras.Model(inputs, [hidden, logits]), reduce_retracing=True)
    optimizer = keras.optimizers.Adam(learning_rate=candidate.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    model.compile(optimizer=optimizer, loss="binary_crossentropy" if binary else "categorical_crossentropy")

    return model, observer, head, optimizer


def measure(
    observer: Observer,
    head: keras.layers.Dense,
    features: Array,
    labels: Array,
    split: int,
) -> Mapping[str, object]:
    """Retain original float32 ratios and stable current-model diagnostics for every ordered batch.

    The exact trace is for the final kernel block of mean cross-entropy, excludes the bias, and is coordinate dependent.
    Binary sigmoid uses p*(1-p); it is never treated as a one-class softmax or subjected to softmax centering.
    """
    kernel = np.asarray(head.get_weights()[0], dtype=np.float32)
    weight_norm = float(np.linalg.norm(kernel))
    weight_norm64 = float(np.linalg.norm(kernel.astype(np.float64)))
    best = -math.inf
    batches, hidden_arrays, logit_arrays = [], [], []

    with np.errstate(all="ignore"):
        for start in range(0, len(features), BATCH_SIZE):
            hidden, logits = observer(features[start : start + BATCH_SIZE], training=False)
            h = np.asarray(hidden, dtype=np.float32)
            hidden_norm = float(np.linalg.norm(h))
            ratio = float(np.divide(np.float32(hidden_norm), np.float32(weight_norm)))

            if ratio > best and ratio != math.inf:
                best = ratio

            batches.append(
                {
                    "start": start,
                    "stop": min(start + BATCH_SIZE, len(features)),
                    "h_f": hidden_norm,
                    "w_f": weight_norm,
                    "h_over_w": ratio,
                    "ratio_finite": bool(np.isfinite(ratio)),
                    "h_f_float64": float(np.linalg.norm(h.astype(np.float64))),
                }
            )
            hidden_arrays.append(h)
            logit_arrays.append(np.asarray(logits, dtype=np.float32))

        h = np.concatenate(hidden_arrays).astype(np.float64)
        z = np.concatenate(logit_arrays).astype(np.float64)
        binary = z.shape[1] == 1

        if binary:
            positive = expit(z[:, 0])
            probabilities = np.column_stack((1 - positive, positive))
            losses = np.logaddexp(0, z[:, 0]) - labels[:, 0] * z[:, 0]
            targets = labels[:, 0]
            curvature = positive * (1 - positive)
        else:
            log_probabilities = z - logsumexp(z, axis=1, keepdims=True)
            probabilities = np.exp(log_probabilities)
            losses = -np.sum(labels * log_probabilities, axis=1)
            targets = labels.argmax(axis=1)
            curvature = 1 - np.sum(probabilities**2, axis=1)

        entropy = -np.sum(xlogy(probabilities, probabilities), axis=1)
        trace_contributions = curvature * np.sum(h**2, axis=1)
        partitions = {}

        for name, selected in (
            ("train", slice(None, split)),
            ("validation", slice(split, None)),
            ("whole_fold", slice(None)),
        ):
            partitions[name] = {
                "cross_entropy": float(np.mean(losses[selected])),
                "accuracy": float(np.mean(probabilities[selected].argmax(axis=1) == targets[selected])),
                "entropy": float(np.mean(entropy[selected])),
                "max_probability": float(np.mean(probabilities[selected].max(axis=1))),
                "hidden_zero_fraction": float(np.mean(h[selected] == 0)),
                "exact_head_kernel_trace": float(np.mean(trace_contributions[selected])),
            }

    return {
        "original_max_ratio": best,
        "max_h_f": max(batch["h_f"] for batch in batches),
        "w_f": weight_norm,
        "w_f_float64": weight_norm64,
        "batches": batches,
        "partitions": partitions,
        "tensors_and_losses_finite": bool(
            np.isfinite(h).all() and np.isfinite(z).all() and np.isfinite(kernel).all() and np.isfinite(losses).all()
        ),
        "invalid_ratio_batches": sum(not batch["ratio_finite"] for batch in batches),
    }


def verify_initialization(model: keras.Model, candidate: Candidate, expected: dict[tuple[int, int], str]) -> str:
    initialization = array_hash(model.get_weights())
    key = candidate.task_id, candidate.seed
    expected.setdefault(key, initialization)

    if initialization != expected[key]:
        message = "Paired initialization hashes differ"
        raise RuntimeError(message)

    return initialization


def verify_epoch(actual_steps: int, expected_steps: int, *, finite: bool) -> None:
    if not finite:
        message = "Nonfinite current-model tensors, weights or cross-entropy"
        raise FloatingPointError(message)

    if actual_steps != expected_steps:
        message = "Optimizer step count does not match the prescribed training batches"
        raise RuntimeError(message)


def run_candidate(
    candidate: Candidate,
    features: Array,
    labels: Array,
    split: int,
    permutations: NDArray[np.int64],
    expected_initial: dict[tuple[int, int], str],
    deadline: float,
) -> str:
    """Attempt a prescribed run and retain terminal failures before allowing the next run.

    A numerical failure means nonfinite weights, hidden activations, logits, or current-model cross-entropy. Degenerate
    ratios alone are retained without terminating otherwise finite training. The shared deadline is checked per epoch.
    """
    record({"event": "candidate_start", **asdict(candidate)})
    model = None
    epoch = 0

    try:
        model, observer, head, optimizer = build(candidate, features.shape[1], labels.shape[1])
        initialization = verify_initialization(model, candidate, expected_initial)

        record(
            {
                "event": "initialization",
                **asdict(candidate),
                "weights_sha256": initialization,
                "shuffle_sequence_sha256": array_hash([permutations]),
            }
        )

        for epoch in range(1, EPOCHS + 1):
            if time.perf_counter() >= deadline:
                _ = checkpoint(model, candidate, f"timeout-after-epoch{epoch - 1}")
                record({"event": "candidate_timeout", **asdict(candidate), "completed_epochs": epoch - 1})

                return "timeout"

            permutation = permutations[epoch - 1]

            for start in range(0, split, BATCH_SIZE):
                selected = permutation[start : start + BATCH_SIZE]
                model.train_on_batch(features[selected], labels[selected])

            metrics = measure(observer, head, features, labels, split)
            all_weights_finite = all(np.isfinite(weight).all() for weight in model.get_weights())
            record(
                {
                    "event": "epoch",
                    **asdict(candidate),
                    "epoch": epoch,
                    **metrics,
                    "all_weights_finite": all_weights_finite,
                    "learning_rate": float(np.asarray(optimizer.learning_rate)),
                    "optimizer_steps": int(np.asarray(optimizer.iterations)),
                    "shuffle_sha256": array_hash([permutation]),
                    "seconds_remaining": deadline - time.perf_counter(),
                }
            )

            verify_epoch(
                int(np.asarray(optimizer.iterations)),
                epoch * math.ceil(split / BATCH_SIZE),
                finite=bool(metrics["tensors_and_losses_finite"]) and all_weights_finite,
            )

            if epoch in (1, EPOCHS) and not checkpoint(model, candidate, f"epoch{epoch}"):
                record({"event": "candidate_artifact_failure", **asdict(candidate), "epoch": epoch})
                return "checkpoint_failed"

    # Prescribed runs must continue after arbitrary individual training failures, with complete failure evidence.
    except Exception as error:  # noqa: BLE001
        record(
            {
                "event": "candidate_failure",
                **asdict(candidate),
                "epoch": epoch,
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
        )

        if model is not None:
            _ = checkpoint(model, candidate, f"failure-epoch{epoch}")

        return "failed"
    else:
        record({"event": "candidate_complete", **asdict(candidate), "completed_epochs": epoch})
        return "complete"
    finally:
        keras.backend.clear_session()


def main() -> None:
    """Execute only with the explicit run flag, retaining the frozen protocol and every prescribed run's status.

    Review the protocol and source before invoking this entry point. A unique execution directory prevents overwriting
    prior evidence. A timeout or any failed/unstarted candidate makes the complete panel unavailable for interpretation.
    """
    if sys.argv[1:] != ["--run"]:
        sys.stdout.write("Prepared only. After protocol review, invoke this script with --run.\n")
        return

    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "sources").mkdir()

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    sources = {}

    for source in (
        Path(__file__),
        protocol_path,
        ROOT / "pyproject.toml",
        DATA / "training-manifest.json",
        DATA / "prepare_training.py",
    ):
        shutil.copyfile(source, OUTPUT / "sources" / source.name)
        sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()

    packages = {name: version(name) for name in ("keras", "tensorflow", "numpy", "scipy", "scikit-learn")}
    record(
        {
            "event": "panel_start",
            "utc": datetime.now(UTC).isoformat(),
            "protocol": protocol,
            "source_sha256": sources,
            "packages": packages,
            "packages_sha256": hashlib.sha256(json.dumps(packages, sort_keys=True).encode()).hexdigest(),
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "command": sys.argv,
            "platform": platform.platform(),
            "cpu_only": True,
        }
    )
    candidates = [Candidate(task, seed, rate) for task in TASKS for seed in SEEDS for rate in RATES]
    statuses = {candidate.name: "unstarted" for candidate in candidates}
    initial_hashes = {}
    started = time.perf_counter()
    deadline = started + SECONDS_LIMIT

    try:
        for task_id in TASKS:
            if time.perf_counter() >= deadline:
                break

            features, labels, split = prepare(task_id, protocol["input_sha256"][str(task_id)])

            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                permutations = np.stack([rng.permutation(split) for _ in range(EPOCHS)])
                np.savez_compressed(OUTPUT / f"task{task_id}-seed{seed}-shuffles.npz", permutations=permutations)

                for rate in RATES:
                    if time.perf_counter() >= deadline:
                        break

                    candidate = Candidate(task_id, seed, rate)
                    statuses[candidate.name] = run_candidate(
                        candidate, features, labels, split, permutations, initial_hashes, deadline
                    )
                    sys.stdout.write(f"{candidate.name}: {statuses[candidate.name]}\n")
                    sys.stdout.flush()

            del features, labels
    finally:
        summary = {
            "panel_complete": all(status == "complete" for status in statuses.values()),
            "statuses": statuses,
            "seconds": time.perf_counter() - started,
            "required_candidates": len(candidates),
            "interpretation": "Prospective mechanism condition; incomplete panels must not be selectively pooled.",
        }
        _ = (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        record({"event": "panel_end", **summary})


if __name__ == "__main__":
    main()
