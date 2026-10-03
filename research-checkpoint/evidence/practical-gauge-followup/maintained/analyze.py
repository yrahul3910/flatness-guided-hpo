"""Measure class-common head components in the complete saved lifecycle panel.

This exploratory follow-up uses the existing validation examples and no official test data. Four reconstructed
initializations must match their recorded whole-model hashes. The 32 saved checkpoints remain unchanged.
"""

import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import keras
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from scipy.stats import spearmanr

from image.src.util import get_mnist_model

if TYPE_CHECKING:
    from collections.abc import Mapping

SOURCE = Path(__file__).resolve().parents[1]
OUT = SOURCE / "replays" / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ")
ROOT = SOURCE.parents[2]
PILOT = SOURCE.parent / "lifecycle-pilot/run-prepared-20261002-v1"
DATA = Path(os.environ["KERAS_HOME"]) / "datasets/mnist.npz"
EXPECTED_CHECKPOINTS = 32
EXPECTED_INITIALIZATIONS = 4
TOLERANCE = 1e-5


def write_json(path: Path, value: "Mapping") -> None:
    """Write finite JSON values to the replay directory."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def weight_hash(model: keras.Model) -> str:
    """Hash the concatenated model-weight bytes in their saved order."""
    digest = hashlib.sha256()

    for weight in model.get_weights():
        digest.update(weight.tobytes())

    return digest.hexdigest()


def measure(model: keras.Model, images: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """Decompose the output kernel and validation logits in float64 and retain the underlying arrays.

    Centering subtracts each kernel row's class mean while keeping the bias and hidden features fixed. The
    mathematical probabilities stay fixed; the numerical check compares both float64 heads and saved float32 output.
    """
    features = keras.Model(model.inputs, [model.layers[-2].output, model.layers[-1].output])
    hidden_tensor, probability_tensor = features(images, training=False)
    hidden = np.asarray(hidden_tensor, dtype=np.float64)
    probability = np.asarray(probability_tensor, dtype=np.float64)
    kernel, bias = (np.asarray(value, dtype=np.float64) for value in model.layers[-1].get_weights())
    common = np.broadcast_to(kernel.mean(axis=1, keepdims=True), kernel.shape)
    centered = kernel - common
    total_norm = float(np.linalg.norm(kernel))
    common_norm = float(np.linalg.norm(common))
    centered_norm = float(np.linalg.norm(centered))
    logits = hidden @ kernel + bias
    centered_kernel_logits = hidden @ centered + bias
    logit_common = logits.mean(axis=1, keepdims=True)
    logit_centered = logits - logit_common
    original_probability = softmax(logits, axis=1)
    centered_probability = softmax(centered_kernel_logits, axis=1)
    difference = float(np.max(np.abs(original_probability - centered_probability)))
    reconstruction_error = float(np.max(np.abs(probability - original_probability)))
    changes = int(np.sum(original_probability.argmax(axis=1) != centered_probability.argmax(axis=1)))

    if difference > TOLERANCE or reconstruction_error > TOLERANCE or changes:
        msg = f"Probability equivalence failed for {name}."
        raise RuntimeError(msg)

    if not np.isclose(total_norm**2, common_norm**2 + centered_norm**2, rtol=1e-12):
        msg = f"Orthogonal decomposition failed for {name}."
        raise RuntimeError(msg)

    np.savez_compressed(
        OUT / "arrays" / f"{name}.npz",
        hidden=hidden,
        kernel=kernel,
        bias=bias,
        probabilities=probability,
        labels=labels,
    )

    return {
        "kernel_shape": list(kernel.shape),
        "kernel_frobenius": total_norm,
        "kernel_common_frobenius": common_norm,
        "kernel_centered_frobenius": centered_norm,
        "common_squared_norm_fraction": common_norm**2 / total_norm**2,
        "common_to_centered_norm": common_norm / centered_norm,
        "centered_score_multiplier": total_norm / centered_norm,
        "kernel_rms_float32": float(np.sqrt(np.mean(kernel.astype(np.float32) ** 2))),
        "bias_mean": float(bias.mean()),
        "bias_centered_norm": float(np.linalg.norm(bias - bias.mean())),
        "logit_common_rms": float(np.sqrt(np.mean(logit_common**2))),
        "logit_centered_rms": float(np.sqrt(np.mean(logit_centered**2))),
        "logit_common_to_centered_rms": float(np.sqrt(np.mean(logit_common**2) / np.mean(logit_centered**2))),
        "logit_kernel_common_rms": float(np.sqrt(np.mean((hidden @ kernel.mean(axis=1)) ** 2))),
        "validation_accuracy": float(np.mean(probability.argmax(axis=1) == labels)),
        "float32_probability_reconstruction_error": reconstruction_error,
        "kernel_centering_probability_difference": difference,
        "kernel_centering_prediction_changes": changes,
    }


def initializations(starts: list[dict], images: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Reconstruct each distinct paired initialization and demand exact whole-model hash agreement.

    No optimizer updates are executed. The source panel used a common initialization for all eight candidates
    within each block/phase. Only config zero needs reconstruction; the saved starts verify all eight hashes match.
    """
    rows = []

    for start in starts:
        if start["config_id"] != 0:
            continue

        block, phase = start["seed_block"], start["phase"]
        peers = [row for row in starts if row["seed_block"] == block and row["phase"] == phase]

        if len({row["initialization_sha256"] for row in peers}) != 1:
            msg = "Recorded paired initialization hashes disagree."
            raise RuntimeError(msg)

        keras.backend.clear_session()
        keras.utils.set_random_seed(start["phase_seed"])
        model = get_mnist_model(start["config"], decay_steps=2400 if phase == "final" else None)
        model.build((None, 28, 28, 1))
        actual_hash = weight_hash(model)

        if actual_hash != start["initialization_sha256"]:
            msg = "Reconstructed initialization does not match recorded weights."
            raise RuntimeError(msg)

        name = f"block{block}-{phase}-reconstructed-initial"
        rows.append(
            {
                "seed_block": block,
                "phase": phase,
                "stage": "reconstructed_initial",
                "initialization_sha256": actual_hash,
                "paired_candidates": len(peers),
                **measure(model, images, labels, name),
            }
        )

    return rows


def trained_checkpoints(ends: list[dict], images: np.ndarray, labels: np.ndarray) -> list[dict]:
    """Read every complete checkpoint and check its recorded file, weights and validation accuracy.

    Probe centering keeps the logged probability/feature numerator and changes only the denominator, so no
    training-set inference or score-batch reshuffling is needed. This is a mathematical denominator correction.
    """
    rows = []

    for end in ends:
        path = PILOT / end["checkpoint"]

        if hashlib.sha256(path.read_bytes()).hexdigest() != end["checkpoint_sha256"]:
            msg = f"Saved checkpoint hash changed: {path}."
            raise RuntimeError(msg)

        keras.backend.clear_session()
        model = keras.models.load_model(path, compile=False)

        if not isinstance(model, keras.Model):
            msg = f"Checkpoint did not contain a Keras model: {path}."
            raise TypeError(msg)

        if weight_hash(model) != end["weights_sha256"]:
            msg = f"Saved model weight hash changed: {path}."
            raise RuntimeError(msg)

        measures = measure(model, images, labels, path.stem)

        if measures["validation_accuracy"] != end["validation_accuracy"]:
            msg = f"Validation classification mismatch: {path}."
            raise RuntimeError(msg)

        row = {
            "seed_block": end["seed_block"],
            "config_id": end["config_id"],
            "config": end["config"],
            "phase": end["phase"],
            "stage": "trained",
            "checkpoint": str(path.relative_to(ROOT)),
            "checkpoint_sha256": end["checkpoint_sha256"],
            "best_epoch": end["best_epoch"],
            "completed_epochs": end["completed_epochs"],
            **measures,
        }

        if end["phase"] == "probe":
            if measures["kernel_rms_float32"] != end["weight_rms"]:
                msg = "Original score denominator did not replay exactly."
                raise RuntimeError(msg)

            row["original_rms_ratio_max"] = end["original_rms_ratio_max"]
            row["centered_rms_ratio_max"] = end["original_rms_ratio_max"] * measures["centered_score_multiplier"]

        rows.append(row)
        write_json(OUT / "partial.json", {"trained_rows": rows, "complete": False})

    return rows


def summarize(rows: list[dict], initials: list[dict]) -> dict:
    """Report all blocks and stages with classification-target associations and deterministic ranks.

    Correlations are descriptive within eight-candidate paired blocks. The validation examples were already used
    for early stopping and earlier diagnostics; these results do not estimate untouched-holdout performance.
    """
    distributions, rankings = [], []

    for stage in ("reconstructed_initial", "probe", "final"):
        group = initials if stage == "reconstructed_initial" else [row for row in rows if row["phase"] == stage]
        metrics = {}

        for key in (
            "common_squared_norm_fraction",
            "common_to_centered_norm",
            "centered_score_multiplier",
            "logit_common_to_centered_rms",
        ):
            values = np.array([row[key] for row in group])
            metrics[key] = dict(
                zip(
                    ("min", "q25", "median", "q75", "max"),
                    np.quantile(values, [0, 0.25, 0.5, 0.75, 1]).tolist(),
                    strict=True,
                )
            )

        distributions.append({"stage": stage, "count": len(group), "metrics": metrics})

    for block in sorted({row["seed_block"] for row in rows}):
        probes = sorted(
            [row for row in rows if row["seed_block"] == block and row["phase"] == "probe"],
            key=lambda row: row["config_id"],
        )
        finals = sorted(
            [row for row in rows if row["seed_block"] == block and row["phase"] == "final"],
            key=lambda row: row["config_id"],
        )
        errors = [1 - row["validation_accuracy"] for row in finals]
        scores = {key: [row[key] for row in probes] for key in ("original_rms_ratio_max", "centered_rms_ratio_max")}
        entry = {
            "seed_block": block,
            "n": len(probes),
            "endpoint": "restored_validation_error",
            "target_errors": errors,
        }

        for key, values in scores.items():
            ranking = sorted(range(len(values)), key=lambda index: (values[index], index))
            entry[key] = {
                "values": values,
                "ranking_config_ids": ranking,
                "spearman_vs_error": float(spearmanr(values, errors).statistic),
                "selected_config_id": ranking[0],
                "selected_errors_of_375": round(375 * errors[ranking[0]]),
                "error_regret": errors[ranking[0]] - min(errors),
            }

        entry["rank_correlation_original_centered"] = float(
            spearmanr(scores["original_rms_ratio_max"], scores["centered_rms_ratio_max"]).statistic
        )
        rankings.append(entry)

    return {"distributions": distributions, "rankings": rankings}


def main() -> None:
    """Reconstruct initialization and checkpoint diagnostics in a fresh replay directory.

    Frozen source and dataset hashes must match before inference. The output records this script and the original
    protocol hashes; existing result directories cannot be overwritten. No optimizer updates or test reads occur.
    """
    started = time.perf_counter()
    protocol = json.loads((SOURCE / "protocol.json").read_text())

    for filename, digest in protocol["source_sha256"].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != digest:
            msg = f"Frozen source mismatch: {filename}."
            raise RuntimeError(msg)

    OUT.parent.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=False)
    (OUT / "arrays").mkdir()
    write_json(
        OUT / "replay-source.json",
        {
            "executed_source": str(Path(__file__).resolve()),
            "executed_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "original_source_sha256": hashlib.sha256((SOURCE / "analyze.py").read_bytes()).hexdigest(),
            "original_protocol_sha256": hashlib.sha256((SOURCE / "protocol.json").read_bytes()).hexdigest(),
        },
    )
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()

    if hashlib.sha256(DATA.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        msg = "Cached dataset hash changed."
        raise RuntimeError(msg)

    with np.load(DATA, allow_pickle=False) as archive:
        images = archive["x_train"][1500:1875].astype(np.float32)[..., None]
        labels = archive["y_train"][1500:1875]

    images = (images / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    events = [json.loads(line) for line in (PILOT / "events.jsonl").read_text().splitlines()]
    starts = [row for row in events if row["event"] == "phase_start"]
    ends = [row for row in events if row["event"] == "phase_end" and row["complete"]]
    initials = initializations(starts, images, labels)
    rows = trained_checkpoints(ends, images, labels)

    if len(rows) != EXPECTED_CHECKPOINTS or len(initials) != EXPECTED_INITIALIZATIONS:
        msg = "Incomplete panel; no aggregate analysis is permitted."
        raise RuntimeError(msg)

    result = {
        "complete": True,
        "seconds": time.perf_counter() - started,
        "training_updates": 0,
        "official_test_examples": 0,
        "initials": initials,
        "trained": rows,
        **summarize(rows, initials),
    }
    write_json(OUT / "results.json", result)


if __name__ == "__main__":
    main()
