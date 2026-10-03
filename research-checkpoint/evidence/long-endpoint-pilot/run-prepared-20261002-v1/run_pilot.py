# Standalone evidence script invoked by path, outside any import package.  # noqa: INP001
"""Run the frozen 100-epoch exploratory panel without using official test arrays.

The command takes the protocol SHA256 as its only argument. Existing results cannot be overwritten, and an
incomplete panel never produces correlations. Source hashes and paired initialization and shuffle hashes are checked.
"""

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.special import logsumexp
from scipy.stats import spearmanr

from image.src.config import Config
from image.src.util import BATCH_SIZE, get_mnist_model

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
DATASET = Path(os.environ["KERAS_HOME"]) / "datasets/mnist.npz"
PROTOCOL_PATH = OUT / "protocol.json"
SEEDS = (71, 89)
EPOCHS = 100
TRAIN = 1500
VALIDATION = 375
STEPS = 2400
SECONDS_LIMIT = 900
RECONSTRUCTION_TOLERANCE = 1e-5
REGRET_EPOCHS = (1, 2, 5, 10, 20, 50)
METRICS = ("original_rms_ratio_max", "relative_head_trace_centered", "temperature_curvature", "validation_loss")
CONFIGS = [
    Config(n_filters=32, kernel_size=2, padding="same", n_blocks=2, n_units=128, learning_rate=lr, weight_decay=wd)
    for lr, wd in product((0.0001, 0.0003, 0.001, 0.003), (0.00001, 0.001))
]


def require(*, condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(event: Mapping[str, object]) -> None:
    with (OUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")


def weight_hash(model: keras.Model) -> str:
    digest = hashlib.sha256()

    for weight in model.get_weights():
        digest.update(weight.tobytes())

    return digest.hexdigest()


def classification(logits: np.ndarray, probability: np.ndarray, labels: np.ndarray) -> dict:
    reconstructed = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    difference = float(np.max(np.abs(reconstructed - probability)))
    require(
        condition=difference <= RECONSTRUCTION_TOLERANCE,
        message="Float64 reconstruction exceeds frozen probability tolerance.",
    )

    return {
        "loss": float(np.mean(logsumexp(logits, axis=1) - np.sum(logits * labels, axis=1))),
        "accuracy": float(np.mean(probability.argmax(axis=1) == labels.argmax(axis=1))),
        "probability_reconstruction_max_abs_difference": difference,
    }


def measure(
    model: keras.Model, inference: tf.types.experimental.PolymorphicFunction, images: np.ndarray, labels: np.ndarray
) -> dict:
    """Compute original float32 RMS batches and the frozen float64 C1/C2 controls.

    The original score includes the final 28-example batch. C1 and C2 use sample-weighted means over all 1500
    examples and stable softmax reconstructed from frozen activations, kernel, and bias. Training CE uses stable,
    reconstructed logits without probability clipping.
    """
    kernel, bias = model.layers[-1].get_weights()
    weight_rms = float(np.sqrt(np.mean(kernel**2)))
    kernel64 = kernel.astype(np.float64)
    centered = kernel64 - kernel64.mean(axis=1, keepdims=True)
    row_scales = np.sum(centered**2, axis=1)
    batches, probabilities, all_logits, c1_parts, c2_parts = [], [], [], [], []

    for start in range(0, TRAIN, BATCH_SIZE):
        probability_tensor, hidden_tensor = inference(images[start : start + BATCH_SIZE], training=False)
        probability, hidden = probability_tensor.numpy(), hidden_tensor.numpy()
        probabilities.append(probability)
        p_rms, h_rms = float(np.sqrt(np.mean(probability**2))), float(np.sqrt(np.mean(hidden**2)))
        batches.append(
            {"start": start, "count": len(hidden), "p_rms": p_rms, "h_rms": h_rms, "score": p_rms * h_rms / weight_rms}
        )

        hidden64 = hidden.astype(np.float64)
        logits = hidden64 @ kernel64 + bias.astype(np.float64)
        logits -= logits.mean(axis=1, keepdims=True)
        all_logits.append(logits)
        exponent = np.exp(logits - logits.max(axis=1, keepdims=True))
        p64 = exponent / exponent.sum(axis=1, keepdims=True)

        c1_parts.append((1 - np.sum(p64**2, axis=1)) * (hidden64**2 @ row_scales))
        c2_parts.append(np.sum(p64 * logits**2, axis=1) - np.sum(p64 * logits, axis=1) ** 2)

    training = classification(np.concatenate(all_logits), np.concatenate(probabilities), labels)

    return {
        "original_rms_ratio_max": max(batch["score"] for batch in batches),
        "relative_head_trace_centered": float(np.concatenate(c1_parts).mean()),
        "temperature_curvature": float(np.concatenate(c2_parts).mean()),
        "training_loss": training["loss"],
        "training_accuracy": training["accuracy"],
        "training_probability_reconstruction_error": training["probability_reconstruction_max_abs_difference"],
        "weight_rms": weight_rms,
        "batches": batches,
    }


def terminal_gradient(model: keras.Model, images: np.ndarray, labels: np.ndarray) -> float:
    """Measure full-network mean training CE gradient, excluding optimizer weight decay."""
    gradients = [np.zeros(variable.shape, dtype=np.float64) for variable in model.trainable_variables]

    for start in range(0, TRAIN, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, TRAIN)

        with tf.GradientTape() as tape:
            probability = model(images[start:stop], training=False)
            loss = tf.reduce_mean(keras.losses.categorical_crossentropy(labels[start:stop], probability))

        for aggregate, gradient in zip(gradients, tape.gradient(loss, model.trainable_variables), strict=True):
            aggregate[...] += gradient.numpy().astype(np.float64) * (stop - start) / TRAIN

    return float(np.sqrt(sum(np.sum(gradient**2) for gradient in gradients)))


def correlation(left: list[float], right: list[float]) -> float | None:
    value = float(spearmanr(left, right).statistic)
    return value if np.isfinite(value) else None


def analyze(rows: list[dict]) -> None:
    """Analyze the complete panel with fixed minimization directions and both declared targets.

    Every seed is analyzed separately. Learning-rate-collapsed points average both weight-decay configurations;
    two observations within a learning rate do not support correlation inference. Selection ties use configuration ID.
    """
    correlations, regrets = [], []

    for seed in SEEDS:
        selected = [row for row in rows if row["seed"] == seed]
        endpoints = {row["config_id"]: row["validation_loss"] for row in selected if row["epoch"] == EPOCHS}
        best = {
            i: min(row["validation_loss"] for row in selected if row["config_id"] == i) for i in range(len(CONFIGS))
        }

        for target_name, targets in (("epoch100_validation_loss", endpoints), ("best_across100_validation_loss", best)):
            for epoch in range(1, EPOCHS + 1):
                current = [row for row in selected if row["epoch"] == epoch]
                final = [targets[row["config_id"]] for row in current]
                rates = [row["config"]["learning_rate"] for row in current]
                unique_rates = sorted(set(rates))

                for metric in METRICS:
                    scores = [row[metric] for row in current]
                    mean_scores = [
                        float(np.mean([s for s, lr in zip(scores, rates, strict=True) if lr == rate]))
                        for rate in unique_rates
                    ]
                    mean_targets = [
                        float(np.mean([s for s, lr in zip(final, rates, strict=True) if lr == rate]))
                        for rate in unique_rates
                    ]
                    correlations.append(
                        {
                            "seed": seed,
                            "epoch": epoch,
                            "metric": metric,
                            "target": target_name,
                            "rho": correlation(scores, final),
                            "n": len(current),
                            "learning_rate_vs_score_rho": correlation(rates, scores),
                            "learning_rate_vs_target_rho": correlation(rates, final),
                            "lr_mean_sensitivity_rho": correlation(mean_scores, mean_targets),
                            "lr_mean_sensitivity_n": len(unique_rates),
                            "learning_rates": unique_rates,
                            "lr_mean_scores": mean_scores,
                            "lr_mean_targets": mean_targets,
                        }
                    )

                    if epoch in REGRET_EPOCHS:
                        chosen = min(current, key=lambda row: (row[metric], row["config_id"]))
                        regrets.append(
                            {
                                "seed": seed,
                                "epoch": epoch,
                                "metric": metric,
                                "target": target_name,
                                "chosen_config_id": chosen["config_id"],
                                "chosen_score": chosen[metric],
                                "selected_target_loss": targets[chosen["config_id"]],
                                "oracle_target_loss": min(targets.values()),
                                "regret": targets[chosen["config_id"]] - min(targets.values()),
                            }
                        )

    write_json(
        OUT / "analysis.json",
        {
            "correlations": correlations,
            "selection_regrets": regrets,
            "interpretation": "Positive rho favors minimization; no p-values or causal claims.",
            "secondary_caveat": "Best-across100 reuses validation; not unbiased test evidence.",
            "lr_caveat": "Four LR means are descriptive; within-LR n=2 is insufficient.",
        },
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for axis, seed in zip(axes, SEEDS, strict=True):
        for metric in METRICS:
            values = [
                row["rho"]
                for row in correlations
                if row["seed"] == seed and row["metric"] == metric and row["target"] == "epoch100_validation_loss"
            ]
            axis.plot(
                range(1, EPOCHS + 1),
                values,
                label={
                    "original_rms_ratio_max": "Original RMS ratio",
                    "relative_head_trace_centered": "C1: relative head trace",
                    "temperature_curvature": "C2: temperature curvature",
                    "validation_loss": "Validation CE",
                }[metric],
            )

        axis.set(title=f"Seed {seed}: eight paired configurations", xlabel="Epoch", ylim=(-1.05, 1.05))
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Spearman rho with epoch-100 validation CE")
    handles, labels = axes[1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    figure.suptitle("Exploratory MNIST: 1500 train / 375 validation; all metrics minimized")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(OUT / "correlations.png", dpi=180)
    figure.savefig(OUT / "correlations.pdf")
    plt.close(figure)


def run_candidate(identity: dict, images: np.ndarray, labels: np.ndarray, state: dict, started: float) -> bool:
    """Preserve each epoch and checkpoint; stop at the deadline without analyzing partial panels.

    Both RNG initialization and all epoch shuffle hashes must match across configurations within a seed. The final
    saved model includes optimizer state. A stopped or failed candidate retains its current model and next RNG state.
    """
    seed, config_id = identity["seed"], identity["config_id"]
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = get_mnist_model(identity["config"], decay_steps=STEPS)
    model.build((None, 28, 28, 1))
    initial = weight_hash(model)
    state["initializations"].setdefault(seed, initial)

    require(condition=initial == state["initializations"][seed], message="Initialization differs within seed.")

    inference = tf.function(
        keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output]), reduce_retracing=True
    )
    rng = np.random.default_rng(seed)
    epoch = 0
    complete = False
    record({"event": "candidate_start", **identity, "initialization_sha256": initial})

    try:
        for next_epoch in range(1, EPOCHS + 1):
            if time.perf_counter() - started >= SECONDS_LIMIT:
                break
            permutation = rng.permutation(TRAIN)
            shuffle_hash = hashlib.sha256(permutation.tobytes()).hexdigest()
            key = (seed, next_epoch)
            state["shuffles"].setdefault(key, shuffle_hash)

            require(condition=shuffle_hash == state["shuffles"][key], message="Epoch shuffle differs within seed.")

            for start in range(0, TRAIN, BATCH_SIZE):
                batch = permutation[start : start + BATCH_SIZE]
                model.train_on_batch(images[batch], labels[batch])

            probability, hidden = inference(images[TRAIN:], training=False)
            kernel, bias = model.layers[-1].get_weights()
            logits = hidden.numpy().astype(np.float64) @ kernel.astype(np.float64) + bias.astype(np.float64)
            validation = classification(logits, probability.numpy(), labels[TRAIN:])
            epoch = next_epoch
            row = {
                **identity,
                "epoch": epoch,
                **measure(model, inference, images[:TRAIN], labels[:TRAIN]),
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_probability_reconstruction_error": validation[
                    "probability_reconstruction_max_abs_difference"
                ],
                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                "learning_rate_after_epoch": float(model.optimizer.learning_rate.numpy()),
                "shuffle_sha256": shuffle_hash,
                "seconds_since_panel_start": time.perf_counter() - started,
            }
            state["rows"].append(row)
            record({"event": "epoch", **row})

            if epoch == 1:
                checkpoint = OUT / f"seed{seed}-config{config_id}-epoch1.keras"
                model.save(checkpoint)
                record(
                    {
                        "event": "early_checkpoint",
                        **identity,
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "weights_sha256": weight_hash(model),
                    }
                )

        if epoch == EPOCHS and int(model.optimizer.iterations.numpy()) == STEPS:
            record(
                {
                    "event": "terminal_gradient",
                    **identity,
                    "training_ce_gradient_l2": terminal_gradient(model, images[:TRAIN], labels[:TRAIN]),
                }
            )
            complete = True
    except BaseException as error:
        record({"event": "candidate_failed", **identity, "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        checkpoint = OUT / f"seed{seed}-config{config_id}-{'epoch100' if complete else 'partial'}.keras"
        model.save(checkpoint)
        record(
            {
                "event": "candidate_complete" if complete else "candidate_incomplete",
                **identity,
                "completed_epochs": epoch,
                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                "weights_sha256": weight_hash(model),
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "shuffle_rng_state": rng.bit_generator.state,
            }
        )

    return complete


def main() -> None:
    """Verify frozen inputs, execute the bounded panel, and analyze only complete results.

    The 900-second guard is checked before every epoch. One active epoch and checkpoint I/O may extend beyond it.
    Failures and interrupted panels retain all epoch records and checkpoints; this command never resumes implicitly.
    """
    protocol_hash = hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()

    require(condition=sys.argv[1:] == [protocol_hash], message="Pass the reviewed frozen protocol SHA256.")

    if (OUT / "metadata.json").exists():
        message = "Existing pilot metadata must be preserved."
        raise FileExistsError(message)

    protocol = json.loads(PROTOCOL_PATH.read_text())

    for relative, expected in protocol["source_sha256"].items():
        source = ROOT / relative

        require(
            condition=hashlib.sha256(source.read_bytes()).hexdigest() == expected,
            message=f"Source hash changed: {relative}",
        )

        destination = OUT / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    require(
        condition=hashlib.sha256(DATASET.read_bytes()).hexdigest() == protocol["dataset_sha256"],
        message="Cached dataset hash changed.",
    )

    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    with np.load(DATASET) as archive:
        images = archive["x_train"][: TRAIN + VALIDATION].astype(np.float32)[..., None]
        labels = keras.utils.to_categorical(archive["y_train"][: TRAIN + VALIDATION], 10)

    images = (images / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    write_json(
        OUT / "metadata.json",
        {
            "protocol_sha256": protocol_hash,
            "started_utc": datetime.now(UTC).isoformat(),
            "command": sys.argv,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy", "scipy", "matplotlib")},
            "array_hashes": {
                "images": hashlib.sha256(images.tobytes()).hexdigest(),
                "labels": hashlib.sha256(labels.tobytes()).hexdigest(),
            },
            "device": "CPU only",
        },
    )
    state = {"initializations": {}, "shuffles": {}, "rows": []}
    completed = []
    candidates = [(seed, i) for seed in SEEDS for i in range(len(CONFIGS))]
    started = time.perf_counter()

    try:
        for seed, config_id in candidates:
            identity = {"seed": seed, "config_id": config_id, "config": CONFIGS[config_id]}

            if not run_candidate(identity, images, labels, state, started):
                break

            completed.append((seed, config_id))
            sys.stdout.write(
                f"Completed seed={seed} config={config_id}; elapsed={time.perf_counter() - started:.1f}s\n"
            )
            sys.stdout.flush()
    finally:
        write_json(
            OUT / "summary.json",
            {
                "complete": completed == candidates,
                "completed": completed,
                "required_candidates": len(candidates),
                "epoch_rows": len(state["rows"]),
                "seconds": time.perf_counter() - started,
                "remaining": [i for i in candidates if i not in completed],
                "interpretation": "Exploratory small-data panel; 100 epochs does not establish convergence.",
            },
        )

    if completed == candidates:
        analyze(state["rows"])


if __name__ == "__main__":
    main()
