# One-off evidence script invoked by path, outside any import package.  # noqa: INP001
"""Measure current-model proxy trajectories on a fixed, exploratory MNIST protocol.

All observations use official training examples. This five-epoch endpoint checks the correlation pipeline;
it cannot establish the relationship with converged, 100-epoch generalization. Every candidate attempt is
recorded, and correlations require the complete preregistered set of sixteen successful candidates.
"""

import hashlib
import json
import math
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
from scipy.stats import spearmanr

from image.src.config import Config
from image.src.util import BATCH_SIZE, get_mnist_model

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
DATASET = Path(os.environ["KERAS_HOME"]) / "datasets/mnist.npz"
SEEDS = (31, 47)
EPOCHS = 5
N_TRAIN = 1500
N_VALIDATION = 375
SECONDS_LIMIT = 180
CONFIGS = [
    Config(
        n_filters=32,
        kernel_size=2,
        padding="same",
        n_blocks=2,
        n_units=128,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    for learning_rate, weight_decay in product((0.0001, 0.0003, 0.001, 0.003), (0.00001, 0.001))
]


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(value: Mapping[str, object]) -> None:
    with (OUTPUT / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def weight_hash(model: keras.Model) -> str:
    digest = hashlib.sha256()

    for weight in model.get_weights():
        digest.update(weight.tobytes())

    return digest.hexdigest()


def measure(model: keras.Model, inference: tf.types.experimental.PolymorphicFunction, images: np.ndarray) -> dict:
    """Measure the repository's batch RMS proxy with the model frozen in inference mode.

    The maximum includes the final partial batch. Every batch's probability and hidden-activation RMS is retained,
    so the p*h/W and reviewer h/W alternatives can be reconstructed without training or choosing another subset.
    """
    weights = model.layers[-1].weights[0].numpy()
    weight_rms = float(np.sqrt(np.mean(weights**2)))
    batches = []

    for start in range(0, len(images), BATCH_SIZE):
        probability, hidden = inference(images[start : start + BATCH_SIZE], training=False)
        probability_rms = float(np.sqrt(np.mean(probability.numpy() ** 2)))
        hidden_rms = float(np.sqrt(np.mean(hidden.numpy() ** 2)))
        batches.append(
            {
                "start": start,
                "stop": min(start + BATCH_SIZE, len(images)),
                "p_rms": probability_rms,
                "h_rms": hidden_rms,
                "proxy": probability_rms * hidden_rms / weight_rms,
                "h_over_w": hidden_rms / weight_rms,
                "h_over_w_frobenius": float(np.linalg.norm(hidden.numpy()) / np.linalg.norm(weights)),
            }
        )

    maximum = max(batches, key=lambda batch: batch["proxy"])

    return {
        "w_rms": weight_rms,
        "proxy": maximum["proxy"],
        "proxy_max_batch_start": maximum["start"],
        "h_over_w": max(batch["h_over_w"] for batch in batches),
        "h_over_w_frobenius": max(batch["h_over_w_frobenius"] for batch in batches),
        "batches": batches,
    }


def analyze(rows: list[dict]) -> None:
    """Require all candidates before computing across-configuration rank correlations.

    Seeds define separate eight-configuration comparisons. Pooled correlations are supplementary because repeated
    configurations and shared data make those observations dependent; no p-values or confirmatory claims are produced.
    """
    correlations = []

    for seed in (*SEEDS, "pooled"):
        selected = [row for row in rows if seed == "pooled" or row["seed"] == seed]
        final_losses = {
            (row["seed"], row["config_id"]): row["validation_loss"] for row in selected if row["epoch"] == EPOCHS
        }

        for epoch in range(1, EPOCHS + 1):
            current = [row for row in selected if row["epoch"] == epoch]
            final = [final_losses[row["seed"], row["config_id"]] for row in current]
            correlations.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "n_candidates": len(current),
                    "proxy_vs_final_loss": float(spearmanr([row["proxy"] for row in current], final).statistic),
                    "h_over_w_vs_final_loss": float(spearmanr([row["h_over_w"] for row in current], final).statistic),
                    "h_over_w_frobenius_vs_final_loss": float(
                        spearmanr([row["h_over_w_frobenius"] for row in current], final).statistic
                    ),
                    "validation_loss_vs_final_loss": float(
                        spearmanr([row["validation_loss"] for row in current], final).statistic
                    ),
                    "learning_rate_vs_proxy": float(
                        spearmanr(
                            [row["config"]["learning_rate"] for row in current], [row["proxy"] for row in current]
                        ).statistic
                    ),
                }
            )

    write_json(
        OUTPUT / "correlations.json",
        {
            "interpretation": "Positive rho associates smaller values with smaller final loss; no significance claim.",
            "pooled_warning": "Supplementary only; configurations repeat across seeds and all runs share data.",
            "rows": correlations,
        },
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for axis, seed in zip(axes, SEEDS, strict=True):
        selected = [row for row in correlations if row["seed"] == seed]

        for metric, label in (
            ("proxy_vs_final_loss", "Current p*h/W proxy"),
            ("h_over_w_vs_final_loss", "Current RMS h/W"),
            ("h_over_w_frobenius_vs_final_loss", "Current Frobenius h/W"),
            ("validation_loss_vs_final_loss", "Current validation loss"),
        ):
            axis.plot(range(1, EPOCHS + 1), [row[metric] for row in selected], marker="o", label=label)

        axis.set(title=f"Initialization seed {seed} (8 configurations)", xlabel="Epoch", ylim=(-1.05, 1.05))
        axis.set_xticks(range(1, EPOCHS + 1))
        axis.axhline(0, color="gray", linewidth=0.7)
        axis.grid(alpha=0.2)

    axes[0].set_ylabel("Spearman rho with epoch-5 validation loss")
    axes[1].legend(fontsize=8, loc="lower right")
    figure.suptitle("Exploratory MNIST trajectory diagnostic: 1,500 train / 375 validation")
    figure.tight_layout()
    figure.savefig(OUTPUT / "correlations.png", dpi=180)
    figure.savefig(OUTPUT / "correlations.pdf")
    plt.close(figure)


def run_candidate(
    seed: int, config_id: int, config: Config, images: np.ndarray, labels: np.ndarray, initial_hashes: dict
) -> list[dict]:
    """Train one complete candidate and preserve its current-model measurements and final checkpoint.

    All configurations in a seed replicate must share the same initial weights. This function raises if initialization
    hashes differ or if the optimizer step count differs from five full epochs, and the caller records that failure.
    """
    candidate_started = time.perf_counter()
    identity = {"seed": seed, "config_id": config_id, "config": config}
    steps = EPOCHS * math.ceil(N_TRAIN / BATCH_SIZE)
    rows = []
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = get_mnist_model(config, decay_steps=steps)
    model.build((None, 28, 28, 1))
    initialization = weight_hash(model)
    initial_hashes.setdefault(seed, initialization)

    if initialization != initial_hashes[seed]:
        message = "Configurations did not receive identical initialization within seed."
        raise RuntimeError(message)

    inference_model = keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output])
    inference = tf.function(inference_model, reduce_retracing=True)
    rng = np.random.default_rng(seed)

    for epoch in range(1, EPOCHS + 1):
        permutation = rng.permutation(N_TRAIN)

        for start in range(0, N_TRAIN, BATCH_SIZE):
            batch = permutation[start : start + BATCH_SIZE]
            model.train_on_batch(images[batch], labels[batch])

        validation = model(images[N_TRAIN:], training=False).numpy()
        measurement = measure(model, inference, images[:N_TRAIN])
        row = {
            **identity,
            "epoch": epoch,
            **measurement,
            "validation_loss": float(
                np.mean(keras.losses.categorical_crossentropy(labels[N_TRAIN:], validation).numpy())
            ),
            "validation_accuracy": float(np.mean(validation.argmax(axis=1) == labels[N_TRAIN:].argmax(axis=1))),
            "optimizer_steps": int(model.optimizer.iterations.numpy()),
            "learning_rate_after_epoch": float(model.optimizer.learning_rate.numpy()),
            "shuffle_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
            "seconds_since_candidate_start": time.perf_counter() - candidate_started,
        }
        rows.append(row)
        record({"event": "epoch", **row})

    if int(model.optimizer.iterations.numpy()) != steps:
        message = "Candidate did not complete exactly the expected optimizer steps."
        raise RuntimeError(message)

    checkpoint = OUTPUT / f"seed{seed}-config{config_id}.keras"
    model.save(checkpoint)
    record(
        {
            "event": "candidate_complete",
            **identity,
            "initialization_sha256": initialization,
            "final_weights_sha256": weight_hash(model),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "seconds": time.perf_counter() - candidate_started,
        }
    )

    return rows


def main() -> None:
    """Run the fixed protocol once, preserving complete, failed, and unstarted candidates.

    A wall-clock guard stops starting candidates after 180 seconds of CPU-only execution, after finishing the active
    candidate. The file-existence guard prevents overwriting results. An exception is recorded and re-raised.
    """
    if (OUTPUT / "metadata.json").exists():
        message = "This pilot already has metadata; preserve the existing results."
        raise FileExistsError(message)

    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    with np.load(DATASET) as archive:
        images = archive["x_train"][: N_TRAIN + N_VALIDATION].astype(np.float32)[..., None]
        labels = keras.utils.to_categorical(archive["y_train"][: N_TRAIN + N_VALIDATION], 10)

    images = (images / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    steps = EPOCHS * math.ceil(N_TRAIN / BATCH_SIZE)
    source_paths = [Path(__file__), ROOT / "image/src/util.py", ROOT / "image/src/config.py", ROOT / "pyproject.toml"]
    sources = {}

    for source in source_paths:
        target = OUTPUT / "sources" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        sources[str(source.relative_to(ROOT))] = hashlib.sha256(source.read_bytes()).hexdigest()

    write_json(
        OUTPUT / "metadata.json",
        {
            "status": "exploratory short-endpoint diagnostic",
            "git_head": "b2d7116c2b154081342f88618be15a144054eb59",
            "started_utc": datetime.now(UTC).isoformat(),
            "question": "Does current proxy rank fifth-epoch validation loss better than current validation loss does?",
            "reviewer": "6ZG2, across-run epochwise correlation request (2024-08-13)",
            "hypothesis": "Lower proxy ranks predict lower final loss; compare rho with validation-loss baseline.",
            "train_indices": [0, N_TRAIN],
            "validation_indices": [N_TRAIN, N_TRAIN + N_VALIDATION],
            "index_interpretation": "half-open intervals in official training partition; test arrays never loaded",
            "normalization": "float32, (pixel/255 - 0.1307)/0.3081, channels_last",
            "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            "source_sha256": sources,
            "command": sys.argv,
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy", "scipy", "matplotlib")},
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "KERAS_HOME",
                    "MPLCONFIGDIR",
                    "XDG_CACHE_HOME",
                    "PYTHONPATH",
                    "TF_NUM_INTRAOP_THREADS",
                    "TF_NUM_INTEROP_THREADS",
                )
            },
            "device": "CPU only",
            "seeds": SEEDS,
            "configs": CONFIGS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "cosine_decay_steps": steps,
            "early_stopping": False,
            "wall_seconds_cap_between_candidates": SECONDS_LIMIT,
            "shuffle": "numpy default_rng(seed), permutation each epoch; identical across configs within seed",
            "initialization": "keras set_random_seed before each model; hashes must agree across configs within seed",
            "architecture_note": "get_mnist_model hardcodes 128 hidden units; explicit n_units=128 matches it",
        },
    )
    started = time.perf_counter()
    rows = []
    completed = []
    initial_hashes = {}
    candidates = list(product(SEEDS, enumerate(CONFIGS)))

    for seed, (config_id, config) in candidates:
        if time.perf_counter() - started >= SECONDS_LIMIT:
            break

        identity = {"seed": seed, "config_id": config_id, "config": config}
        record({"event": "candidate_start", **identity})

        try:
            rows.extend(run_candidate(seed, config_id, config, images, labels, initial_hashes))
            completed.append((seed, config_id))
            sys.stdout.write(
                f"Completed seed={seed} config={config_id}, elapsed={time.perf_counter() - started:.1f}s\n"
            )
            sys.stdout.flush()
        except BaseException as error:
            record({"event": "candidate_failed", **identity, "error": repr(error), "traceback": traceback.format_exc()})
            raise

    complete = len(completed) == len(candidates)
    write_json(
        OUTPUT / "summary.json",
        {
            "complete": complete,
            "completed": completed,
            "required_candidates": len(candidates),
            "epoch_rows": len(rows),
            "seconds": time.perf_counter() - started,
            "unstarted": [
                {"seed": seed, "config_id": config_id}
                for seed, (config_id, _) in candidates
                if (seed, config_id) not in completed
            ],
            "limitation": "Five epochs, small subset, eight configs, two seeds; no converged generalization claim.",
        },
    )

    if complete:
        analyze(rows)


if __name__ == "__main__":
    main()
