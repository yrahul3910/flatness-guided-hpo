# Standalone evidence script invoked by path, outside any import package.  # noqa: INP001
"""Compare one-epoch probe scores against restored-checkpoint outcomes in a frozen panel.

This script requires the reviewed protocol hash on the command line. It uses the installed Keras EarlyStopping
callback with actual Keras validation loss, while retaining stable float64 losses as separate diagnostics.
"""

import hashlib
import json
import os
import platform
import runpy
import shutil
import sys
import time
import traceback
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr

from image.src import util as image_util
from image.src.util import BATCH_SIZE, get_mnist_model

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
HELPER_PATH = OUT.parents[1] / "long-endpoint-pilot/run-prepared-20261002-v1/run_pilot.py"
HELPERS = runpy.run_path(str(HELPER_PATH), run_name="frozen_long_pilot_helpers")
CONFIGS = HELPERS["CONFIGS"]
require, write_json, weight_hash = HELPERS["require"], HELPERS["write_json"], HELPERS["weight_hash"]
classification = HELPERS["classification"]
BLOCKS = (307, 331)
TRAIN, VALIDATION, FINAL_EPOCHS, SECONDS_LIMIT = 1500, 375, 100, 900
PHASES = ("probe", "final")
STEPS_PER_EPOCH = 24
LEGACY_THRESHOLD = 0.1


def record(event: Mapping[str, object], *, primary_error: BaseException | None = None) -> None:
    try:
        with (OUT / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
    except OSError as error:
        if primary_error is None:
            raise
        sys.stderr.write(json.dumps({**event, "event_logging_error": repr(error)}) + "\n")


def check_pair_hash(hashes: dict, key: tuple, value: str) -> None:
    hashes.setdefault(key, value)
    require(condition=hashes[key] == value, message=f"Paired hash differs for {key}.")


def validation_state(
    model: keras.Model, inference: tf.types.experimental.PolymorphicFunction, images: np.ndarray, labels: np.ndarray
) -> dict:
    """Keep actual Keras validation metrics distinct from reconstructed float64 diagnostics.

    Reset compiled metric accumulators, then evaluate every validation batch in order. Keras's test step weights its
    CE accumulator by batch size. The resulting loss is the exact quantity supplied to the installed stopping callback.
    """
    model.reset_metrics()

    for start in range(TRAIN, TRAIN + VALIDATION, BATCH_SIZE):
        logs = model.test_on_batch(
            images[start : start + BATCH_SIZE], labels[start : start + BATCH_SIZE], return_dict=True
        )

    probability, hidden = inference(images[TRAIN:], training=False)
    kernel, bias = model.layers[-1].get_weights()
    logits = hidden.numpy().astype(np.float64) @ kernel.astype(np.float64) + bias.astype(np.float64)
    stable = classification(logits, probability.numpy(), labels[TRAIN:])
    monitor = float(logs["loss"])
    require(condition=np.isfinite(monitor), message="Keras validation CE is nonfinite.")

    return {
        "keras_validation_ce": monitor,
        "keras_validation_accuracy": float(logs["accuracy"]),
        "stable_validation_ce": stable["loss"],
        "validation_accuracy": stable["accuracy"],
        "probability_reconstruction_error": stable["probability_reconstruction_max_abs_difference"],
    }


def probe_score(
    model: keras.Model, inference: tf.types.experimental.PolymorphicFunction, images: np.ndarray, order: np.ndarray
) -> dict:
    """Measure the RMS maximum in probe subset order, including its final 28 examples.

    Every component and invalid batch is retained. Finite maxima use the production positive-score rule; its infinity
    fallback is represented by JSON null plus an explicit status, which prevents partial-panel aggregate analysis.
    """
    denominator = np.sqrt(np.mean(model.layers[-1].get_weights()[0] ** 2))
    batches, finite_scores = [], []

    for start in range(0, TRAIN, BATCH_SIZE):
        probability, hidden = inference(images[order[start : start + BATCH_SIZE]], training=False)
        p_rms = np.sqrt(np.mean(probability.numpy() ** 2))
        h_rms = np.sqrt(np.mean(hidden.numpy() ** 2))

        with np.errstate(divide="ignore", invalid="ignore"):
            score = float(p_rms * h_rms / denominator)

        components = {"p_rms": float(p_rms), "h_rms": float(h_rms), "score": score}
        batches.append(
            {
                "start": start,
                "count": min(BATCH_SIZE, TRAIN - start),
                **{name: value if np.isfinite(value) else None for name, value in components.items()},
                "nonfinite_components": {
                    name: repr(value) for name, value in components.items() if not np.isfinite(value)
                },
            }
        )
        if np.isfinite(score):
            finite_scores.append(score)

    best = max(finite_scores, default=-np.inf)
    value = float(best) if best > 0 else None

    return {
        "original_rms_ratio_max": value,
        "legacy_eligible": value is not None and value > LEGACY_THRESHOLD,
        "score_status": "finite_positive" if value is not None else "positive_infinity_get_convexity_fallback",
        "weight_rms": float(denominator) if np.isfinite(denominator) else None,
        "weight_rms_nonfinite": None if np.isfinite(denominator) else repr(float(denominator)),
        "score_batches": batches,
    }


def save_checkpoint(model: keras.Model, path: Path, identity: dict, primary_error: BaseException | None = None) -> dict:
    """Retain checkpoint failures without replacing an active training exception.

    Save and hashing errors are recorded, with stderr as a fallback if the event file itself cannot be written.
    Without an active primary exception, the checkpoint error propagates and the phase cannot count as successful.
    """
    try:
        model.save(path)
        checkpoint_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        weights_hash = weight_hash(model)
    except Exception as error:
        failure = {
            "event": "checkpoint_failed",
            **identity,
            "checkpoint": path.name,
            "error": repr(error),
            "primary_error": repr(primary_error) if primary_error else None,
        }

        record(failure, primary_error=primary_error or error)

        if primary_error is None:
            raise
        return {"checkpoint": path.name, "checkpoint_saved": False, "checkpoint_error": repr(error)}

    return {
        "checkpoint": path.name,
        "checkpoint_saved": True,
        "checkpoint_sha256": checkpoint_hash,
        "weights_sha256": weights_hash,
    }


def initialize_phase(
    identity: dict, state: dict
) -> tuple[keras.Model, np.ndarray, np.random.Generator, str, keras.callbacks.EarlyStopping]:
    """Pair initial weights and data orders by block and phase, independently of candidate ranking.

    Seeds depend only on the declared block/phase mapping. Probe measurement order is randomized separately from
    training shuffles. All hashes and the explicit order array are retained before any optimizer update.
    """
    block, phase = identity["seed_block"], identity["phase"]
    phase_seed = block * 1000 + (1 if phase == "probe" else 2)
    subset_seed, shuffle_seed = block * 1000 + 3, phase_seed + 10
    keras.backend.clear_session()
    keras.utils.set_random_seed(phase_seed)
    model = get_mnist_model(
        identity["config"], decay_steps=FINAL_EPOCHS * STEPS_PER_EPOCH if phase == "final" else None
    )
    model.build((None, 28, 28, 1))

    initial = weight_hash(model)
    check_pair_hash(state["initializations"], (block, phase), initial)
    order = np.random.default_rng(subset_seed).permutation(TRAIN) if phase == "probe" else np.arange(TRAIN)
    rng = np.random.default_rng(shuffle_seed)
    np.save(OUT / f"block{block}-config{identity['config_id']}-{phase}-order.npy", order)
    record(
        {
            "event": "phase_start",
            **identity,
            "phase_seed": phase_seed,
            "subset_seed": subset_seed,
            "shuffle_seed": shuffle_seed,
            "initialization_sha256": initial,
            "measurement_order_sha256": hashlib.sha256(order.tobytes()).hexdigest(),
        }
    )

    callback = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    callback.set_model(model)
    callback.set_params({"epochs": FINAL_EPOCHS, "steps": STEPS_PER_EPOCH, "verbose": 0})
    callback.on_train_begin()
    model.stop_training = False

    return model, order, rng, initial, callback


def run_phase(identity: dict, images: np.ndarray, labels: np.ndarray, state: dict, started: float) -> dict | None:
    """Use independent probe/final seeds and the installed callback's stopping and restoration logic.

    All configurations share initialization and shuffle prefixes within their block and phase. Final training
    cold-starts, and restores weights only after its last epoch. The optimizer retains terminal state, as in production.
    A time-limited final phase preserves its last weights before restoration and marks the whole panel incomplete.
    """
    block, phase = identity["seed_block"], identity["phase"]
    model, order, rng, initial, callback = initialize_phase(identity, state)
    inference = tf.function(
        keras.Model(model.inputs, [model.layers[-1].output, model.layers[-2].output]), reduce_retracing=True
    )
    epoch, finished, restored = 0, False, False
    prefix = f"block{block}-config{identity['config_id']}-{phase}"
    maximum = 1 if phase == "probe" else FINAL_EPOCHS
    measures = {}
    primary_error = None
    phase_started = time.perf_counter()

    try:
        for next_epoch in range(1, maximum + 1):
            if time.perf_counter() - started >= SECONDS_LIMIT:
                break
            permutation = order[rng.permutation(TRAIN)]
            digest = hashlib.sha256(permutation.tobytes()).hexdigest()
            check_pair_hash(state["shuffles"], (block, phase, next_epoch), digest)
            model.reset_metrics()

            for start in range(0, TRAIN, BATCH_SIZE):
                batch = permutation[start : start + BATCH_SIZE]
                online = model.train_on_batch(images[batch], labels[batch], return_dict=True)

            epoch = next_epoch
            require(
                condition=int(model.optimizer.iterations.numpy()) == epoch * STEPS_PER_EPOCH,
                message="Unexpected optimizer step count.",
            )
            require(
                condition=all(np.isfinite(weight).all() for weight in model.get_weights()),
                message="Model weights became nonfinite.",
            )
            measures = validation_state(model, inference, images, labels)

            if phase == "final":
                callback.on_epoch_end(epoch - 1, {"val_loss": measures["keras_validation_ce"]})

            record(
                {
                    "event": "epoch",
                    **identity,
                    "epoch": epoch,
                    **measures,
                    "online_training_ce": float(online["loss"]),
                    "online_training_accuracy": float(online["accuracy"]),
                    "optimizer_steps": int(model.optimizer.iterations.numpy()),
                    "learning_rate": float(model.optimizer.learning_rate.numpy()),
                    "shuffle_sha256": digest,
                    "best_epoch": callback.best_epoch + 1 if phase == "final" else None,
                    "callback_wait": callback.wait if phase == "final" else None,
                    "callback_stop": bool(model.stop_training),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if model.stop_training:
                break

        reached_end = epoch == maximum or bool(model.stop_training)

        if phase == "final":
            if not reached_end:
                partial = OUT / f"{prefix}-partial-last.keras"
                checkpoint_info = save_checkpoint(model, partial, identity)
                record({"event": "partial_last_checkpoint", **identity, **checkpoint_info})

            callback.on_train_end()
            restored = callback.best_weights is not None
            measures = validation_state(model, inference, images, labels)
        else:
            measures.update(probe_score(model, inference, images, order))

        finished = reached_end
        measures.update(
            {
                "completed_epochs": epoch,
                "restored": restored,
                "best_epoch": callback.best_epoch + 1 if restored else None,
                "best_monitored_ce": float(callback.best) if restored else None,
                "optimizer_steps": int(model.optimizer.iterations.numpy()),
                "shuffle_rng_state": rng.bit_generator.state,
            }
        )
    except BaseException as error:
        primary_error = error
        record(
            {"event": "phase_failed", **identity, "error": repr(error), "traceback": traceback.format_exc()},
            primary_error=error,
        )
        raise
    finally:
        checkpoint = (
            OUT / f"{prefix}-{'restored' if restored else 'current'}-{'complete' if finished else 'partial'}.keras"
        )
        checkpoint_info = save_checkpoint(model, checkpoint, identity, primary_error)
        record(
            {
                "event": "phase_end",
                **identity,
                "complete": finished,
                **measures,
                **checkpoint_info,
                "seconds": time.perf_counter() - phase_started,
                "initialization_sha256": initial,
            },
            primary_error=primary_error,
        )

    return {**identity, **measures} if finished else None


def analyze(results: list[dict]) -> None:
    """Compare both fixed probe scores with both restored targets, separately within each seed block.

    Only complete panels enter this function. All candidates contribute to correlations, regrets and tie counts.
    Learning-rate associations and four-mean sensitivity are descriptive and do not support within-LR inference.
    """
    rows = []
    lr_sensitivity = []

    for block in BLOCKS:
        probe = sorted(
            (r for r in results if r["seed_block"] == block and r["phase"] == "probe"), key=lambda r: r["config_id"]
        )
        final = sorted(
            (r for r in results if r["seed_block"] == block and r["phase"] == "final"), key=lambda r: r["config_id"]
        )

        for target in ("restored_validation_error", "restored_validation_ce"):
            targets = [
                1 - r["validation_accuracy"] if target.endswith("error") else r["keras_validation_ce"] for r in final
            ]

            for metric in ("original_rms_ratio_max", "keras_validation_ce"):
                scores = [r[metric] for r in probe]
                selected = min(range(len(scores)), key=lambda index: (scores[index], index))
                rho = float(spearmanr(scores, targets).statistic)
                rates = [r["config"]["learning_rate"] for r in probe]
                lr_score = float(spearmanr(rates, scores).statistic)
                lr_target = float(spearmanr(rates, targets).statistic)

                unique_rates = sorted(set(rates))
                mean_scores = [
                    float(np.mean([value for lr, value in zip(rates, scores, strict=True) if lr == rate]))
                    for rate in unique_rates
                ]
                mean_targets = [
                    float(np.mean([value for lr, value in zip(rates, targets, strict=True) if lr == rate]))
                    for rate in unique_rates
                ]
                sensitivity = float(spearmanr(mean_scores, mean_targets).statistic)
                lr_sensitivity.append(
                    {
                        "seed_block": block,
                        "target": target,
                        "score": metric,
                        "learning_rates": unique_rates,
                        "mean_scores": mean_scores,
                        "mean_targets": mean_targets,
                        "rho": sensitivity if np.isfinite(sensitivity) else None,
                        "n": len(unique_rates),
                    }
                )
                rows.append(
                    {
                        "seed_block": block,
                        "target": target,
                        "score": metric,
                        "rho": rho if np.isfinite(rho) else None,
                        "n": len(scores),
                        "learning_rate_vs_score_rho": lr_score if np.isfinite(lr_score) else None,
                        "learning_rate_vs_target_rho": lr_target if np.isfinite(lr_target) else None,
                        "scores": scores,
                        "targets": targets,
                        "selected_config_id": selected,
                        "regret": targets[selected] - min(targets),
                        "selected_target": targets[selected],
                        "oracle_target": min(targets),
                        "score_minimum_ties": scores.count(min(scores)),
                        "target_minimum_ties": targets.count(min(targets)),
                        "legacy_eligible_count": sum(r["legacy_eligible"] for r in probe),
                    }
                )

    write_json(
        OUT / "analysis.json",
        {
            "rows": rows,
            "lr_mean_sensitivity": lr_sensitivity,
            "lr_caveat": "Four LR means are descriptive; within-LR n=2 cannot support correlation inference.",
            "status": "Exploratory reused-data development; all scores minimized; no held-out confirmation",
        },
    )


def main() -> None:
    """Check frozen sources and execute all phases without excluding failures or analyzing partial panels.

    The wall-clock guard is checked between epochs; active-epoch work and checkpoint I/O can extend beyond it.
    Previously existing metadata prevents rerunning in this directory. Only official-training arrays are loaded.
    """
    protocol_path = OUT / "protocol.json"
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    require(condition=sys.argv[1:] == [digest], message="Pass the independently reviewed protocol SHA256.")
    require(condition=not (OUT / "metadata.json").exists(), message="Existing results must be preserved.")
    protocol = json.loads(protocol_path.read_text())
    imported_util = Path(image_util.__file__).resolve()
    require(
        condition=imported_util == (ROOT / "image/src/util.py").resolve(),
        message="Imported util is outside frozen clone.",
    )

    for logical, item in protocol["sources"].items():
        source = Path(item["path"])
        require(
            condition=hashlib.sha256(source.read_bytes()).hexdigest() == item["sha256"],
            message=f"Source changed: {logical}",
        )
        destination = OUT / "sources" / logical
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    dataset = Path(os.environ["KERAS_HOME"]) / "datasets/mnist.npz"
    require(
        condition=hashlib.sha256(dataset.read_bytes()).hexdigest() == protocol["dataset_sha256"],
        message="Dataset changed.",
    )
    tf.config.set_visible_devices([], "GPU")
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    with np.load(dataset) as archive:
        images = archive["x_train"][: TRAIN + VALIDATION].astype(np.float32)[..., None]
        labels = keras.utils.to_categorical(archive["y_train"][: TRAIN + VALIDATION], 10)

    images = (images / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    write_json(
        OUT / "metadata.json",
        {
            "protocol_sha256": digest,
            "started_utc": datetime.now(UTC).isoformat(),
            "command": sys.argv,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "imported_util_path": str(imported_util),
            "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy", "scipy")},
            "images_sha256": hashlib.sha256(images.tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        },
    )
    state, results = {"initializations": {}, "shuffles": {}}, []
    failures, attempted = [], []
    required = [(block, i, phase) for block in BLOCKS for i in range(len(CONFIGS)) for phase in PHASES]
    started = time.perf_counter()

    try:
        for block, config_id, phase in required:
            if time.perf_counter() - started >= SECONDS_LIMIT:
                break
            identity = {"seed_block": block, "config_id": config_id, "config": CONFIGS[config_id], "phase": phase}
            attempted.append((block, config_id, phase))

            try:
                result = run_phase(identity, images, labels, state, started)
            except (OSError, ValueError, RuntimeError, tf.errors.OpError) as error:
                failures.append({**identity, "error": repr(error)})
                continue

            if result is None:
                break
            results.append(result)
            sys.stdout.write(
                f"Done block={block} config={config_id} phase={phase}; elapsed={time.perf_counter() - started:.1f}s\n"
            )
            sys.stdout.flush()
    finally:
        metric_failures = [
            {"seed_block": r["seed_block"], "config_id": r["config_id"], "status": r["score_status"]}
            for r in results
            if r["phase"] == "probe" and r["original_rms_ratio_max"] is None
        ]
        comparison_complete = len(results) == len(required) and not metric_failures
        write_json(
            OUT / "summary.json",
            {
                "complete": comparison_complete,
                "successful_phases": len(results),
                "metric_failures": metric_failures,
                "required_phases": len(required),
                "failures": failures,
                "attempted": attempted,
                "unattempted": [item for item in required if item not in attempted],
                "seconds": time.perf_counter() - started,
            },
        )

    if comparison_complete:
        analyze(results)


if __name__ == "__main__":
    main()
