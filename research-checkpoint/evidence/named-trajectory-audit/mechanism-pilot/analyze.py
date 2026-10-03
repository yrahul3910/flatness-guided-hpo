"""Verify saved mechanism checkpoints and summarize every prescribed trajectory without training."""

# Standalone evidence analysis, outside an importable package.

# ruff: noqa: INP001

import csv
import hashlib
import importlib.util
import json
import math
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, Normalizer, OneHotEncoder

HERE = Path(__file__).resolve().parent
EXECUTION = HERE / "execution"
DATA = HERE.parent / "data-recovery"
TOLERANCE = 1e-7
DATASET_NAMES = {53: "vehicle", 10101: "btsc", 146822: "segment"}


def compare(expected: object, actual: object, location: str = "metrics") -> float:
    """Check frozen measurements recursively, returning the largest absolute numeric difference.

    Numeric comparisons use relative tolerance 1e-7 and absolute tolerance 1e-9. Strings retain nonfinite values; key
    sets and list lengths must match before their values are compared.
    """
    if isinstance(expected, Mapping) and isinstance(actual, Mapping) and expected.keys() == actual.keys():
        return max(compare(expected[key], actual[key], f"{location}.{key}") for key in expected)

    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        return max(
            compare(left, right, f"{location}[{index}]")
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
        )

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(expected, actual, rel_tol=TOLERANCE, abs_tol=1e-9):
            return abs(expected - actual)
    elif expected == actual:
        return 0.0

    message = f"Checkpoint measurement mismatch at {location}: expected={expected!r}, actual={actual!r}"
    raise ValueError(message)


def main() -> None:
    """Reload all retained checkpoints, recompute their measurements, and plot all twelve full trajectories.

    The analyzed learning-rate setting is recovered by exact float32 equality with the frozen nominal rates. Checkpoint
    verification failures are recorded independently; no result is removed and the executed runner is never edited.
    """
    freeze = json.loads((HERE / "launch-freeze.json").read_text())
    runner_path = HERE / "run_pilot.py"

    if hashlib.sha256(runner_path.read_bytes()).hexdigest() != freeze["source_sha256"]:
        message = "Executed source differs from the reviewed freeze"
        raise ValueError(message)

    if (
        spec := importlib.util.spec_from_file_location("frozen_mechanism_runner", runner_path)
    ) is None or spec.loader is None:
        message = "Cannot load the frozen measurement definitions"
        raise ImportError(message)

    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    records = [json.loads(line) for line in (EXECUTION / "events.jsonl").read_text().splitlines()]
    epoch_rows = [row for row in records if row["event"] == "epoch"]
    epochs = {}

    for row in epoch_rows:
        nominal_rate = next(rate for rate in runner.RATES if row["learning_rate"] == float(np.float32(rate)))
        epochs[row["task_id"], row["seed"], nominal_rate, row["epoch"]] = row

    verify_checkpoints(runner, records, epochs, freeze)
    summarize_and_plot(runner, epoch_rows, epochs)


def verify_checkpoints(
    runner: ModuleType, records: list[Mapping[str, object]], epochs: Mapping, freeze: Mapping
) -> None:
    verification = []

    for task_id in runner.TASKS:
        with np.load(DATA / f"task-{task_id}-train-repeat0-fold0.npz", allow_pickle=False) as archive:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
            features = Normalizer().fit_transform(encoder.fit_transform(archive["X"]))
            labels = LabelBinarizer().fit_transform(archive["y"]).astype(np.float32)

        split = int(len(features) * 0.8)
        dataset_record = next(row for row in records if row["event"] == "dataset" and row["task_id"] == task_id)

        if (
            runner.array_hash([features]) != dataset_record["features_sha256"]
            or runner.array_hash([labels]) != dataset_record["labels_sha256"]
        ):
            message = "Reload preprocessing differs from the recorded feature or label arrays"
            raise ValueError(message)

        checkpoints = [row for row in records if row["event"] == "checkpoint" and row["task_id"] == task_id]

        for saved in checkpoints:
            entry = {"file": saved["file"], "passed": False}

            try:
                keras.backend.clear_session()
                path = EXECUTION / saved["file"]
                model = keras.models.load_model(path)
                epoch = int(path.stem.rsplit("epoch", 1)[1])
                expected = epochs[task_id, saved["seed"], saved["learning_rate"], epoch]
                head = model.get_layer("head")
                observer = tf.function(keras.Model(model.inputs, [head.input, head.output]), reduce_retracing=True)
                actual = runner.measure(observer, head, features, labels, split)
                difference = compare({key: expected[key] for key in actual}, runner.json_safe(actual))
                checks = {
                    "file_hash": hashlib.sha256(path.read_bytes()).hexdigest() == saved["sha256"],
                    "weights_hash": runner.array_hash(model.get_weights()) == saved["weights_sha256"],
                    "optimizer_steps": int(np.asarray(model.optimizer.iterations)) == expected["optimizer_steps"],
                }
                entry.update(
                    {"passed": all(checks.values()), "checks": checks, "max_absolute_metric_difference": difference}
                )
            # Every checkpoint's loading or numerical verification failure remains in the evidence report.
            except Exception as error:  # noqa: BLE001
                entry.update({"error": repr(error), "traceback": traceback.format_exc()})

            verification.append(entry)
            sys.stdout.write(f"{entry['file']}: {'PASS' if entry['passed'] else 'FAIL'}\n")
            sys.stdout.flush()

        del features, labels

    summary = {
        "all_passed": all(entry["passed"] for entry in verification),
        "checkpoints": verification,
        "count": len(verification),
        "frozen_source_sha256": freeze["source_sha256"],
        "measurement_tolerance": {"relative": TOLERANCE, "absolute": 1e-9},
    }
    _ = (HERE / "checkpoint-verification.json").write_text(json.dumps(summary, indent=2) + "\n")


def summarize_and_plot(runner: ModuleType, epoch_rows: list[Mapping[str, object]], epochs: Mapping) -> None:
    table = []
    figure, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    ce_scale = "log" if all(row["partitions"]["validation"]["cross_entropy"] > 0 for row in epoch_rows) else "linear"

    for row_index, task_id in enumerate(runner.TASKS):
        for seed in runner.SEEDS:
            for rate in runner.RATES:
                rows = [epochs[task_id, seed, rate, epoch] for epoch in range(1, runner.EPOCHS + 1)]
                first, last = rows[0], rows[-1]
                entry = {
                    "dataset": DATASET_NAMES[task_id],
                    "task_id": task_id,
                    "seed": seed,
                    "configured_lr": rate,
                    "recorded_lr": last["learning_rate"],
                    "epochs": len(rows),
                    "q_epoch1": first["original_max_ratio"],
                    "q_epoch30": last["original_max_ratio"],
                    "h_max_epoch1": first["max_h_f"],
                    "h_max_epoch30": last["max_h_f"],
                    "w_epoch1": first["w_f"],
                    "w_epoch30": last["w_f"],
                    "h_factor_30_over_1": last["max_h_f"] / first["max_h_f"],
                    "w_factor_30_over_1": last["w_f"] / first["w_f"],
                    "q_factor_30_over_1": last["original_max_ratio"] / first["original_max_ratio"],
                    "zero_fraction_epoch30": last["partitions"]["whole_fold"]["hidden_zero_fraction"],
                    "entropy_epoch30": last["partitions"]["whole_fold"]["entropy"],
                    "recorded_head_trace_epoch30": last["partitions"]["whole_fold"]["exact_head_kernel_trace"],
                    "val_ce_epoch1": first["partitions"]["validation"]["cross_entropy"],
                    "val_ce_epoch30": last["partitions"]["validation"]["cross_entropy"],
                    "val_accuracy_epoch30": last["partitions"]["validation"]["accuracy"],
                    "invalid_ratio_batches_all_epochs": sum(row["invalid_ratio_batches"] for row in rows),
                }
                table.append(entry)
                label = f"LR {rate:g}, seed {seed}"
                color = "#c44e52" if rate == 1.0 else "#4c72b0"
                style = "-" if seed == runner.SEEDS[0] else "--"
                series = (
                    [row["original_max_ratio"] for row in rows],
                    [row["partitions"]["whole_fold"]["hidden_zero_fraction"] for row in rows],
                    [row["partitions"]["validation"]["cross_entropy"] for row in rows],
                )

                for axis, values in zip(axes[row_index], series, strict=True):
                    axis.plot(range(1, runner.EPOCHS + 1), values, color=color, linestyle=style, label=label)

        axes[row_index, 0].set_ylabel(DATASET_NAMES[task_id])
        axes[row_index, 1].set_ylim(0, 1.03)
        axes[row_index, 2].set_yscale(ce_scale)

        for axis in axes[row_index]:
            axis.set_xlim(1, runner.EPOCHS)
            axis.grid(alpha=0.2)

    with (HERE / "component-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    _ = (HERE / "component-summary.json").write_text(json.dumps(table, indent=2) + "\n")

    for axis, title in zip(
        axes[0],
        (
            "Original batch-max h/W ratio",
            "Hidden-zero fraction (whole fold)",
            f"Validation CE (nats; {ce_scale} scale)",
        ),
        strict=True,
    ):
        axis.set_title(title)

    for axis in axes[-1]:
        axis.set_xlabel("Epoch")

    figure.suptitle("Historical-preprocessing mechanism diagnostic: all 12 prescribed runs", fontsize=14)
    figure.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    figure.text(
        0.5, 0.048, "Paired initialization and shuffles; internal validation; no official test data", ha="center"
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(HERE / "all-trajectories.png", dpi=180)
    figure.savefig(HERE / "all-trajectories.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
