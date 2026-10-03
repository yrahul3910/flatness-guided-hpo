# Standalone evidence script invoked by path, outside any import package.  # noqa: INP001
"""Audit frozen normalization formulas on saved heads using training inputs only.

The protocol fixes the model panel, transformations, formulas, and tolerances before inference. Every case is
retained, including numerical failures. This script never reads labels, validation examples, or test arrays,
and never trains, ranks, or selects models. Aggregate results require all sixteen models and all checks to pass.
"""

import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict

import keras
import numpy as np
import tensorflow as tf

DIRECTORY = Path(__file__).resolve().parent
PROTOCOL = DIRECTORY / "normalization_protocol.json"
PROTOCOL_SHA256 = "0f9364f9d435483dfc3cbe282483b65ceea1686094b6428da85a255acb57b44d"
DATASET = Path(os.environ["KERAS_HOME"]) / "datasets/mnist.npz"
N_EXAMPLES = 1500
BATCH_SIZE = 64
MODEL_LIMIT = 16
CASES_PER_MODEL = 14
SECONDS_LIMIT = 180


class MetricArrays(TypedDict):
    probabilities: np.ndarray
    centered_logits: np.ndarray
    C1: np.ndarray
    C2: np.ndarray
    raw_head_trace: np.ndarray
    centered_weight_row_norm_squared: np.ndarray


@dataclass
class Head:
    hidden: np.ndarray
    weights: np.ndarray
    bias: np.ndarray


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def transformations(head: Head, panel: dict) -> Iterator[tuple[str, Head, dict, dict]]:
    """Yield the protocol's fourteen transformations with their analytic score multipliers.

    Missing multiplier keys mean the protocol makes no invariance prediction for that metric. All transformations
    leave the output bias unchanged. Duplication changes the hidden width and has explicit nonunit expectations.
    """
    yield "baseline", head, {}, {"original": 1.0, "C1": 1.0, "C2": 1.0, "raw_head_trace": 1.0}

    for shift in panel["common_uniform_row_shift"]["values"]:
        changed = Head(head.hidden, head.weights + shift, head.bias)
        yield f"common_shift_{shift}", changed, {"shift": shift}, {"C1": 1.0, "C2": 1.0, "raw_head_trace": 1.0}

    shift = np.linspace(-2, 2, head.weights.shape[0])
    changed = Head(head.hidden, head.weights + shift[:, None], head.bias)
    yield "rowwise_shift", changed, {"shift_sha256": array_hash(shift)}, {"C1": 1.0, "C2": 1.0, "raw_head_trace": 1.0}

    for scale in panel["global_positive_rescale"]["values"]:
        changed = Head(head.hidden / scale, head.weights * scale, head.bias)
        expected = {"original": 1 / scale**2, "C1": 1.0, "C2": 1.0, "raw_head_trace": 1 / scale**2}
        yield f"global_scale_{scale}", changed, {"scale": scale}, expected

    scales = np.geomspace(0.1, 10, head.weights.shape[0])
    changed = Head(head.hidden / scales, head.weights * scales[:, None], head.bias)
    yield "per_unit_scale", changed, {"scales_sha256": array_hash(scales)}, {"C1": 1.0, "C2": 1.0}
    changed = Head(head.hidden[:, ::-1], head.weights[::-1], head.bias)
    yield "reverse_permutation", changed, {}, {"original": 1.0, "C1": 1.0, "C2": 1.0, "raw_head_trace": 1.0}

    for copies in panel["uniform_hidden_duplication"]["factors"]:
        changed = Head(
            np.repeat(head.hidden, copies, axis=1), np.repeat(head.weights, copies, axis=0) / copies, head.bias
        )
        expected = {"original": copies, "C1": 1 / copies, "C2": 1.0, "raw_head_trace": copies}
        yield f"duplicate_{copies}", changed, {"copies": copies}, expected


def metrics(head: Head) -> tuple[dict, MetricArrays]:
    """Calculate all fixed formulas in float64, without labels or probability clipping.

    The original score uses consecutive batches with the final partial batch included. C1, C2, and the diagnostic raw
    head trace are arithmetic means over examples. C2 uses the protocol's explicit second-moment variance expression.
    """
    logits = head.hidden @ head.weights + head.bias
    centered_logits = logits - logits.mean(axis=1, keepdims=True)
    exponentials = np.exp(centered_logits - centered_logits.max(axis=1, keepdims=True))
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    centered_weights = head.weights - head.weights.mean(axis=1, keepdims=True)
    row_norm_squared = (centered_weights**2).sum(axis=1)
    categorical_trace = 1 - (probabilities**2).sum(axis=1)
    hidden_squared = head.hidden**2
    relative_trace = categorical_trace * (hidden_squared @ row_norm_squared)
    temperature_curvature = (probabilities * centered_logits**2).sum(axis=1)
    temperature_curvature -= (probabilities * centered_logits).sum(axis=1) ** 2
    raw_head_trace = categorical_trace * hidden_squared.sum(axis=1)
    weight_rms = float(np.sqrt(np.mean(head.weights**2)))
    batches = []

    for start in range(0, len(head.hidden), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(head.hidden))
        probability_rms = float(np.sqrt(np.mean(probabilities[start:stop] ** 2)))
        hidden_rms = float(np.sqrt(np.mean(hidden_squared[start:stop])))
        batches.append(
            {
                "start": start,
                "stop": stop,
                "p_rms": probability_rms,
                "h_rms": hidden_rms,
                "score": probability_rms * hidden_rms / weight_rms,
            }
        )

    scores = {
        "original": max(batch["score"] for batch in batches),
        "C1": float(relative_trace.mean()),
        "C2": float(temperature_curvature.mean()),
        "raw_head_trace": float(raw_head_trace.mean()),
    }
    arrays = {
        "probabilities": probabilities,
        "centered_logits": centered_logits,
        "C1": relative_trace,
        "C2": temperature_curvature,
        "raw_head_trace": raw_head_trace,
        "centered_weight_row_norm_squared": row_norm_squared,
    }

    return {"scores": scores, "weight_rms": weight_rms, "batches": batches}, arrays


def audit_model(checkpoint: Path, images: np.ndarray, output: Path, protocol: dict) -> dict:
    """Compare one saved model with every transformation, retaining all case results.

    Float32 saved-model probabilities must agree with the baseline float64 reconstruction before transformations are
    scored. Numerical tolerance failures are recorded for every case and prevent aggregation, without excluding models.
    """
    started = time.perf_counter()
    model_output = output / checkpoint.stem
    model_output.mkdir()
    model = keras.models.load_model(checkpoint, compile=False)
    inference_model = keras.Model(model.inputs, [model.layers[-2].output, model.layers[-1].output])
    hidden_parts, prediction_parts = [], []

    for start in range(0, N_EXAMPLES, BATCH_SIZE):
        hidden, prediction = inference_model(images[start : start + BATCH_SIZE], training=False)
        hidden_parts.append(hidden.numpy())
        prediction_parts.append(prediction.numpy())

    kernel, bias = model.layers[-1].get_weights()
    head = Head(np.concatenate(hidden_parts).astype(np.float64), kernel.astype(np.float64), bias.astype(np.float64))
    saved_predictions = np.concatenate(prediction_parts).astype(np.float64)
    baseline, reference_arrays = metrics(head)
    tolerance = protocol["numerical_tolerances"]
    baseline_difference = float(np.max(np.abs(reference_arrays["probabilities"] - saved_predictions)))
    baseline_tolerance = tolerance[
        "baseline_saved_float32_vs_reconstructed_float64_probability_max_absolute_difference"
    ]
    np.savez_compressed(
        model_output / "baseline.npz",
        hidden=head.hidden,
        weights=head.weights,
        bias=head.bias,
        saved_probabilities=saved_predictions,
        **reference_arrays,
    )

    if baseline_difference > baseline_tolerance:
        write_json(
            model_output / "baseline_failure.json",
            {"max_absolute_probability_difference": baseline_difference, "tolerance": baseline_tolerance},
        )
        message = "Saved-model probabilities disagree with the float64 reconstructed head."
        raise RuntimeError(message)

    records = []
    saved_scores = {}
    near_zero = tolerance["metric_analytic_expectation_allclose"]["atol"]

    for name, changed, parameters, expected in transformations(head, protocol["transformation_panel"]):
        measured, arrays = metrics(changed)
        probability_error = float(np.max(np.abs(arrays["probabilities"] - reference_arrays["probabilities"])))
        logit_error = float(np.max(np.abs(arrays["centered_logits"] - reference_arrays["centered_logits"])))
        checks = {
            "probabilities": probability_error <= tolerance["probability_max_absolute_difference"],
            "centered_logits": bool(
                np.allclose(
                    arrays["centered_logits"],
                    reference_arrays["centered_logits"],
                    **tolerance["centered_logits_allclose"],
                )
            ),
        }
        ratios, errors = {}, {}

        for metric, value in measured["scores"].items():
            ratios[metric] = value / baseline["scores"][metric] if baseline["scores"][metric] != 0 else None

            if metric in expected:
                predicted = baseline["scores"][metric] * expected[metric]
                errors[metric] = abs(value - predicted)
                checks[metric] = bool(np.isclose(value, predicted, **tolerance["metric_analytic_expectation_allclose"]))

        counts = {}

        for metric in ("C1", "C2", "raw_head_trace", "centered_weight_row_norm_squared"):
            values = arrays[metric]
            counts[metric] = {
                "zero": int(np.count_nonzero(values == 0)),
                "near_zero": int(np.count_nonzero(np.abs(values) <= near_zero)),
                "negative": int(np.count_nonzero(values < 0)),
                "nonfinite": int(np.count_nonzero(~np.isfinite(values))),
            }
            checks[f"finite_{metric}"] = bool(np.isfinite(values).all())
            saved_scores[f"{name}_{metric}"] = values

        checks["finite_original"] = bool(np.isfinite(measured["scores"]["original"]))
        row = {
            "case": name,
            "parameters": parameters,
            "hidden_width": changed.weights.shape[0],
            **measured,
            "ratios_to_baseline": ratios,
            "expected_multipliers": expected,
            "absolute_score_errors": errors,
            "probability_max_absolute_error": probability_error,
            "centered_logits_max_absolute_error": logit_error,
            "argmax_differences": int(
                np.count_nonzero(
                    arrays["probabilities"].argmax(axis=1) != reference_arrays["probabilities"].argmax(axis=1)
                )
            ),
            "hidden_sha256": array_hash(changed.hidden),
            "kernel_sha256": array_hash(changed.weights),
            "bias_sha256": array_hash(changed.bias),
            "component_counts": counts,
            "checks": checks,
            "all_checks_pass": all(checks.values()),
        }
        records.append(row)

        with (model_output / "cases.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    np.savez_compressed(model_output / "per_example_scores.npz", **saved_scores)
    keras.backend.clear_session()

    return {
        "checkpoint": checkpoint.name,
        "case_count": len(records),
        "all_checks_pass": all(row["all_checks_pass"] for row in records),
        "baseline_probability_max_absolute_difference": baseline_difference,
        "seconds": time.perf_counter() - started,
        "cases": records,
    }


def main() -> None:
    """Create a fresh evidence directory and stop starting models after the fixed CPU time cap.

    Protocol and source snapshots are saved before model inference. Exceptions are recorded and re-raised; incomplete
    or numerically failing panels retain all evidence but do not produce aggregate statistics.
    """
    if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() != PROTOCOL_SHA256:
        message = "Protocol hash differs from the frozen agreement; no evaluation was started."
        raise ValueError(message)

    protocol = json.loads(PROTOCOL.read_text())
    output = DIRECTORY / f"run-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}"
    output.mkdir()
    checkpoints = [
        DIRECTORY.parent / "trajectory-pilot" / f"seed{seed}-config{config}.keras"
        for seed in (31, 47)
        for config in range(8)
    ]
    shutil.copyfile(PROTOCOL, output / PROTOCOL.name)
    shutil.copyfile(__file__, output / "check_saved_models.py")
    write_json(
        output / "metadata.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "protocol_sha256": PROTOCOL_SHA256,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in checkpoints},
            "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            "input_indices_half_open": [0, N_EXAMPLES],
            "data_access": "Only x_train[0:1500]; no labels, validation inputs, endpoint losses, or test arrays",
            "preprocessing": "float32, (x/255-0.1307)/0.3081, channels_last",
            "near_zero_absolute_threshold": 1e-10,
            "command": sys.orig_argv,
            "cwd": str(Path.cwd()),
            "python": platform.python_version(),
            "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy")},
            "device": "CPU only",
            "seconds_limit_between_models": SECONDS_LIMIT,
            "model_limit": MODEL_LIMIT,
            "expected_cases_per_model": CASES_PER_MODEL,
            "evidence_status": "exploratory algebraic audit; no selection or generalization claim",
        },
    )

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)

    with np.load(DATASET) as archive:
        images = archive["x_train"][:N_EXAMPLES].astype(np.float32)[..., None]

    images = (images / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    write_json(output / "input_hash.json", {"preprocessed_inputs_sha256": array_hash(images), "shape": images.shape})
    started = time.perf_counter()
    completed = []
    sys.stdout.write(f"Evidence directory: {output}\n")

    for checkpoint in checkpoints:
        if time.perf_counter() - started >= SECONDS_LIMIT:
            break

        try:
            result = audit_model(checkpoint, images, output, protocol)
            completed.append(result)
            write_json(output / checkpoint.stem / "summary.json", result)
            sys.stdout.write(f"{checkpoint.name}: {result['case_count']} cases, checks={result['all_checks_pass']}\n")
            sys.stdout.flush()
        except BaseException as error:
            write_json(
                output / "run_failure.json",
                {
                    "checkpoint": checkpoint.name,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "completed_models": len(completed),
                },
            )
            raise

    complete = len(completed) == MODEL_LIMIT and all(row["case_count"] == CASES_PER_MODEL for row in completed)
    passed = complete and all(row["all_checks_pass"] for row in completed)
    write_json(
        output / "summary.json",
        {
            "complete": complete,
            "all_checks_pass": passed,
            "models_completed": len(completed),
            "seconds": time.perf_counter() - started,
            "unstarted": [path.name for path in checkpoints[len(completed) :]],
            "failed_cases": [
                {"checkpoint": row["checkpoint"], "case": case["case"], "checks": case["checks"]}
                for row in completed
                for case in row["cases"]
                if not case["all_checks_pass"]
            ],
        },
    )

    if passed:
        aggregate = {}

        for name in (case["case"] for case in completed[0]["cases"]):
            cases = [case for row in completed for case in row["cases"] if case["case"] == name]
            aggregate[name] = {}

            for metric in ("original", "C1", "C2", "raw_head_trace"):
                ratios = [case["ratios_to_baseline"][metric] for case in cases]
                defined = all(ratio is not None for ratio in ratios)
                aggregate[name][metric] = {
                    "all_ratios_defined": defined,
                    "min": min(ratios) if defined else None,
                    "max": max(ratios) if defined else None,
                    "absolute_score_min": min(case["scores"][metric] for case in cases),
                    "absolute_score_max": max(case["scores"][metric] for case in cases),
                }

        write_json(output / "aggregate_ratios.json", aggregate)


if __name__ == "__main__":
    main()
