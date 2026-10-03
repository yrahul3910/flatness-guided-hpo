"""Exploratory, no-training symmetry audit of a saved MNIST smoke checkpoint.

Run with the original project's existing Python venv. Only cached x_train and
y_train arrays are accessed. Transformations are evaluated in float64 using
the real checkpoint's frozen hidden activations and final affine head.
"""
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["KERAS_HOME"] = "/tmp/fghpo-preprocessing-keras"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

started = time.perf_counter()
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
import keras

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "sweep-results/cached-mnist-smoke-mnist-seed31-20261002T154204096257Z"
CHECKPOINT = CHECKPOINT_DIR / "incumbent.keras"
DATA = Path("/tmp/fghpo-preprocessing-keras/datasets/mnist.npz")
SHIFTS = [-100.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 100.0]
RESCALES = [0.1, 0.5, 1.0, 2.0, 10.0]
INDICES = np.arange(128, dtype=np.int64)
OUTPUT = Path(__file__).parent / ("mnist-head-symmetry-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
OUTPUT.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def head_metrics(hidden, probabilities, weights, labels, batch_size=64):
    count, width = hidden.shape
    classes = probabilities.shape[1]
    dot_p = probabilities @ probabilities.T
    square_p = probabilities ** 2
    tr_cc = (dot_p - probabilities @ square_p.T - square_p @ probabilities.T + dot_p ** 2)
    dot_h = hidden @ hidden.T
    norm_squared = np.sum(tr_cc * dot_h ** 2) / count ** 2
    assert norm_squared >= -1e-12
    exact_frobenius = math.sqrt(max(float(norm_squared), 0))
    trace = float(np.mean((1 - square_p.sum(axis=1)) * np.sum(hidden ** 2, axis=1)))
    rank_upper_bound = (classes - 1) * min(count, width)
    lower_from_trace = trace / math.sqrt(rank_upper_bound)
    energy_upper = float((1 - 1 / classes) * np.mean(np.sum(hidden ** 2, axis=1)))
    assert lower_from_trace <= exact_frobenius + 1e-10
    assert exact_frobenius <= trace + 1e-10
    assert trace <= energy_upper + 1e-10
    w_rms = float(np.sqrt(np.mean(weights ** 2)))
    batches = []
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        p = probabilities[start:stop]
        h = hidden[start:stop]
        p_rms = float(np.sqrt(np.mean(p ** 2)))
        h_rms = float(np.sqrt(np.mean(h ** 2)))
        batches.append({"start": start, "stop": stop, "output_rms": p_rms,
                        "hidden_rms": h_rms, "weight_rms": w_rms,
                        "rms_ratio": p_rms * h_rms / w_rms,
                        "raw_frobenius_ratio": float(np.linalg.norm(p) * np.linalg.norm(h) / np.linalg.norm(weights))})
    return {"mean_cross_entropy": float(-np.mean(np.log(probabilities[np.arange(count), labels]))),
            "accuracy": float(np.mean(np.argmax(probabilities, axis=1) == labels)),
            "proxy_max_over_batches": max(b["rms_ratio"] for b in batches),
            "batches": batches, "exact_head_hessian_frobenius": exact_frobenius,
            "exact_head_hessian_trace": trace, "head_rank_upper_bound": rank_upper_bound,
            "trace_based_frobenius_lower": lower_from_trace,
            "trace_based_frobenius_upper": trace, "activation_energy_trace_upper": energy_upper}


# Independent, seeded tiny fixture checks the compressed formula against an
# explicitly materialized Kronecker Hessian (batch=4, classes=3, width=3).
rng = np.random.default_rng(314159)
tiny_h = rng.normal(size=(4, 3))
tiny_p = softmax(rng.normal(size=(4, 3)))
tiny_w = rng.normal(size=(3, 3))
tiny_explicit = sum(np.kron(np.diag(p) - np.outer(p, p), np.outer(h, h))
                    for p, h in zip(tiny_p, tiny_h)) / len(tiny_h)
tiny_metrics = head_metrics(tiny_h, tiny_p, tiny_w, np.zeros(4, dtype=int))
tiny_norm = float(np.linalg.norm(tiny_explicit, ord="fro"))
assert np.isclose(tiny_metrics["exact_head_hessian_frobenius"], tiny_norm, rtol=1e-12, atol=1e-12)
assert np.isclose(tiny_metrics["exact_head_hessian_trace"], np.trace(tiny_explicit), rtol=1e-12, atol=1e-12)

with np.load(DATA, allow_pickle=False) as archive:
    # Do not read x_test or y_test, even transiently.
    raw_images = archive["x_train"][INDICES]
    labels = archive["y_train"][INDICES].astype(np.int64)
images = raw_images.astype(np.float32).reshape(-1, 28, 28, 1)
images /= np.float32(255)
images = (images - np.float32(0.1307)) / np.float32(0.3081)

model = keras.models.load_model(CHECKPOINT, compile=False)
assert model.layers[-2].get_config()["activation"] == "relu"
assert model.layers[-1].get_config()["activation"] == "softmax"
feature_model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output, model.layers[-1].output])
hidden_tensor, model_p_tensor = feature_model(images, training=False)
hidden = np.asarray(hidden_tensor, dtype=np.float64)
original_model_p = np.asarray(model_p_tensor, dtype=np.float64)
weights32, bias32 = model.layers[-1].get_weights()
weights, bias = weights32.astype(np.float64), bias32.astype(np.float64)
baseline_p = softmax(hidden @ weights + bias)
baseline_difference = float(np.max(np.abs(baseline_p - original_model_p)))
assert baseline_difference < 2e-6
baseline = head_metrics(hidden, baseline_p, weights, labels)


def deltas(row, p):
    return {"prediction_max_abs_delta": float(np.max(np.abs(p - baseline_p))),
            "prediction_argmax_changes": int(np.sum(np.argmax(p, axis=1) != np.argmax(baseline_p, axis=1))),
            "loss_abs_delta": abs(row["mean_cross_entropy"] - baseline["mean_cross_entropy"]),
            "head_hessian_frobenius_relative_change": row["exact_head_hessian_frobenius"] / baseline["exact_head_hessian_frobenius"] - 1,
            "proxy_relative_factor": row["proxy_max_over_batches"] / baseline["proxy_max_over_batches"]}


common_shift_results = []
for shift in SHIFTS:
    transformed_weights = weights + shift
    p = softmax(hidden @ transformed_weights + bias)
    row = head_metrics(hidden, p, transformed_weights, labels)
    row.update(shift=shift, transformation="W_new = W + shift * ones(D,K); hidden and bias fixed", **deltas(row, p))
    assert row["prediction_max_abs_delta"] < 1e-9
    assert row["loss_abs_delta"] < 1e-9
    assert abs(row["head_hessian_frobenius_relative_change"]) < 1e-9
    assert row["prediction_argmax_changes"] == 0
    common_shift_results.append(row)

rescale_results = []
for scale in RESCALES:
    transformed_hidden = hidden / scale
    transformed_weights = weights * scale
    p = softmax(transformed_hidden @ transformed_weights + bias)
    row = head_metrics(transformed_hidden, p, transformed_weights, labels)
    row.update(scale=scale, transformation="W_head_new = scale*W_head; hidden_new = hidden/scale, realizable by dividing penultimate ReLU kernel and bias by scale", **deltas(row, p))
    row["expected_proxy_and_head_curvature_factor"] = 1 / scale ** 2
    assert row["prediction_max_abs_delta"] < 1e-12
    assert row["loss_abs_delta"] < 1e-12
    assert np.isclose(row["proxy_relative_factor"], 1 / scale ** 2, rtol=1e-12)
    assert np.isclose(row["exact_head_hessian_frobenius"] / baseline["exact_head_hessian_frobenius"], 1 / scale ** 2, rtol=1e-12)
    rescale_results.append(row)

events = [json.loads(line) for line in (CHECKPOINT_DIR / "events.jsonl").read_text().splitlines()]
incumbent_event = [e for e in events if e.get("event") == "trial" and e.get("incumbent")][-1]
np.savez_compressed(OUTPUT / "head_arrays.npz", indices=INDICES, labels=labels, hidden=hidden,
                    weights=weights, bias=bias, probabilities=baseline_p)
source_snapshot = OUTPUT / Path(__file__).name
source_snapshot.write_bytes(Path(__file__).read_bytes())
results = {
    "evidence_status": "smoke-only exploratory diagnostic; no training or HPO outcome claims",
    "started_at_utc": OUTPUT.name.removeprefix("mnist-head-symmetry-"),
    "elapsed_seconds": time.perf_counter() - started,
    "data_policy": "Only cached official-training x_train/y_train indices 0:128 accessed. These indices are in this checkpoint's training subset 0:409. Official test arrays never accessed.",
    "all_transforms_prespecified": {"common_shifts": SHIFTS, "relu_positive_rescales": RESCALES},
    "training_performed": False,
    "checkpoint_mutated": False,
    "numeric_policy": "One CPU inference of original checkpoint in its saved float32 dtype; frozen hidden activations and affine-head transformations evaluated with NumPy float64. Original and transformed heads all use the same implementation and examples.",
    "curvature_scope": "Hessian with respect to final affine kernel weights only, mean categorical cross-entropy on 128 examples; excludes biases, all earlier layers, regularizers, optimizer, and training-mode stochastic behavior.",
    "proxy_definition": "maximum of RMS(p_batch)*RMS(h_batch)/RMS(W) over two fixed consecutive 64-example batches",
    "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha(CHECKPOINT),
    "checkpoint_metadata_sha256": sha(CHECKPOINT_DIR / "metadata.json"),
    "checkpoint_events_sha256": sha(CHECKPOINT_DIR / "events.jsonl"),
    "incumbent_training_event": incumbent_event,
    "model_architecture": model.get_config(),
    "dataset": str(DATA), "dataset_sha256": sha(DATA),
    "data_indices": INDICES.tolist(), "processed_input_sha256": array_sha(images),
    "labels_sha256": array_sha(labels), "hidden_sha256": array_sha(hidden),
    "preprocessing": {"dtype": "float32", "shape": [-1, 28, 28, 1], "divide": 255, "mean": 0.1307, "std": 0.3081},
    "artifact_sha256": {"head_arrays.npz": sha(OUTPUT / "head_arrays.npz"), "source_script": sha(source_snapshot)},
    "environment": {"python_executable": sys.executable, "python": platform.python_version(), "numpy": np.__version__, "keras": keras.__version__, "tensorflow": tf.__version__, "visible_gpus": [d.name for d in tf.config.get_visible_devices("GPU")], "threads": 2},
    "tiny_explicit_hessian_check": {"seed": 314159, "B": 4, "K": 3, "D": 3,
        "frobenius_explicit": tiny_norm, "frobenius_pairwise": tiny_metrics["exact_head_hessian_frobenius"],
        "absolute_error": abs(tiny_norm - tiny_metrics["exact_head_hessian_frobenius"]), "passed": True},
    "float64_head_vs_original_float32_model_max_abs_prediction_difference": baseline_difference,
    "baseline": baseline,
    "common_shift_results": common_shift_results,
    "relu_rescale_results": rescale_results,
    "assertions": "all passed",
}
(OUTPUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
print(OUTPUT)
print(f"Baseline: proxy={baseline['proxy_max_over_batches']:.9g}; CE={baseline['mean_cross_entropy']:.9g}; head ||H||F={baseline['exact_head_hessian_frobenius']:.9g}; trace={baseline['exact_head_hessian_trace']:.9g}")
for row in common_shift_results:
    print(f"shift={row['shift']:7g}, proxy={row['proxy_max_over_batches']:.9g}, ratio={row['proxy_relative_factor']:.9g}, max probability delta={row['prediction_max_abs_delta']:.3g}, relative head curvature change={row['head_hessian_frobenius_relative_change']:.3g}")
print(f"All assertions passed; elapsed {results['elapsed_seconds']:.3f}s.")
