# Standalone evidence script invoked by path, outside any import package.  # noqa: INP001
"""Deterministic algebra checks; synthetic fixtures only, no training or pilot scores."""

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def require(*, condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


OUT = Path(__file__).resolve().parent
PROTOCOL = OUT / "normalization_protocol.json"
PROTOCOL_SHA256 = "0f9364f9d435483dfc3cbe282483b65ceea1686094b6428da85a255acb57b44d"
require(
    condition=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256,
    message="The normalization protocol hash differs from the frozen protocol.",
)


def metrics(
    h: np.ndarray, w: np.ndarray, b: np.ndarray
) -> tuple[Mapping[str, float | None], np.ndarray, np.ndarray, np.ndarray]:
    z = h @ w + b
    zc = z - z.mean(axis=1, keepdims=True)
    exp = np.exp(zc - zc.max(axis=1, keepdims=True))
    p = exp / exp.sum(axis=1, keepdims=True)
    wc = w - w.mean(axis=1, keepdims=True)
    r2 = np.sum(wc**2, axis=1)

    variance_mass = 1 - np.sum(p**2, axis=1)
    relative = float(np.mean(variance_mass * ((h**2) @ r2)))
    temp = float(np.mean(np.sum(p * zc**2, axis=1) - np.sum(p * zc, axis=1) ** 2))
    rawtrace = float(np.mean(variance_mass * np.sum(h**2, axis=1)))
    original = float(np.sqrt(np.mean(p**2)) * np.sqrt(np.mean(h**2)) / np.sqrt(np.mean(w**2)))
    centered_ratio = (
        float(np.sqrt(np.mean(p**2)) * np.sqrt(np.mean(h**2)) / np.sqrt(np.mean(wc**2))) if np.any(wc) else None
    )
    kernel = (h * np.sqrt(r2)) @ (h * np.sqrt(r2)).T

    return (
        {
            "relative_head_trace_centered": relative,
            "temperature_curvature": temp,
            "unweighted_exact_head_trace": rawtrace,
            "original_rms_ratio": original,
            "diagnostic_only_centered_denominator_ratio": centered_ratio,
        },
        p,
        zc,
        kernel,
    )


rng = np.random.default_rng(20261002)
h = np.abs(rng.normal(size=(6, 4)))
w = rng.normal(size=(4, 3))
b = rng.normal(size=3)
baseline, p0, zc0, kernel0 = metrics(h, w, b)
rows = []


def record(
    name: str,
    ht: np.ndarray,
    wt: np.ndarray,
    *,
    expected_relative: float = 1,
    expected_temperature: float = 1,
    expected_original: float | None = None,
    expected_rawtrace: float | None = None,
    invariant_kernel: bool = False,
) -> None:
    scores, p, zc, kernel = metrics(ht, wt, b)
    require(
        condition=np.allclose(p, p0, rtol=0, atol=1e-10), message="The transformation changed predicted probabilities."
    )
    require(
        condition=np.allclose(zc, zc0, rtol=1e-9, atol=1e-10), message="The transformation changed centered logits."
    )
    require(
        condition=np.isclose(
            scores["relative_head_trace_centered"],
            baseline["relative_head_trace_centered"] * expected_relative,
            rtol=1e-10,
            atol=1e-11,
        ),
        message="C1 does not match its expected transformation factor.",
    )
    require(
        condition=np.isclose(
            scores["temperature_curvature"],
            baseline["temperature_curvature"] * expected_temperature,
            rtol=1e-10,
            atol=1e-11,
        ),
        message="C2 does not match its expected transformation factor.",
    )

    if expected_original is not None:
        require(
            condition=np.isclose(
                scores["original_rms_ratio"], baseline["original_rms_ratio"] * expected_original, rtol=1e-10
            ),
            message="The original RMS ratio does not match its expected transformation factor.",
        )

    if expected_rawtrace is not None:
        require(
            condition=np.isclose(
                scores["unweighted_exact_head_trace"],
                baseline["unweighted_exact_head_trace"] * expected_rawtrace,
                rtol=1e-10,
            ),
            message="The raw head trace does not match its expected transformation factor.",
        )

    if invariant_kernel:
        require(
            condition=np.allclose(kernel, kernel0, rtol=1e-10, atol=1e-11),
            message="The weighted kernel changed under an invariant transformation.",
        )

    rows.append(
        {
            "transformation": name,
            "scores": scores,
            "ratios_to_baseline": {key: value / baseline[key] for key, value in scores.items() if value is not None},
            "max_probability_difference": float(np.max(np.abs(p - p0))),
            "expected_C1_factor": expected_relative,
            "expected_C2_factor": expected_temperature,
            "weighted_kernel_expected_invariant": invariant_kernel,
        }
    )


for c in [-100, -1, 0, 1, 100]:
    record(f"uniform_gauge_shift_{c}", h, w + c, expected_rawtrace=1, invariant_kernel=True)
record("nonuniform_common_row_shift", h, w + np.linspace(-2, 2, 4)[:, None], expected_rawtrace=1, invariant_kernel=True)

for s in [0.1, 1, 10]:
    record(
        f"global_positive_scale_{s}",
        h / s,
        w * s,
        expected_original=1 / s**2,
        expected_rawtrace=1 / s**2,
        invariant_kernel=True,
    )

s = np.geomspace(0.1, 10, 4)
record("positive_diagonal_rescale", h / s, w * s[:, None], invariant_kernel=True)
record("hidden_permutation", h[:, ::-1], w[::-1], expected_original=1, expected_rawtrace=1, invariant_kernel=True)

for r in [2, 4]:
    record(
        f"uniform_neuron_duplication_{r}",
        np.repeat(h, r, axis=1),
        np.repeat(w, r, axis=0) / r,
        expected_relative=1 / r,
        expected_original=r,
        expected_rawtrace=r,
    )

# The candidate C1 equals an explicit relative-coordinate Hessian trace.
wc = w - w.mean(axis=1, keepdims=True)
r2 = np.sum(wc**2, axis=1)
head = sum(np.kron(np.diag(p) - np.outer(p, p), np.outer(hi, hi)) for p, hi in zip(p0, h, strict=True)) / len(h)
row_scaling = np.kron(np.eye(3), np.diag(np.sqrt(r2)))
relative_head = row_scaling @ head @ row_scaling
require(
    condition=np.isclose(np.trace(head), baseline["unweighted_exact_head_trace"], rtol=1e-12),
    message="The raw head trace differs from the explicit Hessian trace.",
)
require(
    condition=np.isclose(np.trace(relative_head), baseline["relative_head_trace_centered"], rtol=1e-12),
    message="C1 differs from the explicit relative-coordinate Hessian trace.",
)

# PSD trace/Frobenius bounds apply to this new local Hessian, not the original ratio.
rank_upper = (3 - 1) * min(h.shape)
relative_frob = float(np.linalg.norm(relative_head, ord="fro"))
require(
    condition=baseline["relative_head_trace_centered"] / np.sqrt(rank_upper)
    <= relative_frob
    <= baseline["relative_head_trace_centered"],
    message="The relative-coordinate Hessian violates its PSD trace bounds.",
)

# C2 equals scalar temperature second derivative, independent of labels.
labels = np.arange(len(h)) % 3


def temperature_loss(alpha: float) -> float:
    za = alpha * zc0
    m = za.max(axis=1)

    return float(np.mean(m + np.log(np.exp(za - m[:, None]).sum(axis=1)) - za[np.arange(len(h)), labels]))


epsilon = 1e-4
temperature_finite_difference = (
    temperature_loss(1 + epsilon) - 2 * temperature_loss(1) + temperature_loss(1 - epsilon)
) / epsilon**2
require(
    condition=np.isclose(temperature_finite_difference, baseline["temperature_curvature"], rtol=1e-6, atol=1e-7),
    message="C2 differs from the finite-difference temperature curvature.",
)

# Permuting classes is not a symmetry of a fixed-label task. These label-free scores do not order supervised loss.
log99 = np.log(99.0)
counterexamples = []

for logits in [[log99 / 2, -log99 / 2], [-log99 / 2, log99 / 2]]:
    scores, p, _, _ = metrics(np.ones((1, 1)), np.array([logits]), np.zeros(2))
    counterexamples.append(
        {
            "fixed_label": 0,
            "probabilities": p[0].tolist(),
            "cross_entropy": float(-np.log(p[0, 0])),
            "accuracy": int(np.argmax(p[0]) == 0),
            "scores": scores,
        }
    )

require(
    condition=np.isclose(counterexamples[0]["scores"]["temperature_curvature"], 0.0099 * log99**2, rtol=1e-12),
    message="The fixed-label counterexample has an unexpected temperature curvature.",
)

for key in ["relative_head_trace_centered", "temperature_curvature", "unweighted_exact_head_trace"]:
    require(
        condition=np.isclose(counterexamples[0]["scores"][key], counterexamples[1]["scores"][key], rtol=1e-12),
        message="The counterexample scores differ after swapping predicted probabilities.",
    )

require(
    condition=counterexamples[0]["cross_entropy"] < counterexamples[1]["cross_entropy"],
    message="The fixed-label counterexample does not increase cross-entropy.",
)

centered_w = np.array([[0.1, -0.1]])
require(
    condition=np.array_equal(centered_w, centered_w - centered_w.mean(axis=1, keepdims=True)),
    message="The original bound counterexample is not already class-centered.",
)
_, centered_p, _, _ = metrics(np.ones((1, 1)), centered_w, np.zeros(2))
centered_bound = float(np.linalg.norm(centered_p) / np.linalg.norm(centered_w))
centered_exact = float(2 * centered_p[0, 0] * centered_p[0, 1])
require(
    condition=centered_bound > centered_exact,
    message="The centered counterexample does not violate the claimed lower bound.",
)

result = {
    "status": "deterministic analytical fixtures, not checkpoint performance evidence",
    "protocol_sha256": PROTOCOL_SHA256,
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "seed": 20261002,
    "fixture": {"h": h.tolist(), "W": w.tolist(), "b": b.tolist()},
    "baseline": baseline,
    "all_transformations": rows,
    "exact_C1_trace_check": {
        "trace_of_explicit_relative_hessian": float(np.trace(relative_head)),
        "formula": baseline["relative_head_trace_centered"],
        "relative_hessian_frobenius": relative_frob,
    },
    "exact_C2_temperature_check": {
        "second_central_difference": temperature_finite_difference,
        "formula": baseline["temperature_curvature"],
        "step": epsilon,
    },
    "fixed_label_loss_ordering_counterexample": counterexamples,
    "already_centered_original_bound_counterexample": {
        "W": centered_w.tolist(),
        "hidden": [1],
        "claimed_lower_bound": centered_bound,
        "exact_head_hessian_frobenius": centered_exact,
        "class_centering_leaves_W_unchanged": True,
        "conclusion": "Output-gauge centering alone does not repair the original lower bound.",
    },
    "conclusion": (
        "C1 removes gauge and positive diagonal rescaling but fails duplication. "
        "C2 removes all tested representation symmetries but neither score guarantees loss/accuracy ordering. "
        "No repair of the published lower bound follows."
    ),
    "all_assertions_passed": True,
}
destination = OUT / "normalization_identity_results.json"
destination.write_text(json.dumps(result, indent=2) + "\n")
sys.stdout.write(f"{destination}\n")
sys.stdout.write("All analytical identity assertions passed.\n")
counterexample_values = [(r["scores"]["temperature_curvature"], r["cross_entropy"]) for r in counterexamples]
sys.stdout.write(f"Fixed-label counterexample: {counterexample_values}\n")
