"""Deterministic exploratory checks; no training, datasets, or external calls.

Run from repository root:
    python3 sweep-results/research-audit-20261002/theory_proxy_checks.py
Writes theory_proxy_checks.json next to this script. These are counterexamples
and identity checks, NOT benchmark results or reviewer-response evidence.
"""
import hashlib
import json
import math
import platform
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def norm(values):
    return math.sqrt(sum(x * x for x in values))


def rms(values):
    return norm(values) / math.sqrt(len(values))


def probs(weights, hidden=1.0):
    logits = [w * hidden for w in weights]
    largest = max(logits)
    exps = [math.exp(x - largest) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def ce_gradient(weights, hidden=1.0):
    p = probs(weights, hidden)
    return [(p[0] - 1.0) * hidden, p[1] * hidden]


def head_check(weights):
    p = probs(weights)
    variance = p[0] * p[1]
    exact = [[variance, -variance], [-variance, variance]]
    step = 1e-5
    finite_difference = [[0.0, 0.0], [0.0, 0.0]]
    for col in range(2):
        plus, minus = list(weights), list(weights)
        plus[col] += step
        minus[col] -= step
        gp, gm = ce_gradient(plus), ce_gradient(minus)
        for row in range(2):
            finite_difference[row][col] = (gp[row] - gm[row]) / (2 * step)
    exact_frob = 2 * variance
    claimed = norm(p) / norm(weights)
    fd_error = max(abs(exact[i][j] - finite_difference[i][j]) for i in range(2) for j in range(2))
    assert fd_error < 1e-8
    return {
        "weights": weights,
        "hidden": 1.0,
        "probabilities": p,
        "cross_entropy_label_0": -math.log(p[0]),
        "exact_hessian": exact,
        "exact_hessian_frobenius": exact_frob,
        "claimed_lower_bound": claimed,
        "bound_over_actual": claimed / exact_frob,
        "claimed_bound_holds": claimed <= exact_frob,
        "finite_difference_step": step,
        "finite_difference_max_abs_error": fd_error,
    }


def existing_filter(values, keep=10):
    # Mirrors image/stcvx.py:52-60 as read at audit time.
    selected = []
    for value in values:
        if len(selected) < keep or (value < max(selected) and value > 0.1):
            selected = sorted(selected + [value])[:keep]
    return selected


def batch_identity(batch_size, classes, width):
    p = [1.0 / classes] * (batch_size * classes)
    hidden = [1.0 + (i % width) / width for i in range(batch_size * width)]
    weights = [0.01 * (1 + i % 7) for i in range(width * classes)]
    raw = norm(p) * norm(hidden) / norm(weights)
    normalized = rms(p) * rms(hidden) / rms(weights)
    assert math.isclose(raw / batch_size, normalized, rel_tol=1e-12)
    return {"B": batch_size, "K": classes, "D": width,
            "frobenius_score": raw, "rms_score": normalized,
            "frobenius_over_rms": raw / normalized}


source_files = ["paper/main.tex", "paper/appendix/proofs.tex",
                "image/src/util.py", "image/stcvx.py", "image/parsers/parse.py"]
high = [0.2 + i / 100 for i in range(10)]
low = [0.01, 0.02]
all_values = high + low
first_order = existing_filter(all_values)
second_order = existing_filter(low + high)
assert first_order != second_order
assert second_order == sorted(all_values)[:10]

counterexamples = [head_check([t, -t]) for t in [0.01, 0.1, 0.5, 1.0, 5.0]]
assert all(not row["claimed_bound_holds"] for row in counterexamples)
gauge = [head_check([0.5 + c, -0.5 + c]) for c in [0.0, 1.0, 10.0, 100.0]]
for row in gauge:
    assert math.isclose(row["exact_hessian_frobenius"], gauge[0]["exact_hessian_frobenius"], abs_tol=1e-12)
    assert math.isclose(row["cross_entropy_label_0"], gauge[0]["cross_entropy_label_0"], abs_tol=1e-12)

epsilon = 0.1
f0 = math.log(2)
actual_sharpness = (math.log1p(math.exp(epsilon)) - f0) / (1 + f0)
hessian_only = epsilon ** 2 * 0.25 / (2 * (1 + f0))

output = {
    "status": "exploratory_audit_not_benchmark",
    "randomness": "none; deterministic stdlib calculations",
    "python": platform.python_version(),
    "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
    "counterexample_setup": "One sample, hidden activation h=1, two-class affine softmax output, categorical cross-entropy. A fully connected penultimate layer can realize h=1. Exact final-weight Hessian = (diag(p)-p p^T) tensor (h h^T).",
    "small_weight_counterexamples": counterexamples,
    "common_logit_shift_invariance": gauge,
    "batch_rms_identity": [batch_identity(b, k, d) for b, k, d in [(1, 2, 1), (28, 100, 32), (64, 100, 32), (64, 10, 512)]],
    "filter_order_dependence": {"high_then_low_input": all_values,
        "low_then_high_input": low + high, "high_then_low_selected": first_order,
        "low_then_high_selected": second_order, "correct_lowest_10": sorted(all_values)[:10]},
    "nonstationary_sharpness_example": {"loss": "f(w)=log(1+exp(w)); center=0",
        "epsilon": epsilon, "gradient_at_center": 0.5, "hessian_at_center": 0.25,
        "actual_normalized_ball_sharpness": actual_sharpness,
        "hessian_only_normalized_approximation": hessian_only,
        "actual_over_hessian_only": actual_sharpness / hessian_only},
}
destination = Path(__file__).with_suffix(".json")
destination.write_text(json.dumps(output, indent=2) + "\n")
print(destination)
print("All deterministic assertions passed.")
print(f"t=0.1: claimed lower bound {counterexamples[1]['claimed_lower_bound']:.6f}; exact Hessian {counterexamples[1]['exact_hessian_frobenius']:.6f}")
print(f"Gauge shift c=0 -> c=100: proxy {gauge[0]['claimed_lower_bound']:.6f} -> {gauge[-1]['claimed_lower_bound']:.6f}, unchanged Hessian and loss")
print(f"Filter outputs differ on the same pool: {first_order} vs {second_order}")
