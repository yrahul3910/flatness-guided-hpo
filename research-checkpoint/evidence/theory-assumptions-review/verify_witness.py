#!/usr/bin/env python3
"""Check the local theorem using two dense layers and exact categorical cross-entropy.

Run this script with the project's existing Python environment. It writes inputs, parameters, forward values, complete
Hessians, source hashes, and numerical checks beside the script. There is no training, randomness, or network access.
"""

import hashlib
import json
import logging
import math
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent
TOLERANCE = 1e-12
LOGGER = logging.getLogger(__name__)

type JsonValue = str | float | int | bool | None | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True)
class Witness:
    name: str
    activation: str
    hidden_weight: float
    hidden_bias: float
    final_weight: float
    labels: tuple[int, ...]
    stationary: bool = False


def forward(parameters: Tensor, inputs: Tensor, activation: str) -> tuple[Tensor, Tensor]:
    """Evaluate affine, hidden activation, and affine logits with all six parameters available for differentiation."""
    hidden = functional.linear(inputs, parameters[:1].reshape(1, 1), parameters[1:2])
    hidden = torch.sigmoid(hidden) if activation == "sigmoid" else torch.relu(hidden)
    logits = functional.linear(hidden, parameters[2:4].reshape(2, 1), parameters[4:6])

    return hidden, logits


def evaluate(witness: Witness) -> Mapping[str, JsonValue]:
    """Compare the analytic final-kernel block with full-network automatic differentiation.

    Save both whole-network and head derivatives, so the final-kernel comparison cannot silently substitute a different
    Hessian. A stationary case must have a zero gradient for every trainable weight and bias within float64 tolerance.
    """
    inputs = torch.ones((len(witness.labels), 1), dtype=torch.float64)
    labels = torch.tensor(witness.labels, dtype=torch.int64)
    values = [witness.hidden_weight, witness.hidden_bias, witness.final_weight, -witness.final_weight, 0.0, 0.0]
    parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)

    def loss(candidate: Tensor) -> Tensor:
        _, candidate_logits = forward(candidate, inputs, witness.activation)
        return functional.cross_entropy(candidate_logits, labels, reduction="mean")

    gradient = torch.autograd.grad(loss(parameters), parameters, create_graph=True)[0]
    full_hessian = torch.stack(
        [torch.autograd.grad(component, parameters, retain_graph=True)[0] for component in gradient]
    )
    hidden, logits = forward(parameters, inputs, witness.activation)
    probabilities = logits.softmax(dim=1)

    covariances = torch.diag_embed(probabilities) - probabilities.unsqueeze(2) * probabilities.unsqueeze(1)
    analytic_head = (covariances * hidden.square().unsqueeze(2)).mean(dim=0)
    autodiff_head = full_hessian[2:4, 2:4]
    head_norm = math.sqrt(analytic_head.square().sum().item())
    per_example_ratio = math.sqrt(probabilities[0].square().sum().item() / parameters[2:4].square().sum().item()) * abs(
        hidden[0, 0].item()
    )
    batched_ratio = math.sqrt(
        probabilities.square().sum().item() * hidden.square().sum().item() / parameters[2:4].square().sum().item()
    )

    torch.testing.assert_close(autodiff_head, analytic_head, atol=TOLERANCE, rtol=TOLERANCE)

    if not per_example_ratio > head_norm:
        message = f"Expected a strict lower-bound violation for {witness.name}"
        raise AssertionError(message)

    if witness.stationary:
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=TOLERANCE, rtol=0)
        label_frequency = torch.bincount(labels, minlength=2).to(torch.float64) / len(labels)
        torch.testing.assert_close(probabilities[0], label_frequency, atol=TOLERANCE, rtol=0)

    result: dict[str, JsonValue] = {
        "setup": asdict(witness),
        "inputs": inputs.tolist(),
        "labels_zero_based": labels.tolist(),
        "parameter_order": [
            "hidden_kernel",
            "hidden_bias",
            "final_kernel_class_0",
            "final_kernel_class_1",
            "bias_0",
            "bias_1",
        ],
        "parameters": parameters.tolist(),
        "hidden": hidden.tolist(),
        "logits": logits.tolist(),
        "probabilities": probabilities.tolist(),
        "mean_cross_entropy": loss(parameters).item(),
        "full_gradient": gradient.tolist(),
        "full_hessian": full_hessian.tolist(),
        "analytic_final_kernel_hessian": analytic_head.tolist(),
        "autodiff_final_kernel_hessian": autodiff_head.tolist(),
        "head_hessian_max_absolute_error": (analytic_head - autodiff_head).abs().max().item(),
        "head_hessian_frobenius": head_norm,
        "full_hessian_frobenius": math.sqrt(full_hessian.square().sum().item()),
        "per_example_claimed_bound": per_example_ratio,
        "batched_frobenius_claimed_bound": batched_ratio,
        "batch_rms_claimed_bound": batched_ratio / len(labels),
        "violation_factor_per_example": per_example_ratio / head_norm,
        "bound_holds": head_norm >= per_example_ratio,
    }

    return result


def main() -> None:
    """Write five fixed witnesses and the manuscript passages used to interpret their scope.

    Source hashes identify the current local files; they do not establish that these files match a reviewed submission.
    Each reproduction overwrites only the result and context JSON files in this fixture directory.
    """
    torch.set_num_threads(1)
    witnesses = (
        Witness("relu_single_label_0", "relu", 1.0, 0.0, 0.1, (0,)),
        Witness("relu_single_label_1", "relu", 1.0, 0.0, 0.1, (1,)),
        Witness("sigmoid_single_label_0", "sigmoid", 0.0, 0.0, 0.1, (0,)),
        Witness("relu_low_loss", "relu", 1.0, 0.0, 5.0, (0,)),
        Witness("relu_stationary_mean", "relu", 1.0, 0.0, math.log(1.5) / 2, (0, 0, 0, 1, 1), stationary=True),
    )
    source_paths = (
        "paper/main.tex",
        "paper/appendix/main.tex",
        "paper/appendix/proofs.tex",
        "paper/appendix/curvature_audit_draft.tex",
        "sweep-results/research-audit-20261002/theory_proxy_checks.py",
        "sweep-results/research-audit-20261002/theory_proxy_checks.json",
        "sweep-results/research-audit-20261002/theory_proxy_audit.txt",
    )
    source_hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}
    excerpts = []

    for path, start, end in (
        ("paper/main.tex", 206, 226),
        ("paper/main.tex", 360, 368),
        ("paper/main.tex", 437, 535),
        ("paper/appendix/proofs.tex", 81, 110),
    ):
        lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
        excerpts.append(
            {"path": path, "first_line": start, "last_line": end, "text": "\n".join(lines[start - 1 : end])}
        )

    result = {
        "scope": "Exact local th:ff final-kernel Hessian; unregularized sample-mean categorical cross-entropy",
        "network": "1 input -> Dense(1) -> hidden activation -> Dense(2) -> softmax",
        "dtype": "float64",
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tolerance": TOLERANCE,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "witnesses": [evaluate(witness) for witness in witnesses],
    }
    _ = (DESTINATION / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _ = (DESTINATION / "source-context.json").write_text(json.dumps(excerpts, indent=2) + "\n", encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info("Five lower-bound violations verified; analytic and autodiff head blocks agree within %.1e.", TOLERANCE)


if __name__ == "__main__":
    main()
