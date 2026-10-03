"""Recompute classification comparisons from retained pilot records without loading models.

The output distinguishes endpoint and checkpoint policies. Every comparison retains all eight candidates in its seed
block; no correlations across seed blocks or inferential claims are made.
"""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from scipy.stats import spearmanr

SOURCE = Path(__file__).resolve().parents[1]
OUTPUT = SOURCE / "replays" / datetime.now(UTC).strftime("run-%Y%m%dT%H%M%S%fZ")
AUDIT = SOURCE.parent
VALIDATION_SIZE = 375
N_SEEDS = 2
ACCURACY_TOLERANCE = 1e-12
PILOTS = {
    "five_epoch": "trajectory-pilot",
    "hundred_epoch": "long-endpoint-pilot/run-prepared-20261002-v1",
    "cold_retrain": "lifecycle-pilot/run-prepared-20261002-v1",
}


def require(*, condition: bool, message: str) -> None:
    """Reject input records that violate an explicitly checked audit invariant."""
    if not condition:
        raise ValueError(message)


def error_count(row: dict) -> int:
    """Recover the integer number of mistakes from saved prediction accuracy."""
    count = round(VALIDATION_SIZE * (1 - row["validation_accuracy"]))
    require(
        condition=abs(row["validation_accuracy"] - (1 - count / VALIDATION_SIZE)) < ACCURACY_TOLERANCE,
        message="Saved accuracy is inconsistent with an integer error count on 375 examples.",
    )

    return count


def comparison(probes: list[dict], endpoints: list[dict], metric: str) -> dict:
    """Compare a fixed minimized probe metric with all eight classification endpoints.

    Spearman ranks use integer mistake counts and average ties. Selection breaks score ties by the lowest configuration
    ID, independently of the outcome.
    """
    scores = [row[metric] for row in probes]
    counts = [error_count(row) for row in endpoints]
    selected = min(range(8), key=lambda index: (scores[index], index))
    minimum = min(counts)

    return {
        "metric": metric,
        "spearman_vs_error_count": float(spearmanr(scores, counts).statistic),
        "selected_config_id": selected,
        "selected_errors": counts[selected],
        "selected_accuracy": 1 - counts[selected] / VALIDATION_SIZE,
        "best_available_errors": minimum,
        "excess_errors": counts[selected] - minimum,
        "best_available_tie_count": counts.count(minimum),
        "minimum_score_tie_count": scores.count(min(scores)),
        "n_candidates": 8,
        "n_validation": VALIDATION_SIZE,
    }


def analyze_block(pilot: str, events: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Extract each endpoint policy and compare it with the same one-epoch probe.

    Continuation panels have epoch rows; the cold-retrain panel additionally has restored phase-end rows. Restored
    metrics must match the recorded minimum-monitored-CE epoch.
    """
    lifecycle = pilot == "cold_retrain"
    seed_key = "seed_block" if lifecycle else "seed"
    score_key = "proxy" if pilot == "five_epoch" else "original_rms_ratio_max"
    ce_key = "keras_validation_ce" if lifecycle else "validation_loss"
    rows = [row for row in events if row[seed_key] == seed]
    probes, groups = [], []

    for config_id in range(8):
        candidate = [row for row in rows if row["config_id"] == config_id]
        epoch_rows = [
            row for row in candidate if row["event"] == "epoch" and (not lifecycle or row["phase"] == "final")
        ]
        probe = next(
            row
            for row in candidate
            if (lifecycle and row["event"] == "phase_end" and row["phase"] == "probe")
            or (not lifecycle and row["event"] == "epoch" and row["epoch"] == 1)
        )
        require(
            condition=len(epoch_rows) == len({row["epoch"] for row in epoch_rows}), message="Duplicate candidate epoch."
        )
        probes.append({**probe, "early_validation_error": error_count(probe) / VALIDATION_SIZE})
        groups.append(epoch_rows)

    targets = {
        "last_executed_epoch": [max(group, key=lambda row: row["epoch"]) for group in groups],
        "minimum_validation_ce_epoch": [min(group, key=lambda row: (row[ce_key], row["epoch"])) for group in groups],
    }

    if lifecycle:
        targets["restored_minimum_validation_ce_checkpoint"] = sorted(
            [row for row in rows if row["event"] == "phase_end" and row["phase"] == "final"],
            key=lambda row: row["config_id"],
        )

        for saved, measured in zip(
            targets["restored_minimum_validation_ce_checkpoint"], targets["minimum_validation_ce_epoch"], strict=True
        ):
            require(condition=saved["complete"] and saved["restored"], message="Incomplete restored checkpoint.")
            require(
                condition=saved["best_epoch"] == measured["epoch"],
                message="Restored epoch differs from minimum CE epoch.",
            )
            require(condition=saved[ce_key] == measured[ce_key], message="Restored CE differs from minimum CE epoch.")
            require(
                condition=saved["validation_accuracy"] == measured["validation_accuracy"],
                message="Restored accuracy differs from minimum CE epoch.",
            )

    metrics = [score_key, ce_key, "early_validation_error"]

    if pilot == "hundred_epoch":
        metrics.extend(["relative_head_trace_centered", "temperature_curvature"])

    comparisons, candidate_rows = [], []

    for target, endpoints in targets.items():
        require(
            condition=[row["config_id"] for row in endpoints] == list(range(8)),
            message="Incomplete candidate population.",
        )
        identity = {"pilot": pilot, "seed": seed, "target": target}
        comparisons.extend({**identity, **comparison(probes, endpoints, metric)} for metric in metrics)

        for probe, endpoint in zip(probes, endpoints, strict=True):
            candidate_rows.append(
                {
                    **identity,
                    "config_id": endpoint["config_id"],
                    "config": endpoint["config"],
                    "epoch": endpoint.get("epoch", endpoint.get("best_epoch")),
                    "validation_errors": error_count(endpoint),
                    "n_validation": VALIDATION_SIZE,
                    "validation_accuracy": endpoint["validation_accuracy"],
                    "validation_ce": endpoint[ce_key],
                    "probe_metrics": {metric: probe[metric] for metric in metrics},
                }
            )

    return comparisons, candidate_rows


def main() -> None:
    """Write auditable summaries derived only from completed, unchanged pilot event files.

    Every input has a SHA256 digest in the output. The script imports no training framework, reads no model checkpoint
    or dataset, and requires complete source event populations.
    """
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=False)
    provenance = {
        "executed_source": str(Path(__file__).resolve()),
        "executed_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "original_source_sha256": hashlib.sha256((SOURCE / "analyze.py").read_bytes()).hexdigest(),
    }
    (OUTPUT / "replay-source.json").write_text(json.dumps(provenance, indent=2) + "\n")

    comparisons, candidate_rows, sources = [], [], {}
    expected_epochs = {"five_epoch": 80, "hundred_epoch": 1600, "cold_retrain": 832}

    for pilot, relative in PILOTS.items():
        path = AUDIT / relative / "events.jsonl"
        raw = path.read_bytes()
        events = [json.loads(line) for line in raw.splitlines()]
        require(
            condition=sum(row["event"] == "epoch" for row in events) == expected_epochs[pilot],
            message="Unexpected epoch population.",
        )
        source = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "epoch_rows": expected_epochs[pilot]}
        sources[pilot] = source
        seed_key = "seed_block" if pilot == "cold_retrain" else "seed"
        seeds = sorted({row[seed_key] for row in events})

        require(condition=len(seeds) == N_SEEDS, message="Unexpected seed population.")

        for seed in seeds:
            block_comparisons, block_candidates = analyze_block(pilot, events, seed)
            comparisons.extend(block_comparisons)
            candidate_rows.extend(block_candidates)

    summary = {
        "analysis_status": "Post-hoc classification reanalysis; lifecycle restored error was originally primary.",
        "direction": "All probe metrics and endpoint errors minimized; positive rho supports the ranking direction.",
        "holdout": "All panels reused the same 375 development validation examples. No independent test results.",
        "sources": sources,
        "comparisons": comparisons,
        "new_training_updates": 0,
        "new_checkpoint_reloads": 0,
    }
    (OUTPUT / "metric-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (OUTPUT / "candidate-endpoints.json").write_text(json.dumps(candidate_rows, indent=2, allow_nan=False) + "\n")
    sys.stdout.write(
        json.dumps(
            {"comparisons": len(comparisons), "candidate_endpoint_rows": len(candidate_rows), "sources": sources}
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
