# Standalone evidence verifier invoked by path, outside an import package.  # noqa: INP001
"""Verify completed lifecycle evidence without importing its runner or loading models.

Pass the cached MNIST archive as the sole argument. This reads official-training arrays only, independently rebuilds
shuffle and callback histories, and recomputes every statistical summary. Failed checks raise before a success report
is written. Reports use unique timestamps and include input and verifier hashes. No executed artifact is modified.
"""

import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "run-prepared-20261002-v1"
PROTOCOL_HASH = "cb579865478345bdb4ce0098e17b322da1c177803e78cda5250d1deb97b94435"
SOURCE_HASH = "36168984699eba72876ce55b6362b3c6928e57a92401d91805b32721117f6c39"
BLOCKS = (307, 331)
PHASES = ("probe", "final")
METRICS = ("original_rms_ratio_max", "keras_validation_ce")
TARGETS = ("restored_validation_error", "restored_validation_ce")
RESTORED_RTOL, RESTORED_ATOL = 1e-5, 1e-6
RECONSTRUCTION_TOLERANCE = 1e-5
PATIENCE, EPOCHS, LEGACY_THRESHOLD = 25, 100, 0.1


def expect(actual: object, *, expected: object) -> None:
    if actual != expected:
        message = f"Evidence mismatch: actual={actual!r}; expected={expected!r}"
        raise AssertionError(message)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(row: dict) -> tuple[int, int, str]:
    return row["seed_block"], row["config_id"], row["phase"]


def ranks(values: Sequence[float]) -> list[float]:
    """Assign one-based average ranks, retaining exact ties in the saved values."""
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    start = 0

    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[ordered[stop]] == values[ordered[start]]:
            stop += 1
        for index in ordered[start:stop]:
            output[index] = (start + stop + 1) / 2
        start = stop

    return output


def correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Compute Pearson correlation of average ranks without SciPy or pilot helpers."""
    a, b = ranks(left), ranks(right)
    a_mean, b_mean = math.fsum(a) / len(a), math.fsum(b) / len(b)
    numerator = math.fsum((x - a_mean) * (y - b_mean) for x, y in zip(a, b, strict=True))
    denominator = math.sqrt(math.fsum((x - a_mean) ** 2 for x in a) * math.fsum((y - b_mean) ** 2 for y in b))

    return numerator / denominator if denominator else None


def close(actual: float | None, expected: float | None) -> None:
    if actual is None or expected is None:
        expect(actual, expected=expected)
        return

    expect(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), expected=True)


def verify_phase(start: dict, end: dict, epochs: list[dict]) -> dict:
    """Reconstruct each prescribed phase's seeds, shuffles, callback decisions, and restored targets.

    Restoration comparisons use fixed rtol=1e-5 and atol=1e-6 for floating-point CE, while prediction accuracy,
    callback decisions, epochs, and hashes must match exactly. Checkpoint contents are hashed but never deserialized.
    """
    block, config_id, phase = identity(start)
    phase_seed = block * 1000 + (1 if phase == "probe" else 2)
    subset_seed, shuffle_seed = block * 1000 + 3, phase_seed + 10
    expect(
        (start["phase_seed"], start["subset_seed"], start["shuffle_seed"]),
        expected=(phase_seed, subset_seed, shuffle_seed),
    )
    order = np.random.default_rng(subset_seed).permutation(1500) if phase == "probe" else np.arange(1500)
    saved_order = np.load(RUN / f"block{block}-config{config_id}-{phase}-order.npy", allow_pickle=False)

    np.testing.assert_array_equal(saved_order, order)
    expect(start["measurement_order_sha256"], expected=hashlib.sha256(order.tobytes()).hexdigest())
    expect([row["epoch"] for row in epochs], expected=list(range(1, len(epochs) + 1)))
    expect((end["completed_epochs"], end["optimizer_steps"]), expected=(len(epochs), len(epochs) * 24))
    expect(end["config"], expected=start["config"])
    expect(end["complete"] and end["checkpoint_saved"], expected=True)
    expect(digest(RUN / end["checkpoint"]), expected=end["checkpoint_sha256"])
    expect(end["initialization_sha256"], expected=start["initialization_sha256"])

    rng = np.random.default_rng(shuffle_seed)
    best, best_epoch, wait = math.inf, 0, 0

    for epoch, row in enumerate(epochs, 1):
        expect(identity(row), expected=identity(start))
        expect(row["config"], expected=start["config"])
        permutation = order[rng.permutation(1500)]
        expect(row["shuffle_sha256"], expected=hashlib.sha256(permutation.tobytes()).hexdigest())
        expect(row["optimizer_steps"], expected=epoch * 24)

        rate = start["config"]["learning_rate"]
        expected_rate = rate if phase == "probe" else rate * (1 + math.cos(math.pi * epoch / 100)) / 2
        expect(math.isclose(row["learning_rate"], expected_rate, rel_tol=2e-4, abs_tol=1e-9), expected=True)
        expect(row["probability_reconstruction_error"] <= RECONSTRUCTION_TOLERANCE, expected=True)
        expect(
            all(
                math.isfinite(row[key]) for key in ("keras_validation_ce", "stable_validation_ce", "online_training_ce")
            ),
            expected=True,
        )

        if phase == "final":
            wait += 1
            if row["keras_validation_ce"] < best:
                best, best_epoch, wait = row["keras_validation_ce"], epoch, 0
            expect((row["best_epoch"], row["callback_wait"]), expected=(best_epoch, wait))
            expect(row["callback_stop"], expected=wait >= PATIENCE and epoch > 1)
        else:
            expect(row["best_epoch"], expected=None)
            expect(row["callback_wait"], expected=None)
            expect(row["callback_stop"], expected=False)

        if epoch < len(epochs):
            expect(row["callback_stop"], expected=False)

    expect(end["shuffle_rng_state"], expected=rng.bit_generator.state)

    if phase == "final":
        expect(len(epochs) == EPOCHS or epochs[-1]["callback_stop"], expected=True)
        expect((end["restored"], end["best_epoch"], end["best_monitored_ce"]), expected=(True, best_epoch, best))
        selected = epochs[best_epoch - 1]

        for field in ("keras_validation_ce", "stable_validation_ce"):
            expect(
                math.isclose(end[field], selected[field], rel_tol=RESTORED_RTOL, abs_tol=RESTORED_ATOL), expected=True
            )

        expect(end["validation_accuracy"], expected=selected["validation_accuracy"])
    else:
        expect((len(epochs), end["restored"]), expected=(1, False))
        expect(
            (end["keras_validation_ce"], end["validation_accuracy"]),
            expected=(epochs[0]["keras_validation_ce"], epochs[0]["validation_accuracy"]),
        )
        verify_probe_components(end)

    return {
        "identity": identity(start),
        "epochs": len(epochs),
        "best_epoch": end["best_epoch"],
        "checkpoint": end["checkpoint"],
        "checkpoint_sha256": end["checkpoint_sha256"],
    }


def verify_probe_components(end: dict) -> None:
    """Recompute float32 batch scores and the positive maximum without loading a model."""
    expect(end["score_status"], expected="finite_positive")
    expect([row["count"] for row in end["score_batches"]], expected=[64] * 23 + [28])
    expect([row["start"] for row in end["score_batches"]], expected=list(range(0, 1500, 64)))
    scores = []

    for batch in end["score_batches"]:
        expect(batch["nonfinite_components"], expected={})
        value = float(np.float32(batch["p_rms"]) * np.float32(batch["h_rms"]) / np.float32(end["weight_rms"]))
        expect(batch["score"], expected=value)
        scores.append(value)

    expect(end["original_rms_ratio_max"], expected=max(scores))
    expect(end["legacy_eligible"], expected=max(scores) > LEGACY_THRESHOLD)


def verify_analysis(analysis: dict, ends: dict) -> None:
    """Recompute all correlations, targets, tie counts, regrets, and four-LR sensitivity rows.

    Every block/target/score combination must appear exactly once. All eight configurations contribute, and
    the lower configuration ID resolves score ties. Mismatched saved values raise before a success report is written.
    """
    expected_keys = set(product(BLOCKS, TARGETS, METRICS))
    seen = set()
    sensitivities = {(row["seed_block"], row["target"], row["score"]): row for row in analysis["lr_mean_sensitivity"]}
    expect(set(sensitivities), expected=expected_keys)
    expect(len(analysis["lr_mean_sensitivity"]), expected=len(expected_keys))

    for row in analysis["rows"]:
        block, target, metric = row["seed_block"], row["target"], row["score"]
        key = block, target, metric
        expect(key not in seen, expected=True)
        seen.add(key)

        probe = [ends[block, config_id, "probe"] for config_id in range(8)]
        final = [ends[block, config_id, "final"] for config_id in range(8)]
        scores = [item[metric] for item in probe]
        targets = [
            1 - item["validation_accuracy"] if target.endswith("error") else item["keras_validation_ce"]
            for item in final
        ]
        rates = [item["config"]["learning_rate"] for item in probe]
        chosen = min(range(8), key=lambda index: (scores[index], index))

        expect((row["scores"], row["targets"], row["n"]), expected=(scores, targets, 8))
        expect(
            (row["selected_config_id"], row["selected_target"], row["oracle_target"]),
            expected=(chosen, targets[chosen], min(targets)),
        )
        expect(row["regret"], expected=targets[chosen] - min(targets))
        expect(row["score_minimum_ties"], expected=scores.count(min(scores)))
        expect(row["target_minimum_ties"], expected=targets.count(min(targets)))
        expect(row["legacy_eligible_count"], expected=sum(item["legacy_eligible"] for item in probe))

        close(row["rho"], correlation(scores, targets))
        close(row["learning_rate_vs_score_rho"], correlation(rates, scores))
        close(row["learning_rate_vs_target_rho"], correlation(rates, targets))

        mean_scores = [math.fsum(scores[index : index + 2]) / 2 for index in range(0, 8, 2)]
        mean_targets = [math.fsum(targets[index : index + 2]) / 2 for index in range(0, 8, 2)]
        sensitivity = sensitivities[key]
        expect(sensitivity["learning_rates"], expected=sorted(set(rates)))
        expect(
            (sensitivity["mean_scores"], sensitivity["mean_targets"], sensitivity["n"]),
            expected=(mean_scores, mean_targets, 4),
        )
        close(sensitivity["rho"], correlation(mean_scores, mean_targets))

    expect(seen, expected=expected_keys)


def main() -> None:
    """Check complete evidence and write uniquely named verification reports beside this script.

    Assertions are independent of runner code. A failed source or statistical check raises and produces no success
    report. Only file hashes and recorded restoration evidence are checked here; the author owns model reloads.
    """
    expect(digest(RUN / "protocol.json"), expected=PROTOCOL_HASH)
    expect(digest(RUN / "run_pilot.py"), expected=SOURCE_HASH)
    protocol = json.loads((RUN / "protocol.json").read_text())
    metadata = json.loads((RUN / "metadata.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    analysis = json.loads((RUN / "analysis.json").read_text())
    records = [json.loads(line) for line in (RUN / "events.jsonl").read_text().splitlines()]
    expect(metadata["protocol_sha256"], expected=PROTOCOL_HASH)

    for logical, item in protocol["sources"].items():
        expect(digest(Path(item["path"])), expected=item["sha256"])
        expect(digest(RUN / "sources" / logical), expected=item["sha256"])

    dataset = Path(sys.argv[1])
    expect(digest(dataset), expected=protocol["dataset_sha256"])

    with np.load(dataset, allow_pickle=False) as archive:
        images = archive["x_train"][:1875].astype(np.float32)[..., None]
        labels = np.eye(10)[archive["y_train"][:1875]]

    images = (images / np.float32(255) - np.float32(0.1307)) / np.float32(0.3081)
    expect(hashlib.sha256(images.tobytes()).hexdigest(), expected=metadata["images_sha256"])
    expect(hashlib.sha256(labels.tobytes()).hexdigest(), expected=metadata["labels_sha256"])

    expected = list(product(BLOCKS, range(8), PHASES))
    expect(summary["complete"], expected=True)
    expect(summary["successful_phases"], expected=len(expected))
    expect(summary["required_phases"], expected=len(expected))
    expect([tuple(row) for row in summary["attempted"]], expected=expected)
    expect((summary["metric_failures"], summary["failures"], summary["unattempted"]), expected=([], [], []))

    starts = {identity(row): row for row in records if row["event"] == "phase_start"}
    ends = {identity(row): row for row in records if row["event"] == "phase_end"}
    expect(set(starts), expected=set(expected))
    expect(set(ends), expected=set(expected))
    counts = Counter(row["event"] for row in records)
    epoch_rows = [row for row in records if row["event"] == "epoch"]
    expect(counts, expected={"phase_start": 32, "epoch": len(epoch_rows), "phase_end": 32})
    phase_facts = [
        verify_phase(starts[key], ends[key], [row for row in epoch_rows if identity(row) == key]) for key in expected
    ]

    for block, phase in product(BLOCKS, PHASES):
        hashes = {starts[block, config_id, phase]["initialization_sha256"] for config_id in range(8)}
        expect(len(hashes), expected=1)

    initial_hashes = {starts[block, 0, phase]["initialization_sha256"] for block, phase in product(BLOCKS, PHASES)}
    expect(len(initial_hashes), expected=4)
    expect(len(list(RUN.glob("*.keras"))), expected=32)
    verify_analysis(analysis, ends)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    report = {
        "all_checks_pass": True,
        "created_utc": timestamp,
        "scope": "No training or model deserialization",
        "verifier_sha256": digest(Path(__file__)),
        "protocol_sha256": PROTOCOL_HASH,
        "source_sha256": SOURCE_HASH,
        "source_files_verified": len(protocol["sources"]),
        "input_sha256": {
            name: digest(RUN / name) for name in ("events.jsonl", "metadata.json", "summary.json", "analysis.json")
        },
        "event_counts": dict(counts),
        "phase_facts": phase_facts,
        "analysis_rows_verified": len(analysis["rows"]),
        "lr_sensitivity_rows_verified": len(analysis["lr_mean_sensitivity"]),
        "restoration_ce_tolerance": {"rtol": RESTORED_RTOL, "atol": RESTORED_ATOL},
        "independent_statistics": "Average tied ranks and Pearson formula, without SciPy or runner imports",
        "verified_analysis": analysis,
    }
    path = HERE / f"verification-{timestamp}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    note = (
        f"PASS: {len(expected)} phases, {len(epoch_rows)} epoch rows, 32 checkpoint file hashes.\n"
        "Verified seeds/orders/shuffles, callback wait/best/stop history and restored recorded metrics.\n"
        "Independently recomputed all 8 correlations/regrets and 8 learning-rate sensitivity rows.\n"
        "No training, model loading, or executed-file edits. Author owns checkpoint reload verification.\n"
        f"Details: {path.name}\n"
    )
    path.with_suffix(".txt").write_text(note)
    path.with_suffix(".py").write_bytes(Path(__file__).read_bytes())
    sys.stdout.write(str(path) + "\n" + note)


if __name__ == "__main__":
    main()
