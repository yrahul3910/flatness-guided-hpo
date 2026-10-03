"""Forecast prospective CIFAR cost using timing-only records from every frozen calibration case.

Arguments: PLAN PROTOCOL PROTOCOL_SHA TIMING_DIRECTORY PRICING LAUNCH_ELIGIBILITY NEW_OUTPUT. Prices and parent-verified
launch evidence are explicit inputs. A passing estimate does not authorize launch or guarantee runtime or memory use.
"""

import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PLAN_SHA = "dc4d2a1bd1107b6080f89656c782b9e35086d4015c2009c0871f9c2dbb79fc38"
CASES, UPDATES = 9, 18
DIGEST_HEX_LENGTH = 64
SCIENTIFIC_CAP, OTHER_CAP = 42, 4
OTHER_KEYS = {"storage", "network", "finalization", "contingency"}
MAC_KEY = "forward_multiply_accumulates_per_image_excluding_BN_activation_augmentation"
TIMED_TRAIN = tuple(f"train_timed_{index:02d}" for index in range(2, 18))
REQUIRED = (
    *TIMED_TRAIN,
    "inference_timed_1",
    "inference_timed_2",
    "train_warmup_00",
    "train_warmup_01",
    "inference_warmup_0",
    "probe_score_1500",
    "build_initialize",
    "load_validate_arrays",
    "initial_weight_hash",
    "final_weight_hash",
    "checkpoint_save",
    "checkpoint_hash",
)


def forecast(
    plan: Mapping[str, Any], records: list[Mapping[str, Any]], pricing: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Charge every candidate one probe and a possible 100-epoch final fit.

    Callers must authenticate the plan and case identities first. Nonfinite timings or prices fail; incomplete cases
    and absent memory measurements produce a blocked gate. All nine cases contribute to each maximum slope.
    """
    invalid = [
        index
        for index, row in enumerate(records)
        if row["status"] != "complete"
        or row.get("error")
        or row.get("memory_error")
        or row.get("optimizer_counter_error")
        or row.get("optimizer_updates") != UPDATES
        or not row.get("gpu_peak_bytes")
    ]

    if len(records) != CASES or invalid:
        return {
            "decision": "stop_after_calibration",
            "reason": "Incomplete or unusable calibration",
            "invalid_cases": invalid,
        }

    timings = [row["timings"] for row in records]
    values = [float(row[key]) for row in timings for key in REQUIRED]
    costs = [pricing[key] for key in ("worker_usd_per_hour", "paid_startup_idle_shutdown_seconds")]
    costs += [pricing["other_usd"][key] for key in OTHER_KEYS]

    if set(pricing["other_usd"]) != OTHER_KEYS or not all(
        math.isfinite(value) and value >= 0 for value in values + costs
    ):
        message = "Every required timing and price must be finite and nonnegative."
        raise ValueError(message)

    if (
        pricing["worker_usd_per_hour"] <= 0
        or not pricing["quote_sources"]
        or not math.isfinite(pricing["gpu_memory_limit_bytes"])
        or pricing["gpu_memory_limit_bytes"] <= 0
    ):
        message = "A positive worker rate and explicit provider quote sources are required."
        raise ValueError(message)

    macs = [row["case"]["forward_mac_per_image"] for row in records]
    parameters = [row["parameters"] for row in records]
    train = max(row[key] / mac for row, mac in zip(timings, macs, strict=True) for key in TIMED_TRAIN)
    inference = max(
        row[key] / (16 * mac)
        for row, mac in zip(timings, macs, strict=True)
        for key in ("inference_timed_1", "inference_timed_2")
    )
    score = max(row["probe_score_1500"] / mac for row, mac in zip(timings, macs, strict=True))
    overhead_keys = (
        "build_initialize",
        "train_warmup_00",
        "train_warmup_01",
        "inference_warmup_0",
        "initial_weight_hash",
        "final_weight_hash",
        "checkpoint_save",
        "checkpoint_hash",
    )
    overhead_max = {key: max(row[key] for row in timings) for key in overhead_keys}
    overhead_slopes = {
        key: max(row[key] / size for row, size in zip(timings, parameters, strict=True)) for key in overhead_keys
    }
    load = max(row["load_validate_arrays"] for row in timings)
    other_usd = sum(pricing["other_usd"].values())
    memory_ok = all(0 < row["gpu_peak_bytes"] <= pricing["gpu_memory_limit_bytes"] for row in records)
    candidates = []

    for candidate in plan["architecture_estimates"]:
        mac = candidate[MAC_KEY]
        size = candidate["trainable_parameters"] + candidate["nontrainable_parameters"]
        overhead = {key: max(overhead_max[key], overhead_slopes[key] * size) for key in overhead_keys}
        components = {
            "training": 62524 * mac * train,
            "inference": 78695 * mac * inference,
            "probe_score": mac * score,
            "two_phase_build_warmup_checkpoint_hash": 2 * sum(overhead.values()),
            "epoch_weight_copy_check_and_restore": (4 * 101 + 1)
            * max(overhead["initial_weight_hash"], overhead["final_weight_hash"]),
            "data_io_validation_array_hash_proxy": 3 * load * (11500 + 50000) / 2524,
        }
        candidates.append(
            {
                "dataset": candidate["dataset"],
                "block": candidate["block"],
                "candidate_id": candidate["candidate_id"],
                "seconds_before_margin": components,
            }
        )

    prefixes = []

    for count in (2, 4):
        rows = [row for row in candidates if row["block"] in plan["repetitions_per_dataset"][:count]]
        backend_seconds = 2 * sum(sum(row["seconds_before_margin"].values()) for row in rows)
        paid_overhead = pricing["paid_startup_idle_shutdown_seconds"]
        worker_usd = (backend_seconds + paid_overhead) / 3600 * pricing["worker_usd_per_hour"]
        prefixes.append(
            {
                "repetitions_per_dataset": count,
                "candidates": len(rows),
                "backend_seconds_with_2x_margin": backend_seconds,
                "paid_startup_idle_shutdown_seconds": paid_overhead,
                "scientific_worker_usd": worker_usd,
                "other_usd": other_usd,
                "total_usd": worker_usd + other_usd,
                "gate_pass": memory_ok and worker_usd <= SCIENTIFIC_CAP and other_usd <= OTHER_CAP,
            }
        )

    eligible = [row["repetitions_per_dataset"] for row in prefixes if row["gate_pass"]]

    return {
        "decision": f"prefix_{max(eligible)}" if eligible else "stop_after_calibration",
        "launch_authorized": False,
        "calibration_peak_memory_gate": memory_ok,
        "slopes_seconds_per_batch_per_forward_mac": {"training": train, "inference": inference},
        "probe_score_seconds_per_forward_mac": score,
        "prefixes": prefixes,
        "candidates": candidates,
        "limitations": (
            "Measured extrema and overhead proxies are planning assumptions, not runtime or memory guarantees. "
            "Unprofiled candidate memory feasibility and a provider hard spending cap still require launch review."
        ),
    }


def main() -> None:
    plan_name, protocol_name, expected_protocol, timing_name, pricing_name, eligibility_name, output_name = sys.argv[1:]
    paths = {"plan": Path(plan_name), "protocol": Path(protocol_name), "pricing": Path(pricing_name)}
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}

    if hashes["plan"] != PLAN_SHA or hashes["protocol"] != expected_protocol:
        message = "Frozen plan or calibration protocol hash differs."
        raise ValueError(message)

    plan, protocol, pricing = (json.loads(paths[name].read_text()) for name in ("plan", "protocol", "pricing"))
    directory = Path(timing_name)
    records = []
    timing_hashes = {}

    for index, case in enumerate(protocol["cases"]):
        path = directory / f"case-{index:02d}-{case['dataset']}-{case['candidate_id']}" / "timing.json"

        if not path.is_file():
            continue

        raw = path.read_bytes()
        row = json.loads(raw)

        if row["case"] != case:
            message = "Calibration case identity differs from the frozen protocol."
            raise ValueError(message)

        timing_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        records.append(row)

    completion_path, runtime_path = directory / "completion.json", directory / "runtime.json"
    eligibility_path = Path(eligibility_name)

    if not all(path.is_file() for path in (completion_path, runtime_path, eligibility_path)):
        result = {"decision": "stop_after_calibration", "reason": "Missing completion, GPU or verified launch evidence"}
    else:
        completion, runtime = (json.loads(path.read_text()) for path in (completion_path, runtime_path))
        eligibility = json.loads(eligibility_path.read_text())
        digest_keys = ("verified_evidence_sha256", "calibration_source_sha256", "payload_sha256")
        launch_ok = all(
            eligibility.get(key) is True for key in ("single_attempt", "runtime_finished", "modal_app_stopped")
        )
        launch_ok = launch_ok and all(
            isinstance(eligibility.get(key), str)
            and len(eligibility[key]) == DIGEST_HEX_LENGTH
            and all(char in "0123456789abcdef" for char in eligibility[key])
            for key in digest_keys
        )
        source_key = "sweep-results/research-audit-20261002/prospective-image-study/calibration/calibrate.py"
        verified = (
            completion == {"complete": True, "cases": CASES, "calibration_only": True}
            and launch_ok
            and eligibility["calibration_source_sha256"] == protocol["source_sha256"][source_key]
            and runtime["protocol_sha256"] == expected_protocol
            and bool(pricing["gpu_name_contains"])
            and pricing["gpu_name_contains"] in runtime["gpu"]["device_name"]
        )
        result = (
            forecast(plan, records, pricing)
            if verified
            else {"decision": "stop_after_calibration", "reason": "Completion, GPU identity or launch evidence differs"}
        )
        hashes |= {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (completion_path, runtime_path, eligibility_path)
        }

    result = {
        **result,
        "inputs_sha256": hashes,
        "timing_sha256": timing_hashes,
        "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pricing": pricing,
    }

    with Path(output_name).open("x") as stream:
        _ = stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
