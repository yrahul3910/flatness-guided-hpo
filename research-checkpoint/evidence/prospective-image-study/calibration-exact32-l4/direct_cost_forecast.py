"""Apply the frozen direct-candidate cost rule without opening models or data arrays.

Run with RESULTS_DIRECTORY LAUNCH_ELIGIBILITY NEW_OUTPUT. Every one of the 32 frozen cases and the independently
verified launch evidence must pass. Invalid or missing evidence writes a stop decision. Existing outputs are refused.
"""

import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROTOCOL_SHA = "113b8af3c5e1596a39ea7395d3a08de94fb8458b65c157b66c388e9cf5ea51a9"
RULE_SHA = "901f515c4c7ee1a2e6ec733a966d85e2afda99b5e58be606d45ae7d96ec6a1f7"
ADDENDUM_SHA = "4fa2019a69330d484a16446807c83bf543022daad794c4f8380b318f8fc61047"
CASES, UPDATES, MEMORY_LIMIT = (32, 18, 15032385536)
RATE, PAID_OVERHEAD, SCIENCE_CAP, OTHER_RESERVE, PRIOR_HOLDS = (1.021392, 15616, 42, 2, 6)
TRAIN_KEYS = tuple(f"train_timed_{index:02d}" for index in range(2, 18))
OVERHEAD_KEYS = (
    "build_initialize",
    "train_warmup_00",
    "train_warmup_01",
    "inference_warmup_0",
    "initial_weight_hash",
    "final_weight_hash",
    "checkpoint_save",
    "checkpoint_hash",
)
REQUIRED = (
    *TRAIN_KEYS,
    *OVERHEAD_KEYS,
    "inference_timed_1",
    "inference_timed_2",
    "probe_score_1500",
    "load_validate_arrays",
)


def require(reason: str, *, condition: bool) -> None:
    if not condition:
        raise ValueError(reason)


def read_document(path: Path, hashes: dict[str, str]) -> Mapping[str, Any]:
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()

    return json.loads(raw)


def components(timing: Mapping[str, float]) -> dict[str, float]:
    """Return pre-margin seconds for one probe and a possible 100-epoch final fit."""
    train = max(timing[key] for key in TRAIN_KEYS)
    inference = max(timing["inference_timed_1"], timing["inference_timed_2"]) / 16
    require("Training and inference timings must be positive.", condition=train > 0 and inference > 0)

    return {
        "training": 62524 * train,
        "inference": 78695 * inference,
        "probe_score": timing["probe_score_1500"],
        "two_phase_build_warmup_checkpoint_hash": 2 * sum(timing[key] for key in OVERHEAD_KEYS),
        "epoch_weight_copy_check_and_restore": 405 * max(timing["initial_weight_hash"], timing["final_weight_hash"]),
        "data_io_validation_array_hash_proxy": 3 * timing["load_validate_arrays"] * 61500 / 2524,
    }


def main() -> None:
    """Require the frozen cohort, authenticated metadata and all usable timing records.

    Prices, infrastructure allowance and reserves are fixed by the approved scope. The output binds every evidence
    file read and preserves a blocked decision for missing, malformed, failed or over-budget cohorts.
    """
    results_name, eligibility_name, output_name = sys.argv[1:]
    folder = Path(__file__).resolve().parent
    results = Path(results_name)
    hashes: dict[str, str] = {}
    result: dict[str, Any] = {"decision": "stop_after_calibration", "launch_authorized": False}

    try:
        protocol_path = folder / "timing-protocol.json"
        protocol = read_document(protocol_path, hashes)
        require("Frozen calibration protocol changed.", condition=hashes[str(protocol_path)] == PROTOCOL_SHA)

        for name, expected in (
            ("direct-forecast-rule.json", RULE_SHA),
            ("eligibility-addendum.json", ADDENDUM_SHA),
        ):
            path = folder / name
            _ = read_document(path, hashes)
            require("Frozen formula or eligibility addendum changed.", condition=hashes[str(path)] == expected)

        completion = read_document(results / "completion.json", hashes)
        runtime = read_document(results / "runtime.json", hashes)
        eligibility = read_document(Path(eligibility_name), hashes)
        require(
            "Incomplete cohort.", condition=completion == {"complete": True, "cases": CASES, "calibration_only": True}
        )
        require(
            "GPU, protocol or TF32 precision mismatch.",
            condition=runtime["protocol_sha256"] == PROTOCOL_SHA
            and "L4" in runtime["gpu"]["device_name"]
            and runtime.get("tensor_float_32_execution_enabled") is False,
        )

        require(
            "Launch evidence does not verify one completed and stopped remote attempt.",
            condition=all(
                eligibility[key] is True for key in ("single_attempt", "runtime_finished", "modal_app_stopped")
            ),
        )
        require(
            "Invalid evidence hashes.",
            condition=all(
                re.fullmatch("[0-9a-f]{64}", eligibility[key])
                for key in ("verified_evidence_sha256", "calibration_source_sha256", "payload_sha256")
            ),
        )
        source_key = "sweep-results/research-audit-20261002/prospective-image-study/calibration-exact32-l4/calibrate.py"
        require(
            "Calibrator source mismatch.",
            condition=eligibility["calibration_source_sha256"] == protocol["source_sha256"][source_key],
        )
        require(
            "Exactly 32 frozen timing records are required.",
            condition=len(protocol["cases"]) == CASES and len(list(results.glob("case-*/timing.json"))) == CASES,
        )
        candidates = []

        for index, case in enumerate(protocol["cases"]):
            path = results / f"case-{index:02d}-{case['dataset']}-{case['candidate_id']}" / "timing.json"
            row = read_document(path, hashes)
            require(
                "Case identity, completion or optimizer count differs.",
                condition=row["case"] == case and row["status"] == "complete" and (row["optimizer_updates"] == UPDATES),
            )
            require(
                "Case error present.",
                condition=all(row[key] is None for key in ("error", "memory_error", "optimizer_counter_error")),
            )
            peak = row["gpu_peak_bytes"]
            require(
                "GPU peak memory is missing, invalid or over limit.",
                condition=type(peak) is int and 0 < peak <= MEMORY_LIMIT,
            )
            timing = row["timings"]
            require(
                "Missing or invalid timing stage.",
                condition=all(
                    type(timing[key]) in (int, float) and math.isfinite(timing[key]) and (timing[key] >= 0)
                    for key in REQUIRED
                ),
            )

            candidates.append(
                {
                    "dataset": case["dataset"],
                    "repetition_id": case["repetition_id"],
                    "candidate_id": case["candidate_id"],
                    "seconds_before_margin": components(timing),
                    "gpu_peak_bytes": peak,
                }
            )

        totals = {
            key: 2 * sum(row["seconds_before_margin"][key] for row in candidates)
            for key in candidates[0]["seconds_before_margin"]
        }
        backend_seconds = sum(totals.values())
        scientific_usd = (backend_seconds + PAID_OVERHEAD) * RATE / 3600
        require("Forecast arithmetic overflowed.", condition=math.isfinite(scientific_usd))
        result |= {
            "decision": "prefix_2" if scientific_usd <= SCIENCE_CAP else "stop_after_calibration",
            "candidates": candidates,
            "component_seconds_after_2x": totals,
            "backend_seconds_after_2x": backend_seconds,
            "scientific_worker_usd": scientific_usd,
            "prior_and_calibration_holds_usd": PRIOR_HOLDS,
            "other_reserve_usd": OTHER_RESERVE,
            "cumulative_forecast_usd": PRIOR_HOLDS + scientific_usd + OTHER_RESERVE,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["reason"] = f"{type(error).__name__}: {error}"

    result |= {
        "input_sha256": hashes,
        "calculator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "worker_usd_per_hour": RATE,
        "paid_startup_idle_shutdown_seconds": PAID_OVERHEAD,
        "limits": "Short-run timings with 2x margin are planning estimates, not observed study costs or guarantees.",
    }

    with Path(output_name).open("x") as stream:
        _ = stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
