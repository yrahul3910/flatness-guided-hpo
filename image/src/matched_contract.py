"""Shared CIFAR proposal, seed, and selection contract.

This module does not load data. Its conditional architecture prior preserves the CIFAR random sampler. The matched
runner remains a development implementation until its scientific protocol and resource plan are approved.
"""

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Literal, TypedDict

from image.src.config import Config

N_COORDINATES = 7
PADDING_OPTIONS: tuple[Literal["valid", "same"], ...] = ("valid", "same")


class Candidate(TypedDict):
    candidate_id: str
    coordinates: list[float]
    config: Config
    seeds: dict[str, int]


class Manifest(TypedDict):
    dataset: str
    repetition_id: int
    shared_seeds: dict[str, int]
    candidates: list[Candidate]


@dataclass(frozen=True)
class Probe:
    candidate_id: str
    status: Literal["ok", "failed"]
    rms: float | None
    ce: float | None
    head_trace: float | None


@dataclass(frozen=True)
class Selection:
    selected: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]


def phase_seed(dataset: str, repetition: int, candidate_id: str, phase: str, role: str) -> int:
    payload = ["matched-cifar-v1", dataset, repetition, candidate_id, phase, role]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).digest()

    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def integer_bin(coordinate: float, lower: int, upper: int) -> int:
    return min(lower + int(coordinate * (upper - lower + 1)), upper)


def decode(coordinates: list[float]) -> Config:
    """Decode seven unit coordinates, retaining conditional valid-padding depth.

    Explicit endpoints are supported for adapter checks. Production draws from random.Random are in [0, 1).
    """
    if len(coordinates) != N_COORDINATES or any(not math.isfinite(x) or not 0 <= x <= 1 for x in coordinates):
        msg = "Expected seven finite unit coordinates."
        raise ValueError(msg)

    filters, kernel, padding, depth, units, learning_rate, weight_decay = coordinates
    kernel_size = integer_bin(kernel, 2, 6)
    padding_value = PADDING_OPTIONS[integer_bin(padding, 0, len(PADDING_OPTIONS) - 1)]
    maximum_depth = {2: 4, 3: 3, 4: 3, 5: 2, 6: 2}[kernel_size] if padding_value == "valid" else 5

    return Config(
        n_filters=integer_bin(filters, 32, 128),
        kernel_size=kernel_size,
        padding=padding_value,
        n_blocks=integer_bin(depth, 2, maximum_depth),
        n_units=[32, 64, 128, 256, 512][integer_bin(units, 0, 4)],
        learning_rate=10 ** (-4 + 2 * learning_rate),
        weight_decay=10 ** (-5 + 2 * weight_decay),
    )


def make_manifest(dataset: str, repetition: int, count: int) -> Manifest:
    """Create occurrence IDs before scoring and fail on any seed collision.

    Duplicate configurations keep separate IDs. Method names, ranks, and execution order never enter a seed.
    """
    if dataset not in {"cifar10", "cifar100"} or repetition < 0 or count < 1:
        msg = "Expected CIFAR dataset, nonnegative repetition, and positive candidate count."
        raise ValueError(msg)

    shared_roles = [
        ("proposal", "rng"),
        ("probe", "subset"),
        ("probe", "shuffle"),
        ("final", "shuffle"),
        ("selection", "random"),
    ]
    shared = {f"{phase}/{role}": phase_seed(dataset, repetition, "shared", phase, role) for phase, role in shared_roles}
    rng = random.Random(shared["proposal/rng"])
    candidates: list[Candidate] = []
    all_seeds = list(shared.values())

    for index in range(count):
        candidate_id = f"c{index:06d}"
        coordinates = [rng.random() for _ in range(N_COORDINATES)]
        seeds = {
            f"{phase}/{role}": phase_seed(dataset, repetition, candidate_id, phase, role)
            for phase in ["probe", "final"]
            for role in ["init", "augmentation"]
        }
        candidates.append(
            Candidate(candidate_id=candidate_id, coordinates=coordinates, config=decode(coordinates), seeds=seeds)
        )
        all_seeds.extend(seeds.values())

    if len(all_seeds) != len(set(all_seeds)):
        msg = "Seed collision; freeze a revised seed contract before execution."
        raise ValueError(msg)

    return Manifest(dataset=dataset, repetition_id=repetition, shared_seeds=shared, candidates=candidates)


def shortlist(
    manifest: Manifest,
    probes: list[Probe],
    count: int,
    method: Literal["rms", "ce", "trace", "random"],
    threshold: float | None = None,
) -> Selection:
    """Rank a complete common probe population and retain explicit exclusions.

    A threshold is an explicitly named sensitivity policy. Failed/nonfinite rows, nonpositive RMS, or negative CE/trace
    are excluded for every method; trace zero is valid. No resampling is allowed. Save exclusions with their costs.
    """
    expected = {candidate["candidate_id"] for candidate in manifest["candidates"]}
    actual = [probe.candidate_id for probe in probes]

    if len(actual) != len(set(actual)) or set(actual) != expected:
        msg = "Every proposed occurrence must have exactly one probe result."
        raise ValueError(msg)

    if (
        count < 1
        or method not in {"rms", "ce", "trace", "random"}
        or (threshold is not None and not math.isfinite(threshold))
    ):
        msg = "Invalid shortlist size, method, or threshold."
        raise ValueError(msg)

    eligible: list[tuple[str, float, float, float]] = []
    excluded: list[tuple[str, str]] = []

    for probe in sorted(probes, key=lambda item: item.candidate_id):
        if probe.status == "failed":
            excluded.append((probe.candidate_id, "failed"))
        elif probe.rms is None or probe.ce is None or probe.head_trace is None:
            msg = f"Successful probe {probe.candidate_id} is missing a metric."
            raise ValueError(msg)
        elif (
            not all(math.isfinite(value) for value in (probe.rms, probe.ce, probe.head_trace))
            or probe.rms <= 0
            or probe.ce < 0
            or probe.head_trace < 0
        ):
            excluded.append((probe.candidate_id, "invalid_metric"))
        elif threshold is not None and probe.rms <= threshold:
            excluded.append((probe.candidate_id, "threshold"))
        else:
            eligible.append((probe.candidate_id, probe.rms, probe.ce, probe.head_trace))

    if len(eligible) < count:
        msg = f"Only {len(eligible)} common eligible probes for {count} finalists; no replacement."
        raise ValueError(msg)

    if method == "random":
        random.Random(manifest["shared_seeds"]["selection/random"]).shuffle(eligible)
    else:
        metric_index = {"rms": 1, "ce": 2, "trace": 3}[method]
        eligible.sort(key=lambda item: (item[metric_index], item[0]))

    return Selection(tuple(item[0] for item in eligible[:count]), tuple(excluded))
