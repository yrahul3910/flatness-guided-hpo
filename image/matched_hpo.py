"""Run a development-only matched CIFAR screening comparison from fit/validation arrays.

This entry point never loads official test data. All four arms reuse the same probes and candidate-keyed final fits.
The scientific framing, eligibility policy, publication scale, and remote resource plan remain author decisions.
"""

import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np

from image.src.matched_contract import Candidate, Manifest, Probe, decode, make_manifest, shortlist
from image.src.matched_training import FitSettings, PhaseResult, PhaseSeeds, TrainingArrays, array_hash, fit_phase

METHODS: tuple[Literal["rms", "ce", "trace", "random"], ...] = ("rms", "ce", "trace", "random")
CLASS_MATRIX_RANK = 2


@dataclass(frozen=True)
class RunSettings:
    shortlist_size: int = 2
    probe_size: int = 1500
    max_epochs: int = 100
    patience: int = 25
    batch_size: int = 64
    max_seconds: float = 600.0
    proposal_mode: Literal["generated", "fixed_fixture"] = "generated"


@dataclass
class ArmResult:
    status: str
    selected: tuple[str, ...]
    incumbent: str | None
    validation_accuracy: float | None
    validation_ce: float | None
    checkpoint: str | None
    logical_phase_seconds: float
    ranking_seconds: float
    logical_epochs: int
    logical_optimizer_steps: int


def augmentation_seeds(root: int) -> tuple[int, int]:
    values: list[int] = []

    for layer in ("translation", "flip"):
        payload = json.dumps(["matched-cifar-augmentation-v1", root, layer], separators=(",", ":")).encode()
        values.append(int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF)

    return values[0], values[1]


def _write(path: Path, value: object, *, append: bool = False) -> None:
    with path.open("a" if append else "w") as stream:
        stream.write(json.dumps(value, allow_nan=False, default=str, sort_keys=True) + "\n")


class MatchedRun:
    """Save a complete proposal population before training and keep every attempted phase.

    The caller supplies already separated, preprocessed arrays and their provenance. Deadline checks are cooperative;
    a provider or subprocess deadline remains necessary for a hard resource cap. Calling run twice is forbidden.
    """

    def __init__(self, data: TrainingArrays, manifest: Manifest, settings: RunSettings, directory: Path) -> None:
        """Validate identities and preserve source/input hashes before any model allocation."""
        self.data, self.manifest, self.settings, self.directory = data, manifest, settings, directory
        self.started = time.perf_counter()
        self.closed = False
        self.costs: dict[tuple[str, str], float] = {}
        self.results: dict[tuple[str, str], PhaseResult] = {}
        self._validate()
        directory.mkdir(parents=True, exist_ok=False)
        _write(directory / "manifest.json", manifest)
        _write(directory / "settings.json", asdict(settings))
        source_root = Path(__file__).resolve().parents[1]
        hashes = {}

        for source in [
            *sorted((source_root / "image").rglob("*.py")),
            source_root / "pyproject.toml",
            source_root / "uv.lock",
        ]:
            relative = source.relative_to(source_root)
            target = directory / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()

        _write(
            directory / "metadata.json",
            {
                "evidence_status": "development_only",
                "test_policy": "No test arrays or final test evaluation in this runner.",
                "source_sha256": hashes,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": {name: version(name) for name in ("keras", "tensorflow", "numpy")},
                "input_arrays": {
                    name: {"shape": array.shape, "dtype": str(array.dtype), "sha256": array_hash(array)}
                    for name, array in vars(data).items()
                },
                "augmentation_seed_recipe": (
                    "SHA256 compact JSON [matched-cifar-augmentation-v1,root,layer]; "
                    "4-byte prefix big-endian & 0x7fffffff"
                ),
                "cost_policy": "Each arm pays every probe and its finalists; shared physical work is recorded once.",
            },
        )

    def _validate(self) -> None:
        if (
            self.settings.shortlist_size < 1
            or self.settings.probe_size < 1
            or not math.isfinite(self.settings.max_seconds)
            or self.settings.max_seconds <= 0
            or self.settings.proposal_mode not in {"generated", "fixed_fixture"}
        ):
            msg = "Expected positive shortlist, probe size and finite wall limit."
            raise ValueError(msg)

        expected = make_manifest(
            self.manifest["dataset"], self.manifest["repetition_id"], len(self.manifest["candidates"])
        )
        expected_classes = 10 if self.manifest["dataset"] == "cifar10" else 100

        if self.data.fit_y.ndim != CLASS_MATRIX_RANK or self.data.fit_y.shape[1] != expected_classes:
            msg = "Dataset identity and training class dimension disagree."
            raise ValueError(msg)

        if expected["shared_seeds"] != self.manifest["shared_seeds"]:
            msg = "Shared seeds disagree with the frozen seed contract."
            raise ValueError(msg)

        seeds = list(expected["shared_seeds"].values())

        for candidate, reference in zip(self.manifest["candidates"], expected["candidates"], strict=True):
            if (
                candidate["candidate_id"] != reference["candidate_id"]
                or candidate["seeds"] != reference["seeds"]
                or candidate["config"] != decode(candidate["coordinates"])
                or (self.settings.proposal_mode == "generated" and candidate != reference)
            ):
                msg = "Candidate identity, seed map, or coordinates disagree with the declared contract."
                raise ValueError(msg)

            seeds.extend(candidate["seeds"].values())

            for phase in ("probe", "final"):
                seeds.extend(augmentation_seeds(candidate["seeds"][f"{phase}/augmentation"]))

        if len(seeds) != len(set(seeds)):
            msg = "A parent or augmentation-child seed collision requires a prospective protocol revision."
            raise ValueError(msg)

    def _phase(self, candidate: Candidate, phase: Literal["probe", "final"], data: TrainingArrays) -> PhaseResult:
        identifier = candidate["candidate_id"]
        remaining = self.settings.max_seconds - (time.perf_counter() - self.started)
        seeds = PhaseSeeds(
            candidate["seeds"][f"{phase}/init"],
            self.manifest["shared_seeds"][f"{phase}/shuffle"],
            augmentation_seeds(candidate["seeds"][f"{phase}/augmentation"]),
        )
        started = time.perf_counter()
        _write(
            self.directory / "events.jsonl",
            {"event": "phase_start", "candidate_id": identifier, "phase": phase, "seeds": asdict(seeds)},
            append=True,
        )

        result = PhaseResult(status="failed")

        try:
            result = (
                fit_phase(
                    data,
                    candidate["config"],
                    phase,
                    seeds,
                    FitSettings(self.settings.max_epochs, self.settings.patience, self.settings.batch_size, remaining),
                    self.directory / phase / identifier,
                )
                if remaining > 0
                else PhaseResult(status="incomplete", metadata={"reason": "deadline_before_start"})
            )
        except BaseException as error:
            result = PhaseResult(status="failed", metadata={"error": repr(error)})
            saved = self.directory / phase / identifier / "result.json"

            if saved.exists():
                try:
                    retained = json.loads(saved.read_text())
                    result.epochs, result.optimizer_steps = retained["epochs"], retained["optimizer_steps"]
                    result.metadata["backend_result"] = str(saved)
                except (OSError, ValueError, KeyError, TypeError) as retention_error:
                    result.metadata["backend_result_read_error"] = repr(retention_error)
                    raise error from retention_error

            if not isinstance(error, ValueError | RuntimeError):
                raise
        finally:
            self.results[(phase, identifier)] = result
            self.costs[(phase, identifier)] = time.perf_counter() - started
            _write(
                self.directory / "events.jsonl",
                {
                    "event": "phase_end",
                    "candidate_id": identifier,
                    "phase": phase,
                    "seconds": self.costs[(phase, identifier)],
                    "result": asdict(result),
                },
                append=True,
            )

        return result

    def _finals(self, probes: list[Probe]) -> str:
        selections = {}
        ranking_times = {}

        for method in METHODS:
            started = time.perf_counter()
            selections[method] = shortlist(self.manifest, probes, self.settings.shortlist_size, method)
            ranking_times[method] = time.perf_counter() - started

        _write(self.directory / "shortlists.json", {method: asdict(result) for method, result in selections.items()})
        union = sorted({identifier for selection in selections.values() for identifier in selection.selected})
        candidates = {candidate["candidate_id"]: candidate for candidate in self.manifest["candidates"]}
        final = {identifier: self._phase(candidates[identifier], "final", self.data) for identifier in union}
        arms: dict[str, ArmResult] = {}

        for method, selection in selections.items():
            completed: list[tuple[str, float]] = []

            for identifier in selection.selected:
                result = final[identifier]

                if result.status == "complete":
                    if result.validation_accuracy is None or result.validation_ce is None or result.checkpoint is None:
                        msg = "A completed phase must supply restored validation metrics and a checkpoint."
                        raise ValueError(msg)

                    completed.append((identifier, result.validation_accuracy))

            incumbent = min(completed, key=lambda item: (-item[1], item[0]))[0] if completed else None
            arms[method] = ArmResult(
                "complete" if len(completed) == len(selection.selected) else "degraded" if completed else "failed",
                selection.selected,
                incumbent,
                final[incumbent].validation_accuracy if incumbent else None,
                final[incumbent].validation_ce if incumbent else None,
                str(final[incumbent].checkpoint) if incumbent else None,
                sum(value for (phase, _), value in self.costs.items() if phase == "probe")
                + sum(self.costs[("final", identifier)] for identifier in selection.selected),
                ranking_times[method],
                sum(result.epochs for (phase, _), result in self.results.items() if phase == "probe")
                + sum(final[identifier].epochs for identifier in selection.selected),
                sum(result.optimizer_steps for (phase, _), result in self.results.items() if phase == "probe")
                + sum(final[identifier].optimizer_steps for identifier in selection.selected),
            )

        _write(self.directory / "arms.json", {method: asdict(result) for method, result in arms.items()})

        if all(arm.status == "complete" for arm in arms.values()):
            return "complete"
        return "degraded" if any(arm.incumbent is not None for arm in arms.values()) else "failed"

    def run(self) -> None:
        """Run the frozen pool once, preserving raw rows before selection can fail.

        Incomplete probes prevent selection of a partial population. Final failures consume their shortlist slots.
        Unexpected exceptions abort the run with its existing phase artifacts intact, without replacing candidates.
        """
        if self.closed:
            msg = "This matched run is already closed."
            raise RuntimeError(msg)

        self.closed = True
        status = "failed"
        error = None

        try:
            rng = np.random.default_rng(self.manifest["shared_seeds"]["probe/subset"])
            order = rng.permutation(len(self.data.fit_x))[: self.settings.probe_size]
            np.save(self.directory / "probe-order.npy", order)
            data = TrainingArrays(
                self.data.fit_x[order], self.data.fit_y[order], self.data.validation_x, self.data.validation_y
            )
            probes: list[Probe] = []
            incomplete = False

            for candidate in self.manifest["candidates"]:
                result = self._phase(candidate, "probe", data)
                incomplete |= result.status == "incomplete"
                probe = Probe(
                    candidate["candidate_id"],
                    "ok" if result.status == "complete" else "failed",
                    result.rms if result.rms is not None else math.inf,
                    result.validation_ce,
                    result.head_trace,
                )
                probes.append(probe)
                _write(
                    self.directory / "probes.jsonl",
                    {
                        "candidate_id": probe.candidate_id,
                        "status": result.status,
                        "rms": result.rms,
                        "score_status": result.score_status,
                        "ce": result.validation_ce,
                        "head_trace": result.head_trace,
                        "legacy_threshold_eligible": result.metadata.get("legacy_threshold_eligible"),
                    },
                    append=True,
                )

            status = "incomplete" if incomplete else self._finals(probes)
        except BaseException as exception:
            error = repr(exception)
            raise
        finally:
            _write(
                self.directory / "run-end.json",
                {
                    "status": status,
                    "error": error,
                    "seconds": time.perf_counter() - self.started,
                    "physical_phase_seconds": sum(self.costs.values()),
                    "physical_epochs": sum(result.epochs for result in self.results.values()),
                    "physical_optimizer_steps": sum(result.optimizer_steps for result in self.results.values()),
                    "test_evaluations": 0,
                },
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arrays", type=Path, required=True, help="Preprocessed NPZ with fit_x,fit_y,validation_x,validation_y only"
    )
    parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument("--wall-seconds", type=float, required=True)
    arguments = parser.parse_args()

    with np.load(arguments.arrays, allow_pickle=False) as arrays:
        if set(arrays.files) != set(TrainingArrays.__dataclass_fields__):
            msg = "NPZ must contain only the four fit/validation arrays."
            raise ValueError(msg)

        training = TrainingArrays(**{name: arrays[name] for name in arrays.files})

    manifest = make_manifest(arguments.dataset, arguments.repetition, arguments.candidates)
    settings = RunSettings(shortlist_size=arguments.finalists, max_seconds=arguments.wall_seconds)
    MatchedRun(training, manifest, settings, arguments.output).run()
