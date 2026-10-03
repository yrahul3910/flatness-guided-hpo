"""Seal complete matched runs before opening their held-out classification archives.

The caller supplies the assignment frozen before scientific training and frozen archive provenance. A failed
evaluation consumes the attempt. Every assigned dataset/repetition must complete before any evaluation archive opens.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np

from image.src.matched_training import (
    FitSettings,
    FloatArray,
    PhaseResult,
    TrainingArrays,
    _validate,
    array_hash,
    weights_hash,
)

METHODS = {"rms", "trace", "ce", "random"}


@dataclass(frozen=True)
class Winner:
    run: str
    dataset: str
    method: str
    candidate: str
    checkpoint: str
    weights_sha256: str


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_incumbent(arm: tuple[Path, float, float], phase: tuple[Path, float | None, float | None]) -> None:
    _, accuracy, ce = arm

    if arm != phase or not 0 <= accuracy <= 1 or not np.isfinite(ce) or ce < 0:
        msg = "Incumbent identity and finite validation metrics must match the saved phase."
        raise ValueError(msg)


def _verify_phase(path: Path) -> tuple[PhaseResult, Path, float]:
    result = PhaseResult(**json.loads(path.read_text()))
    checkpoint = (path.parent / "model.keras").resolve()

    if (
        result.status != "complete"
        or result.checkpoint is None
        or Path(result.checkpoint).resolve() != checkpoint
        or result.metadata["checkpoint_sha256"] != _hash(checkpoint)
        or result.validation_ce is None
        or not np.isfinite(result.validation_ce)
        or result.validation_ce < 0
        or result.validation_accuracy is None
        or not 0 <= result.validation_accuracy <= 1
    ):
        msg = "Every shortlisted phase must retain its complete, verified checkpoint."
        raise ValueError(msg)

    return result, checkpoint, result.validation_accuracy


def _verify_json_digest(value: object, expected: str) -> None:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    if hashlib.sha256(canonical).hexdigest() != expected:
        msg = "Run manifest or settings differ from their prospectively frozen canonical JSON hash."
        raise ValueError(msg)


def seal_suite(run_dirs: list[Path], directory: Path, assignment_path: Path) -> Path:
    """Write a hash-bound lock for every supplied complete run, without opening evaluation arrays.

    All four arms and all their shortlisted final phases must be complete. Recorded checkpoints must match their
    files and arm metrics. `assignment_path` lists dataset/repetition pairs and canonical manifest/settings hashes
    frozen before scientific training. Canonical JSON uses sorted keys, compact separators and no trailing newline.
    Existing output directories, attempted runs, malformed records, omitted assignments and partial suites fail.
    """
    runs = [path.resolve() for path in run_dirs]

    if not runs or len(set(runs)) != len(runs):
        msg = "Supply a nonempty list of distinct assigned runs."
        raise ValueError(msg)

    assignment = json.loads(assignment_path.read_text())
    assigned = [(row["dataset"], row["repetition_id"]) for row in assignment["runs"]]
    assigned_rows = {(row["dataset"], row["repetition_id"]): row for row in assignment["runs"]}
    observed: list[tuple[str, int]] = []
    artifacts = {str(path.resolve()): _hash(path) for path in (Path(__file__), assignment_path)}
    winners: list[Winner] = []

    for run in runs:
        if (run / "evaluation-attempt.json").exists():
            msg = "This run already has an evaluation attempt."
            raise FileExistsError(msg)

        records = {
            name: json.loads((run / f"{name}.json").read_text())
            for name in ("arms", "run-end", "manifest", "settings", "shortlists")
        }
        arms, manifest = records["arms"], records["manifest"]
        observed.append((manifest["dataset"], manifest["repetition_id"]))
        reference = assigned_rows.get(observed[-1], {})
        _verify_json_digest(manifest, reference.get("manifest_sha256", ""))
        _verify_json_digest(records["settings"], reference.get("settings_sha256", ""))

        if records["run-end"]["status"] != "complete" or set(arms) != METHODS:
            msg = "Every assigned run must complete all four arms before sealing."
            raise ValueError(msg)

        phases = {
            path: _verify_phase(path)
            for path in {
                run / "final" / candidate / "result.json" for arm in arms.values() for candidate in arm["selected"]
            }
        }
        artifacts.update({str(path): _hash(path) for path in phases})
        artifacts.update(
            {str(checkpoint): str(result.metadata["checkpoint_sha256"]) for result, checkpoint, _ in phases.values()}
        )

        for method, arm in arms.items():
            selected = arm["selected"]

            if (
                arm["status"] != "complete"
                or len(selected) != records["settings"]["shortlist_size"]
                or len(set(selected)) != len(selected)
                or arm["incumbent"] not in selected
                or selected != records["shortlists"][method]["selected"]
                or arm["incumbent"]
                != min(
                    selected, key=lambda candidate: (-phases[run / "final" / candidate / "result.json"][2], candidate)
                )
            ):
                msg = "A complete arm requires its entire frozen shortlist and selected incumbent."
                raise ValueError(msg)

            for candidate in selected:
                phase_path = run / "final" / candidate / "result.json"
                result, checkpoint, _ = phases[phase_path]

                if candidate == arm["incumbent"]:
                    _verify_incumbent(
                        (Path(arm["checkpoint"]).resolve(), arm["validation_accuracy"], arm["validation_ce"]),
                        (checkpoint, result.validation_accuracy, result.validation_ce),
                    )
                    winners.append(
                        Winner(
                            str(run),
                            manifest["dataset"],
                            method,
                            candidate,
                            str(checkpoint),
                            str(result.metadata["weights_sha256"]),
                        )
                    )

        paths = [*(run / f"{name}.json" for name in records), run / "metadata.json", run / "probes.jsonl"]
        artifacts.update({str(path): _hash(path) for path in paths})
        artifacts.update({str(path): _hash(path) for path in (run / "source").rglob("*") if path.is_file()})

    if len(set(assigned)) != len(assigned) or len(set(observed)) != len(observed) or set(assigned) != set(observed):
        msg = "Completed runs must match every prospectively assigned dataset/repetition exactly once."
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=False)
    lock = directory / "suite.json"
    _ = lock.write_text(
        json.dumps(
            {
                "artifacts": artifacts,
                "winners": [asdict(row) for row in winners],
                "evaluation_archives": assignment["evaluation_archives"],
            },
            sort_keys=True,
        )
    )
    _ = (directory / "suite.sha256").write_text(_hash(lock))

    return lock


def _classify(winner: Winner, images: FloatArray, labels: FloatArray, output: Path) -> dict[str, int | float | str]:
    """Restore one sealed checkpoint and retain classification predictions without choosing any model.

    Checkpoint type, restored weight hash, output dimensions and probability validity must all agree. Invalid outputs
    raise before prediction artifacts or reported metrics are written; the caller has already consumed the attempt.
    """
    model = keras.models.load_model(winner.checkpoint)

    if not isinstance(model, keras.Model) or weights_hash(model) != winner.weights_sha256:
        msg = "Selected checkpoint type or restored weights disagree with the seal."
        raise ValueError(msg)

    probability = np.asarray(model.predict(images, batch_size=64), dtype=np.float32)

    if (
        probability.shape != labels.shape
        or not np.isfinite(probability).all()
        or np.any(probability < 0)
        or np.any(probability > 1)
        or not np.allclose(probability.sum(axis=1), 1, atol=1e-5)
    ):
        msg = "Selected checkpoint did not produce finite class probabilities."
        raise ValueError(msg)

    prediction, truth = probability.argmax(axis=1), labels.argmax(axis=1)
    errors, count = int(np.count_nonzero(prediction != truth)), len(truth)
    np.savez_compressed(output, probability=probability, prediction=prediction, truth=truth)
    keras.backend.clear_session()

    return {
        "errors": errors,
        "count": count,
        "error_rate": errors / count,
        "accuracy": 1 - errors / count,
        "predictions": str(output),
        "evaluation_x_sha256": array_hash(images),
        "evaluation_y_sha256": array_hash(labels),
    }


def finalize_suite(lock_path: Path, evaluation_archives: dict[str, Path]) -> None:
    """Verify sealed decisions, then evaluate each distinct selected checkpoint once.

    Archives contain only float32 `evaluation_x` and one-hot `evaluation_y`. Their absolute paths and byte hashes must
    match the assignment frozen before training. Attempt markers precede hashing/loading and persist on failure.
    Predictions, labels and classification counts are retained; evaluation never selects models.
    """
    if _hash(lock_path) != lock_path.with_suffix(".sha256").read_text():
        msg = "Suite lock changed after sealing."
        raise ValueError(msg)

    lock = json.loads(lock_path.read_text())

    if any(_hash(Path(path)) != digest for path, digest in lock["artifacts"].items()):
        msg = "A sealed run artifact changed before evaluation."
        raise ValueError(msg)

    winners = [Winner(**row) for row in lock["winners"]]

    expected = {dataset: Path(record["path"]) for dataset, record in lock["evaluation_archives"].items()}

    if (
        set(expected) != {row.dataset for row in winners}
        or not set(expected) <= {"cifar10", "cifar100"}
        or {dataset: path.resolve() for dataset, path in evaluation_archives.items()}
        != {dataset: path.resolve() for dataset, path in expected.items()}
        or not all(path.is_absolute() for path in expected.values())
    ):
        msg = "Supply exactly the CIFAR datasets assigned in the sealed suite."
        raise ValueError(msg)

    for directory in [lock_path.parent, *sorted({Path(row.run) for row in winners})]:
        with (directory / "evaluation-attempt.json").open("x") as stream:
            _ = stream.write(json.dumps({"suite_sha256": _hash(lock_path)}))

    if any(_hash(path) != lock["evaluation_archives"][dataset]["sha256"] for dataset, path in expected.items()):
        msg = "Evaluation archive bytes disagree with the prospectively frozen assignment."
        raise ValueError(msg)

    evaluated: dict[str, dict[str, int | float | str]] = {}

    for dataset, archive_path in evaluation_archives.items():
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive) != {"evaluation_x", "evaluation_y"}:
                msg = "Evaluation archive must contain only evaluation_x and evaluation_y."
                raise ValueError(msg)

            images, labels = archive["evaluation_x"], archive["evaluation_y"]

        _validate(TrainingArrays(images, labels, images, labels), FitSettings(), "final")

        if labels.shape[1] != (10 if dataset == "cifar10" else 100):
            msg = "Evaluation labels disagree with the sealed dataset."
            raise ValueError(msg)

        for winner in {row.checkpoint: row for row in winners if row.dataset == dataset}.values():
            output = lock_path.parent / f"predictions-{len(evaluated)}.npz"
            evaluated[winner.checkpoint] = _classify(winner, images, labels, output)

    _ = (lock_path.parent / "classification.json").write_text(
        json.dumps(
            [{**asdict(winner), **evaluated[winner.checkpoint]} for winner in winners], allow_nan=False, sort_keys=True
        )
    )
