"""Freeze the assigned four-block study and its explicit training-only upload archive.

Run once from the project root after the executor and budget plan are frozen. Outputs are created exclusively;
existing assignments or payloads are never replaced. Data archives are hashed as bytes, without loading arrays.
"""

import hashlib
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

STUDY = Path("sweep-results/research-audit-20261002/prospective-image-study")
EXECUTION = STUDY / "scientific-execution"
CLOUD = Path(".cloudexp/research-audit-20261002/prospective-image-study/scientific-execution")
COHORT = [(dataset, repetition) for dataset in ("cifar10", "cifar100") for repetition in (2000, 2001)]


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_hash(value: object) -> str:
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


def save(path: Path, value: object) -> None:
    with path.open("x") as stream:
        _ = stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def build_archive(paths: list[Path]) -> dict[str, str | int]:
    """Create an exclusive allowlisted archive and retain each input's byte hash and size."""
    members = [{"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in paths]
    payload = CLOUD / "payload.tar.gz"

    with tarfile.open(payload, "x:gz", compresslevel=1) as archive:
        for path in paths:
            archive.add(path, arcname=str(path), recursive=False)

    record = {"path": str(payload), "bytes": payload.stat().st_size, "sha256": file_hash(payload)}
    save(CLOUD / "payload-manifest.json", {"archive": record, "files": members})

    return record


def main() -> None:
    budget_path = EXECUTION / "block-budget-plan.json"
    budget = json.loads(budget_path.read_text())
    blocks = budget["blocks"]

    if [(row["dataset"], row["repetition_id"]) for row in blocks] != COHORT:
        msg = "Budget rows must match the exact ordered four-block cohort."
        raise ValueError(msg)

    (EXECUTION / "settings").mkdir(exist_ok=False)
    CLOUD.mkdir(parents=True, exist_ok=True)
    sources = [
        *sorted(Path("image").rglob("*.py")),
        Path("pyproject.toml"),
        Path("uv.lock"),
        EXECUTION / "modal_launch.py",
        EXECUTION / "run_block.py",
        Path(__file__).resolve().relative_to(Path.cwd()),
        STUDY / "calibration/requirements-t4.txt",
        STUDY / "data-provenance/prepare_cifar.py",
    ]
    source_hashes = {str(path): file_hash(path) for path in sources}

    for path in sources:
        destination = EXECUTION / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    inputs = {}
    payload_files = list(sources)

    for dataset in ("cifar10", "cifar100"):
        provenance_path = STUDY / "data-provenance/prepared" / dataset / "manifest.json"
        provenance = json.loads(provenance_path.read_text())
        arrays_path = provenance_path.with_name("training.npz")
        digest = file_hash(arrays_path)

        with zipfile.ZipFile(arrays_path) as archive:
            names = archive.namelist()

        if digest != provenance["training_npz_sha256"] or sorted(names) != [
            "fit_x.npy",
            "fit_y.npy",
            "validation_x.npy",
            "validation_y.npy",
        ]:
            msg = "Training archive hash or four-member allowlist disagrees with its provenance."
            raise ValueError(msg)

        inputs[dataset] = {
            "path": str(arrays_path),
            "sha256": digest,
            "bytes": arrays_path.stat().st_size,
            "arrays": provenance["arrays"],
            "array_sha256_recipe": "SHA256 of C-order contiguous array bytes; shape and dtype checked separately",
            "provenance_path": str(provenance_path),
            "provenance_sha256": file_hash(provenance_path),
        }
        payload_files.extend((arrays_path, provenance_path))

    rows = []

    for block in blocks:
        dataset, repetition = block["dataset"], block["repetition_id"]
        manifest_path = EXECUTION / "manifests" / f"{dataset}-{repetition}.json"
        manifest = json.loads(manifest_path.read_text())

        if manifest_path.read_bytes() != (STUDY / "candidate-plan" / manifest_path.name).read_bytes():
            msg = "Scientific candidate manifest changed after the candidate plan was frozen."
            raise ValueError(msg)

        settings = {
            "shortlist_size": 2,
            "probe_size": 1500,
            "max_epochs": 100,
            "patience": 25,
            "batch_size": 64,
            "max_seconds": float(block["run_settings_max_seconds"]),
            "proposal_mode": "generated",
        }
        settings_path = EXECUTION / "settings" / manifest_path.name
        save(settings_path, settings)
        rows.append(
            {
                "dataset": dataset,
                "repetition_id": repetition,
                "manifest_path": str(manifest_path),
                "manifest_sha256": canonical_hash(manifest),
                "settings_path": str(settings_path),
                "settings_sha256": canonical_hash(settings),
                "arrays_path": inputs[dataset]["path"],
                "output": str(STUDY / "runs" / f"{dataset}-{repetition}"),
                "hard_seconds": block["hard_seconds"],
                "provider_seconds": block["provider_seconds"],
                "max_seconds": block["run_settings_max_seconds"],
                "max_return_bytes": 3221225472,
            }
        )
        payload_files.extend((manifest_path, settings_path))

    assignment = {
        "runs": rows,
        "inputs": inputs,
        "source_sha256": source_hashes,
        "budget_plan": {"path": str(budget_path), "sha256": file_hash(budget_path)},
        "canonical_recipe": "SHA256 UTF-8 JSON; sorted keys, compact separators, allow_nan=False, no newline",
        "precision": {"float_policy": "float32", "tensor_float_32_execution_enabled": False},
        "test_policy": "Workers receive no evaluation arrays; all four blocks must complete before local evaluation.",
    }
    assignment_path = EXECUTION / "training-only-assignment.json"
    save(assignment_path, assignment)
    local_assignment = {
        **assignment,
        "evaluation_archives": json.loads((STUDY / "data-provenance/evaluation-archives.json").read_text()),
    }
    save(EXECUTION / "assigned-suite.json", local_assignment)
    payload_files.extend((assignment_path, budget_path))
    payload_files = sorted(set(payload_files))
    archive_record = build_archive(payload_files)
    save(
        EXECUTION / "training-job.json",
        {
            "payload_path": str((CLOUD / "payload.tar.gz").resolve()),
            "payload_sha256": archive_record["sha256"],
            "payload_manifest_path": str((CLOUD / "payload-manifest.json").resolve()),
            "training_assignment_path": str(assignment_path),
            "assignment_sha256": file_hash(assignment_path),
            "blocks": rows,
            "source_sha256": source_hashes,
        },
    )
    save(
        EXECUTION / "payload-preparation.json",
        {
            "status": "prepared_not_launched",
            "archive": archive_record,
            "member_count": len(payload_files),
            "source_sha256": source_hashes,
            "training_bytes": sum(row["bytes"] for row in inputs.values()),
            "evaluation_arrays_uploaded": False,
            "research_arrays_loaded": False,
            "model_or_cloud_calls": 0,
            "assigned_suite_sha256": file_hash(EXECUTION / "assigned-suite.json"),
            "training_assignment_sha256": file_hash(assignment_path),
        },
    )


if __name__ == "__main__":
    main()
