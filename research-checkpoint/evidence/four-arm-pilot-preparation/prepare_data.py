"""Materialize the predeclared KMNIST splits without models or outcome summaries."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BASE = Path(__file__).resolve().parent
CLASSES = 10


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_source(name: str, expected_shape: tuple[int, ...]) -> NDArray[np.uint8]:
    """Read a publisher archive with a fixed image or label schema, without pickle support."""
    with np.load(BASE / "raw" / name, allow_pickle=False) as archive:
        values = archive["arr_0"]

    if values.shape != expected_shape or values.dtype != np.uint8:
        msg = f"Unexpected source schema for {name}."
        raise ValueError(msg)

    return values


def main() -> None:
    """Create a new prepared-data directory and preserve exact source, index, and array hashes.

    This process never trains or evaluates a model. Repeated execution refuses to overwrite the prepared dataset.
    Evaluation labels are copied mechanically, with no label-dependent choice of indices or preprocessing.
    """
    plan_path = BASE / "data-plan.json"
    plan = json.loads(plan_path.read_text())
    output = BASE / "prepared-data"
    output.mkdir(exist_ok=False)
    train_rule, eval_rule = plan["fit_tuning_split"], plan["evaluation_split"]
    train_order = np.random.Generator(np.random.PCG64(train_rule["seed"])).permutation(train_rule["population"])
    eval_order = np.random.Generator(np.random.PCG64(eval_rule["seed"])).permutation(eval_rule["population"])
    indices = {
        "fit": train_order[slice(*train_rule["fit_permutation_slice"])],
        "tuning": train_order[slice(*train_rule["tuning_permutation_slice"])],
        "evaluation": eval_order[slice(*eval_rule["permutation_slice"])],
    }
    np.savez(output / "indices.npz", **indices)
    index_hash = file_hash(output / "indices.npz")
    _ = (output / "index-freeze.json").write_text(
        json.dumps({"created_utc": datetime.now(UTC).isoformat(), "indices_sha256": index_hash}) + "\n"
    )

    train_images = read_source("kmnist-train-imgs.npz", (60000, 28, 28))
    train_labels = read_source("kmnist-train-labels.npz", (60000,))
    test_images = read_source("kmnist-test-imgs.npz", (10000, 28, 28))
    test_labels = read_source("kmnist-test-labels.npz", (10000,))
    arrays: dict[str, NDArray[np.float32]] = {}

    for split, row_ids in indices.items():
        images, labels = (test_images, test_labels) if split == "evaluation" else (train_images, train_labels)

        if np.any(labels[row_ids] >= CLASSES):
            msg = "Labels fall outside the fixed ten-class schema."
            raise ValueError(msg)

        arrays[f"{split}_x"] = images[row_ids].astype(np.float32)[..., None] / np.float32(255)
        arrays[f"{split}_y"] = np.eye(CLASSES, dtype=np.float32)[labels[row_ids]]

    np.savez_compressed(output / "dataset.npz", **arrays)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "data_plan_sha256": file_hash(plan_path),
        "preparation_source_sha256": file_hash(Path(__file__)),
        "numpy_version": np.__version__,
        "sources": {
            path.name: {
                "url": plan["download_base"] + path.name,
                "bytes": path.stat().st_size,
                "sha256": file_hash(path),
            }
            for path in sorted((BASE / "raw").glob("*.npz"))
        },
        "indices_sha256": index_hash,
        "dataset_sha256": file_hash(output / "dataset.npz"),
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
            for name, value in arrays.items()
        },
        "model_evaluations": 0,
        "training_updates": 0,
        "outcome_or_class_distribution_summaries": 0,
    }
    _ = (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _ = sys.stdout.write(json.dumps({"status": "prepared", "dataset_sha256": manifest["dataset_sha256"]}) + "\n")


if __name__ == "__main__":
    main()
