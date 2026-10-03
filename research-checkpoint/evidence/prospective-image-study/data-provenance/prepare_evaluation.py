"""Freeze official CIFAR evaluation inputs locally, without model use or outcome inspection.

These files must remain off training/calibration workers. The suite finalizer may read them only after every
assigned block is locked. This mechanical preparation does not make historical CIFAR test use disappear.
"""

import argparse
import hashlib
import json
import pickle
import sys
import tarfile
from pathlib import Path

import numpy as np

from .prepare_cifar import SOURCES, file_hash


def prepare(dataset: str, archive: Path, training_manifest: Path, output: Path) -> None:
    """Authenticate the archive and save all 10,000 official test rows in fixed order.

    Class names come from the completed training preparation. No model, class count, sample image, or quality
    metric is computed. A successful manifest follows the NPZ write; existing output directories are rejected.
    """
    source = SOURCES[dataset]
    training = json.loads(training_manifest.read_text())

    if file_hash(archive) != source.sha256 or training["source"]["sha256"] != source.sha256:
        message = "Archive and training manifest must match the pinned official SHA256."
        raise ValueError(message)

    if training["status"] != "complete" or training["dataset"] != dataset:
        message = "A completed preparation manifest for this dataset is required."
        raise ValueError(message)

    output.mkdir(parents=True, exist_ok=False)
    member_name = f"{source.directory}/{'test_batch' if dataset == 'cifar10' else 'test'}"

    with tarfile.open(archive, "r:gz") as bundle:
        if (stream := bundle.extractfile(member_name)) is None:
            message = "Official evaluation member is missing."
            raise ValueError(message)

        with stream:
            raw = stream.read()

    # Official CIFAR pickle bytes are decoded only after verifying the complete archive's pinned SHA256.
    payload = pickle.loads(raw, encoding="bytes")  # noqa: S301
    pixels, labels = payload[b"data"], np.asarray(payload[source.label_key])

    if pixels.shape != (10000, 3072) or pixels.dtype != np.uint8 or labels.shape != (10000,):
        message = "Official evaluation shapes or pixel dtype are invalid."
        raise ValueError(message)

    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    images = (images / np.float32(255) - np.asarray(source.mean, np.float32)) / np.asarray(source.std, np.float32)
    targets = np.eye(source.classes, dtype=np.float32)[labels]
    arrays = {"evaluation_x": images, "evaluation_y": targets}
    np.savez(output / "evaluation.npz", **arrays)
    indices = np.arange(10000, dtype=np.int64)
    np.save(output / "official-test-indices.npy", indices, allow_pickle=False)

    manifest = {
        "status": "complete",
        "dataset": dataset,
        "archive": str(archive.resolve()),
        "archive_sha256": source.sha256,
        "member": member_name,
        "member_sha256": hashlib.sha256(raw).hexdigest(),
        "row_namespace": f"{dataset}/test",
        "indices_half_open": [0, 10000],
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "class_names": training["class_names"],
        "training_manifest_sha256": file_hash(training_manifest),
        "preprocessor_sha256": file_hash(Path(__file__)),
        "shared_source_sha256": file_hash(Path(__file__).with_name("prepare_cifar.py")),
        "numpy": np.__version__,
        "python": sys.version,
        "command": sys.argv,
        "test_member_opened_for_mechanical_preparation": True,
        "model_use_or_outcome_inspection": False,
        "normalization": training["normalization"],
        "mean": source.mean,
        "std": source.std,
        "arrays": {
            key: {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
            for key, value in arrays.items()
        },
        "evaluation_npz_sha256": file_hash(output / "evaluation.npz"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("archive", type=Path)
    parser.add_argument("training_manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.dataset, args.archive, args.training_manifest, args.output)
