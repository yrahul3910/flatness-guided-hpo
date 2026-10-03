"""Prepare CIFAR fit/tuning arrays without opening the archive's test member.

The archive must match its pinned official SHA256 before any pickle is decoded. Existing output directories are
rejected; failed preparation retains its partial output. Only a completed manifest authorizes downstream use.
"""

import argparse
import hashlib
import json
import pickle
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Source:
    sha256: str
    directory: str
    training: tuple[str, ...]
    metadata: str
    label_key: bytes
    names_key: bytes
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    classes: int


SOURCES = {
    "cifar10": Source(
        "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
        "cifar-10-batches-py",
        tuple(f"data_batch_{i}" for i in range(1, 6)),
        "batches.meta",
        b"labels",
        b"label_names",
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616),
        10,
    ),
    "cifar100": Source(
        "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
        "cifar-100-python",
        ("train",),
        "meta",
        b"fine_labels",
        b"fine_label_names",
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761),
        100,
    ),
}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def training_payloads(archive: Path, source: Source) -> dict[str, bytes]:
    """Read only the explicit training and class-name allowlist from a verified archive."""
    payloads: dict[str, bytes] = {}

    with tarfile.open(archive, "r:gz") as bundle:
        for name in (*source.training, source.metadata):
            member = bundle.getmember(f"{source.directory}/{name}")

            if not member.isfile() or (stream := bundle.extractfile(member)) is None:
                message = f"Expected regular archive member: {member.name}"
                raise ValueError(message)

            with stream:
                payloads[name] = stream.read()

    return payloads


def prepare(dataset: str, archive: Path, output: Path) -> None:
    """Verify provenance, decode training members, and persist the fixed 40k/10k split.

    Dataset identity is fixed by SHA256 and class metadata. Arrays are float32 NHWC with one-hot labels. The
    official test member is never extracted or decoded. Existing directories and malformed training shapes fail.
    """
    source = SOURCES[dataset]

    if file_hash(archive) != source.sha256:
        message = "Archive SHA256 differs from the pinned official dataset."
        raise ValueError(message)

    output.mkdir(parents=True, exist_ok=False)
    payloads = training_payloads(archive, source)
    # Official CIFAR uses pickle; the complete archive was authenticated against its pinned SHA256 above.
    decoded = {name: pickle.loads(raw, encoding="bytes") for name, raw in payloads.items()}  # noqa: S301
    pixels = np.concatenate([decoded[name][b"data"] for name in source.training])
    labels = np.concatenate([np.asarray(decoded[name][source.label_key]) for name in source.training])
    names = [name.decode("utf-8") for name in decoded[source.metadata][source.names_key]]

    if pixels.shape != (50000, 3072) or pixels.dtype != np.uint8 or labels.shape != (50000,):
        message = "Training shapes or pixel dtype do not match official CIFAR."
        raise ValueError(message)

    if len(names) != source.classes or not np.all((labels >= 0) & (labels < source.classes)):
        message = "Class metadata or labels do not match the selected dataset."
        raise ValueError(message)

    images = pixels.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    images = (images / np.float32(255) - np.asarray(source.mean, np.float32)) / np.asarray(source.std, np.float32)
    targets = np.eye(source.classes, dtype=np.float32)[labels]
    arrays = {
        "fit_x": images[:40000],
        "fit_y": targets[:40000],
        "validation_x": images[40000:],
        "validation_y": targets[40000:],
    }
    np.savez(output / "training.npz", **arrays)
    spec = asdict(source) | {"label_key": source.label_key.decode(), "names_key": source.names_key.decode()}
    manifest = {
        "status": "complete",
        "dataset": dataset,
        "archive": str(archive.resolve()),
        "source": spec,
        "archive_url": f"https://www.cs.toronto.edu/~kriz/cifar-{source.classes}-python.tar.gz",
        "members_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()},
        "class_names": names,
        "fit_official_train_rows": [0, 40000],
        "tuning_official_train_rows": [40000, 50000],
        "intervals": "half-open",
        "row_namespace": f"{dataset}/train",
        "loader": "prepare_cifar.training_payloads plus SHA256-authenticated standard-library pickle decoding",
        "normalization": "float32 NHWC; (pixel/255 - per-channel mean)/per-channel std; no learned statistics",
        "test_member_opened": False,
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "preprocessor_sha256": file_hash(Path(__file__)),
        "arrays": {
            key: {
                "shape": value.shape,
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
            }
            for key, value in arrays.items()
        },
        "training_npz_sha256": file_hash(output / "training.npz"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.dataset, args.archive, args.output)
