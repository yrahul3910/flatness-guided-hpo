"""Copy deterministic calibration subsets from the authenticated training-only NPZs.

Run as a module from the repository root. The new output directory is created exclusively; a completed manifest
is written only after both datasets pass checksum and byte-equality checks. No evaluation archive is opened.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .prepare_cifar import file_hash

NAMESPACE = "fghpo/prospective-image-study/calibration-inputs/v1"
MANIFEST_HASHES = {
    "cifar10": "c5c4f5342e1bb30d338ff4bc53fa8f3e1c865c2446e22de3c56fa43a613401ab",
    "cifar100": "fa6506ac30f914e2a3a23ccff0066d8c589d77e964b97a76e939981d8570f51d",
}
KEYS = {"fit_x", "fit_y", "validation_x", "validation_y"}


def prepare() -> None:
    provenance = Path(__file__).resolve().parent
    output = provenance.parent / "calibration-inputs"
    output.mkdir(exist_ok=False)
    records = {}

    for dataset, expected_manifest_hash in MANIFEST_HASHES.items():
        source_manifest = provenance / "prepared" / dataset / "manifest.json"
        source_npz = source_manifest.with_name("training.npz")

        if file_hash(source_manifest) != expected_manifest_hash:
            message = "Training provenance manifest changed."
            raise ValueError(message)

        metadata = json.loads(source_manifest.read_text())
        expected_npz_hash = metadata["training_npz_sha256"]

        if metadata["status"] != "complete" or file_hash(source_npz) != expected_npz_hash:
            message = "Training NPZ checksum or completion status failed."
            raise ValueError(message)

        indices = {}
        selections = {}

        for split, population, count, offset in (("fit", 40000, 1500, 0), ("validation", 10000, 1024, 40000)):
            namespace = f"{NAMESPACE}/{dataset}/{split}"
            seed = int.from_bytes(hashlib.sha256(namespace.encode("ascii")).digest()[:8], "big")
            indices[split] = np.random.Generator(np.random.PCG64(seed)).permutation(population)[:count]
            index_file = output / f"{dataset}-{split}-official-train-indices.npy"
            np.save(index_file, indices[split] + offset, allow_pickle=False)
            selections[split] = {
                "namespace": namespace,
                "seed": seed,
                "rng": "PCG64; permutation(population)[:count]; no label stratification",
                "population": population,
                "count": count,
                "official_train_offset": offset,
                "indices_path": str(index_file),
                "indices_sha256": file_hash(index_file),
            }

        _ = (output / f"{dataset}-selection.json").write_text(json.dumps(selections, indent=2) + "\n")

        with np.load(source_npz, allow_pickle=False) as source:
            if set(source.files) != KEYS:
                message = "Training archive keys are not the four authorized arrays."
                raise ValueError(message)

            arrays = {key: source[key][indices[key.rsplit("_", 1)[0]]] for key in sorted(KEYS)}

        destination = output / f"{dataset}.npz"
        np.savez(destination, **arrays)

        with np.load(destination, allow_pickle=False) as saved:
            if set(saved.files) != KEYS or any(not np.array_equal(saved[key], arrays[key]) for key in KEYS):
                message = "Calibration output differs from selected training rows."
                raise ValueError(message)

        records[dataset] = {
            "source_npz": str(source_npz),
            "source_npz_sha256": expected_npz_hash,
            "source_manifest_sha256": expected_manifest_hash,
            "selections": selections,
            "path": str(destination),
            "sha256": file_hash(destination),
            "bytes": destination.stat().st_size,
            "arrays": {
                key: {
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                    "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                }
                for key, value in arrays.items()
            },
        }

    manifest = {
        "status": "complete",
        "purpose": "Timing and memory calibration only; no quality-based study choices",
        "source_sha256": file_hash(Path(__file__)),
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "evaluation_archives_opened": [],
        "normalization_and_class_maps": "Unmodified copies; inherited from authenticated source manifests",
        "datasets": records,
    }
    _ = (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    prepare()
