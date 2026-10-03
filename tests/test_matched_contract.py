"""Check reproducibility and selection boundaries without loading any dataset."""

import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from image.src.matched_contract import Probe, decode, make_manifest, phase_seed, shortlist


class SharedContractTests(unittest.TestCase):
    def test_manifest_replay_and_phase_identity(self) -> None:
        """Check replay, occurrence IDs, and rank-independent distinct phase seeds."""
        manifest = make_manifest("cifar10", 503, 100)
        np.testing.assert_equal(manifest, json.loads(json.dumps(manifest)))
        np.testing.assert_equal(manifest, make_manifest("cifar10", 503, 100))
        np.testing.assert_equal(
            manifest["candidates"] != make_manifest("cifar10", 509, 100)["candidates"], desired=True
        )
        np.testing.assert_equal(
            manifest["candidates"] != make_manifest("cifar100", 503, 100)["candidates"], desired=True
        )
        seeds = list(manifest["shared_seeds"].values())

        for candidate in reversed(manifest["candidates"]):
            seeds.extend(candidate["seeds"].values())

            for label, seed in candidate["seeds"].items():
                phase, role = label.split("/")
                np.testing.assert_equal(seed, phase_seed("cifar10", 503, candidate["candidate_id"], phase, role))

        np.testing.assert_equal(len(seeds), len(set(seeds)))
        np.testing.assert_equal(manifest["candidates"][0]["candidate_id"], "c000000")
        np.testing.assert_equal(manifest["candidates"][-1]["candidate_id"], "c000099")

        with patch("image.src.matched_contract.random.Random.random", return_value=0.0):
            duplicates = make_manifest("cifar10", 503, 2)["candidates"]

        np.testing.assert_equal(duplicates[0]["config"], duplicates[1]["config"])
        np.testing.assert_equal(duplicates[0]["candidate_id"] != duplicates[1]["candidate_id"], desired=True)
        np.testing.assert_equal(duplicates[0]["seeds"] != duplicates[1]["seeds"], desired=True)

    def test_domain_endpoints_and_geometry(self) -> None:
        """Exercise support boundaries and calculate spatial validity independently."""
        lower = decode([0.0] * 7)
        upper = decode([1.0] * 7)
        middle = decode([0.5] * 7)
        np.testing.assert_equal(
            set(lower), {"n_filters", "kernel_size", "padding", "n_blocks", "n_units", "learning_rate", "weight_decay"}
        )
        np.testing.assert_equal(
            (lower["n_filters"], lower["kernel_size"], lower["n_blocks"], lower["n_units"]), (32, 2, 2, 32)
        )
        np.testing.assert_equal(
            (upper["n_filters"], upper["kernel_size"], upper["n_blocks"], upper["n_units"]), (128, 6, 5, 512)
        )
        np.testing.assert_equal((middle["learning_rate"], middle["weight_decay"]), (1e-3, 1e-4))
        np.testing.assert_equal((lower["learning_rate"], upper["learning_rate"]), (1e-4, 1e-2))
        np.testing.assert_equal((lower["weight_decay"], upper["weight_decay"]), (1e-5, 1e-3))

        for kernel, max_depth in zip(range(2, 7), [4, 3, 3, 2, 2], strict=True):
            for padding in [0.0, 1.0]:
                config = decode([0.0, (kernel - 2) / 5, padding, 1.0, 0.0, 0.0, 0.0])
                np.testing.assert_equal(config["n_blocks"], max_depth if padding == 0 else 5)
                size = 32

                for _ in range(config["n_blocks"]):
                    size = (size - (kernel - 1) if padding == 0 else size) // 2
                    np.testing.assert_equal(size > 0, desired=True)

                if padding == 0:
                    np.testing.assert_equal((size - (kernel - 1)) // 2 <= 0, desired=True)

        for bad in [[0.0] * 6, [math.nan] * 7, [math.inf] * 7, [-0.1] * 7, [1.1] * 7]:
            with np.testing.assert_raises(ValueError):
                _ = decode(bad)

    def test_common_population_selection_and_failure_accounting(self) -> None:
        """Check score direction, ties, random independence, and complete failure accounting."""
        manifest = make_manifest("cifar10", 503, 9)
        probes = [
            Probe(f"c{i:06d}", "ok", rms, ce, float(i))
            for i, (rms, ce) in enumerate(
                [
                    (0.2, 0.8),
                    (0.2, 0.1),
                    (0.1, 0.4),
                    (0.0, 0.2),
                    (math.nan, 0.2),
                    (math.inf, 0.2),
                    (0.3, math.inf),
                    (-1.0, 0.2),
                ]
            )
        ]
        probes.append(Probe("c000008", "failed", None, None, None))
        rms = shortlist(manifest, probes, 2, "rms")
        ce = shortlist(manifest, probes, 2, "ce")
        random_result = shortlist(manifest, probes, 2, "random")
        trace = shortlist(manifest, probes, 2, "trace")
        np.testing.assert_equal(trace.selected, ("c000000", "c000001"))
        np.testing.assert_equal(trace.excluded, rms.excluded)
        np.testing.assert_equal(rms.selected, ("c000002", "c000000"))
        np.testing.assert_equal(ce.selected, ("c000001", "c000002"))
        np.testing.assert_equal(rms.excluded, ce.excluded)
        np.testing.assert_equal(rms.excluded, random_result.excluded)
        np.testing.assert_equal(len(rms.excluded), 6)
        np.testing.assert_equal(shortlist(manifest, list(reversed(probes)), 2, "random"), random_result)
        changed = [
            Probe(p.candidate_id, p.status, 10.0, 100.0, p.head_trace)
            if p.candidate_id in {"c000000", "c000001", "c000002"}
            else p
            for p in probes
        ]
        np.testing.assert_equal(shortlist(manifest, changed, 2, "random"), random_result)
        np.testing.assert_equal(shortlist(manifest, probes, 2, "rms", threshold=0.1).selected, ("c000000", "c000001"))

        for invalid_trace in [math.nan, math.inf, -1.0]:
            changed = [Probe("c000000", "ok", 0.2, 0.8, invalid_trace), *probes[1:]]
            selections = [shortlist(manifest, changed, 2, method) for method in ("rms", "trace", "ce", "random")]
            np.testing.assert_equal(
                all(selection.excluded == selections[0].excluded for selection in selections), desired=True
            )

        for bad, count in [(probes[:-1], 2), ([*probes, probes[0]], 2), (probes, 4)]:
            with np.testing.assert_raises(ValueError):
                _ = shortlist(manifest, bad, count, "rms")

        for missing in [Probe("c000000", "ok", None, 0.1, 0.0), Probe("c000000", "ok", 0.2, 0.1, None)]:
            with np.testing.assert_raises(ValueError):
                _ = shortlist(manifest, [missing, *probes[1:]], 2, "ce")


if __name__ == "__main__":
    _ = unittest.main()
