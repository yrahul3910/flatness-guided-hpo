# Paused research artifacts

[RESEARCH_CHECKPOINT.md](../RESEARCH_CHECKPOINT.md) describes the findings and unfinished work. Research is paused; no scientific CIFAR block was launched.

`evidence/` contains selected review copies from the local `sweep-results/research-audit-20261002/` directory, retaining their relative paths. `cloud-calibration/` contains scalar timing records and verification reports from the three completed Modal calibrations. `artifact-manifest.json` records each original local path, original SHA256, review-copy SHA256, and whether identifiers were redacted. The original artifacts are unchanged. Hashes embedded inside a sanitized report refer to the original artifacts, so they may differ from its review copy. Paths to omitted data and checkpoints are historical provenance, not files promised by this branch.

`history.md` is a sanitized copy of the working notes. It includes superseded plans and earlier intermediate conclusions; the checkpoint and pause-status report state the final status. `pause-status.json` records the final read-only resource verification and qualifies the account billing snapshot.

The review branch includes current image runner code, tests, the existing manuscript source and separate draft appendices. It does not claim that the unfinished executor is runnable or that the manuscript's original theorem has been repaired in place. The scientific executor draft requires a missing `run_block.py` and has unresolved validation, cleanup and durability defects.

The following remain preserved locally and are deliberately excluded from the public branch:

- `.cloudexp/`, including account configuration, client environments, billing snapshots, original launch logs, operational resource IDs and result archives. Curated scalar calibration evidence is copied here with provenance.
- Raw reviewer PDF and extracted reviewer text, private Codex session inventory, unrelated biological manuscript/checkpoint metadata, and personal paths or notification endpoints.
- All dataset arrays and archives, checkpoints, prediction arrays, virtual environments, caches and build outputs. Full scientific training arrays were prepared locally but never uploaded; only compact training/tuning calibration arrays were uploaded. Official evaluation arrays were never uploaded.
- Large historical `results/` logs, `image-rs/data.tar.gz`, unrelated root experiment/configuration files, and legacy cloud launch scripts. The original checkout, its inherited changes, and all local research artifacts remain intact.
- Audit files not selected for this review snapshot, including redundant source snapshots and third-party dependency copies. The retained reports name their original locations and hashes; no excluded artifact was deleted.

The local research root is the isolated `flatness-guided-hpo` clone in the task workspace. Its `sweep-results/research-audit-20261002/` directory contains the full evidence, and its `.cloudexp/research-audit-20261002/` directory contains operational records. The sibling `inherited-status.txt` and `inherited-changes.patch` preserve the starting checkout state.

No tests or scientific experiments were rerun for this documentation checkpoint. Earlier test reports distinguish real model replay, mocked checks and static warnings. The last executor draft is explicitly untested. No paper submission, public review comment, pull request or merge was made.
