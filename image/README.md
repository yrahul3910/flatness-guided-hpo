# File description

Research is paused as of 2026-10-03. See [the review checkpoint](../RESEARCH_CHECKPOINT.md) for current results, limitations, and unfinished work. No scientific CIFAR study has been launched.

* `*_hpo.py`: Implementation of HPO using the algorithm in the filename.
* `common.py`: Shared functions. Probably should move to src/util.
* `stats.pysh`: Code to run statistics.
* `stcvx.py`: STCVX algorithm (strong convexity-based HPO)
* `parsers/`: Result parsers for each output file
* `results/`: The raw results. Warning: the files are compressed, and expand to about 4GB.
* `src/`: Utility functions

Please download the SVHN dataset, in the cropped digits format, from http://ufldl.stanford.edu/housenumbers/, and place the files here.

## Seeded searches and evidence

The Python `random`, `stcvx`, `bohb`, `hebo`, `hyperopt`, and `turbo` entry points now select candidates on validation accuracy and evaluate the saved winner once on the official test set. Every initialized `SearchRun` records metadata/source snapshots and attempted-trial/run events; `incumbent.keras` exists only after a successful eligible candidate. BOHB also writes `adapter.json` and `bohb-proposals.jsonl`. The separate matched runner follows the validation-only contract below. Existing output directories are rejected. Use `DATASET` and `FGHPO_SEED` explicitly; `CLOUD_EXP_DATASET` and `SWEEP_DATASET` are accepted when `DATASET` is absent. `FGHPO_OUTPUT_DIR` names the exact new run directory; otherwise a unique directory is created under `SWEEP_RESULTS_DIR` or `sweep-results`.

For example, after the final experimental protocol and compute allocation are fixed:

```sh
DATASET=cifar10 FGHPO_SEED=31 FGHPO_OUTPUT_DIR=sweep-results/random-cifar10-seed31 python -m image.random_hpo
DATASET=cifar10 FGHPO_SEED=31 FGHPO_OUTPUT_DIR=sweep-results/stcvx-cifar10-seed31 python -m image.stcvx
```

These commands run full searches, not smoke tests. STCVX allows at most `FGHPO_MAX_PROBE_ATTEMPTS` attempts (default 1,000) to collect 100 candidates above its fixed 0.1 threshold. Insufficient eligible probes fail the run. A `final_test` event is the selected result; candidate scores and proxy attempts are not independent HPO repetitions. Legacy text parsers apply to historical outputs and must not be used to aggregate the new event logs. Results remain exploratory by default; final evidence also requires the outstanding fairness, theory, and reproducibility checks in `NOTES.md`.

Remote jobs must configure their uploader to include the explicit run directory. The legacy cloud startup script is not safe to reuse: it deletes its VM and does not invoke the STCVX entry point correctly. No remote launch is performed by the Python entry points themselves.

## Matched screening development runner

`matched_hpo.py` implements the draft RMS, exact head-trace, same-probe validation-CE, and random-shortlist comparison. It accepts an NPZ containing only `fit_x`, `fit_y`, `validation_x`, and `validation_y`: preprocessed float32 32x32 RGB images and float32 one-hot labels with 10 or 100 columns. It has no official-test input or evaluation path. Preparing the official training split, normalization, row IDs, and upstream dataset provenance remains the caller's responsibility. The proposed 40,000/10,000 CIFAR split and final benchmark scale have not been launched.

Run `python -m image.matched_hpo --help` for the array, dataset, output, repetition, candidate, finalist, and required wall-limit arguments. The CLI defaults to eight candidate occurrences and two finalists per arm, with the proposed 100-epoch maximum, patience 25, batch size 64, and at most 1,500 probe examples. It applies common validity exclusions without the legacy 0.1 threshold. The random-shortlist control shares the probes; it is not the standalone 30-fit random baseline. `RunSettings` supports explicitly recorded engineering fixtures with shorter horizons; `proposal_mode="fixed_fixture"` distinguishes an externally supplied fixed panel from replay of generated proposals. Every output is labeled development-only. Existing directories and repeated execution of the same run object are rejected.

The runner saves the proposal/seed manifest, input/source hashes, probe subset order, all raw probe outcomes before selection, all three shortlists, and one fresh final fit for each candidate in their union. Finalist failures consume slots without replacements. Each arm selects restored-checkpoint validation accuracy; the checkpoint itself is selected by the actual Keras validation-loss callback. Per-phase histories retain permutations, callback state, metrics, costs, and checkpoint hashes. Augmentation and shuffle RNG state is recorded separately; the saved `.keras` file alone is not a continuation interface. Logical phase/ranking costs and physical elapsed time are separate quantities. The wall limit is cooperative between batches, so a hard subprocess/provider deadline is still required.

`src/matched_contract.py` supplies inclusive integer bins, categorical choices, log-uniform LR/WD decoding, conditional convolution geometry, and candidate/phase seeds. `bohb_hpo.py` uses this decoder through five normalized numeric coordinates and two native categorical coordinates, with method label `bohb-unit-coordinate`. Its append-only proposal ledger retains latent coordinates, decoded configurations, budgets, and failures. This adapter corrects the installed KDE sampler's native-unit clipping defect and changes the earlier sampling laws; historical BOHB results must retain their original label. Its 199-call cold-fit schedule remains unchanged and is not compute-matched to 30-call baselines.

The bounded local validation uses synthetic images only. It verifies implementation behavior, not CIFAR performance or the paper's empirical claim. Scientific framing, primary threshold policy, final study scale, actual training-data preparation, and remote resource limits remain separate decisions recorded in `NOTES.md`.
