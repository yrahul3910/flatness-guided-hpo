# TuRBO image integration audit

The image adapter now decodes all seven hyperparameters deterministically, returns native Python values, and preserves the optimizer's NumPy state across `SearchRun.evaluate`. Two focused regression tests pass, and an unmodified official TuRBO optimizer completed exactly 30 calls through the actual image entrypoint using an analytic stand-in for `SearchRun`. No neural training, datasets, test-set evaluations, package installation, dependency edits, or paid compute were involved.

## Source and environment

The intended implementation is [Uber's TuRBO](https://github.com/uber-research/TuRBO), package version 0.0.1. No installed `turbo` module or local TuRBO source was found in the bounded repository/cache search. Existing NumPy 1.26.4, Torch 2.10.0, GPyTorch 1.15.1, and linear-operator 0.6 were sufficient for the smoke checks.

Root approved a source-only acquisition capped at 1 MiB. The official package files, license, and setup metadata were downloaded at commit `de0db39f481d9505bb3610b7b7aa0ebf7702e4a5`; total download was 40,592 bytes. URLs, byte counts, and hashes are in `upstream/acquisition-manifest.json`. The first sandbox attempt failed at DNS resolution; the approved network-enabled retry succeeded. No unrelated package named `turbo` was used. The upstream source remains unchanged.

## Adapter behavior

The prior five-dimensional point omitted `padding` and `n_units`, which were randomly drawn on each evaluation. Therefore, repeated points could represent different architectures. Its NumPy integer `n_units` also violated the installed Keras Dense constructor's Python-integer requirement. Both defects are removed by deterministic list indexing.

| Coordinate | Bounds | Decoding |
| --- | --- | --- |
| `n_filters` | 32 to 128 | Round to Python integer |
| `kernel_size` | 2 to 6 | Round to Python integer |
| `padding` | 0 to 2 | Equal-width bins for `valid`, `same` |
| `n_blocks` | 2 to 5 | Round to Python integer |
| `n_units` | 0 to 5 | Equal-width bins for 32, 64, 128, 256, 512 |
| `learning_rate` | 0.0001 to 0.01 | Python float, linear scale |
| `weight_decay` | 0.00001 to 0.001 | Python float, linear scale |

List coordinates use `min(int(x), len(choices) - 1)` on the nonnegative bounded interval, including its exact upper endpoint. Numeric bounds and linear scales are preserved. This is an ordinal continuous relaxation with arbitrary ordering for nominal choices, not a categorical TuRBO kernel or a completed shared-domain comparison protocol.

The minimization objective remains `1 - validation_accuracy`; recoverable `ValueError`/`RuntimeError` evaluations retain loss 100. NumPy state is restored in `finally` because `SearchRun` reseeds global NumPy per training trial. Torch receives the run seed. Current settings are dimension 7, `n_init=15`, `max_evals=30`, and explicit `batch_size=1`.

## Verification

- The final focused tests fail against the preserved prior entrypoint: five-coordinate bounds and altered NumPy state are detected. They pass against the updated source, covering repeated-point decoding, all list bins, boundaries, native integer types, objective sign, constructor budgets, Torch seed propagation, and NumPy restoration after success and handled failure.
- An initial test fixture imported TensorFlow inside a temporary `sys.modules` patch, then removed those imports on exit; a second import aborted in TensorFlow's native registry. The fixture now imports dependencies before that patch. The failed source/log remain in `test-initial-crash.py.txt` and `test-initial-crash.log`.
- Real upstream optimizer smoke: seeds 31, 31, 47; two dimensions; five initial plus three GP-guided evaluations per run; 30 GP fitting steps; 24 total analytic evaluations. Exact same-seed replay and different-seed variation passed in 0.626 seconds. Every trajectory is retained.
- Actual image entrypoint smoke: seed 31; seven dimensions; 15 initial plus 15 GP-guided calls; default 50 GP fitting steps; exactly 30 analytic evaluations in 0.969 seconds. Raw coordinates, decoded configurations, trial seeds, objectives, import path, and source hashes are retained. This is one integration check, not a seven-dimensional replay or HPO efficacy result.
- The existing full mocked image integration suite passes without fixture changes. Its prior source/log and new output are preserved here.
- Production entrypoint and focused tests pass Ruff, formatting, and no-sloppy. Scoped-source Basedpyright reports 0 errors and 21 warnings concerning untyped dynamic configuration/dependency values and existing unused return values. The executed acquisition/smoke scripts are retained verbatim; their separate `audit-scripts-style.log` records 16 findings, including standalone-script/printing/URL-audit rules, two exception-message style findings, and a synchronous loop callback warning. These scripts are not claimed lint-clean.

Both real optimizer smokes ran in supervised subprocesses with a 60-second wall limit and one CPU thread. The entrypoint command was equivalent to the following from the repository root; exact resolved paths and environment are in `entrypoint-smoke-command.json`:

```sh
PYTHONPATH="$PWD/sweep-results/research-audit-20261002/turbo-integration/upstream:$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <ORIGINAL_CHECKOUT>/.venv/bin/python -u sweep-results/research-audit-20261002/turbo-integration/entrypoint_smoke.py
```

Existing event files are rejected on rerun to preserve evidence. The upstream import is scoped to this audit source directory; `pyproject.toml`, `uv.lock`, and the installed environment remain unchanged. Ordinary production invocation still requires an explicitly selected dependency/source path. This audit does not register TuRBO as an installed project dependency.

The exact 30-call result applies to these fixed settings. Upstream restarts can initialize a whole new batch of `n_init` points without clamping it to the remaining budget; different budgets/settings need their own cap audit. With dimension 7 and batch 1, contracting the initial trust region below its minimum requires at least 49 unsuccessful guided calls, beyond this run's 15-call guided budget. The official implementation targets noise-free objectives; stochastic image-training behavior and any fully matched mixed-domain protocol remain separate research decisions.
