# BOHB unit-coordinate adapter

`image/bohb_hpo.py` now uses the shared `image.src.matched_contract.decode` implementation and records method `bohb-unit-coordinate`. Five numeric dimensions use continuous unit coordinates. Padding and dense width retain the installed BOHB ordered-discrete KDE representation, with unit-bin centers as their decoded values. The BOHB dependency and its schedule were not modified.

## Defect and correction

The installed `bohb-hpo` 0.5.2 KDE sampler uses truncated-normal numeric proposals bounded by absolute values 0 and 1. Its hyperparameter setters clamp those values to each parameter's declared bounds. Native filter, kernel, and depth coordinates all had lower bounds greater than 1. The earlier retained fixture exercised this actual branch: all 128 considered candidates had `n_filters=32`, `kernel_size=2`, and `n_blocks=2`. That evidence and source remain unchanged; copies and hashes are in `before/` and `before-manifest.json`.

| Coordinate | BOHB representation | Shared decoder |
| --- | --- | --- |
| `x0` | Uniform `[0,1]` | 32 through 128 filters, inclusive equal-width bins |
| `x1` | Uniform `[0,1]` | Kernel 2 through 6, inclusive equal-width bins |
| `x2` | Native categorical values `[0.25,0.75]` | `valid`, `same` |
| `x3` | Uniform `[0,1]` | Depth 2 through conditional maximum, inclusive equal-width bins |
| `x4` | Native categorical values `[0.1,0.3,0.5,0.7,0.9]` | 32, 64, 128, 256, 512 dense units |
| `x5` | Uniform `[0,1]` | Learning rate `10**(-4+2*x5)` |
| `x6` | Uniform `[0,1]` | Weight decay `10**(-5+2*x6)` |

For valid padding, maximum depths for kernels 2 through 6 are 4, 3, 3, 2, 2; same padding allows depth 5. Exact unit endpoint 1 is supported. Parameter creation follows coordinate order because the installed `Configuration` sorts parameters by their creation IDs. `to_dict()` returns categorical bin centers for decoding; `to_list()` returns their ordered category indices for KDE. The actual KDE type string remains `ccococc`.

This changes the adapter's geometry and initial sampling laws: inclusive integer bins, conditional depth, and logarithmic learning-rate/weight-decay sampling replace the prior native-unit behavior. Padding still receives an artificial order in the installed ordered-discrete KDE. This is explicitly a changed adapter, not a claim that historical BOHB results used the corrected contract or that all optimizers now share a completed benchmark protocol.

## Randomness, budgets, and evidence

The initial `ConfigurationSpace` private generator receives `run.seed`. The objective saves and restores NumPy's legacy global state in `finally`, isolating installed BOHB/SciPy proposal randomness from `SearchRun` model reseeding. The inspected proposal path does not use Python's `random` stream.

The minimization objective stays `1 - validation_accuracy`. Existing budget mapping is preserved, including raw 99 mapped to 100 epochs and selection eligibility only at the full mapped budget. Recoverable model `ValueError`/`RuntimeError` failures retain loss 100; other exceptions are recorded and propagated. Every objective call starts a fresh model, including repeated configurations and promoted rungs. This edit does not convert BOHB into weight-preserving promotion.

Each production run writes `adapter.json` and append-only `bohb-proposals.jsonl`. The proposal ledger contains stable objective-call IDs, latent values, decoded configurations, raw budget, mapped epochs, selection eligibility, start/end events, loss, and errors. Strict JSON is used; nonfinite coordinates/losses are represented explicitly as strings. Decoder failures retain an end record with the latent proposal and error before any model fit. Missing latent keys are outside the trusted installed sampler contract and are not treated as an external-input validation interface.

## Checks and retained outcomes

- Three focused tests pass. Initial sampler seeds 31/31 replay and seed 47 varies. Callback cases check exact lower/midpoint/upper decoder agreement, native integer types, raw budgets 1/99/100, corresponding selection flags, NumPy restoration, and a retained controlled model failure. The shared decoder's exhaustive bin/geometry checks remain in `tests/test_matched_contract.py`.
- The actual installed KDE branch was forced with `random_percent=0` on two fixed, nondegenerate 24-row synthetic populations. Seeds 607/607/619 generated 192 considered candidates through the unchanged 64-candidate density-ratio rule. All five continuous dimensions remained in `[0,1]` and varied. Both padding categories and all five unit categories occurred. Seed 607 replayed exactly, while seed 619 differed. No objective or neural training was called in this KDE fixture.
- Decoding all retained candidates produced filters ranging from 32 to 125, all kernels 2 through 6, all depths 2 through 5, both paddings, and all listed dense widths. The finite fixture is not an exhaustive visitation claim for every filter value. Raw observations, fitted bandwidths, every considered point, and returned proposals are in `kde_evidence.json`; callback records are in `callback_evidence.json`.
- The earlier native-unit and corrected fixtures have different dimensions and synthetic populations. Their comparison establishes the coordinate/clamping defect and its removal; it is not an optimizer-performance comparison or a replay of a trained search.
- A copied integration harness passes BOHB import, success, and controlled-failure cases with 0/7/7 mocked evaluation calls. Only the copy was adapted for the run directory, shared decoder, and latent callback arguments; the saved original harness remains unchanged. An initial evidence-serialization attempt encountered a callback object; its log is retained, and the final evidence explicitly serializes evaluation events and event order.
- Ruff and formatting pass. No-sloppy reports zero errors and one advisory for the required PDF-recording callback. Basedpyright reports zero errors and 34 warnings from untyped dependency/dynamic values and unused results. Explicit typed ledger/evidence structures resolve the substantive local typing issues. The final typing-only test rerun reproduces all retained callback/KDE values exactly.
- Independent review verified all 192 retained candidates and 64 additional actual-KDE candidates at fixed seed 647. All 256 decoded configurations satisfy the discrete domains and convolution geometry; categorical encoding and removal of integer collapse were confirmed. The reviewer also checked callback budgets, terminal losses, ledger pairing, and the final production-source hash. Evidence is in `independent-review/kde-verification.json`; no remaining fix blocker was reported.

No dataset access, neural training, full adaptive search, test-set evaluation, package installation, dependency change, or paid compute occurred. Full adaptive replay under trained objectives and benchmark efficacy remain untested. Raw logs, initial passing-source snapshots, failure logs, source hashes, and the prior 128-proposal evidence are retained.
