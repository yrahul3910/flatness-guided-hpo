# Reviewer-named trajectory evidence audit

Audited 2026-10-02 without training, downloads, cloud access, or edits to paper/production code. The original worktree was read-only. The notebook and vector figure survive locally; raw per-epoch proxy components and checkpoints do not appear in the inspected repository paths or notebook outputs. Approximate plotted values are recoverable, but the cause of the growth is not established.

## Sources and provenance

- `hpobench/Flatness Over Time.ipynb` contains the generating recipe, a printed configuration, execution counts, warning/runtime output, and an embedded PNG. It does not serialize `history`, `mus`, model weights, activations, probability vectors, or gradients. The source extraction and original embedded PNG are retained in this audit directory.
- `paper/appendix/oneepoch.tex:3` includes `paper/img/hpobench.svg`. That file is byte-identical to `sharpness_over_epochs/hpobench.svg`: SHA256 `1f4fc37b2e55bdb99053cff17ffaf61fd43b6be83679b3e657cb0904e212f3fb`. Both are also identical to the read-only original repository copies. The SVG metadata gives `2024-08-05T05:59:43.541644` and Matplotlib 3.8.4. The notebook output identifies a Tesla T4, Python 3.10.14, and a GCloud Workbench TensorFlow 2-11 kernel.
- Commit `2c3cfdd7e2ac3070079e89e78922830e2238c500`, dated 2024-11-30, first adds the notebook and `sharpness_over_epochs` figures in this local Git history. Its notebook source is retained separately. The later lint commit `309da1b` removed printed timing/configuration statements while preserving their saved output. Thus current notebook source and output are not a clean recorded execution. The initial-commit source explains those prints, but the full interactive kernel history, imported file versions at execution, initialization state, and exact submitted paper revision remain unverified.
- `source-manifest.json` records hashes and sizes. The current notebook SHA256 is `3bbd432ffb7f0d77178928da22f089e7de4265c5bf082052b838306001209624`.

## Recorded recipe

The following is evidence about the saved recipe and output, not a claim that an independent rerun would reproduce the original model states.

| Item | Local evidence |
|---|---|
| Datasets | Notebook cell 13 (`:819`) maps vehicle to OpenML task 53, btsc to 10101, segment to 146822. The full plot loops over eight tasks. |
| Configuration | Cell 10, execution 93, prints `Config(depth=3, width=64, batch_size=128, alpha=1e-06, learning_rate_init=1.0)`. The same `config` is passed through the figure loop. |
| Initialization | `set_random_seed(13)` precedes configuration sampling, but a standalone vehicle run has execution 96 and the plot has execution 116. There is no per-dataset reset, saved random state, or checkpoint. These are not independently seeded repeats. |
| Data | Cell 6 uses OpenML repeat 0, fold 0; fits `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` then row `Normalizer` on the whole OpenML training fold; transforms the held-out fold; label-binarizes object labels. |
| Internal validation | `validation_split=0.2` in `fit`. The callback evaluates the proxy over all `data.x_train`, including the internal validation slice. Preprocessing also sees that slice. The external OpenML test fold is loaded but is not evaluated by this plotting function. |
| Model | Imported `hpobench/util/model.py` at the initial-figure commit and current tree uses three width-64 ReLU Dense layers, then sigmoid for binary labels or softmax for multiclass labels; Adam with constant configured learning rate. It never uses `alpha`, so `alpha=1e-6` is not evidence that weight decay or L2 regularization was applied. |
| Endpoint | Cell 9 (`:557`) specifies up to 200 epochs and `EarlyStopping(monitor="loss", patience=5, min_delta=1e-3)`, with no explicit best-weight restoration. The figure's retained endpoints are much shorter. |
| Figure | Cell 13 plots red `history["val_loss"]` and green callback `mus` on the same linear y-axis; x is `range(1, len(val_loss)+1)`. There is no y-axis label or unit definition. |

The callback's original score is

`q(t) = max_b ||H_b(t)||_F / ||W_out(t)||_F`,

where `H_b` is the penultimate ReLU activation matrix for an ordered batch of at most 128 training-fold examples and `W_out` is the final Dense kernel, excluding its bias. `np.linalg.norm` has no `ord` or `axis` argument, so these are unnormalized Frobenius norms. There is no probability-norm factor, batch averaging, or division by width. For fixed activation distribution, its numerator carries a square-root batch-size dependence. The plotted ratio has activation/weight coordinate units; no calibrated Hessian or strong-convexity units are supplied. The red curve is the model's cross-entropy validation loss and has a different meaning.

Cell 7 (`:509`) excludes positive-infinite batch ratios with `mu != np.inf`; NaNs cannot win the comparison. It retains only the maximum eligible batch ratio, and retains `-inf` if none qualify. Invalid batch counts are not logged. A finite drawn score therefore cannot certify that every batch score was finite.

The initial-figure commit's `hpobench/util/stcvx.py` also uses `||H_b||_F / ||W_out||_F`. Commit `682d6905af942d0e4dfb1bccf2e003cc228dd946` on 2024-12-26 adds the output-probability norm to the production helper. The current helper's `||H_b||_F ||P_b||_F / ||W_out||_F` is not the saved plotting formula. The image implementation's RMS form is another distinct formula and must not be substituted when explaining this figure.

## Recovered figure values

`recover_svg.py` calibrates coordinates against the SVG ticks and exports every retained red/green vertex for the three named datasets to `approximate-svg-vertices.csv` and `.json`. It checks the known source hash, recognizes linear `M`/`L` paths and their colors, and verifies integer epoch coordinates. All six named paths contain contiguous vertices from epoch 1 through the listed endpoint. No interior epoch gap is visible in these paths. Matplotlib can simplify paths generally; this observation does not restore omitted nonfinite batch values, unplotted training records, or precision lost during SVG serialization. These are approximate retained plot vertices, not raw experimental measurements.

| Dataset | Retained endpoint | Proxy at epoch 1 | Largest proxy (epoch) | Proxy at endpoint | Minimum validation loss (epoch) | Validation loss at endpoint |
|---|---:|---:|---:|---:|---:|---:|
| vehicle | 22 | 62.698 | 134.355 (11) | 131.391 | 1.388 (10) | 1.421 |
| btsc | 15 | 25.024 | 326.571 (15) | 326.571 | 0.433 (2) | 4.974 |
| segment | 12 | 67.000 | 125.272 (12) | 125.272 | 1.961 (6) | 1.974 |

The first-epoch validation losses are approximately 42.711, 33.662, and 78.806, respectively. The minimum proxy values are 44.464, 4.342, and 21.763, all at epoch 3. The stored embedded PNG was inspected and agrees visually with the recovered shapes. In particular, btsc's later validation deterioration is present in the figure; it should remain in any discussion of the diagnostic.

## What can explain the growth?

- Learning rate 1.0 is a documented configuration and a plausible stressor. There is no controlled learning-rate comparison, saved optimizer state, or component trajectory here that establishes it as the cause.
- Denominator collapse cannot be diagnosed: no per-epoch `||W_out||_F` values survive. Activation growth also cannot be separated from denominator changes because no `||H_b||_F` values survive. Their ratio alone cannot identify either mechanism.
- Confidence is not an explicit multiplicative component of this plotted formula. Confidence could affect training dynamics, but probability vectors and entropy were not recorded. The probability multiplier added later cannot explain an earlier curve as an observed component.
- The plot establishes neither a divergent limit nor curvature growth. All recovered named paths are finite, short, and stopped before epoch 23. Vehicle and segment flatten visibly; btsc is still increasing over its retained endpoint. A changing coordinate-dependent ratio is not evidence that the full Hessian grew.
- These are one-configuration trajectories, not the reviewer's requested across-configuration correlations against a common final endpoint. Neither their shapes nor a within-run correlation closes that reviewer request.

## Remaining recovery and reproduction path

No relevant `.npz`, `.npy`, `.pkl`, `.pickle`, `.parquet`, `.keras`, or `.h5` data/checkpoint file was found in the inspected original/isolated repository paths outside dependency files and new audit runs. The default macOS OpenML cache `~/.openml` is absent; the checked alternative `~/.cache/openml`, `~/openml`, `~/scikit_learn_data`, `~/Library/Caches/openml`, `~/Library/Application Support/openml`, and repository `hpobench/openml_cache` locations are absent. `OPENML_CACHE_DIR` and `XDG_CACHE_HOME` are unset. This is a bounded local search, not proof that no copy exists on other disks or accounts.

Existing `hpobench/runtimes/{btsc,vehicle}/stcvx.log` files contain ten single screening `mu` entries each and many later training-loss lines, but no `val_loss` entries or per-epoch component values. They describe a different HPO/runtime experiment, not the plotted callback history. There is no corresponding segment runtime folder. The six `paper/landscapes/{vehicle,btsc,segment}-{low,high}.svg` files and `paper/appendix/landscapes.tex` concern separate low/high landscape examples; their configurations must not be assigned to this trajectory figure.

Older raw HPO result archives were removed from the worktree in January 2024 by commit `71d8272` with a README pointer to `s3://stcvx-results/`. Git tree `71d8272^` still names `hpobench/results/vehicle/stcvx.tar.gz` (12,326,072 bytes), `blood-transfusion-service-center/stcvx.tar.gz` (14,763,090 bytes), and `segment/stcvx.tar.gz` (25,408,906 bytes). These predate the August figure, so they are not evidence of its original trajectories. They were not downloaded or expanded in this audit.

A minimal later diagnostic would first recover the OpenML task data and exact fold indices, then preserve this original configuration and formula in a separate run with explicit seeds. Log every batch's numerator, denominator, ratio and validity, full train/validation losses, current learning rate, output confidence, and checkpoints at each epoch. Retain the historical preprocessing as a labelled reproduction condition; keep an internal-training-only preprocessing/proxy condition separate. Any lower-learning-rate control or corrected metric must be fixed before examining results and retained regardless of outcome. Without original random state and environment, call this a new mechanism diagnostic, not an exact reproduction. No such run was launched here.

Verification: the extraction ran successfully and the six series have 98 retained vertices total. Ruff format/check passed. Basedpyright reports zero errors and one partially unknown JSON-report warning. After confirming that the `no-sloppy` CLI skips the ignored `sweep-results` path, both checker layers were invoked directly on the actual source using `run_ruff([source], repository_pyproject)` and `run_rules([source])`, with the repository's 120-column limit. The executed source and output hashes were preserved before fixing two prose-wrapping findings. The completed direct check reports one checked path, zero errors, zero warnings, and zero heuristics. Only the SVG extraction was rerun; CSV and JSON outputs remain byte-identical. `no-sloppy-before.log`, `no-sloppy-after.log`, `recover_svg.before-style-check.txt`, `style-only.patch`, and `style-verification.json` retain the evidence. Original sources, negative figure evidence, and formula variants were preserved.
