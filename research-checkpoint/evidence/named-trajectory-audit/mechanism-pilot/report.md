# Prespecified learning-rate mechanism diagnostic

All 12 prescribed runs completed 30 epochs: three datasets, seeds 211 and 223, and paired learning rates 1.0 and 0.001. The recorded CPU loop took 18.94 seconds, within the approximate 300-second bound. There were no failed or timed-out runs, invalid batch ratios, or nonfinite current-model tensors/losses. All 24 epoch-1/epoch-30 checkpoints loaded successfully; file hashes, weight hashes, and optimizer step counts matched their records. Recomputed frozen measurements matched all retained checkpoint measurements exactly, with maximum absolute difference 0.0.

This is a new mechanism diagnostic with the historical whole-training-fold preprocessing and scoring condition. Internal validation rows contribute to preprocessing and proxy measurement but receive no gradient updates. No official test data was opened. It is not an exact replay of the 2024 plots, unbiased validation, an HPO comparison, or confirmatory generalization evidence. The reviewed protocol and executed source remained unchanged throughout training and verification.

## Outcome

The experiment does not support denominator collapse as the explanation in these new runs. Between epochs 1 and 30, the output-kernel Frobenius norm grew in every run. At learning rate 1.0, the ratio increased on vehicle, decreased on btsc, and decreased or became zero on segment. At learning rate 0.001, it grew strongly in every run. Thus a growing ratio also occurs during the lower-rate training condition; growth alone is not a diagnosis of failure or curvature growth.

Learning rate 1.0 produced 99.879% to 100% zero penultimate activations at epoch 30 across the observed whole training fold. In segment seed 223, every measured penultimate activation was zero from epoch 3 through epoch 30, yielding an exactly zero numerator and ratio despite finite nonzero output weights and finite loss. This describes the historical plotted score on these inputs. It does not imply that the current STCVX search would select that configuration: its eligibility filter excludes zero/nonpositive scores.

The validation result is mixed. Vehicle and segment have lower endpoint validation CE at learning rate 0.001 in both seeds. Btsc is the exception: learning rate 1.0 has lower endpoint validation CE in both seeds, despite nearly all-zero hidden activations. Its validation accuracy is 114/135 = 0.84444 in both seeds, equal to the observed internal-validation majority-class frequency. That equality is a descriptive class-imbalance check, not an additional trained baseline or evidence that degeneracy uniformly worsens validation loss.

## Every prescribed run

`q` is the original float32 maximum batch ratio `||h_batch||_F / ||W_out||_F`. The component factors compare epoch 30 with epoch 1; `h` is the largest ordered-batch activation norm and `W` is the output-kernel norm. The final column is stable current-model cross-entropy on the fixed internal validation rows. Full precision, absolute component values, entropy, accuracy and recorded head trace remain in `component-summary.csv` and `.json`.

| Dataset | Seed | LR | q at epoch 1 | q at epoch 30 | h30/h1 | W30/W1 | Hidden zeros at epoch 30 | Validation CE at epoch 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vehicle | 211 | 1 | 46.403 | 56.621 | 2.645 | 2.168 | 99.959% | 1.378 |
| vehicle | 211 | 0.001 | 0.444 | 19.303 | 59.287 | 1.363 | 30.071% | 1.288 |
| vehicle | 223 | 1 | 55.044 | 77.698 | 3.263 | 2.312 | 99.879% | 23.081 |
| vehicle | 223 | 0.001 | 0.478 | 20.065 | 57.552 | 1.370 | 20.947% | 1.353 |
| btsc | 211 | 1 | 21.092 | 6.831 | 0.813 | 2.510 | 99.986% | 0.449 |
| btsc | 211 | 0.001 | 2.649 | 16.639 | 8.679 | 1.382 | 28.610% | 0.540 |
| btsc | 223 | 1 | 29.050 | 3.268 | 0.300 | 2.670 | 99.995% | 0.502 |
| btsc | 223 | 0.001 | 2.544 | 20.947 | 11.411 | 1.386 | 29.625% | 0.553 |
| segment | 211 | 1 | 113.909 | 98.735 | 1.260 | 1.453 | 99.975% | 1.963 |
| segment | 211 | 0.001 | 0.362 | 22.230 | 86.384 | 1.406 | 32.220% | 0.996 |
| segment | 223 | 1 | 70.292 | 0.000 | 0.000 | 1.382 | 100.000% | 1.985 |
| segment | 223 | 0.001 | 0.349 | 20.710 | 85.381 | 1.440 | 16.121% | 1.004 |

For vehicle at LR 1.0, activation-norm growth exceeded kernel-norm growth, explaining the net ratio increase. For btsc, the maximum activation norm shrank while the kernel norm grew, explaining the ratio decrease. For segment seed 211, both grew but the denominator grew faster; seed 223's numerator became zero. At LR 0.001, activation norms grew by factors 8.68 to 86.38 while kernel norms grew by factors 1.36 to 1.44. These component statements follow the fixed ratio's arithmetic; they do not identify the corresponding full-network Hessian or establish what happened in the missing 2024 component histories.

![All 12 complete trajectories](all-trajectories.png)

The [one-page PDF](all-trajectories.pdf) and PNG show every retained epoch for every prescribed run. Ratio and zero-fraction axes are linear. Validation CE is strictly positive for all retained records and is shown on an explicitly labelled logarithmic axis. No interval, seed or dataset was removed. Both formats were rendered and visually checked.

## Integrity and interpretation limits

- `launch-freeze.json`, `protocol.json`, `executed-command.txt`, `execution.log` and `execution/sources/` retain the reviewed design and execution provenance. Source SHA256 is `ebe1ca0b206d14e2fc7d6d3c587f3f910275a4d3daecfdc84800a18afb71945a`; protocol SHA256 is `170b9b0d831d480edc5211f06b60be1c491b5eb15dedc500da296b5dddd15007`.
- `execution/events.jsonl` retains all 360 epoch rows, every ordered batch's norms/ratio/validity, current-model losses, output diagnostics, paired initialization/shuffle hashes and checkpoint hashes. All 12 lifecycle outcomes are in `execution/summary.json`. The independent raw audit verified candidate accounting, source snapshots, inputs, splits, optimizer steps, paired hashes, original batch-ratio arithmetic and weighted partition means.
- Epoch events record the actual optimizer learning-rate scalar in the `learning_rate` field, replacing the nominal value in that event. In particular, configured `0.001` is recorded as `0.0010000000474974513`. Analysis maps each epoch to its unique frozen nominal rate by exact `float(np.float32(rate))` equality, preserves both values in the summary, and does not edit the source or original records.
- `checkpoint-verification.json` records all 24 reload checks. Verification rebuilt preprocessing independently, matched feature and label hashes, and called only the frozen read-only measurement functions. It did not call the runner's data-preparation, event-writing or checkpoint-writing functions. `analysis-manifest.json` hashes the analysis source, raw event file, tables and plots.
- The additional trace is the mean-cross-entropy final-kernel block trace, excluding bias, using `p*(1-p)` for binary sigmoid and `1-sum(p**2)` for multiclass softmax. It is not the full-Hessian trace or a parameterization-invariant quantity. Recorded zero or tiny traces can also reflect finite-precision cancellation/saturation in those formulas; they are not proof of a flat minimum. The observed all-zero segment activations and numerator do not depend on that curvature arithmetic.
- The historical one-hot/row-L2 preprocessing was intentionally fitted on the entire recovered training fold, including internal validation. The new runner uses float32 preprocessing, paired fresh seeds/shuffles, current TensorFlow/Keras, separate output Activation layers exposing logits, and a fixed 30-epoch endpoint. The original figure had an unknown interactive random state and training-loss early stopping. Its precise numerator/denominator trajectories therefore remain unrecovered.
- Runner Ruff and direct no-sloppy checks pass; compilation and the non-running entry point passed. Runner typecheck retains one sklearn dtype-annotation error and 47 warnings; an actual encoder-only check confirmed that the installed library honors float32. The separate executed analysis passes Ruff and has zero no-sloppy errors with two density warnings; its typecheck retains 14 errors and 54 warnings concerning incomplete JSON/Mapping annotations, sklearn's inferred dtype, and the Keras loader's broad return type. These scratch-script static checks are not reported as fully clean. The actual analysis completed and all checkpoint measurements replayed exactly.

No additional training, threshold changes, seed replacements, selected-subset reporting, paid compute, or official-test evaluation was performed. The useful next scientific decision is how to frame the ratio's empirical role alongside its parameterization dependence; these new mechanism trajectories do not repair the invalid theoretical bound or close the across-configuration final-endpoint comparison requirement.
