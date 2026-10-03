# Classification endpoints in the saved pilots

The saved 100-epoch results do not support carrying the favorable loss correlation over to classification. Classification accuracies were recorded at every epoch, so this follow-up recovers exact mistake counts without training or reloading models. These are descriptive comparisons on reused development validation data. The five-epoch and 100-epoch protocols originally targeted loss; their classification comparisons below are post-hoc. The later cold-retrain protocol already declared restored-checkpoint classification error as primary.

| Pilot | Probe and subsequent training | Stopping and checkpoint policy | Classification endpoint |
| --- | --- | --- | --- |
| Five epochs, seeds 31/47 | First epoch of each candidate's single continuous five-epoch fit; cosine horizon 120 updates | No early stopping; only epoch-5 checkpoint saved | Actual epoch-5 validation predictions; not a converged endpoint |
| 100 epochs, seeds 71/89 | First epoch of each candidate's single continuous 100-epoch fit; cosine horizon 2,400 updates | No early stopping and no restoration; epoch-1 and epoch-100 checkpoints saved | Actual epoch-100 predictions; separately, predictions recorded at the minimum-validation-CE epoch across 100 |
| Cold retrain, blocks 307/331 | Separate one-epoch constant-LR probe; fresh model and optimizer for final fit, with independent final initialization seed; no probe weights reused | At most 100 final epochs, cosine horizon 2,400 updates; actual Keras EarlyStopping monitors validation CE, patience 25, restores minimum-CE weights | Actual predictions after restoration; terminal pre-restoration epoch is a separate diagnostic |

All panels use eight fixed LR/weight-decay configurations per seed: LR in {0.0001, 0.0003, 0.001, 0.003}, weight decay in {0.00001, 0.001}, LR outermost, IDs 0 through 7. Architecture is fixed: 32 filters, kernel 2, same padding, two convolutional blocks, 128 hidden units. Each panel pairs initial weights and shuffle sequences across configurations within a seed block. The cold-retrain panel runs all eight final fits so every selection outcome can be examined; it is not a timed execution of the production eligible-100-to-10 search.

| Endpoint | Seed/block | Epoch-1 RMS vs endpoint error rho | Epoch-1 validation CE vs endpoint error rho | RMS-selected errors | CE-selected errors | Best available errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Epoch 5, last | 31 | -0.9577 | +0.9577 | 95/375 (ID 1) | 18/375 (ID 7) | 18/375 |
| Epoch 5, last | 47 | -0.0894 | +0.6002 | 26/375 (ID 6) | 26/375 (ID 4) | 26/375 |
| Epoch 100, last | 71 | -0.1565 | +0.1304 | 19/375 (ID 1) | 19/375 (ID 4) | 18/375 |
| Epoch 100, last | 89 | -0.9207 | +0.9207 | 21/375 (ID 1) | 17/375 (ID 4) | 16/375 |
| Minimum-CE epoch across 100 | 71 | +0.2224 | +0.3459 | 19/375 (ID 1) | 17/375 (ID 4) | 17/375 |
| Minimum-CE epoch across 100 | 89 | -0.7488 | +0.7979 | 22/375 (ID 1) | 18/375 (ID 4) | 17/375 |
| Cold retrain, restored minimum CE | 307 | -0.2169 | +0.1084 | 17/375 (ID 1) | 18/375 (ID 7) | 11/375 |
| Cold retrain, restored minimum CE | 331 | -0.3593 | +0.3713 | 20/375 (ID 1) | 14/375 (ID 4) | 14/375 |
| Cold retrain, terminal before restoration | 307 | -0.7783 | +0.8401 | 17/375 (ID 1) | 14/375 (ID 7) | 14/375 |
| Cold retrain, terminal before restoration | 331 | -0.5302 | +0.6266 | 19/375 (ID 1) | 14/375 (ID 4) | 14/375 |

Each rho uses all eight candidates in one block, with average ranks for tied error counts. Lower scores and errors are preferred, so positive rho supports the intended ranking direction. Top-1 ties use the lowest configuration ID; no direction reversal or candidate exclusions were introduced. The JSON retains exact values, endpoint epochs, all candidate outcomes, tie counts, and secondary early-error/C1/C2 comparisons. These small, dependent comparisons provide no significance or broad efficacy claim.

The 100-epoch fit did not restore its best-loss checkpoint. The separate minimum-CE-across-100 endpoint uses the classification accuracy already recorded at that selected epoch. Most of those checkpoints were not saved; the measurements exist, but replaying those particular models is unavailable. Calling this a restored-model result would be inaccurate. The later cold-retrain pilot did restore and save the selected weights, and every restored accuracy and CE exactly matches its recorded minimum-monitored-CE epoch. Prior checkpoint verification covers all 32 probe/final checkpoints in that pilot.

Choosing the minimum-CE checkpoint can change accuracy in either direction. In long seed 71, the CE-selected candidate has 19 errors at epoch 100 and 17 at its minimum-CE epoch; in seed 89 it has 17 and 18 respectively. In cold-retrain block 307, the early-CE-selected candidate has 14 errors at its terminal epoch but 18 after minimum-CE restoration. This is a target-policy distinction, not a reason to select the better-looking policy after seeing results. A common predeclared CE-based stopping/checkpoint policy remains compatible with classification as the final task metric; it does not optimize accuracy directly.

All three pilots train on official MNIST training rows [0,1500) and validate on [1500,1875), after the same fixed float32 preprocessing. The sets are disjoint, but the same 375 validation examples are reused for early-validation screening, across many epochs/configurations, and across subsequent research decisions. In the restored panels, validation also selects the checkpoint. Therefore the reported outcomes are tuning/development outcomes, not an unbiased generalization estimate. Fresh seeds do not create a fresh data holdout. No official test arrays were read in these pilots or this follow-up.

A valid next comparison should freeze (1) fitting/probe data, (2) tuning/stopping validation data, and (3) an independent final evaluation split, together with the task metric, direction, tie policy, candidate population, stopping/checkpoint policy, and budget. Fit/probe data produce the score; tuning validation supplies the early-CE baseline, stopping signal, and finalist choice. Once all methods and candidates are frozen, evaluate only the selected models on the independent split, report classification metrics there, and do not feed those outcomes back into method or checkpoint selection. If development continues after inspecting that split, it becomes development data and further confirmation needs another untouched evaluation set. This is a proposed design; no independent evaluation has been run here.

## Secondary named-data check

The previously saved tabular mechanism pilot contains validation classification accuracies. The table uses epoch 30 for both LRs, with the same seed and recovered training-fold inner split. It is a mechanism diagnostic, not historical-run reproduction, held-out-test evidence, or a tuned HPO comparison.

| Dataset (OpenML task) | Seed | LR 1 errors | LR 0.001 errors |
| --- | ---: | ---: | ---: |
| Vehicle (53) | 211 | 124/153 | 53/153 |
| Vehicle (53) | 223 | 111/153 | 56/153 |
| Blood transfusion (10101) | 211 | 21/135 | 28/135 |
| Blood transfusion (10101) | 223 | 21/135 | 33/135 |
| Segment (146822) | 211 | 350/416 | 127/416 |
| Segment (146822) | 223 | 363/416 | 135/416 |

High-LR hidden-feature collapse accompanies much worse classification on Vehicle and Segment, but better raw accuracy on this blood-transfusion validation split. The latter matches the majority-class count (114/135 correct), including seed 223 where all validation hidden activations are zero. Feature collapse therefore does not imply worse raw accuracy on every dataset; label imbalance and the requested task metric matter. No balanced-accuracy or per-class inference is supplied without saved predictions.

## Sources and checks

`analyze.py` reads only `events.jsonl` from `trajectory-pilot`, `long-endpoint-pilot/run-prepared-20261002-v1`, and `lifecycle-pilot/run-prepared-20261002-v1`. Their SHA256s and complete populations (80, 1,600, and 832 epoch rows) are in `metric-summary.json`. The original source and protocol files in each directory establish preprocessing, optimizer schedule, prediction accuracy calculation, checkpoint policy, and phase seeds. Exact derived records are in `candidate-endpoints.json`; the named-data extraction and its input hash are in `named-classification.json`.

Reanalysis completed with 50 comparisons and 112 candidate-endpoint records, zero training updates, zero model reloads, and no reads of dataset/test arrays. Ruff formatting and Ruff checks pass for the final analyzer. Direct no-sloppy invocation skipped the ignored audit path; no successful no-sloppy result is claimed from that invocation. Original event files and main manuscript were not edited.
