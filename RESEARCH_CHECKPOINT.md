# FGHPO research checkpoint - paused 2026-10-03

The user paused research and compute before the prospective CIFAR scientific study launched. This document records existing evidence for review; it does not authorize resumption. The local exploratory studies and three Modal timing calibrations completed. The final image benchmark, manuscript revision, and reviewer response remain unfinished.

Research work was isolated on `research/reviewer-experiments-20261002`, preserving inherited changes and results. The publication checkpoint targets `research/fghpo-review-checkpoint-20261003`. The original checkout and main manuscript matched their recorded baselines at the last preservation check. Executed protocols, failed attempts, seeds, configurations, logs, checkpoints, and source hashes were retained. Nothing was submitted, merged, or posted to OpenReview. The links below target sanitized evidence copies under `research-checkpoint/`; raw data, model checkpoints, and operational logs remain preserved locally.

## Timeline and review provenance

- **October 2:** Audited code, local logs, prior-session evidence, and two stopped GCP machines with surviving disks. No verified newer experiment results were recovered from those disks. Direct OpenReview access encountered a verification challenge; a user-supplied five-page forum PDF subsequently supplied all 13 replies without bypassing it.
- **October 2:** Built the reviewer-to-evidence matrix, corrected image experiment bookkeeping and selection defects, and ran bounded local theory, trajectory, normalization, lifecycle, and baseline-adapter checks.
- **October 3, 00:30 UTC:** Ran the frozen four-arm KMNIST exploratory comparison below.
- **October 3:** Froze a prospective CIFAR-10/CIFAR-100 design and separate training/evaluation inputs. GCP recovery/start attempts failed for capacity; no new GCP machine or disk was created. Three subsequent Modal calibrations completed after authorization.
- **October 3, pause:** L4 timing supported the smaller four-block cost plan, but scientific execution preparation was incomplete. No four-block search or new CIFAR heldout inference launched.

The recovered review record concerns effectiveness across datasets, early-score versus eventual-performance evidence, parameterization dependence, theory/ranking gaps, and fair runtime/baseline comparisons. The exact reviewed manuscript revision is still unresolved: the review, local theorem, tabular callback, and image RMS implementation contain different expressions. See the [review matrix and chronological ledger](research-checkpoint/history.md), and [reviewer status](research-checkpoint/evidence/four-arm-pilot/reviewer-status.json).

## Engineering and theory findings

Image search previously returned official-test accuracy during candidate selection. The corrected [experiment recorder](image/src/experiment.py) separates fitting, validation selection, and one final saved-incumbent evaluation, records attempts/failures/seeds/source hashes, and refuses output overwrite. Other fixes cover SVHN labels/layout, float32 preprocessing, geometry checks, bounded FGHPO eligibility attempts, and full-budget BOHB incumbent eligibility. Historical test-exposed results remain exploratory.

BOHB's private sampler seed and native-unit/KDE coordinate defects were corrected in the explicitly named `bohb-unit-coordinate` adapter. TuRBO decoding and optimizer RNG preservation were checked with pinned upstream code and analytic objectives; its dependency remains scoped rather than installed. These are engineering checks, not completed trained baseline comparisons. The installed BOHB schedule still means 199 cold-fit objective calls and 2,222 requested epochs, not a 30-trial budget. See [BOHB evidence](research-checkpoint/evidence/bohb-coordinate-fix/report.md) and [TuRBO evidence](research-checkpoint/evidence/turbo-integration/report.md).

The [matched runner](image/matched_hpo.py), [training backend](image/src/matched_training.py), and [local suite finalizer](image/finalize_matched.py) received focused fixture and synthetic-model verification. They preserve candidate/phase seeds, fresh final initialization, CE-based checkpoint restoration, and a complete-suite lock before evaluation. These checks do not validate the unfinished cloud executor. Targeted checks passed with documented warnings; no package-wide clean type-check claim is made.

The local paper's original Hessian lower-bound theorem is false under its written assumptions. A stationary global-minimum witness has claimed bound **2.5151458962** but exact mean-CE head Hessian Frobenius norm **0.48**. The decisive derivative error treats hidden features as depending on the independent final kernel. The separate exact identity

`T = mean_i [(1 - ||p_i||_2^2) ||h_i||_2^2]`

is valid for the output-kernel Hessian trace with features and bias fixed. It supports a restricted local head-sensitivity statement, not full-network curvature, generalization, or an ordering of eventual classification accuracy. No theorem supplies the required model-ranking guarantee. See the [assumptions review](research-checkpoint/evidence/theory-assumptions-review/review.md) and [repair analysis](research-checkpoint/evidence/theory-repair-followup/theory-repair-review.txt).

Function-preserving transformations exposed score dependence on parameterization. However, removing ordinary class-common output weights from saved lifecycle models preserved every candidate rank in both blocks; the artificial extreme-shift counterexample is not evidence that normal training causes a practical ranking failure. Proposed normalized quantities also have explicit limitations. Separate [curvature](paper/appendix/curvature_audit_draft.tex) and [operational](paper/appendix/operational_protocol_audit_draft.tex) appendix drafts remain unincluded in the main manuscript.

## Completed local empirical evidence

MNIST five-epoch, 100-epoch, and probe/fresh-fit/restoration panels implemented the requested across-configuration comparisons. Conclusions changed with the endpoint: favorable loss correlations did not carry over reliably to classification or restored checkpoints. On the reused 375-row tuning split, the cold-retrain RMS versus early-CE selections made **17 versus 18 errors** in block 307 and **20 versus 14** in block 331. These are development results, not independent generalization estimates. The named Vehicle/blood-transfusion/Segment mechanism panel retained all 12 runs/360 epochs; feature collapse accompanied worse classification on two tasks but majority-class accuracy on blood transfusion. See [classification endpoint reanalysis](research-checkpoint/evidence/endpoint-metric-followup/report.md) and [practical gauge analysis](research-checkpoint/evidence/practical-gauge-followup/report.txt).

The KMNIST pilot froze 1,500 fitting, 375 tuning, and 1,000 official-test rows before training/evaluation, with eight fixed LR/weight-decay candidates per block and two finalists per arm. Final fits started fresh, used a maximum of 100 epochs and patience-25 validation-CE restoration, then selected by tuning accuracy with predeclared tie breaks. Every decision was locked before model evaluation on the test subset.

| Arm | Block 401: errors / 1,000 | Block 419: errors / 1,000 |
| --- | ---: | ---: |
| Original RMS | 215 (21.5%) | 219 (21.9%) |
| Exact head trace | 219 (21.9%) | 219 (21.9%) |
| Early tuning CE | 212 (21.2%) | 202 (20.2%) |
| Seeded random shortlist | 221 (22.1%) | 202 (20.2%) |

Exact trace showed no advantage over RMS here. Early CE had fewer errors in both blocks, but two initializations on one small shared split do not establish superiority. Block 419 contains shared selected checkpoints, not independent confirmations. The evaluation rows are now spent for adaptive follow-up.

The local CPU run took **118.337 seconds**, with 16 probes, 10 distinct final fits, 566 epochs, and 13,584 updates; no paid compute was used. Independent saved-record verification reproduced all eight error counts from six distinct prediction files, along with seeds, hashes, checkpoint restoration, decisions, and costs. Full details: [pilot report](research-checkpoint/evidence/four-arm-pilot/report.txt) and [verification](research-checkpoint/evidence/four-arm-pilot/independent-review/verification.json).

## Calibration, spending, and preserved resources

Calibration used training/tuning subsets only and measured timing/memory, not scientific efficacy. Each case made 18 updates. All three returned verified artifacts and were subsequently confirmed stopped with zero running tasks. The first controller's cleanup-status parser error was preserved and corrected separately; that calibration was not rerun to conceal the error.

| Completed calibration | Cases / updates | Frozen forecast outcome |
| --- | ---: | --- |
| T4 representative cases | 9 / 162 | MAC extrapolation failed: about $498 for two repetitions/task and $1,030 for four, including that rule's allowances. |
| T4 exact smaller cohort | 32 / 576 | Direct timings failed the $42 science gate: $52.20 worker; $60.20 with holds/other reserves. |
| L4 exact same cohort, TF32 explicitly disabled | 32 / 576 | Direct timings passed: **$38.31 worker; $46.31 cumulative forecast** including $6 holds and $2 other reserves. |

These are full-study **forecasts, not actual spending**. The direct rule requires all 32 cases, each candidate's own timings, the full possible 100-epoch workload, all fit/tuning inference, overhead allowances, and a 2x backend margin. It assumes no early-stopping or shared-finalist savings. The prior failed gates remain preserved. See [nine-case report](research-checkpoint/evidence/prospective-image-study/calibration/completed-calibration-report.json), [T4 forecast](research-checkpoint/evidence/prospective-image-study/calibration-exact32/direct-cost-forecast-20261003.json), and [L4 forecast](research-checkpoint/evidence/prospective-image-study/calibration-exact32-l4/direct-cost-forecast-20261003.json).

The latest retained account snapshot reports **$0.18159692 metered and $0 billed**, after adjustments/credits. This is account-level, potentially lagging data, not a settled per-run charge or proof that incremental spending was zero. All **$6 of reservations** remain held inside the cumulative **$50** authorization. Provider alerts are not a hard task spending cap. A read-only check at 03:31 UTC confirmed all three Modal apps stopped with zero tasks, both original GCP VMs terminated, and both original 100 GiB disks ready and attached. Their contents remain uninspected; no disk was deleted and no new persistent storage remains. Existing preserved disks may continue their pre-existing storage charges. [Sanitized resource and financial checkpoint](research-checkpoint/pause-status.json).

## Unlaunched study and remaining blockers

The [prospective design](research-checkpoint/evidence/prospective-image-study/protocol/design_protocol.json) specifies CIFAR-10/CIFAR-100, repetitions 2000/2001, eight candidate attempts per block, and four arms with two finalists each. It uses official training rows `[0,40000)` for fitting and `[40000,50000)` for tuning. Separate evaluation archives remain local. Historical official-test exposure prevents a globally untouched-test claim; this would be a prospectively sealed rerun. Mechanical evaluation-array preparation is disclosed in the [data provenance report](research-checkpoint/evidence/prospective-image-study/data-provenance/report.txt).

The paused [block budget draft](research-checkpoint/evidence/prospective-image-study/scientific-execution/block-budget-plan.json) assigns 135,012 aggregate exposure seconds at the fixed L4 quote, or $38.30560464 after rounding. Its [concurrency amendment](research-checkpoint/evidence/prospective-image-study/scientific-execution/concurrency4-and-transfer-amendment.json) proposes four independent one-GPU apps, unchanged individual deadlines, stop-all on any cohort failure, and checkpoint-derived transfer caps totaling 9.625 GiB after a 2x transport allowance ($0.385). The sequential draft is archived. These operational estimates are not guarantees or evidence of execution.

At pause, the scientific `modal_launch.py` and `prepare_payload.py` refer to **missing `run_block.py`**. The new streaming executor has not been tested end to end. Assignment/payload completion, one-use launch controls, durable artifact/ACK handling, concurrent aggregate accounting, deadline/cleanup enforcement, and independent executor review remain blockers. Prepared manifests and fixture checks do not make this package launch-ready. All four fixed blocks must finish and seal before any local evaluation; no partial cohort may be reported as the completed study.

The earlier **20-repetition-per-task, 100-candidate/10-finalist** publication experiment is deferred; it was neither executed nor costed by the smaller study's gate. Broader task/architecture coverage, matched adaptive-baseline domains and effective budgets, complete trajectory evidence, manuscript-version reconciliation, and the final empirical claim remain open. Resume only on explicit user direction; preserve the frozen evidence and complete the missing engineering/review before any scientific launch.
