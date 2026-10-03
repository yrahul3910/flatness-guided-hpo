# Matched runner integration review

Prospective engineering review, 2026-10-02. The earlier scientific decision draft remains unchanged at SHA256 `3f5e23558befa40d74eec8b264307c0b750a32fe936b597387b64998841de7af`. This review does not authorize a benchmark or revise that draft's scientific choices.

The approved synthetic backend fixture has 128 fit rows, 65 validation rows, and 65 probe rows, all synthetic 32x32x3 images with ten classes. Its architecture uses 32 initial filters, kernel 2, same padding, two blocks, and 32 hidden units. Batch size is 64. One probe epoch, two fresh-final epochs, and one probe replay total eight optimizer updates. The outer bound is 120 seconds on CPU with two intra-op and one inter-op thread. It reads no real dataset or official-test arrays. This reviewer will inspect the author's evidence and will not duplicate model execution without a specific unresolved concern.

Required implementation properties:

- Training APIs accept fit/validation arrays only. The orchestrator persists all probe attempts and costs before selection. An insufficient common pool leaves a failed block with complete evidence.
- Proposal, subset, shuffle, model initialization, augmentation, and random selection have separate recorded RNG roles. Candidate execution order does not alter its phase seed or outcome. Capture initialization hashes after the model is fully built and before the first update.
- Original RMS and actual compiled Keras validation CE come from the same post-probe checkpoint. Score batch order is the frozen subset order, including the partial batch; training shuffle is separate. Preserve invalid batch components and the original returned-score rule.
- Final fits start fresh, use the declared cosine horizon and installed validation-CE EarlyStopping callback, restore its chosen weights, and expose restored validation accuracy for candidate selection. Keep final optimizer state and restored model-weight semantics explicit.
- Retain histories, shuffle identities, actual updates, checkpoints, errors, incomplete attempts, and elapsed work. Reusing union fits saves physical execution but does not erase an arm's logical cost.

The short synthetic fixture can validate replay, checkpoint restoration plumbing, and partial-batch metrics. Two final epochs do not exercise the patience-25 stopping boundary; that limitation should remain explicit rather than expanding this integration check into another research experiment.

Source review before execution found two blocking paths. First, a checkpoint-save exception escaped before writing the terminal result; the author moved result writing and cleanup into a nested `finally` that preserves the original training exception. Second, augmentation-state logging used `layer.seed_generator`, but the installed random translation and flip layers expose `layer.generator`. That accessor is corrected. One-hot label validation now checks finite binary entries and row sums as well as shapes.

Installed Keras distinguishes layer weights from generator state: `Layer.variables` includes seed generators, while `save_own_variables` saves trainable/nontrainable variables. The backend therefore retains initial, per-epoch and terminal augmentation counters plus shuffle state explicitly in JSON. Model checkpoints contain restored weights and terminal optimizer state as Keras provides them; this review does not call the model archive alone a complete resumable RNG state.

## Backend evidence

The approved fixture completed under `backend/run-20261002T223958949122Z/`: three phases, four completed epoch rows, eight optimizer updates, three checkpoints, and no timeout. Frozen backend SHA256 is `1dd1a156c03f8b6d04bf1f5f37034eeea8245adb8e309b038bd3853376e2d236`; protocol SHA256 is `b5e63c5e730eadfb9a7137a3eb08345d0c2975d1c69c35c88d243be7a217ece7`.

`backend-verification.json` records the independent raw audit. Source copies and live hashes match, synthetic arrays reconstruct exactly from the recorded seed, all phase input hashes and saved shuffle arrays match, and checkpoint file hashes agree. Probe/replay weights, metrics, float32 batch-score arithmetic, augmentation counters and final shuffle state match exactly. The two final epoch rows reproduce callback best/wait decisions and restored CE/accuracy. The author performed the three model reload checks; this reviewer did not duplicate inference or training. The synthetic numerical losses are engineering checks, not dataset results.

The final fixture's best epoch is its last epoch, so this result does not behaviorally distinguish an earlier-best restoration from keeping the terminal weights. The source calls the correct restoration hook. Earlier-best restoration and patience termination require their separate evidence; neither is claimed from this short fixture.

## Orchestration review before its fixture

The first orchestration draft accepted edited proposal coordinates under generated-manifest metadata and did not enforce the dataset's class dimension. The parent corrected both: generated mode requires exact proposal replay, while explicit `fixed_fixture` mode permits bounded engineering configurations without claiming that they came from the benchmark prior. CIFAR-10/100 class dimensions are checked. Another corrected edge now marks the run failed if every arm lacks an incumbent. Ranking time is recorded separately from phase time.

Source review found no blocker for the separately approved orchestration fixture: synthetic CIFAR-100-shaped arrays, 128 fit rows, 65 validation rows, two fixed minimum configurations with distinct occurrence IDs/seeds, 65 probe rows, shortlist size one, and one final epoch. At most four phases/eight updates were allowed, with CPU two/one threads, a hard 120-second subprocess cap, and cooperative 90-second run limit. The parent assigned its execution and checkpoint reloads to the backend author.

## Completed orchestration audit

The frozen fixture at `orchestrator-fixture/run-20261002T224932530385Z/` passed. Its protocol SHA256 is `008eed76dd62c9f0483038efefa1bae620583629eb6123f88398d1087cfc12ee`, and orchestration source SHA256 is `34543ebf279db38e2a3aae55124084c7135e8b8283543971d1f9187ee59902ff`. It completed four phases, four epoch rows, eight updates and four saved checkpoints. RMS and CE selected occurrence `c000001`; random selected `c000000`. Thus two unique final fits served three arm references, with one shared reference. This checks selection/reuse plumbing, not screening efficacy.

`orchestrator-verification.json` records the independent audit. All 29 source hashes match live files and both saved source sets. Raw array/archive/manifest hashes match, and the synthetic data reconstruct exactly from seed 703. The fixed-fixture declaration, class count, shared subset, per-phase shuffles, candidate initialization seeds, augmentation child seeds, generator counters, float32 probe components, selected union, terminal statuses, and checkpoint file hashes agree. The four initialization hashes are distinct. Every arm points to its selected candidate's existing checkpoint and metrics. Physical phase time and each arm's logical phase time recompute exactly from phase-end records; ranking time remains separately recorded. Test-evaluation count is zero. The author reloaded the two selected checkpoints and checked 100 output columns and metrics; this reviewer did not repeat model work.

The expanded cohort preflight also passed independently: 42 proposed manifests contain 32,338 distinct parent/layer seeds for 32,338 required roles. `expanded-seed-verification.json` retains that result separately from the earlier parent-only preflight.

No remaining implementation blocker was identified within the reviewed engineering scope. The benchmark still requires the scientific decisions, dataset/proposal/source freeze, failure and runtime policies, and resource authorization described in the unchanged decision draft. These synthetic successes supply no new CIFAR accuracy, runtime advantage, or empirical-theory claim. No further training or broad checks were performed by this reviewer.
