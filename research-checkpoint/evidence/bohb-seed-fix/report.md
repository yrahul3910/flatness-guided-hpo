# Image BOHB sampler seed correction

`image/bohb_hpo.py` now constructs `ConfigurationSpace(space, seed=run.seed)`. This is the only production-code change in this audit. The installed `bohb-hpo` 0.5.2 implementation creates a private `np.random.default_rng(seed)`, with `seed=None` by default, and uses it for configuration sampling. The existing `SearchRun` global seeding therefore did not initialize this private generator. Copies of the pre-change entrypoint and installed dependency sources are retained here with hashes.

The new `tests/test_image_bohb.py` regression executes the image module entrypoint using the real BOHB constructor, `get_sample`, and configuration space. It replaces `SearchRun` and `optimize`, so its optimization callback only draws 16 initial proposals. It checks that the constructor receives the run seed, that seeds 31 and 31 yield exactly equal proposal sequences, and that seed 47 yields a different sequence. It also checks that no training evaluation occurs and that the entrypoint calls `finish` once. The unchanged final test failed before the production fix because the seed argument was absent (`None != 31`), then passed after it. See `test-before-final.log` and `test-after.log`; `test-before.log` retains the earlier equivalent assertion's output.

The saved integration harness already accepts arbitrary keyword arguments in its `ConfigurationSpace` fake. Its source and prior execution log were copied here without changing the originals. The full cheap mocked integration suite passed against the updated entrypoint, including BOHB budget handling and final-selection boundaries. See `integration-after.log` and `integration-after.json`; the harness hash remains `e7025d586077647660fb397b35177cf40a2c1e0a504a95cd7610ab962f68374d`.

Validation:

- Focused regression: 1 test passed; the pre-fix version failed as expected.
- Existing mocked image integration suite: all checks passed, exit 0.
- Ruff check and format check: passed for both changed files.
- No-sloppy with repository Ruff configuration: 0 errors, 1 advisory warning. The warning concerns the short sampling callback, which is passed to the mock and invoked for all three seed cases; it is a necessary test callback rather than a production helper.
- Basedpyright: 0 errors, 16 warnings. Fourteen are in the existing BOHB entrypoint's untyped dependency calls and unused results; two concern unused `runpy`/`unittest` return values in the test. No suppressions or typing framework were introduced.

This verifies initial configuration-sampler replay and entrypoint seed propagation. It does not establish reproducibility of complete trained BOHB searches or KDE proposal trajectories. Existing experimental results and frozen lifecycle sources were preserved. No model training, dataset access, downloads, test-set evaluation, or cloud execution occurred.
