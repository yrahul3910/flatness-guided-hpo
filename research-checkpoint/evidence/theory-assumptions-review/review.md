# Independent assumptions review

The proposed witness disproves the final-kernel Hessian lower bound in the exact local `paper/main.tex`, Theorem `th:ff`. I found no assumption that excludes it. The conclusion applies to this file, whose SHA-256 matches the manuscript hash in the original `theory_proxy_checks.json`; it does not establish what appeared in the reviewed 2024 revision. The appendix draft is a proposal, not evidence of the original theorem's scope.

The manuscript defines unspecified matrix norms as Frobenius norms and defines its lower-bound notation as a pointwise inequality for every point in the function's domain (`main.tex:216-219`). Assumption 1 permits the input space R. Assumption 2 bounds activation derivatives by 1 (`:223`). The theorem requires fully connected final two layers, eligible hidden activations, final softmax, and cross-entropy (`:486-492`). It specifies differentiation twice with respect to `W[L]`. It imposes no minimum weight norm, stationarity, minimum, small-loss, full-rank, separability, distinct-input, or positive-strong-convexity condition.

The fixture constructs `x -> Dense(1) -> ReLU -> Dense(2) -> softmax`, with trainable biases present and initially zero. For the original witness, `x=1`, hidden kernel `u=1`, hidden bias 0, final kernel `(0.1,-0.1)`, and either integer class label give `h=1`. Both layers are actual affine dense maps; the ReLU preactivation is 1, away from its kink. Additional active identity-weight ReLU layers can precede these layers if more depth is desired. The kernel is already class-centered and nonzero. A separate globally smooth sigmoid witness uses hidden kernel and bias 0, giving `h=1/2`, so the conclusion also survives a strict differentiability reading of Assumption 2.

For fixed hidden activation `h`, output bias `b`, and `p=softmax(W^T h+b)`, one example has

```text
grad_W E = h (p-e_y)^T
H_W = (diag(p)-p p^T) tensor (h h^T)
```

For the sample-mean loss, average these Hessians over samples. This block is the partial derivative with all other model parameters fixed, even when those parameters are trainable in the full model. Flattening the rank-4 weight Hessian into a matrix preserves its Frobenius norm. Transposing the kernel convention only permutes coordinates. Labels disappear from this block because the label-dependent term is linear in the logits. A bias value of zero is allowed; including trainable biases in the fixture does not mean differentiating them inside `H_W`.

| Fixture | Exact `||H_W||_F` | Claimed per-example bound | Consequence |
| --- | ---: | ---: | --- |
| ReLU, `(0.1,-0.1)`, label 0 | 0.4950331454 | 5.0247729032 | Original violation |
| Same network, label 1 | 0.4950331454 | 5.0247729032 | Label invariance |
| Smooth sigmoid, `h=1/2` | 0.1246880201 | 2.5031178548 | No ReLU differentiability dependence |
| ReLU, `(5,-5)`, label 0 | 0.0000907916 | 0.1414149362 | Violation at loss about 0.0000454 |
| Stationary five-example mean | 0.4800000000 | 2.5151458962 | Stationarity alone does not repair the bound |

The original one-example witness is not stationary. That is not a mismatch with the stated theorem, but should be disclosed if the surrounding minima discussion motivates an objection. The additional stationary witness has five identical inputs `x=1`, labels `(0,0,0,1,1)`, `h=1`, and `W=(t,-t)` with `t=log(3/2)/2`. Every example has `p=(0.6,0.4)`, matching the empirical label frequencies. The mean logit gradient is zero, hence every weight and bias gradient is zero. Since every deterministic network gives the same prediction for these identical inputs, this prediction globally minimizes their mean cross-entropy. The maximum absolute computed gradient is below `9e-17`. Conflicting duplicate labels are allowed by the written assumptions; this witness does not claim separability or arbitrarily small minimum loss.

The manuscript's loss notation omits the example index on its activations (`:469-481`). The single-example witness avoids that ambiguity entirely. The stationary witness also uses identical activations and predictions, exactly matching that common-output interpretation. For five identical rows, the unnormalized batched Frobenius numerator is five times the per-example numerator. The batch RMS expression equals the per-example ratio. Both exceed the mean-loss Hessian norm here. The fixture records all three values explicitly; it does not treat a sum-loss Hessian as a mean-loss Hessian.

The proof's decisive error is treating the hidden activation as varying through an inverse output-activation Jacobian while differentiating with respect to the independent final weights (`:498-535`). In that partial derivative, the hidden activation is constant. The omitted label term makes the displayed gradient wrong, although that omission alone does not change this block's Hessian. The original audit's separate pseudoinverse criticism needs care: the softmax Jacobian is singular, but a rectangular product with the output kernel can still have full column rank, including in this witness. Singularity alone is therefore not the reason the witness works. The invalid dependence and the directly computed inequality are sufficient.

No restricted version of the ratio theorem is established here. The exact Hessian identity is valid for a frozen-feature, affine softmax head with unregularized mean cross-entropy at every finite parameter value, without stationarity. The appendix's smooth, strongly convex gradient-descent theorem has its own hypotheses (`proofs.tex:81-110`); it does not add those hypotheses to `th:ff`. Positive strong convexity in all unconstrained final-kernel coordinates is impossible because common-logit directions are null. A whole-network Hessian contains other blocks, so a generic head-only counterexample does not automatically disprove a whole-network claim. The fixture preserves full Hessians to make that distinction inspectable. Its result does not establish general ranking or HPO performance claims.

Keep score provenance separate. Local `main.tex:365` uses a different numerator, effectively `||h||^2/||W||`, from the theorem's `||p|| ||h||/||W||`. The recovered historical trajectory callback uses `max_batch ||H_batch||_F/||W||_F`, without a probability factor, as documented in `../named-trajectory-audit/audit.md`. The image implementation uses a maximum of batch RMS ratios with a probability factor. These are different formulas and aggregation rules. This review establishes no equivalence among them or identity with the reviewed 2024 manuscript.

Reproduce from the repository root:

```sh
<ORIGINAL_CHECKOUT>/.venv/bin/python sweep-results/research-audit-20261002/theory-assumptions-review/verify_witness.py
```

`results.json` preserves inputs, labels, all six parameter values and their ordering, hidden activations, logits, probabilities, mean losses, gradients, full and final-kernel Hessians, numerical errors, and source hashes. `source-context.json` preserves the decisive local passages. All five analytic/autodiff comparisons pass at absolute and relative tolerance `1e-12`; the largest entrywise error is below `6e-17`. Checks and their exit codes are in `verification.json`. The no-sloppy check uses an exact executable copy in a temporary Git repository because its ordinary explicit-path mode skips this ignored evidence directory. No manuscript, existing audit evidence, training data, or production code was changed. No training, external request, paid compute, or broad test suite was used.
