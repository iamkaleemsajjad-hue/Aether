# Quality measurements that require actual evidence

The evaluation validator now rejects missing, mismatched and nonfinite evidence,
scores fixed next-token labels correctly, and drives append-only engine caches
without duplicating their prefixes. This is a correction to the evaluation API.
**Mandatory compiler/publication and runtime-admission quality gating is still
open**: the compiler's existing `core.QualityReport` is a different type, and no
new production compiler gate is claimed here.

Read scope covered the complete evaluation validator and exports, its importers,
compiler quality/report contracts, evaluation tests, actual CPU/Torch forward-step
interfaces, diagnostic harness and worker memory/admission helpers. The initial
83-case regression selection returned 77 failures and six passes against the
original implementation. Later focused runs added edge, mutable-result and real
engine checks. All 102 validator cases pass in the 234-test focused selection.

## Mathematical and execution contracts

KL uses `sum(P * (log(P) - log(Q)))`, with the reference distribution first.
The primary derivation read was Blei, Kucukelbir and McAuliffe,
[arXiv:1601.00670v9, section 2.2, equation (11)](https://arxiv.org/pdf/1601.00670).
The implementation was checked against the actual installed and tagged
[PyTorch 2.9.1 KL implementation](https://github.com/pytorch/pytorch/blob/v2.9.1/aten/src/ATen/native/Loss.cpp).
Log-space calculation preserves low-probability evidence: the regression with
opposed logits separated by 1000 measures KL 1000, rather than an epsilon-capped
value near 23.03. The gate uses maximum per-position KL and minimum cosine;
means remain available for inspection. Equal zero vectors use an explicit
cosine-one convention; unequal zero/nonzero vectors do not.

Fixed-label likelihood follows the autoregressive factorization in
[Radford et al. (2019), equation (1)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf),
the actual installed/tagged
[Transformers 4.57.3 causal-loss alignment](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/loss/loss_utils.py),
and the [HF perplexity definition](https://huggingface.co/docs/transformers/en/perplexity).
`PPL = exp(mean NLL)` implies a relative change of
`expm1(candidate_mean_nll - reference_mean_nll)`. The gate compares that log
ratio with `log1p(allowed_change)` without capping NLL. Absolute overflow remains
explicit in strict JSON while finite means and the log-ratio verdict survive.
The inaccessible original 1951 KL paper is not claimed as read.

Greedy suffix mismatches count against agreement; an empty comparison cannot
pass. Activation checks require equal nonempty layer sets, exact shapes and
finite values. A changed zero reference has an infinite relative error and
fails, with an explicit JSON marker. Thresholds are validated at construction
and use. Empty or inconsistent mutable result objects cannot assert success.
These DTO checks do not authenticate a model or an arbitrary caller's data.

`compare_engines` prefills each isolated cache once, then appends one ID per
decode step. Candidate logit comparison uses the reference history, and explicit
continuation labels use separate teacher-forced caches for both engines. Rows
are copied before a reusable engine scratch buffer can change them. Without
fixed labels, PPL is absent, rather than the likelihood of each model's own
argmax. This is a fixed-count probe with no EOS stopping, not corpus evaluation.
The same real AEG passes all three checks on NumPy CPU and FP32 Torch CPU.

## Executed diagnostic and losses

`tools/audit_quality_validator.py` compares the explicit `3ebcb26` validator
source with the current source in fresh serial workers. The original source is
read from Git; source and numerical evidence bytes are hashed. The input is the
already executed original-FP32 HF/tiny-AEG evidence from the default-policy
audit. HF is **not re-executed** in this diagnostic. The untrained tiny checkpoint
has 26,784 parameters; no trained or mid-size quality claim follows from it.

Three blocks alternate baseline/current ordering, with two warmups and 31 metric
or seven engine-comparison repeats per worker. There are 93 retained metric and
21 comparison samples per revision. Profiling occurs separately after timing.
Hardware: physical i5-1334U CPU, Windows build 26200, Python 3.10.11, NumPy 1.26.4,
one BLAS/OpenMP thread. The host was memory constrained and not exclusive.

| Observation | Baseline | Current |
|---|---|---|
| Metric checks, median [min, max] ms | 0.8784 [0.3806, 2.0079] | **1.0140 [0.4851, 1.5568]**, slower |
| Metric checks, population standard deviation ms | 0.3182 | 0.3084 |
| Full engine comparison, median [min, max] ms | 125.7457 [55.9654, 145.7594] | 83.8309 [68.0316, 104.7568] |
| Comparison block medians, ms | 75.1254 / 128.9655 / 132.8864 | 76.5632 / 80.3650 / 94.9219; first block slower |
| Comparison population standard deviation ms | 28.4349 | 10.1271 |
| Final reference cache logical length, 4 prompt + 32 predicted IDs | 624 | 35 consumed positions |
| Final actual allocated KV arrays | 313,344 bytes | 17,408 bytes |
| Generated IDs match independent fresh-prefix AEG oracle | False in every block | True in every block |
| Engine comparison without fixed labels | Own-argmax PPL 95.2644, reported passing | PPL absent; scope explicitly diagnostic |
| Sampled main-worker RSS peak across blocks | 61,280,256-61,566,976 bytes | 58,904,576-59,301,888 bytes |
| Sampled process-tree RSS peak across blocks | 65,855,488-66,072,576 bytes | 63,225,856-63,647,744 bytes |

These timings establish neither a clean-host speed guarantee nor a serving TTFT
or tokens/s improvement. The stronger metric checks cost more in this run, and
the first comparison block regressed slightly. The cache reduction and repaired
history are directly observed correctness/memory changes. RSS sampling can miss
brief peaks, and process-tree RSS can double-count shared pages; these are not
weight/activation component measurements.

On the archived original-FP32 HF evidence, actual maximum KL is
`5.44895715e-8`, minimum cosine `0.9999953809`, greedy agreement `16/16`, and
aggregated fixed-label PPL rises `0.01122948%` over eight labels. The old/new
formulas agree on these benign measurements within rounding. Adversarial tests
establish the corrected failure behavior; benign HF evidence alone would not.

Two earlier attempts were partially RAM-blocked and remain unsuccessful. Their
successful tiny workers measured process-tree RSS below 64 MiB. Only the tiny
NumPy quality action now reserves 256 MiB plus its tensor estimate, keeping the
independent 128 MiB diagnostic headroom and live termination check. Other action
and fixture estimates are unchanged. In the complete three-block run all worker
identities were verified as owned, and minimum measured host availability was
386,002,944 bytes. Estimates remain admission policy, not guaranteed memory.

Evidence: `build/mission/quality-validator-before.*`, `quality-validator-101.*`,
the retained edge/DTO/consistency failures, `quality-combined-focused.*`,
`quality-validator-audit/`, `quality-validator-audit-retry/` and
`quality-validator-audit-bounded/`. The last diagnostic exited zero; both partial
attempts exited one. Raw samples, cache traces, actual outputs, source/evidence
hashes, profiling records and verified-worker RSS remain in those directories.

## Remaining integration

Production admission must bind original source/config/tokenizer/sample identities
to the candidate graph, weights, execution profile and actual HF measurements,
then recompute all required gates before publishing or replacing an artifact.
The current optional compiler evaluator runs after publication and cannot serve
as that contract. A source-FP32 oracle must not be substituted with a BF16 AEG or
a low-precision reference merely to fit memory. Unsupported references and
unavailable memory must remain clear refusals. CLI, SDK and serving admission
still need this mandatory end-to-end gate.

The [full collection report](mission-validation.md),
[ledger correction](mission-calibration-persistence.md),
[Torch shape correction](mission-forward-step-contract.md), and unchanged
[22-row pass dispositions](mission-default-policy.md) keep the scope separate.
Physical CPU evidence is current. ONNX and accelerator parity, larger/trained
models, remaining pass/algorithm correctness, engine memory/speed and CI work
remain open; no hardware tier is promoted by this evaluation correction.
