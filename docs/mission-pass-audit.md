# Executed pass audit: first CPU checkpoint

Run date: 2026-10-18. This is an incomplete mission checkpoint, not a benefit certification.

The tiny fixture has 26,784 real untrained parameters, two layers, width 32 and
vocabulary 128. All 22 flags were explicitly disabled for the baseline, then
each was enabled alone; seven combinations also ran. Native CPU execution on
the i5-1334U used one BLAS/OpenMP thread, two warmups and seven timed repeats,
eight generated tokens per prompt. Prompt A has four IDs; B has 64. Timings
below show prompt B and are **memory-constrained diagnostics** with a 128 MiB
free-memory reserve. No pass speedup is credited from these observations.

KL, cosine, greedy agreement and fixed-label PPL below use the original FP32
Hugging Face oracle from the successful UTF-8 merging run. Every source fixture
file was hash-compared before reusing that oracle. Logit metrics are teacher-forced
full-prefix results; generated-token agreement additionally exercises the generation
cache. They are not language/corpus quality or compiler certification.

Peak RSS is sampled for the verified generation worker, excluding its Windows
launcher; sampling can miss brief peaks. No component-level or GPU memory
attribution is claimed. Artifact bytes include every file. The full raw samples,
two prompts, inventories, status/errors and idempotency differences remain in
`build/mission/tiny-pass-matrix/report.json`, `merge-utf8-audit/report.json` and
`tiny-original-hf-comparison.json`. The first matrix exited 1 (KV idempotency
failure and a logging-encoding refusal); the separate merging run exited 0.

| Pass | Compile result | Artifact bytes | Decode tok/s, median [min,max] | TTFT ms, median [min,max] | Peak RSS MiB | KL max | Cosine min | Greedy % | PPL delta % |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| Baseline | all off | 192640 | 2366.5 [1497.4,2545.6] | 0.936 [0.729,1.681] | 56.3164 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 1: operator_fusion | applied | 187014 | 2489.2 [2397.9,2725.8] | 0.891 [0.784,1.048] | 55.9023 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 2: sensitivity_analysis | applied | 195049 | 2355.2 [1702.2,2745.6] | 0.925 [0.734,1.281] | 56.1484 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 3: precision_assignment | applied | 162534 | 2332.1 [1267.5,2650.9] | 0.857 [0.749,2.465] | 56.125 | 7.14821e-05 | 0.993556 | 87.5 | 0.465969 |
| 4: kv_cache_structuring | applied | 193994 | 2466.5 [2198.4,2688.9] | 0.862 [0.802,1.699] | 56.0156 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 5: moe_routing | skipped | 192640 | 2174.7 [596.3,2620.2] | 0.985 [0.889,1.084] | 56.0195 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 6: parallelism_discovery | applied | 194809 | 2696.4 [2435.7,2737.7] | 0.872 [0.758,0.989] | 56.0391 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 7: reasoning_graph | applied | 194771 | 2135.4 [1675.2,2538.3] | 0.977 [0.785,1.124] | 56.1875 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 8: sparse_attention | applied | 194754 | 1492.2 [1400.8,1605.2] | 3.208 [3.111,4.289] | 55.9492 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 9: pruning_sparsity | applied | 205792 | 2503.7 [1852.2,2752.5] | 0.996 [0.770,1.175] | 56.0781 | 0.000372663 | 0.967423 | 0 | 0.781797 |
| 10: mtp_head_compilation | skipped | 192640 | 2468.0 [1961.9,2681.2] | 0.964 [0.755,1.135] | 55.8516 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 11: grammar_constraint_compilation | applied | 193681 | 2626.9 [2494.0,2751.2] | 0.951 [0.775,0.989] | 55.9062 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 12: model_merging | applied | 193944 | 1964.9 [1363.9,2349.7] | 1.264 [0.802,2.232] | 56.1953 | 5.15869e-08 | 0.999995 | 100 | 0.014962 |
| 13: ttt_fast_weight_injection | applied | 201130 | 2562.2 [2116.4,2636.4] | 0.901 [0.785,1.204] | 55.9961 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 14: semantic_kv_compression | applied | 194542 | 1557.4 [537.7,1776.6] | 1.421 [1.205,4.612] | 56.3945 | 5.19109e-08 | 0.999995 | 25 | 0.0150864 |
| 15: cross_layer_kv_sharing | applied | 193390 | 2666.4 [2084.3,2716.9] | 0.915 [0.800,1.019] | 55.9102 | 0.00402475 | 0.606427 | 0 | 1.01842 |
| 16: green_energy_compilation | applied | 200873 | 2538.2 [2067.7,2759.1] | 0.936 [0.877,1.003] | 56.1016 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 17: tee_kernel_wrapping | skipped | 192640 | 2435.9 [2112.8,2536.1] | 0.889 [0.788,1.067] | 55.8281 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 18: mdlm_drafter_compilation | skipped | 192640 | 2338.3 [2132.8,2564.9] | 0.984 [0.782,1.629] | 56.3203 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 19: sub2bit_quantization | applied | 155716 | 2050.1 [1431.3,2195.5] | 1.171 [0.927,1.922] | 56.0781 | 0.00284316 | 0.737414 | 0 | 1.9671 |
| 20: video_token_compression | skipped | 192640 | 2448.0 [2204.8,2595.1] | 0.937 [0.820,1.085] | 55.8906 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 21: advanced_peft_compilation | skipped | 192640 | 2132.2 [1790.4,2564.8] | 0.887 [0.775,1.043] | 56.2109 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |
| 22: rlvr_verifier_head_injection | applied | 194087 | 2111.1 [1882.7,2322.0] | 1.114 [0.854,1.346] | 55.832 | 5.19109e-08 | 0.999995 | 100 | 0.0150864 |

The sparse pass's 16-token trigger is a forced diagnostic override; this is
not a long-context sparse-attention result. Precision requests Q4_K_M; pruning
requests 50% magnitude sparsity. Both are opt-in test cases, not a quality-preserved
default. Pass 12 uses its separate UTF-8 run; other rows use the first matrix.

Observed consumers: transformed dense weights for 3/9/12/19, and attached CPU cache plans
for 8/14/15. Fusion/KV IR, routing tiers, sharding, reasoning, TTT, power and RLVR
emission do not establish the specialized runtime behavior. Ordinary text requests
do not activate grammar or MTP. No working verdict is inferred from compilation.

Repeated application failed for KV structuring (duplicate node ID). Fusion, TTT,
semantic KV, cross-layer KV, green profiles, sub-2-bit and RLVR changed graph
state on reapplication. Detailed differences and reports are retained; none is
stripped to make idempotency pass.

The first merging attempt refused before the algorithm ran: unconfigured library
logging tried to emit Greek lambda to a Windows cp1252 pipe. The UTF-8 rerun
executes the same source and real merge. The general logging defect remains open.

Additional deterministic fixtures were created twice and byte-compared: mid
27,271,680 parameters, MoE 63,904, MTP 34,976 and adapter 26,784. Mid compilation
was blocked by admission: about 1.233 GB available versus 1.275 GB estimated worker
plus 128 MiB reserve. This is an estimate, not a measured mid-model footprint.

MoE and MTP fixtures compiled, loaded and generated; routing-tier and actual MTP
request behavior remain unproved. The adapter initially hit the same cp1252
logging failure; its separate UTF-8 run exited 0, selected `adapter_demo` in the
numerical worker and executed changed logits. This does not prove source-adapter
scaling correctness or public adapter request behavior. Full functional grammar/MTP/adapter
probes, the complete mid/conditional matrix, ordering checks, hardware execution
and clean-host/pretrained quality/performance comparisons remain required.

The subsequent compiler-admission correction rejects any reported optimizer
failure before targeting or publication, preserving the original failure details
and an existing destination. Direct pipeline idempotency diagnostics still retain
their reports. The measured table above remains the original checkpoint; this
correction establishes no new per-pass speed, memory or quality benefit. An
`applied` report still does not establish numerical quality certification. See
[the executed regression and complete-suite results](mission-validation.md#correction-reject-reported-optimizer-failures-before-publication).
