# Aether Optimizer Passes

Stage 2 registers 22 passes in the order below. **Only Pass 3, precision
assignment, is enabled by default.** Passes 1, 2, 4, 5, 6, 7 and 8 are now
experimental opt-ins; Passes 9 through 22 remain opt-in. Changing
`optimization_level` does not select these flags or enable every pass.

The pipeline returns one `PassReport` per registration, including disabled or
unsupported passes. `applied` describes a compiler transformation, not a runtime
benefit or a model-quality guarantee. `OptimizerPipeline.run()` retains failed
reports for inspection; public `Compiler.compile()` rejects a failed report
before targeting and publication. A benign skip remains distinct from failure.
CLI feature requests also refuse success when the requested pass was skipped.

## Defaults, consumers and experimental status

The flags below are the canonical `CompilerConfig` fields. All manual opt-ins
remain available. A runtime consumer means that some emitted data reaches an
execution path; it does not certify a latency, memory, energy or quality gain.

| # | Pass/report name | Enable flag | Default | Actual consumer and disposition |
|---|---|---|---|---|
| 1 | `operator_fusion` | `enable_fusion` | Off | Rewrites AEG graph groups and records actual fusion patterns. The CPU/Torch transformer schedules are not interpreters of these fused IR groups. Experimental IR transformation; no demonstrated reduction in runtime launches or attributed speed benefit. |
| 2 | `sensitivity_analysis` | `enable_sensitivity` | Off | Annotates nodes and a sensitivity map using weight/calibration statistics. P9 can use the scores. P3 currently selects BF16 in its default mode with or without this map. Experimental diagnostic; scores are not measured perplexity or quality certification. |
| 3 | `precision_assignment` | `enable_precision_assignment` | **On** | Packaging consumes the precision map, writes encoded tensors, and loaders reconstruct execution weights. Default `sensitivity` mode conservatively assigns BF16; explicit uniform/manual formats can change numerical results. Real storage transformation, without automatic quality certification. |
| 4 | `kv_cache_structuring` | `enable_kv_cache_structuring` | Off | Adds KV IR nodes and placement/block hints. Ordinary runtime KV caching does not require this pass. Experimental plan; emitted hints do not establish paged allocation, tiered offload or cache savings. |
| 5 | `moe_routing` | `enable_moe_routing` | Off | Adds a global routing/tier plan. Executable source MoE routing separately uses each layer's authenticated router weights, experts and source top-k. Experimental placement hints; no demonstrated tier migration, adaptive routing or benefit. |
| 6 | `parallelism_discovery` | `enable_parallelism_discovery` | Off | Adds sharding plans to graph metadata; packaging persists those plans only when this pass applies. Runtime exposes them for inspection, without consuming them for device placement. Experimental discovery; no distributed execution, scaling or memory benefit is established. |
| 7 | `reasoning_graph` | `enable_reasoning_graph` | Off | Emits reasoning metadata and `graph/reasoning_graph.aeg-ir`. No demonstrated serving consumer executes these nodes to improve reasoning or early exit. Experimental graph artifact. |
| 8 | `sparse_attention` | `enable_sparse_attention` | Off | Loader validates `attention_head_patterns` and the CPU attention path applies per-head masks. This is executable approximation, not just metadata. Experimental: no validated long-context quality or speed benefit; plan presence can also bypass native decode attention. |
| 9 | `pruning_sparsity` | `enable_pruning` | Off | Computes masks from available weights; packaging applies them to stored tensors. Execution uses the resulting weights, without proof of a specialized sparse kernel. Experimental lossy transform; measured synthetic regressions preclude a quality-preserved default. |
| 10 | `mtp_head_compilation` | `enable_mtp_head` | Off | Serializes supplied MTP projection heads under `speculation/`; Runtime can load them for speculative requests. Missing real heads are skipped. Experimental: generic generation is not evidence of head use, correct target verification, acceptance rate or speed. |
| 11 | `grammar_constraint_compilation` | `enable_grammar_constraint` | Off | Compiles a supplied constraint and tokenizer-aware FSM under `grammar/`; public constrained generation can consume token masks. Experimental feature opt-in: ordinary generation does not prove the requested constraint was enforced, and no decode-overhead claim is certified. |
| 12 | `model_merging` | `enable_model_merging` | Off | Merges supplied weight sources; persisted tensors are consumed by normal execution. Runtime also exposes AEG merging/task-vector selection. Experimental, intentionally changes the model; graph emission or a successful merge is not task-quality evidence. |
| 13 | `ttt_fast_weight_injection` | `enable_ttt` | Off | Emits fast-weight configuration loaded by Runtime's TTT engine; the CPU request path can request adaptation. Experimental: loading slots or reporting adaptation loss does not prove useful transformer adaptation or improved model quality. |
| 14 | `semantic_kv_compression` | `enable_semantic_kv` | Off | Persists a compression plan consumed by the CPU KV cache, changing retention during generation. Experimental lossy cache policy; teacher-forced prefix logits alone miss generation/cache regressions. No general semantic-preservation or speed guarantee. |
| 15 | `cross_layer_kv_sharing` | `enable_cross_layer_kv` | Off | Persists sharing groups consumed by CPU cache aliasing. Experimental approximation: shared pointers are real, but the synthetic audit observed numerical/generation regressions. No quality-preserved memory-saving default. |
| 16 | `green_energy_compilation` | `enable_green_energy` | Off | Writes a profile loaded by Runtime's green power manager for request accounting. TDP-duration energy estimates are labeled estimates. Experimental metadata/accounting; no demonstrated device power control or measured energy reduction. |
| 17 | `tee_kernel_wrapping` | `enable_tee` | Off | Requires a concrete backend-emitted TEE kernel bundle. The current tree has no such executable backend, so ordinary requests skip and CLI requests are refused. Unsupported execution; hashes/configuration are not enclave isolation or attestation. |
| 18 | `mdlm_drafter_compilation` | `enable_mdlm_drafter` | Off | Requires a supplied, validated MDLM head bundle; Runtime can reload it into the diffusion drafter. Missing weights skip. Experimental trained-head integration; synthetic head execution does not prove draft acceptance, target correctness or throughput benefit. |
| 19 | `sub2bit_quantization` | `enable_sub2bit` | Off | Public packaging supports BitNet/ternary storage and the CPU reference path dequantizes for dense execution. Other accepted method names are refused at packaging when no runtime codec exists. Experimental lossy storage; reconstruction error is not PPL, nor proof of addition-only kernels or speed. |
| 20 | `video_token_compression` | `enable_video_compression` | Off | Gated on a vision-language architecture; text checkpoints skip. Public video generation requires a backend advertising executable vision support; the CPU AEG backend refuses it. Experimental/unvalidated end-to-end; emitted plans establish no token-reduction, streaming-memory or quality benefit. |
| 21 | `advanced_peft_compilation` | `enable_advanced_peft` | Off | Requires real adapter tensors. Packaging persists supported LoRA A/B data; CPU execution can select a compiled adapter via public `adapter_id`. Experimental: selection and changed logits alone do not validate source scaling, every PEFT variant or fused dispatch. |
| 22 | `rlvr_verifier_head_injection` | `enable_rlvr_verifier` | Off | Emits verifier/GRPO configuration consumed by the separate Runtime training interface, which requires caller-supplied policy/optimizer callbacks for real updates. Ordinary generation does not validate training; unconfigured human feedback skips. Experimental training integration, without demonstrated reward, convergence or model-quality benefit. |

## Precision and source-model semantics

The default P3 implementation assigns BF16 to transformer layers, embeddings
and the output head. Enabling P2 alongside it currently leaves that default
BF16 choice unchanged. Uniform mode uses the requested `precision` (or Q4_K_M
when uniform mode is selected without one); manual mode reads
`manual_precision_map`. These are explicit numerical changes, not an optimal
sensitivity-to-bit allocation or an automatically enforced perplexity budget.

An all-passes-off AEG can still use packaging's BF16 fallback. It is therefore
not an original FP32 oracle. Audit comparisons against unoptimized AEG and
against actual original-precision HF execution must be labeled separately.
`quality_budget` without a supplied evaluation callback does not certify
quality; deployment evaluation uses `Compiler.compile()`'s explicit evaluator
and benchmark/baseline inputs.

Disabling P5 removes that pass's global planning annotations, **not source MoE
routing**. Ingestion still binds real per-layer router/expert tensors and the
execution engines retain source top-k behavior. Specialized ingestion can add
its own tier annotations independently; turning off P5 does not erase every
MoE annotation. Similarly, P4 off does not disable runtime KV caches. P6 off
omits sharding-plan files from newly compiled artifacts; existing artifacts
can still load their stored plans for inspection.

P8's default context threshold is 32768. A declared long context selects plan
compilation; actual masks depend on the request length and configured windows.
The audit's forced threshold of 16 with a 64-token prompt is a diagnostic,
not validation of long-context accuracy or sparse-kernel speed. Even below
mask activation, attaching a plan can change native attention dispatch.

## Explicit SDK configuration

Use `CompilerConfig` to request compiler features. These flags are independent
of optimization level and do not silently enable prerequisite passes.

```python
from aether import Compiler, CompilerConfig

# Current default: P3 enabled; the other 21 canonical pass flags are false.
compiler = Compiler(CompilerConfig(targets=["cpu_avx2"]))

# Explicit graph/annotation inspection; no implied serving speed benefit.
inspection = CompilerConfig(
    targets=["cpu_avx2"],
    enable_fusion=True,
    enable_sensitivity=True,
    enable_kv_cache_structuring=True,
    enable_moe_routing=True,
    enable_parallelism_discovery=True,
    enable_reasoning_graph=True,
    enable_sparse_attention=True,
)

# Explicit lossy precision selection; requires independent quality evaluation.
quantized = CompilerConfig(precision="Q4_K_M")

# Sensitivity-aware pruning needs both opt-ins; it does not enable P2 itself.
pruning = CompilerConfig(enable_sensitivity=True, enable_pruning=True)

# A grammar request also needs the actual constraint.
structured = CompilerConfig(
    enable_grammar_constraint=True,
    grammar_schema='root ::= "hello"',
)

# Supported adapter files are required; select the adapter at runtime.
adapted = CompilerConfig(
    enable_advanced_peft=True,
    peft_adapter_paths=["path/to/adapter"],
)
```

Direct pass invocation explicitly exercises that pass. Pipeline-local
`enable_pass()` / `disable_pass()` selection is distinct from configuring a
public `Compiler.compile()` invocation, which creates its optimization
pipeline from its compile configuration. Pass the required flags to
`CompilerConfig` for public compilation.

## CLI examples

These are actual `aether compile` options. Use a local checkpoint directory
containing its configuration and weights; paths below must exist.

```bash
# Current defaults, including conservative BF16 assignment.
aether compile path/to/checkpoint --target cpu_avx2 --output build/default.aeg

# Explicit lossy layer precision; embeddings/head remain BF16.
aether compile path/to/checkpoint --target cpu_avx2 --precision Q4_K_M \
    --output build/q4.aeg

# Experimental grammar feature; supply a real supported constraint file.
aether compile path/to/checkpoint --target cpu_avx2 \
    --grammar-schema path/to/constraint.ebnf --output build/grammar.aeg

# Experimental MTP: checkpoint must contain usable prediction-head weights.
aether compile path/to/mtp-checkpoint --target cpu_avx2 --mtp \
    --output build/mtp.aeg

# Experimental MDLM: requires a real head bundle in Aether's supported format.
aether compile path/to/checkpoint --target cpu_avx2 --mdlm-drafter \
    --mdlm-weights path/to/head.npz --mdlm-K 8 --mdlm-T 6 \
    --output build/mdlm.aeg

# Experimental BitNet storage, executed by dense dequantization on CPU.
aether compile path/to/checkpoint --target cpu_avx2 --sub2bit ternary \
    --output build/ternary.aeg
```

The CLI has no general switch for each of P1/P2/P4/P5/P6/P7/P8; use the SDK
flags in the table for those opt-ins. There is no `--opt-level` pass-selection
option. Other explicit CLI features include `--ttt`, `--green`, `--tee` and
`--video-compression`; an accepted option still requires a supported source,
consumer and successful pass report. In particular, `--tee` does not turn
missing enclave execution into successful compilation.

## Ordering, evidence and compatibility

The table is the canonical pipeline order: 1 through 22. Disabled passes still
occupy their report position. P2 precedes P3 and P9, but no automatic dependency
activation occurs: P3 works without P2; P9 uses sensitivity only when supplied.
Grammar is P11, model merging P12, MDLM P18, sub-2-bit P19 and adapters P21.
Scheduling, MCP tool integration and long-context adaptation are not alternate
names for those passes.

Changes to default selection alter IR, optional artifact inventories and
hashes. AEG merging requires compatible graph semantics, so artifacts built
with older fusion defaults may require rebuilding with matching explicit flags.
Existing package loading is separate from the default compile policy.

The [executed pass audit](mission-pass-audit.md) records a historical CPU
checkpoint and its unresolved gaps; its uniform-Q4 P3 row does not describe
the conservative BF16 default. The [default-policy comparison](mission-default-policy.md)
retains before/after raw observations, repeated-build diffs and dispatch profiles.
Synthetic exact-logit/token agreement is numerical evidence on those fixtures,
not natural-language quality. Mid-model compilation/reference work blocked by
memory admission remains blocked, not successful coverage. No individual pass
receives a speed or energy claim from an IR rewrite, metadata emission,
modeled estimate or unrelated timing difference.
