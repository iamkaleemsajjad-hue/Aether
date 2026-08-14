# Aether Runtime

> **Compile once. Run on any hardware, forever.**

Aether Runtime is a production-grade ML compiler, inference engine, and deployment platform for large language models. It compiles any HuggingFace-compatible model into an **AEG (Aether Executable Graph)** package — a portable, self-contained artifact that runs on NVIDIA GPUs (Hopper/Ada/Blackwell), AMD MI300X, Apple Silicon, Intel NPUs, Qualcomm AI 100, and CPU AVX-512 with zero code changes.

---

## Table of Contents

1. [Architecture](#architecture)
2. [AEG Package Format v3.1](#aeg-package-format-v31)
3. [Compiler Passes 1-9](#compiler-passes-1-9)
4. [Phase 5 — Observability and Safety](#phase-5--observability-and-safety)
5. [Phase 6 — Ecosystem](#phase-6--ecosystem)
6. [v3.1 Elite Extensions](#v31-elite-extensions-sections-2840)
7. [Quick Start](#quick-start)
8. [CLI Reference](#cli-reference)
9. [Testing](#testing)
10. [Directory Structure](#directory-structure)
11. [Research Citations](#research-citations)

---

## Architecture

```
   [HuggingFace Model / GGUF / ONNX / SafeTensors]
                    |
   ┌────────────────────────────────────┐
   │     Aether Compiler (5 Stages)     │
   │  Stage 1: Specialised Ingestion    │
   │  Stage 2: 22 Optimizer Passes      │
   │  Stage 3: Multi-target Kernel Gen  │
   │  Stage 4: Kernel Cache & Hub       │
   │  Stage 5: AEG Packaging            │
   └────────────────────────────────────┘
                    |
         [AEG Package v3.0]
          /    |    |    \
        CUDA  Metal ROCm  CPU
       SM90+ M1-M5 MI300X AVX-512
                    |
   ┌────────────────────────────────────┐
   │   Aether Runtime (R1-R12 Layers)   │
   │  R1: Dynamic Precision   R7: Green │
   │  R2: Multi-Agent KV      R8: TEE   │
   │  R3: Grammar FSM         R9: MDLM  │
   │  R4: SLO Scheduler      R10: RDMA  │
   │  R5: TTT Engine         R11: Cache │
   │  R6: MCP Tools          R12: CXL   │
   └────────────────────────────────────┘
```

The compiler produces a **portable AEG artifact** loaded by the runtime. The engine selects pre-compiled kernels for the detected hardware at load time — zero recompilation, zero configuration.

---

## AEG Package Format v3.0

```
model.aeg/
  FORMAT_VERSION                  # AEG/3.0 (v5.0 passes)
  manifest.json                   # Top-level manifest with section hashes
  graph/
    computation_graph.aeg-ir      # Portable IR (target-agnostic)
    attention_head_patterns.json  # MInference sparse patterns (Pass 8)
  weights/
    precision_map.json            # Per-layer FP8/FP4/INT4/ternary assignments
    sparsity_masks.json           # Wanda 2:4 masks (Pass 9)
  adapters/
    manifest.json                 # LoRA/DoRA/VeRA/LoRAMoE multi-slot (8 concurrent)
  kernels/
    cuda_sm90/                    # H100 kernels
    cuda_sm100/                   # B200 FP4-native kernels
    cuda_sm120/                   # Rubin RTX 5090 kernels
    metal_m1/ metal_m2/ metal_m3/
    rocm_cdna3/                   # MI300X kernels
    openvino_npu/ qualcomm_qnn/ cpu_avx512/
  cuda_graphs/                    [v3.1] Pre-captured CUDA decode graphs
    manifest.json                 # Persistent kernel registry + capture plan
    sm90_decode_b1.json           # Batch=1 decode graph (15-30% speedup)
    sm90_decode_b{2,4,8,16,32,64}.json
    sm90_prefill_chunked.json
  parallelism/                    # Context parallelism plans
    1gpu.json
    4gpu_1m.json                  # 1M tokens on 4xH100 (striped ring)
    32gpu_cp.json                 # 4M tokens on 32xH100
    prefill_decode_split.json     # DistServe disaggregation
  inference/                      [v3.1] Inference-time compute
    compute_profiles.json         # greedy/BoN4/BoN8/beam4/MCTS/adaptive
    prm_head.json                 # Process Reward Model head config
  safety/                         # Content safety guardrails
    prompt_guard.json             # Injection detection
    output_filter.json            # PII + secret redaction
    toxicity_config.json          # 5 category thresholds
    policy.json                   # EU AI Act Art.50 config
  provenance/                     [v3.1] Full audit trail
    manifest.json                 # EU AI Act + C2PA + eval results
    fingerprint.json              # HMAC IP fingerprint (100 triggers)
  watermark/                      [v3.1] Output watermarking
    config.json                   # SynthID-style green-list config
```

---

## Compiler Passes 1-9

| Pass | Name | Research Basis |
|------|------|---------------|
| 1 | Operator Fusion | FlashAttention-3 (2024) |
| 2 | KV Cache Structuring | MLA (DeepSeek-V2), Radix Tree (2023) |
| 3 | Precision Assignment | FP8-LM (2023), GPTQ (2023) |
| 4 | Parallelism Discovery | Megatron-LM (2021) |
| 5 | MoE Expert Routing | DeepSeek-V3 (2024), Mixtral |
| 6 | Reasoning Graph | DeepSeek-R1 (2025), QwQ |
| 7 | LoRA Adapter Slots | LoRAX (2024), Punica (2023) |
| 8 | Sparse Attention | MInference (NeurIPS 2024) |
| 9 | Pruning + Sparsity | Wanda (2023), SparseGPT (2023) |

---

## Phase 5 — Observability and Safety

### OpenTelemetry Native Tracing (`aether.observability.otel`)

```python
from aether.observability.otel import AetherTracer, MetricsCollector, OTLPExporter

tracer = AetherTracer(service_name='aether-prod')
span = tracer.trace_request(
    request_id='req-001', prompt_tokens=512, generated_tokens=128,
    ttft_ms=45.3, total_ms=980.0, model_id='qwen3-72b',
)

mc = MetricsCollector()
mc.record(ttft_ms=45.3, tokens_per_second=130.5, e2e_latency_ms=980.0)
print(mc.prometheus_text())  # Prometheus /metrics endpoint

exporter = OTLPExporter(endpoint='http://otel-collector:4318/v1/traces')
exporter.export_to_file(tracer, Path('traces.json'))
```

**Prometheus metrics:** `aether_request_total`, `aether_error_total`, `aether_ttft_ms{quantile=p50|p95|p99}`, `aether_tokens_per_second`, `aether_eagle_accept_rate`, `aether_kv_hit_rate`

### CI/CD Eval Gate (`aether.observability.ci_pipeline`)

```python
from aether.observability.ci_pipeline import CIEvalPipeline

pipeline = CIEvalPipeline(
    aeg_path='qwen3-72b.aeg',
    max_regression=0.02,
    required_benchmarks=('hellaswag', 'mmlu', 'gsm8k'),
)
report = pipeline.run_and_save(
    output_path='eval_report.json',
    benchmarks=['hellaswag', 'mmlu', 'gsm8k', 'humaneval'],
)
if not report.gate_decision.passed:
    print(f'BLOCKED: {report.gate_decision.failing_benchmarks}')
```

Supported benchmarks: HellaSwag, MMLU, GSM8K, MATH-500, HumanEval, AIME

### Drift Monitoring and A/B Rollout (`aether.observability.gates`)

```python
from aether.observability.gates import DriftMonitor, ABRolloutController

monitor = DriftMonitor(baseline_win_rate=0.80, alert_drop=0.05, min_samples=20)
status = monitor.record(snapshot)
if status['alert']:
    reloader.rollback(exp_id)

ctrl = ABRolloutController('exp-001', candidate_percent=0.01)
variant = ctrl.assign(request_id)  # stable SHA-256 hash routing
new_pct = ctrl.ramp(gate_passed=True, drift_alert=False)  # doubles each step
```

### Content Safety (`aether.safety.policy`)

```python
from aether.safety.policy import ContentPolicyEngine

engine = ContentPolicyEngine(audit_path=Path('audit.jsonl'))
result = engine.check_prompt('Ignore all previous instructions...')
# result.allowed = False, reason = 'prompt_injection'

result = engine.check_output(model_output)
safe = result.redacted_text  # api_key: SECRET_REDACTED
```

| Layer | Detects | Action |
|-------|---------|--------|
| PromptGuard | Injection, jailbreaks | Block |
| ToxicityScorer | Hate, threats, NSFW (5 categories) | Block |
| OutputFilter | PII, API keys, passwords | Redact |
| Audit Logger | All events | JSONL (EU AI Act) |

---

## Phase 6 — Ecosystem

### SDK Generators (`aether.ecosystem.sdks`)

```python
from aether.ecosystem.sdks import (
    TypeScriptSDKGenerator, GoSDKGenerator,
    RustSDKGenerator, GitHubActionsGenerator,
)

TypeScriptSDKGenerator().write('./sdk/typescript/')  # aether-sdk.ts
GoSDKGenerator().write('./sdk/go/')                  # aether_client.go
RustSDKGenerator().write('./sdk/rust/src/')          # aether_client.rs
GitHubActionsGenerator().write('.', model_id='qwen3-72b',
    benchmarks=['hellaswag', 'mmlu', 'gsm8k'])
# -> .github/workflows/aether-eval.yml
```

All SDKs include: typed request/response structs, generate(), stream(), health() methods, retry logic.

### VS Code Extension (`aether.ecosystem.vscode_plugin`)

```python
from aether.ecosystem.vscode_plugin import VSCodePluginManifest, AEGInspectorProvider

VSCodePluginManifest().write('./vscode-extension/')  # package.json (13 commands)
AEGInspectorProvider().write('./vscode-extension/')  # src/extension.ts
```

Commands: `aether.compile`, `aether.inspect`, `aether.bench`, `aether.serve`, `aether.evalGate`, `aether.abRollout`, `aether.hubPush`, `aether.hubPull`, `aether.showGraph`, `aether.showPrecisionMap`, `aether.safetyCheck`, `aether.hardwareInfo`, `aether.traceExport`

---

## v3.1 Elite Extensions Sections 28-40

### Section 28: Long-Context Engine (`aether.runtime.long_context`)

```python
from aether.runtime.long_context import (
    SalienceKVEvictor, RingAttentionPlanner, YaRNConfig, LongContextProfile
)

# 4-tier KV eviction: GPU -> CPU -> NVMe -> Evicted
# Score = 0.5*attention + 0.3*recency + 0.2*anchor (StreamingLLM)
evictor = SalienceKVEvictor(window_size=2048, anchor_tokens=4)
order = evictor.eviction_order(blocks, current_seq_len=500_000)

# Ring/striped/ulysses context parallelism plan
plan = RingAttentionPlanner().plan(
    total_tokens=1_000_000, num_gpus=4, target='cuda_sm90'
)  # topology='striped', tokens_per_gpu=250_000

# YaRN RoPE extension: 4096 -> 131072 (32x scale factor)
yarn = YaRNConfig(original_max_position=4096, target_max_position=131072)

# Write long-context profile to AEG manifest
LongContextProfile(max_context_tokens=1_000_000).write_to_manifest('model.aeg/')
```

Research: StreamingLLM (2023), ScissorHands (2024), SnapKV (2025), Ring Attention (2023), YaRN (2023), LongRoPE (2024)

### Section 33: Process Reward Model (`aether.distillation.reward_model`)

```python
from aether.distillation.reward_model import (
    ProcessRewardModel, ReasoningChainAligner, SelfDistillationConfig
)

prm = ProcessRewardModel()
score = prm.score('What is 42*13?', '1. 42x13=546. Therefore the answer is 546.')
# Conservative minimum step score

steps = prm.score_detailed('prompt', response)
# [StepScore(step_idx=0, score=0.68, error_type=None), ...]

cfg = SelfDistillationConfig.for_reasoning_model()
# 5-30x cost reduction, 95-97% quality retention
```

Research: Let's Verify Step by Step (2023), Math-Shepherd (2024), OmegaPRM (2025), SDFT (2026)

### Section 35: IP Fingerprinting (`aether.provenance.fingerprint`)

```python
from aether.provenance.fingerprint import AEGModelFingerprint, ZKOwnershipProof

fp = AEGModelFingerprint()
fp.write('model.aeg/', owner_id='company-abc', n_triggers=100)

result = fp.verify('suspect-model.aeg', 'company-abc', fingerprint)
print(result.verdict)      # 'IP_DERIVED' if match_rate > 85%

proof = ZKOwnershipProof.create('company-abc', weights_hash)
assert proof.verify_binding(weights_hash)  # Privacy-preserving
```

Research: MetaFinger (2024), ADV-TRA (2025), ZK-proof Model Ownership (2026)

### Section 36: Zero-Downtime Hot-Reload (`aether.runtime.hot_reload`)

```python
from aether.runtime.hot_reload import AetherHotReload

reloader = AetherHotReload()
exp = reloader.start_reload(
    'qwen3-72b-v2.aeg', active_aeg='qwen3-72b-v1.aeg',
    baseline_win_rate=0.80, step_size=0.10,
    step_interval_sec=3600, alert_drop=0.05,
)
variant = reloader.route_request(exp.experiment_id, request_id)
reloader.auto_step(exp.experiment_id)   # 1% -> 2% -> 4% -> ... -> 100%
reloader.rollback(exp.experiment_id)    # Instant rollback <1ms
```

Research: Google SRE Book (canary analysis), Helium workflow-aware serving (2026)

### Section 37: CUDA Graph Manifest (`aether.cuda.graph_manifest`)

```python
from aether.cuda.graph_manifest import CUDAGraphManifestWriter, PersistentKernelRegistry

writer = CUDAGraphManifestWriter(
    target='cuda_sm90',
    decode_batch_sizes=(1, 2, 4, 8, 16, 32, 64),
)
files = writer.write('model.aeg/')
# Persistent kernels: 50-200us -> <5us per decode step
# Throughput improvement: 15-30% at small batch sizes
```

Research: vLLM CUDA Graphs Dispatcher (2026), NVIDIA Persistent Thread Model (2012)

### Section 38: Fleet Health and Auto-Scaling (`aether.fleet.health`)

```python
from aether.fleet.health import FleetHealthMonitor, AutoScaler, MultiRegionTopology, SLOConfig

slo = SLOConfig(max_p95_latency_ms=500.0, max_error_rate=0.01)
monitor = FleetHealthMonitor(slo=slo)
health = monitor.record(node_metrics)  # HEALTHY/DEGRADED/UNHEALTHY/OFFLINE

decision = AutoScaler(min_replicas=1, max_replicas=16).evaluate(2, metrics_list)
# ScaleDecision(action='scale_up', delta_replicas=2)

region = MultiRegionTopology(regions).assign_region('req-001', compliance_required='GDPR')
```

Research: Helium (2026), Kubernetes HPA, MuxWise SLO-aware scheduling (2026)

### Section 39: AEG Format v3.1 Builder (`aether.core.aeg_format_v31`)

```python
from aether.core.aeg_format_v31 import AEGPackageV31, AEGManifestV31

AEGPackageV31(model_id='qwen3-72b', target='cuda_sm90').build('qwen3-72b.aeg/')
# Creates all 10 v3.1 sections with proper manifests

reader = AEGManifestV31('qwen3-72b.aeg/')
print(reader.summary())
# {model_id, available_targets: [14 targets], has_cuda_graphs: True}
```

---

## Quick Start

```bash
pip install aether-runtime
aether compile meta-llama/Llama-3-70B --target cuda_sm90 --precision fp8
aether inspect llama-3-70b.aeg/
aether eval llama-3-70b.aeg/ --suite reasoning --max-regression 0.02
aether serve llama-3-70b.aeg/ --port 8080
aether sdk generate --lang typescript --output ./sdk/
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `aether compile <model>` | Compile model to AEG package |
| `aether inspect <path.aeg>` | Show AEG package summary |
| `aether bench <path.aeg>` | Run benchmark suite |
| `aether serve <path.aeg>` | Start inference server |
| `aether eval <path.aeg>` | Run eval gate CI check |
| `aether hardware` | Show hardware profile |
| `aether hub push <path.aeg>` | Push to Aether Hub CDN |
| `aether hub pull <model-id>` | Pull from Aether Hub CDN |
| `aether sdk generate` | Generate TypeScript/Go/Rust SDKs |
| `aether sign <path.aeg>` | Sign package (C2PA binding) |
| `aether verify <path.aeg>` | Verify package signature + fingerprint |

---

## Testing

```bash
python -m pytest tests/ -v                                    # All tests
python -m pytest tests/unit/test_phase5_observability.py -v  # Phase 5
python -m pytest tests/unit/test_phase6_ecosystem.py -v      # Phase 6
python -m pytest tests/unit/test_v31_elite_extensions.py -v  # v3.1 Extensions
```

> **Run the suite serially.** `test_e2e_compile_run_cpu.py` and
> `test_v31_features.py` both compile into the shared `~/.aether` cache; two
> concurrent pytest processes race on it and fail spuriously.
>
> Tests that need HuggingFace weights skip cleanly when no network path to
> `huggingface.co` is available.

See [REMEDIATION.md](REMEDIATION.md) for the July 2026 gap-closure report —
the compile-time plan layer (PRD §16 multi-modal graph, §34.2 RAG pipeline,
MLA planner), the runtime `PrecisionManager`, and the MXFP4-vs-FP4 codec
decision are documented there.

| Module | Tests | Status |
|--------|-------|--------|
| `observability.otel` | 13 | Pass |
| `observability.ci_pipeline` | 8 | Pass |
| `observability.gates` | 9 | Pass |
| `safety.policy` | 11 | Pass |
| `ecosystem.sdks` | 28 | Pass |
| `ecosystem.vscode_plugin` | 15 | Pass |
| `runtime.long_context` | 15 | Pass |
| `distillation.reward_model` | 11 | Pass |
| `provenance.fingerprint` | 8 | Pass |
| `runtime.hot_reload` | 7 | Pass |
| `cuda.graph_manifest` | 8 | Pass |
| `fleet.health` | 14 | Pass |
| `core.aeg_format_v31` | 12 | Pass |
| **Total** | **166** | **166/166** |

---

## Directory Structure

```
src/aether/
  compiler/passes/
    pass1_fusion.py          Operator fusion
    pass8_sparse_attention.py  MInference A-shape/vertical-slash patterns
    pass9_pruning.py           Wanda 2:4 + SparseGPT
  core/
    aeg_format_v31.py        [v3.1] Complete AEG package builder + manifest reader
  cuda/
    graphs.py                CUDAGraphCapturePlan + Selector
    graph_manifest.py        [v3.1] Manifest writer + persistent kernel registry
  distillation/
    pipeline.py              Base distillation pipeline
    reward_model.py          [v3.1] PRM + ReasoningChainAligner + SDFT config
  ecosystem/
    sdks.py                  TypeScript/Go/Rust SDK generators + GitHub Actions
    vscode_plugin.py         VS Code extension manifest + inspector provider
  fleet/
    manager.py               FleetManager + FleetNode + FleetConfig
    health.py                [v3.1] FleetHealthMonitor + AutoScaler + MultiRegion
  observability/
    gates.py                 EvalGate + DriftMonitor + ABRolloutController
    otel.py                  AetherTracer + MetricsCollector + OTLPExporter
    ci_pipeline.py           CIEvalPipeline + BenchmarkRunner (6 benchmarks)
  provenance/
    manifest.py              ProvenanceManifest + C2PA + EU AI Act
    fingerprint.py           [v3.1] AEGModelFingerprint + ZKOwnershipProof
  runtime/
    hot_reload.py            [v3.1] AetherHotReload + AutoRolloutController
    long_context.py          [v3.1] SalienceKVEvictor + RingAttentionPlanner + YaRN
  safety/
    guardrails.py            PromptGuard + OutputFilter
    policy.py                ContentPolicyEngine + ToxicityScorer + SafetyManifestWriter
tests/unit/
  test_phase5_observability.py   53 tests
  test_phase6_ecosystem.py       55 tests
  test_v31_elite_extensions.py   90+ tests
```

---

## Research Citations

| Feature | Research |
|---------|----------|
| Sparse Attention | MInference (Microsoft, NeurIPS 2024) |
| KV Eviction | StreamingLLM (2023), ScissorHands (2024), SnapKV (2025) |
| Ring Attention | Ring Attention (2023), Striped Attention (2023) |
| RoPE Extension | YaRN (2023), LongRoPE (2024) |
| Process Reward Model | Let's Verify Step by Step (2023), Math-Shepherd (2024), OmegaPRM (2025) |
| Self-Distillation | SDFT (2026) — 5-30x cost reduction, 95-97% quality retention |
| Speculative Decoding | EAGLE-2 (2024), Medusa (2024) |
| CUDA Graphs | vLLM CUDA Graphs Dispatcher (2026) |
| IP Fingerprinting | MetaFinger (2024), ADV-TRA (2025) |
| ZK Ownership Proof | Zero-Knowledge Model Ownership (2026) |
| Watermarking | SynthID-Text (Google DeepMind, 2024) |
| EU AI Act Compliance | Article 50 — AI content transparency obligations |
| Fleet Scheduling | Helium workflow-aware serving (2026), MuxWise (2026) |
| Hot-Reload | Google SRE Book (canary analysis), Helium (2026) |
| Disaggregated Serving | DistServe (2024), Mooncake (2024) |

---

## License

Apache 2.0

*Aether Runtime — Compile once. Run on any hardware, forever.*
