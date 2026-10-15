# Runtime execution and cache integration

This page describes the execution paths added and verified from the local
repository. A configuration flag, device listing, stored descriptor, or SDK
installation is not evidence that an inference workload executed successfully.
The hardware report keeps those facts separate.
The [completion verification report](runtime-completion-verification.md)
records the final source checkpoint, complete-suite results and remaining limits.

## Implemented contracts

| Feature | Actual operation | Supported boundary |
|---|---|---|
| Typed cache handoff | Copies named tensors into a versioned, lossless snapshot, checks model identity and checksums, restores owned state and aliases | Known CPU cache schemas and nonsharded Torch attention/hybrid caches; destination geometry and precision must match |
| Local disk cache | Commits actual snapshot bytes, deduplicates identical blobs, reloads after process restart, removes unreferenced blobs | Ordinary filesystem storage; physical NVMe and GPU-direct IO are not inferred |
| Hub cache | Authenticated private namespace upload/download/delete of actual snapshot bytes, with strict hash/size acknowledgement and restart persistence | Explicit HTTP client configuration; no offline fallback, implicit network lookup, or RDMA claim |
| Separate prefill/decode workers | Executes actual forward calls on separate engine instances and hands off typed state, with bounded admission, cancellation and streaming | In-process Python workers; placement must have a supported adapter factory; no automatic arbitrary-device placement or speedup claim |
| Precision adjustment | Changes actual mutable execution tensors, measures calibration logits and latency, restores exact originals on rejection or request | Compact NumPy projection storage retains FP32 arithmetic; nonsharded Torch adapter can change supported tensor arithmetic with explicit lossy permission |
| Explicit draft speculation | Uses a loaded draft model, verifies proposals with target inference, applies greedy or probability-corrected acceptance | Compatible full vocabulary/special-token mapping and supported incremental cache; phase-pool combination is rejected until a draft-worker adapter exists |
| Dense AEG to ONNX | Exports real prefill and incremental-decode operators with shared external weights, then executes the selected provider | Supported dense decoder geometry and FP32 arithmetic; unsupported operators/plans fail explicitly |

CPU handoff schemas are `attention_kv_v1`, `mamba_v1`, `mamba2_v1`, `rwkv_v1`,
`hybrid_v1`, and `mla_replay_v1`. The MLA schema stores the token replay state
used by the current CPU engine; it does not claim a compressed latent cache.
Torch snapshots currently cover nonsharded attention and hybrid caches. Exact
FP16, BF16, and supported FP8 tensor bytes can be transported; transporting a
dtype is separate from executing a model with that dtype. BF16/FP8 do not have
a lossless built-in NumPy representation and require a suitable tensor adapter
for restoration.
CPU-to-Torch and Torch-to-CPU attention interchange requires contiguous,
uncompressed state and matching arithmetic dtype/geometry. Hybrid interchange
between those distinct adapters and multi-shard handoff remain unsupported.

The dense ONNX exporter rejects routed MoE, MLA, SSM, encoder models, unsupported
dynamic/LongRoPE schemes, active adapter/sparse plans, semantic-KV compression,
and cross-layer KV plans. Its context bound is explicit. Other execution
engines and compiler emitters remain available under their existing contracts.

## Persistent sessions through Runtime

Use an executable AEG containing its weights and tokenizer. These examples
choose CPU explicitly so a different host does not change the restoration
adapter implicitly.

```python
from aether import Runtime, RuntimeConfig

model = "./model.aeg"
config = RuntimeConfig(
    backend_name="aether_cpu",
    execution_devices=["cpu"],
    kv_cache_path="./state-cache",
    enable_semantic_cache=False,
    speculative_decoding=False,
)

with Runtime(config) as runtime:
    runtime.generate(model, "Hello", max_tokens=8, temperature=0,
                     aether_kv_session_id="conversation")
    saved = runtime.persist_session(model, "conversation", tier="disk")
    snapshot_key = saved["key"]

# A new runtime reads committed state, rather than an in-memory client cache.
with Runtime(config) as runtime:
    runtime.restore_session(model, "conversation", snapshot_key)
```

Persistence releases the live session only after the destination commit or
matching Hub acknowledgement succeeds. A failed persistence keeps the live
session available. Restoration checks the integrity-checked model fingerprint,
logical token prefix, cache schema, tensor state and precision epoch before
installing the resumed cache. Changing execution precision invalidates old
state. Explicitly restoring incompatible state raises an error.

Live and persisted sessions also bind the selected adapter and normalized task
coefficients to that execution contract. A request with different weights cannot
reuse base-model state. Legacy live caches without provenance are recomputed;
restoration never silently activates saved adapters or task weights. Shared
prefix lookup partitions by the same execution inputs and storage adapter.
The R2 coordinator bounds these partitions within its registered prefix cache;
this in-process reuse does not establish CUDA IPC or RDMA transport.

Explicit-draft sessions persist the exact target cache and its consumed token
prefix through the same CPU/Torch storage codecs. The draft cache is acceleration
state and is rebuilt from that prefix on the next speculative request; it is
not stored as a process-specific composite object. The result reports
`speculative_resume: {"target_state": "lossless", "draft_state":
"rebuild_from_token_prefix"}`. Restored requests can use another compatible,
tokenizer-validated draft while reusing the authoritative target state. This
avoids storing a second model's cache, but the first resumed speculative request
pays for draft prefill. Unsupported target cache schemas still fail explicitly.
Request-local TTT fast weights are not persisted as reusable base-model state;
older session state is cleared before those weights execute.

The manager's serialized host copy can be released after the execution adapter
owns the restored tensors using `release_payload_residency(block_id)`. This
requires verified, unchanged backing data and avoids rewriting the snapshot.
It reports the manager's backing tier; it does not infer the engine's residency.
Terminal runtime cleanup also calls `discard_resident_payloads()` to drop the
manager's owned snapshot bytes without writing or deleting disk/Hub data.
Committed snapshots remain available to a new manager after restart. This is
shutdown cleanup, not ordinary eviction: unpersisted snapshot bytes cannot be
recovered afterward. The returned byte count describes the manager's logical
snapshot ownership; caller-held references can keep shared buffers allocated.

Hub session storage requires an explicitly configured client:

```python
import os
from aether.hub.client import HubClient

client = HubClient("http://127.0.0.1:8081", auth_token=os.environ["AETHER_HUB_TOKEN"],
                   allow_local_cache=False)
with Runtime(config) as runtime:
    runtime.configure_cache_hub(client, namespace="alice")
    runtime.generate(model, "Hello", max_tokens=8, temperature=0,
                     aether_kv_session_id="conversation")
    saved = runtime.persist_session(model, "conversation", tier="hub")

with Runtime(config) as runtime:
    runtime.configure_cache_hub(client, namespace="alice")
    runtime.restore_session(model, "conversation", saved["key"])
```

Binding the client does not initiate IO. A fresh runtime can bind a client to
the same persisted registry and explicitly restore the returned key. Cache
snapshots are private to the permitted namespace; model-public visibility does
not make cache tensors public. Production network deployments must configure
their transport/security boundary separately; localhost tests do not certify
a production service.

## Cache storage without Runtime

Inference adapters exchange `CachePayload`; the generic storage layer does not
interpret model-specific cache geometry or execute arbitrary serialized objects.
There is no pickle loading and no implicit precision conversion.

```python
from aether.runtime import KVPayloadStore, KVCacheManager
from aether.runtime.cache_handoff import export_cpu_cache, restore_cpu_cache
from aether.core.types import MemoryTier

# cache is actual state returned by the engine's prefill forward call.
snapshot = export_cpu_cache(cache, model_fingerprint=verified_model_fingerprint)
store = KVPayloadStore("./cache-store", max_bytes=2 * 1024**3,
                       max_entry_bytes=128 * 1024**2, max_entries=4096)
manager = KVCacheManager(payload_store=store)
block = manager.import_payload(snapshot, prefix_hash=actual_prefix_hash)
key = manager.payload_key(block.block_id)
manager.swap(block.block_id, MemoryTier.L3_NVME)

restarted = KVCacheManager(payload_store=KVPayloadStore(
    "./cache-store", max_bytes=2 * 1024**3,
    max_entry_bytes=128 * 1024**2, max_entries=4096))
block = restarted.block_for_payload_key(key)
snapshot = restarted.export_payload(
    block.block_id, expected_model_fingerprint=verified_model_fingerprint)
resumed_cache = restore_cpu_cache(
    snapshot, expected_model_fingerprint=verified_model_fingerprint)
restarted.release_payload_residency(block.block_id)
```

`KVPayloadStore` exposes `put`, `get`, `put_bytes`, `get_bytes`, `entries`,
`delete`, `cleanup`, and `stats`. Keys are opaque metadata, never filesystem
paths. Immutable blob filenames are SHA-256 hashes. Entry and aggregate quotas
count framed blob bytes; ancillary metadata and entry count are bounded too.
Writes stage in the cache directory, flush actual bytes, atomically publish
the blob, and commit SQLite references. Interrupted/uncommitted files are
cleaned on restart. Read failures, corrupted bytes and quota failures cannot
be reported as successful reloads or uploads. These tests establish process
restart and fault handling, not a filesystem/firmware power-loss certificate.
The persistent UUID key is authoritative. Block integers are local to a
manager; recovery assigns distinct local IDs when independent managers stored
colliding integers, preserving every snapshot key across subsequent restarts.

Legacy `KVCacheBlock` descriptors remain supported. Their enum changes and
token counters are labelled `descriptor_transitions`; they are not evidence
of tensor IO. Actual snapshot metrics include `payload_transfers`,
`transferred_tensor_bytes`, `transferred_serialized_bytes`,
`payload_transfers_by_route`, and `resident_tensor_bytes`. No estimated head
count or element width is substituted for measured payload bytes.
The Runtime's legacy R10 NIXL/RDMA availability flags describe that specialized
transport. HTTP Hub snapshots are accounted separately through actual payload
routes and byte counters; they do not establish NIXL, UCCL or RDMA availability.

Hub byte routes are:

| Method | Route | Result |
|---|---|---|
| PUT | `/v1/kv/{namespace}/{key}` | Validates bounded payload and required `X-Content-Hash`; returns matching identity/hash/size metadata |
| GET | `/v1/kv/{namespace}/{key}` | Returns verified binary payload with `X-Content-Hash` and `Cache-Control: private, no-store` |
| DELETE | `/v1/kv/{namespace}/{key}` | Deletes the scoped reference and unreferenced bytes; returns an explicit boolean acknowledgement |
| GET | `/v1/kv/{namespace}` | Lists only that permitted namespace's cache metadata |

Corresponding explicit client methods are `put_kv`, `get_kv`, `delete_kv` and
`list_kv`. They never inherit the model client's optional development-memory
fallback. Retry of an upload uses the same key and identical bytes; a lost
HTTP acknowledgement is not proof that the server did not commit it.

## Precision and worker configuration

`RuntimeConfig` exposes `precision_storage_dtype`, `precision_allow_lossy`,
`precision_max_logit_error`, `precision_max_slowdown`,
`precision_calibration_tokens`, and `precision_cooldown_seconds`.
`Runtime.adjust_precision(model, calibration_tokens)` returns the actual
measured result; `restore=True` restores originals. Lossy permission defaults
to false. Calibration results apply to the supplied sequences, not every
possible prompt. Changes wait for active readers and invalidate live cache
and compiled/batched selections that depended on the prior precision.

Separate workers use `disaggregate_prefill_decode`, `prefill_devices`,
`decode_devices`, `disaggregate_prefill_workers`,
`disaggregate_decode_workers`, `disaggregate_queue_capacity`, and
`prefill_chunk_size`. Default factories must match the loaded engine's supported
placement. Unsupported active adapters/TTT/speculation combinations fail
explicitly instead of ignoring their requested behavior. Runtime lifecycle
closes owned worker pools and discards restoration backups when unloading the
owned engine. It avoids expanding compact tensors merely to destroy them.
Explicit precision restoration remains verified and transactional. Arithmetic
precision is reported from the execution contract; compact NumPy projection
storage is separately reported and still uses FP32 arithmetic. Automatic
calibration on an unsupported engine reports `unsupported` without changing
tensors; an explicit adjustment raises the contract error. Precision `saved_bytes`
describes the changed logical execution tensors; other references and allocator
reservations can prevent an equivalent reduction in physical RAM or device memory.

Use `with Runtime(config)` or explicitly call `close()`. Shutdown rejects new
operations, closes owned paused streams and waits for executing operations
before unloading. An executing operation that exceeds `close(timeout=10.0)`
raises a timeout; the runtime remains closed to new admission and shutdown can
be retried. A closed runtime is terminal: construct another runtime for a new
lifetime. Requests using the same session serialize access to its mutable
cache; distinct sessions can execute concurrently.

The CLI exposes the same controls through the global `--runtime-config` option.
Save this JSON as `runtime.json` for separate CPU workers:

```json
{
  "backend_name": "aether_cpu",
  "execution_devices": ["cpu"],
  "enable_semantic_cache": false,
  "speculative_decoding": false,
  "disaggregate_prefill_decode": true,
  "prefill_devices": ["cpu"],
  "decode_devices": ["cpu"],
  "prefill_chunk_size": 128
}
```

```powershell
aether --runtime-config runtime.json run ./model.aeg --prompt "Hello" --max-tokens 8 --temperature 0 --raw --non-interactive
```

The option goes before the subcommand. The JSON must be an object no larger
than 1 MiB and contain known `RuntimeConfig` fields; misspelled/unknown controls
fail with a CLI error. Environment configuration supplies initial values,
the JSON overlays them, and command-specific overrides are applied afterward.
Hub client binding and explicit session persistence/restoration are Python
operations; this option does not add a cache-upload CLI command.

## Task weights and request behavior

The mapping API accepts task names that also happen to be Python parameter
names, such as `weights`, `self` or `model_id`:

```python
with Runtime(config) as runtime:
    runtime.set_model_task_weights(model, {"legal": 2.0, "weights": 1.0})
    result = runtime.generate(model, "Hello", max_tokens=8)
```

The artifact must contain the requested task vectors. `set_task_weights` remains
a convenience API; its mapping-only form supplies global defaults. Per-model
weights override those defaults, and explicit request `task_weights` override
both. Explicit request `task_weights=None` selects base weights. Each admitted
request owns a copied coefficient snapshot. Values must be finite nonnegative
numbers, excluding booleans and strings; normalization avoids overflowing when
several individually finite values are large. Unknown task names fail explicitly.
The CLI and REST reweight routes use the mapping API; the local AetherClient SDK exposes
`set_task_weights_mapping(mapping)` alongside its keyword convenience method.

CPU task clones retain every model, layer and expert execution setting,
including normalization, attention scaling and RoPE geometry. Unchanged tensors
are shared; requested tensor replacements own their arrays. Native CPU streaming
uses the same adapter/task/TTT preparation as ordinary generation and releases
request-local state when the iterator completes, fails or closes.

The NumPy AEG loader decodes persisted low-bit projections once into execution
matrices. `storage_quant_meta` describes their original storage format;
`quant_meta` is reserved for actual encoded tensors passed to a compatible
execution kernel. Storage metadata cannot trigger a second decode or implicit
GPU selection. Task arithmetic on manually encoded execution tensors is rejected
until a typed conversion is provided; it is never added to integer code values.

Semantic response caching bypasses requests whose active execution controls are
not represented by its key. Session, adapter, task-weight and other unsupported
controlled requests execute their requested operation. True batching remains
available for supported ordinary requests; unsupported stateful/adaptation
combinations raise a capability error before entering the batched engine.

## Research and verification scope

The snapshot/connector separation follows the engine-independent extraction
and retrieval approach described in [LMCache's primary technical
report](https://lmcache.ai/tech_report.pdf). The real cache remains typed and
model-bound during persistence. [PagedAttention](https://arxiv.org/abs/2309.06180)
motivates keeping cache coordinates and storage allocation separate; this
implementation does not claim its GPU kernel or paging benchmarks.
[CacheGen](https://arxiv.org/abs/2310.07240) studies compression with quality
tradeoffs; current persistence deliberately preserves exact bytes and does not
claim those compression gains. Draft verification uses target inference and
the probability-corrected acceptance described in [speculative
decoding](https://arxiv.org/abs/2211.17192).
Execution-aware prefix keys follow the reuse boundary described by the official
[vLLM prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/).
[SGLang](https://arxiv.org/abs/2312.07104) also motivates reusing only compatible
prefix state; this implementation does not claim its scheduling or throughput.

Verification is reproducible with:

```powershell
python -m pytest tests/unit/test_kv_payload_storage.py tests/integration/test_hub_kv_payload.py tests/integration/test_runtime_execution_features.py --no-cov
python -m pytest tests/unit/test_disaggregated_execution.py tests/unit/test_execution_precision.py tests/unit/test_torch_cache_handoff.py --no-cov
python -m pytest tests/unit/test_speculative_state_execution.py --no-cov
python -m pytest tests/unit/test_aeg_onnx_export.py tests/unit/test_onnx_provider_contract.py --no-cov
```

Actual cache tests include noncontiguous/endian NumPy and FP16/BF16 Torch byte
roundtrips; checksum/identity/malformed-shape rejection; concurrent deduplication;
entry/body/storage quotas; interrupted writes and database-publication rollback;
restart recovery and cleanup; scoped localhost Hub authentication and actual
binary transport; and exact decode logits after attention/hybrid disk reload
and dense Hub server restart. Torch-only cases are skipped in the clean test
environment without Torch.

The actual CPU handoff/worker tests also execute dense attention, hybrid,
Mamba-1, Mamba-2, RWKV and MLA forward calls, serialize/restore their state and
compare resumed decoding with uninterrupted execution. Cache handoff logits
match bitwise; separately chunked worker execution uses a 5e-7 absolute bound
for changed batch grouping. Speculative tests exercise real dense/hybrid
target and draft inference, rejected-proposal rollback, token budgets, EOS,
grammar and session state, plus a 2,000-trial CPU sampling comparison with the
target marginal distribution. These fixtures demonstrate the exercised
contracts; they do not certify every checkpoint or workload.

The [completion verification report](runtime-completion-verification.md) records
complete-suite and focused results with their exact scopes. Focused runs overlap
and must not be summed as one suite. Both environments exercise independent-manager
local-ID collision recovery. Interrupted or quota-inclusive command wall clocks
are not inference benchmarks; calibration gates measure their own workloads.

Physical execution evidence currently includes CPU and the local Intel UHD
DirectML provider. DirectML tests disable CPU fallback and inspect actual node
profiles. NVIDIA CUDA, AMD HIP, Apple MPS/Metal, vendor NPUs, CXL hardware and
multi-host RDMA were not physically exercised on this machine. Native CUDA and
planning-only vendor/TensorRT adapters must not be reported as executed merely
because a driver, DLL or SDK is present. Supported generic adapters and preserved
compiler emitters remain subject to their reported dependencies and capabilities.
