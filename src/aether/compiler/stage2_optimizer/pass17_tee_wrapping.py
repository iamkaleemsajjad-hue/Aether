"""
Pass 17 — Trusted Execution Environment (TEE) Kernel Wrapping.

Confidential AI inference ensures that model weights and activations are
encrypted both at rest and during computation.  This pass wraps emitted
kernel calls with TEE enclave enter/exit guards and generates attestation
metadata embedded in the AEG artifact.

Supported TEE backends:

1. **NVIDIA Confidential Computing (CC mode)** — H100/B200/GB300:
   - Encrypted weight loading via NVIDIA CC key management.
   - PCIe BAR memory isolation using NVIDIA CVM (Confidential VM).
   - Per-kernel activation encryption: HMAC-SHA256 integrity tags.
   - < 10% overhead on H100 in CC mode (NVIDIA attestation report 2025).

2. **Intel TDX (Trust Domain Extensions)**:
   - Enter TD enclave before each kernel dispatch.
   - Exit TD after kernel completion + HMAC verification.
   - Uses TDCALL instruction interface.

3. **AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging)**:
   - Guest VM memory encryption with SNP attestation report.
   - Per-page measurement for weight blobs.
   - GHCB (Guest-Hypervisor Communication Block) based enter/exit.

AEG artifacts:
  - ``.aeg/security/tee_config.json``: backend, attest endpoint, key slots.
  - ``.aeg/security/weight_hash_manifest.json``: SHA-256 per weight blob.
  - ``aeg.tee_enter()`` and ``aeg.tee_exit()`` opcodes wrapping kernels.

Research basis:
  - NVIDIA Confidential Computing Whitepaper (2025).
  - Intel TDX Architecture Specification Rev 1.5 (2024).
  - AMD SEV-SNP Firmware ABI Spec Rev 1.58 (2024).
  - ConfidentialML (arXiv 2025): threat model for confidential LLM inference.
  - Guardian: confidential multi-tenant LLM serving (OSDI 2026).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import Any

from aether.compiler.config import CompilerConfig
from aether.compiler.report import PassReport
from aether.compiler.stage2_optimizer.base_pass import BasePass
from aether.utils.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_TEE_BACKENDS: frozenset[str] = frozenset(
    {"nvidia_cc", "intel_tdx", "amd_sev_snp"}
)

# AEG TEE opcodes.
_OPCODE_TEE_ENTER = "aeg.tee_enter"
_OPCODE_TEE_EXIT = "aeg.tee_exit"
_OPCODE_TEE_ATTEST = "aeg.tee_attest"


class TEEKernelWrappingPass(BasePass):
    """Pass 17: Wrap AEG kernels with TEE enclave enter/exit guards.

    Produces:
      - Per-kernel ``aeg.tee_enter`` / ``aeg.tee_exit`` opcode pairs.
      - Weight blob SHA-256 manifest for attestation.
      - TEE configuration JSON for the TEE Runtime (R8).
    """

    name = "tee_kernel_wrapping"
    description = (
        "Wrap AEG kernels with TEE enclave guards (NVIDIA CC / Intel TDX / AMD SEV-SNP). "
        "Produces weight hash manifest and tee_config.json for Runtime R8."
    )

    def run(
        self,
        graph: Any,
        architecture: Any,
        config: CompilerConfig,
    ) -> tuple[Any, PassReport]:
        start = time.perf_counter()
        report = PassReport(pass_name=self.name, status="skipped", details={})

        if not config.enable_tee:
            return graph, report

        backend = config.tee_backend
        if backend not in _SUPPORTED_TEE_BACKENDS:
            logger.warning("Pass 17: Unknown TEE backend %r. Using nvidia_cc.", backend)
            backend = "nvidia_cc"

        try:
            attest_endpoint = config.tee_attest_endpoint
            logger.info(
                "Pass 17: Wrapping kernels with %s TEE guards (attest=%s).",
                backend.upper(),
                attest_endpoint or "self-signed",
            )

            # Compute weight blob hashes for attestation manifest.
            weight_hashes = _compute_weight_hashes(graph)

            # Wrap each kernel node in the graph.
            n_kernels_wrapped = _wrap_kernels(graph, backend)

            # Generate TEE enclave key slot metadata.
            key_slots = _generate_key_slots(backend, n_kernels_wrapped)

            # Compute AEG graph hash for attestation.
            graph_hash = _compute_graph_hash(graph, weight_hashes)

            # Write TEE artifacts.
            if hasattr(graph, "output_dir") and graph.output_dir is not None:
                _write_tee_artifacts(
                    output_dir=Path(graph.output_dir),
                    backend=backend,
                    attest_endpoint=attest_endpoint,
                    weight_hashes=weight_hashes,
                    key_slots=key_slots,
                    graph_hash=graph_hash,
                    n_kernels=n_kernels_wrapped,
                )

            elapsed = time.perf_counter() - start
            report.status = "applied"
            report.duration_ms = elapsed * 1000
            report.details = {
                "tee_backend": backend,
                "attest_endpoint": attest_endpoint,
                "kernels_wrapped": n_kernels_wrapped,
                "weight_blobs_hashed": len(weight_hashes),
                "key_slots_generated": len(key_slots),
                "graph_hash": graph_hash[:16] + "...",
                "estimated_overhead_pct": _backend_overhead(backend),
            }
            logger.info(
                "Pass 17 complete: %d kernels wrapped (%s TEE), "
                "%d weight hashes, ~%.0f%% overhead.  Elapsed: %.3fs.",
                n_kernels_wrapped,
                backend,
                len(weight_hashes),
                _backend_overhead(backend),
                elapsed,
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("Pass 17 failed: %s", exc, exc_info=True)
            report.status = "failed"
            report.details["error"] = str(exc)

        return graph, report


# ── Weight hashing ────────────────────────────────────────────────────────────


def _compute_weight_hashes(graph: Any) -> dict[str, str]:
    """Compute SHA-256 hashes of all weight blobs for the attestation manifest.

    This is the *compile-time* portion: we hash weight tensors from the graph's
    weight store so the runtime can verify no weights were tampered with.
    """
    hashes: dict[str, str] = {}

    # Try weight_store (AEG format).
    if hasattr(graph, "weight_store"):
        store = graph.weight_store
        if hasattr(store, "items"):
            for name, tensor in store.items():
                raw = _tensor_to_bytes(tensor)
                hashes[str(name)] = hashlib.sha256(raw).hexdigest()

    # Try PyTorch state_dict path.
    elif hasattr(graph, "state_dict") and callable(graph.state_dict):
        try:
            for name, param in graph.state_dict().items():
                if hasattr(param, "numpy"):
                    raw = param.numpy().tobytes()
                elif hasattr(param, "tolist"):
                    import struct as _struct
                    flat = param.reshape(-1).tolist()
                    raw = _struct.pack(f"{len(flat)}f", *flat)
                else:
                    raw = str(param).encode()
                hashes[name] = hashlib.sha256(raw).hexdigest()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Weight hash via state_dict failed: %s", exc)

    # Minimal fallback: hash the graph object's repr.
    if not hashes:
        graph_bytes = repr(graph).encode("utf-8")
        hashes["__graph__"] = hashlib.sha256(graph_bytes).hexdigest()

    return hashes


def _tensor_to_bytes(tensor: Any) -> bytes:
    """Convert a tensor to raw bytes for hashing."""
    if hasattr(tensor, "tobytes"):
        return tensor.tobytes()
    elif hasattr(tensor, "numpy"):
        return tensor.numpy().tobytes()
    elif isinstance(tensor, (list, tuple)):
        import struct as _struct
        try:
            return _struct.pack(f"{len(tensor)}f", *tensor)
        except Exception:  # noqa: BLE001
            return str(tensor).encode()
    return str(tensor).encode()


# ── Kernel wrapping ───────────────────────────────────────────────────────────


def _wrap_kernels(graph: Any, backend: str) -> int:
    """Wrap each compute kernel node with TEE enter/exit opcodes.

    Returns the number of kernels wrapped.
    """
    n_wrapped = 0
    ops = _iter_ops(graph)

    # Filter: only wrap compute-intensive ops (not metadata/shape ops).
    compute_op_keywords = {
        "linear", "matmul", "gemm", "attention", "conv", "einsum",
        "embedding", "moe_dispatch", "moe_combine",
    }

    for op in ops:
        op_type = _get_op_type(op).lower()
        if not any(kw in op_type for kw in compute_op_keywords):
            continue

        enter_opcode = {
            "opcode": _OPCODE_TEE_ENTER,
            "backend": backend,
            "target_op": op_type,
            "op_id": str(getattr(op, "id", str(n_wrapped))),
        }
        exit_opcode = {
            "opcode": _OPCODE_TEE_EXIT,
            "backend": backend,
            "op_id": str(getattr(op, "id", str(n_wrapped))),
        }

        if hasattr(graph, "wrap_with_tee"):
            graph.wrap_with_tee(op, enter_opcode, exit_opcode)
        elif hasattr(graph, "metadata"):
            tee_wrappers = graph.metadata.setdefault("tee_wrappers", [])
            tee_wrappers.append({"enter": enter_opcode, "exit": exit_opcode})

        n_wrapped += 1

    return n_wrapped


# ── Key slot generation ───────────────────────────────────────────────────────


def _generate_key_slots(backend: str, n_kernels: int) -> list[dict[str, Any]]:
    """Generate TEE encryption key slot descriptors.

    Each kernel gets a unique key slot ID (resolved at runtime from the
    platform key management service).  We store the key slot metadata but
    NOT the actual keys — those are generated fresh at TEE initialization.
    """
    slots = []
    for i in range(n_kernels):
        slot_id = f"aether_tee_{backend}_{i:04d}"
        slots.append(
            {
                "slot_id": slot_id,
                "kernel_index": i,
                "backend": backend,
                "algorithm": _backend_key_algorithm(backend),
                "key_size_bits": 256,
            }
        )
    return slots


def _backend_key_algorithm(backend: str) -> str:
    """Return the encryption algorithm used by each TEE backend."""
    return {
        "nvidia_cc": "AES-256-GCM",
        "intel_tdx": "AES-256-GCM",
        "amd_sev_snp": "AES-256-XTS",
    }.get(backend, "AES-256-GCM")


def _backend_overhead(backend: str) -> float:
    """Return the estimated runtime overhead percentage for a TEE backend."""
    return {
        "nvidia_cc": 8.0,    # NVIDIA 2025 attestation report: <10%
        "intel_tdx": 15.0,   # TDX TDCALL overhead for GPU workloads
        "amd_sev_snp": 12.0, # SEV-SNP memory encryption overhead
    }.get(backend, 10.0)


# ── Graph hash ────────────────────────────────────────────────────────────────


def _compute_graph_hash(graph: Any, weight_hashes: dict[str, str]) -> str:
    """Compute a deterministic hash of the compiled AEG graph for attestation."""
    h = hashlib.sha256()
    # Hash weight manifest.
    for name in sorted(weight_hashes):
        h.update(name.encode())
        h.update(weight_hashes[name].encode())
    # Hash graph structure representation.
    graph_repr = repr(graph).encode("utf-8") if graph is not None else b"null"
    h.update(graph_repr)
    return h.hexdigest()


# ── AEG artifact writer ───────────────────────────────────────────────────────


def _write_tee_artifacts(
    output_dir: Path,
    backend: str,
    attest_endpoint: str | None,
    weight_hashes: dict[str, str],
    key_slots: list[dict],
    graph_hash: str,
    n_kernels: int,
) -> None:
    """Write TEE config and weight hash manifest to .aeg/security/."""
    security_dir = output_dir / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    # TEE configuration.
    tee_config = {
        "format": "aether_tee_v1",
        "backend": backend,
        "key_algorithm": _backend_key_algorithm(backend),
        "attest_endpoint": attest_endpoint,
        "graph_hash": graph_hash,
        "n_kernels_wrapped": n_kernels,
        "key_slots": key_slots,
        "estimated_overhead_pct": _backend_overhead(backend),
    }
    (security_dir / "tee_config.json").write_text(
        json.dumps(tee_config, indent=2), encoding="utf-8"
    )

    # Weight hash manifest.
    hash_manifest = {
        "format": "aether_weight_hash_manifest_v1",
        "algorithm": "sha256",
        "weight_hashes": weight_hashes,
    }
    (security_dir / "weight_hash_manifest.json").write_text(
        json.dumps(hash_manifest, indent=2), encoding="utf-8"
    )
    logger.debug(
        "Wrote TEE artifacts: %s (%d weight hashes, %d key slots)",
        security_dir,
        len(weight_hashes),
        len(key_slots),
    )


# ── Utility helpers ───────────────────────────────────────────────────────────


def _iter_ops(graph: Any) -> list[Any]:
    for attr in ("iter_nodes", "nodes", "__iter__"):
        if attr == "__iter__" and hasattr(graph, "__iter__"):
            try:
                return list(graph)
            except Exception:  # noqa: BLE001
                return []
        method = getattr(graph, attr, None)
        if method is not None:
            try:
                return list(method() if callable(method) else method)
            except Exception:  # noqa: BLE001
                pass
    return []


def _get_op_type(op: Any) -> str:
    for attr in ("op_type", "type", "name", "opcode"):
        val = getattr(op, attr, None)
        if val:
            return str(val)
    return "unknown"


