"""Fleet management and multi-node orchestration primitives."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FleetNode:
    """A physical or virtual host available for AEG serving."""

    node_id: str
    target: str
    gpu_count: int = 0
    memory_gb: float = 0.0
    region: str = "local"
    labels: dict[str, str] = field(default_factory=dict)

    def capacity_score(self) -> float:
        gpu_score = self.gpu_count * 100.0
        memory_score = self.memory_gb
        accelerator_bonus = 100.0 if self.target.startswith(("cuda", "rocm", "metal")) else 0.0
        return gpu_score + memory_score + accelerator_bonus

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "target": self.target,
            "gpu_count": self.gpu_count,
            "memory_gb": self.memory_gb,
            "region": self.region,
            "labels": dict(self.labels),
            "capacity_score": round(self.capacity_score(), 4),
        }


@dataclass(frozen=True)
class FleetConfig:
    """Desired deployment policy for a model AEG."""

    replicas: int
    min_gpu_memory_gb: float = 0.0
    preferred_targets: tuple[str, ...] = ("cuda_sm90", "cuda_sm100", "rocm_gfx942", "metal_m3", "cpu_avx512")
    rollout_percent: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "replicas": self.replicas,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "preferred_targets": list(self.preferred_targets),
            "rollout_percent": self.rollout_percent,
        }


@dataclass(frozen=True)
class DeploymentHandle:
    """Stable deployment assignment produced by the fleet manager."""

    deployment_id: str
    model_aeg: str
    assignments: tuple[dict[str, Any], ...]
    config: FleetConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "model_aeg": self.model_aeg,
            "config": self.config.to_dict(),
            "assignments": list(self.assignments),
        }


class AetherFleetManager:
    """Place AEG replicas across heterogeneous targets."""

    def __init__(self, nodes: list[FleetNode] | None = None) -> None:
        self.nodes = list(nodes or [])

    def register(self, node: FleetNode) -> None:
        self.nodes.append(node)

    def deploy(self, model_aeg: str, config: FleetConfig) -> DeploymentHandle:
        candidates = [node for node in self.nodes if self._eligible(node, config)]
        candidates.sort(key=lambda node: (self._target_rank(node.target, config), -node.capacity_score(), node.node_id))
        if not candidates:
            raise ValueError("no eligible fleet nodes for deployment")
        assignments = []
        for replica in range(config.replicas):
            node = candidates[replica % len(candidates)]
            assignments.append(
                {
                    "replica": replica,
                    "node_id": node.node_id,
                    "target": node.target,
                    "region": node.region,
                    "backend": self._backend_for_target(node.target),
                    "kernel_profile": self._kernel_profile(node.target),
                }
            )
        digest = hashlib.sha256(f"{model_aeg}:{config.to_dict()}:{assignments}".encode("utf-8")).hexdigest()[:16]
        return DeploymentHandle(f"deploy_{digest}", model_aeg, tuple(assignments), config)

    def plan_manifest(self, model_aeg: str, config: FleetConfig) -> dict[str, Any]:
        return self.deploy(model_aeg, config).to_dict()

    def _eligible(self, node: FleetNode, config: FleetConfig) -> bool:
        target_ok = not config.preferred_targets or node.target in config.preferred_targets
        memory_ok = node.memory_gb >= config.min_gpu_memory_gb or node.gpu_count == 0 and config.min_gpu_memory_gb == 0
        return target_ok and memory_ok

    def _target_rank(self, target: str, config: FleetConfig) -> int:
        try:
            return config.preferred_targets.index(target)
        except ValueError:
            return len(config.preferred_targets)

    def _backend_for_target(self, target: str) -> str:
        if target.startswith("cuda"):
            return "trtllm_or_vllm"
        if target.startswith("rocm"):
            return "vllm_rocm"
        if target.startswith("metal"):
            return "mlx"
        if target.startswith("openvino"):
            return "openvino"
        if target.startswith("qualcomm"):
            return "qnn"
        return "torch"

    def _kernel_profile(self, target: str) -> dict[str, Any]:
        if target in {"cuda_sm100", "cuda_sm120"}:
            return {"attention": "flash_attention_4", "precision": "fp4_or_fp8", "cuda_graphs": True}
        if target.startswith("cuda"):
            return {"attention": "flash_attention_3", "precision": "fp8", "cuda_graphs": True}
        if target.startswith("rocm"):
            return {"attention": "aiter_flash_attention", "precision": "fp8", "cuda_graphs": False}
        return {"attention": "portable", "precision": "int4_or_bf16", "cuda_graphs": False}


class HotReloadRouter:
    """Route traffic between active and candidate AEGs during hot reload."""

    def __init__(self, active_aeg: str, candidate_aeg: str | None = None, candidate_percent: float = 0.0) -> None:
        self.active_aeg = active_aeg
        self.candidate_aeg = candidate_aeg
        self.candidate_percent = candidate_percent

    def route(self, request_id: str) -> str:
        if not self.candidate_aeg or self.candidate_percent <= 0:
            return self.active_aeg
        digest = hashlib.sha256(f"hotreload:{request_id}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return self.candidate_aeg if bucket < self.candidate_percent else self.active_aeg

    def promote(self) -> str:
        if self.candidate_aeg:
            self.active_aeg = self.candidate_aeg
        self.candidate_aeg = None
        self.candidate_percent = 0.0
        return self.active_aeg

    def rollback(self) -> str:
        self.candidate_aeg = None
        self.candidate_percent = 0.0
        return self.active_aeg

    def manifest(self) -> dict[str, Any]:
        return {
            "active_aeg": self.active_aeg,
            "candidate_aeg": self.candidate_aeg,
            "candidate_percent": self.candidate_percent,
            "routing": "stable_hash",
            "rollback": "instant_switch_to_active",
        }
