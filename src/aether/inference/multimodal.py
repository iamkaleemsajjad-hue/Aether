"""Multi-modal unified graph planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModalityEncoder:
    """Encoder participating in a unified multimodal AEG graph."""

    modality: str
    model_id: str
    parallelism: str = "data_parallel"
    token_budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "model_id": self.model_id,
            "parallelism": self.parallelism,
            "token_budget": self.token_budget,
        }


@dataclass(frozen=True)
class MultiModalGraphPlan:
    """Compiled graph for text, image, audio, and video inputs."""

    llm_model: str
    encoders: tuple[ModalityEncoder, ...] = field(default_factory=tuple)
    projector_op: str = "aeg.multimodal_project"
    llm_parallelism: str = "tensor_parallel"

    def to_graph(self) -> dict[str, Any]:
        stages: list[dict[str, Any]] = []
        for encoder in self.encoders:
            stages.append({"id": f"encode_{encoder.modality}", "op": "aeg.modality_encode", **encoder.to_dict()})
        stages.append({"id": "project", "op": self.projector_op, "inputs": [stage["id"] for stage in stages]})
        stages.append({"id": "generate", "op": "aeg.llm_generate", "model": self.llm_model, "parallelism": self.llm_parallelism})
        return {
            "version": "multimodal_graph/1.0",
            "stages": stages,
            "optimizations": {
                "vit_data_parallel": True,
                "llm_tensor_parallel": self.llm_parallelism == "tensor_parallel",
                "mm_sparse_attention": any(encoder.modality in {"video", "image"} for encoder in self.encoders),
            },
        }


def default_multimodal_plan(llm_model: str = "llm.aeg") -> MultiModalGraphPlan:
    return MultiModalGraphPlan(
        llm_model=llm_model,
        encoders=(
            ModalityEncoder("image", "vision_encoder.aeg", "data_parallel", 4096),
            ModalityEncoder("audio", "audio_encoder.aeg", "data_parallel", 2048),
            ModalityEncoder("video", "video_encoder.aeg", "data_parallel", 65536),
        ),
    )
