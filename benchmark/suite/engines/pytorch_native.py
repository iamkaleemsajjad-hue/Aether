"""Plain PyTorch: a hand-written decode loop, with no generation framework.

Distinct from the Transformers baseline in exactly one way that matters, and it is
worth isolating: this engine calls ``model(...)`` itself and advances the KV cache
itself, so none of Hugging Face's generation machinery is in the loop - no logits
processors, no stopping-criteria list, no per-step bookkeeping over a
``GenerationConfig``.

That makes it the control for a specific question: how much of a framework's
per-token cost is the model, and how much is the framework around it. It is the
same weights, the same kernels and the same arithmetic as the baseline.
"""

from __future__ import annotations

from typing import Any

from benchmark.backend_transformers import TransformersBackend
from benchmark.backends import GenerationOutcome, UnsupportedConfiguration, set_seed
from benchmark.suite.engines import base

SPEC = base.EngineSpec(
    key="pytorch_native",
    display="PyTorch native decode loop",
    taxonomy=(base.RUNTIME, base.EXECUTION_ENGINE),
    summary=(
        "The same eager PyTorch modules as the Transformers baseline, driven by a "
        "hand-written prefill-then-decode loop that advances the KV cache "
        "directly. No generation framework, no logits processors, no compilation."
    ),
    package="torch",
    requires=("torch", "transformers"),
    has_build_phase=False,
    artifact_persistence=base.ARTIFACT_NONE,
    ttft_method="single_token_call",
    notes=(
        "Isolates framework overhead from model execution: identical weights and "
        "kernels to the Transformers row, minus generate()'s per-step machinery.",
    ),
)


class Engine(TransformersBackend):
    """Prefill once, then step the cache one token at a time."""

    spec = SPEC

    def __init__(self, device: str = "cuda", **_: Any) -> None:
        super().__init__(device=device)
        self.name = SPEC.key

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record.update(
            engine_key=SPEC.key,
            taxonomy=list(SPEC.taxonomy),
            generation="hand-written greedy/sampling loop over model(...) with a KV cache",
            representation=(
                f"published checkpoint cast to {self._precision or '?'} tensors"
            ),
            weight_storage_bits=32 if self._precision == "fp32" else 16,
            weight_storage_format=self._precision,
            quantized=False,
            ttft_method=SPEC.ttft_method,
        )
        return record

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        batch_size: int = 1,
    ) -> GenerationOutcome:
        import torch

        set_seed(seed)
        inputs = self._encode(prompt, batch_size)
        prompt_len = int(inputs["input_ids"].shape[1])
        model = self._model
        produced: list[list[int]] = [[] for _ in range(batch_size)]

        with torch.no_grad():
            output = model(**inputs, use_cache=True)
            cache = output.past_key_values
            attention = inputs.get("attention_mask")
            for _ in range(max_new_tokens):
                step = self._pick(output.logits[:, -1, :], temperature, top_p, top_k)
                for row in range(batch_size):
                    produced[row].append(int(step[row].item()))
                if attention is not None:
                    attention = torch.cat(
                        [attention, torch.ones_like(attention[:, :1])], dim=1
                    )
                # min_new_tokens is effectively pinned: the loop always runs the
                # full count so every iteration does identical work, which is what
                # the Transformers row is also configured to do.
                output = model(
                    input_ids=step.unsqueeze(-1),
                    attention_mask=attention,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = output.past_key_values

        ids = produced[0][:max_new_tokens]
        return GenerationOutcome(
            text=self._tokenizer.decode(ids, skip_special_tokens=True),
            token_ids=[int(value) for value in ids],
            prompt_tokens=prompt_len,
            completion_tokens=len(ids),
            backend_metrics={
                "batch_size": batch_size,
                "returned_rows": batch_size,
                "engine": "pytorch-native-loop",
                "row_completion_tokens": [len(row) for row in produced],
            },
        )

    @staticmethod
    def _pick(logits: Any, temperature: float, top_p: float, top_k: int) -> Any:
        """Choose the next token from a logit row, greedily or by sampling.

        Greedy is the primary configuration; the sampling branch exists so the
        engine honours the same generation settings as every other row rather than
        quietly ignoring them.
        """
        import torch

        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        scaled = logits.float() / temperature
        if top_k and top_k > 0:
            kth = torch.topk(scaled, top_k, dim=-1).values[..., -1, None]
            scaled = scaled.masked_fill(scaled < kth, float("-inf"))
        probabilities = torch.softmax(scaled, dim=-1)
        if top_p and top_p < 1.0:
            ordered, indices = torch.sort(probabilities, descending=True, dim=-1)
            cumulative = torch.cumsum(ordered, dim=-1)
            keep = cumulative - ordered <= top_p
            ordered = ordered * keep
            ordered = ordered / ordered.sum(dim=-1, keepdim=True)
            drawn = torch.multinomial(ordered, 1)
            return indices.gather(-1, drawn).squeeze(-1)
        return torch.multinomial(probabilities, 1).squeeze(-1)

    def generate_mixed(self, prompts: list[str], **_: Any) -> Any:
        raise UnsupportedConfiguration(
            "the native loop is measured on uniform batches only; ragged batching "
            "is a scheduler feature, and implementing one here would make this row "
            "a different engine than the control it exists to be"
        )

    def first_token_latency(self, prompt: str, *, max_new_tokens: int, seed: int) -> float:
        """Time the prefill plus the first decode step, with no streamer involved.

        This engine has no stream to subscribe to, so time-to-first-token is timed
        as what it is here: the work done before the first token exists. The method
        is declared in the spec so the report does not present it as equivalent to a
        streaming measurement.
        """
        import time

        import torch

        from benchmark import metrics

        set_seed(seed)
        inputs = self._encode(prompt, 1)
        metrics.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = self._model(**inputs, use_cache=True)
            token = self._pick(output.logits[:, -1, :], 0.0, 1.0, 0)
            _ = int(token[0].item())
        metrics.synchronize()
        return time.perf_counter() - start


def probe(hardware: Any, model_id: str, precision: str, options: Any) -> base.Availability:
    return base.generic_probe(SPEC, hardware)


def build(hardware: Any, model_id: str, precision: str, options: Any) -> Engine:
    return Engine(device="cuda" if hardware.nvidia else "cpu")
