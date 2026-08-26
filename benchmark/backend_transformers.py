"""The Hugging Face Transformers baseline.

This is deliberately an ordinary, competent Transformers setup: the published
checkpoint, the library's own KV cache, its default attention implementation for
the device (which selects SDPA, and FlashAttention within SDPA where the GPU
supports it), and ``model.generate`` for generation.  Nothing is disabled to
make the comparison easier, and nothing exotic is enabled that a normal user
would not get.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.backends import (
    GenerationOutcome,
    LoadOutcome,
    UnsupportedConfiguration,
    resolve_dtype,
    set_seed,
)


class TransformersBackend:
    """Runs a model through ``transformers`` on the selected device."""

    name = "transformers"

    def __init__(self, device: str = "cuda", attn_implementation: str | None = None) -> None:
        self.device = device
        self.attn_implementation = attn_implementation
        self._model: Any = None
        self._tokenizer: Any = None
        self._precision: str | None = None
        self._model_id: str | None = None
        self._resolved_attn: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": self.device,
            "precision": self._precision,
            "attn_implementation_requested": self.attn_implementation,
            "attn_implementation_resolved": self._resolved_attn,
            "generation": "model.generate with use_cache=True (library default KV cache)",
            "weight_source": "published checkpoint, loaded at the benchmark precision",
        }

    def load(self, model_id: str, precision: str) -> LoadOutcome:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = resolve_dtype(precision)
        self._precision = precision
        self._model_id = model_id

        # Fetch first and time it separately: a download is network latency, not
        # a property of either runtime.
        download_start = time.perf_counter()
        AutoTokenizer.from_pretrained(model_id)
        download_s = time.perf_counter() - download_start

        load_start = time.perf_counter()
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"dtype": dtype}
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        try:
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        except TypeError:
            # Older releases spell the dtype argument differently.
            kwargs.pop("dtype")
            kwargs["torch_dtype"] = dtype
            self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self._model.to(self.device)
        self._model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_s = time.perf_counter() - load_start

        self._resolved_attn = getattr(self._model.config, "_attn_implementation", None)
        if self._tokenizer.pad_token_id is None:
            # Required for any batched call; does not affect batch size 1.
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return LoadOutcome(
            download_s=download_s,
            prepare_s=None,
            load_s=load_s,
            total_s=download_s + load_s,
            notes={
                "attn_implementation": self._resolved_attn,
                "dtype": str(dtype),
                "context_length": getattr(self._model.config, "max_position_embeddings", None),
                "parameters": sum(p.numel() for p in self._model.parameters()),
            },
        )

    def tokenizer(self) -> Any:
        return self._tokenizer

    def _encode(self, prompt: str, batch_size: int) -> Any:
        encoded = self._tokenizer([prompt] * batch_size, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def prefill(self, prompt: str) -> Any:
        import torch

        inputs = self._encode(prompt, 1)
        with torch.no_grad():
            output = self._model(**inputs)
        return output.logits[0, -1].detach().float().cpu()

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
        sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,  # fixed work per iteration
            "do_sample": sample,
            "use_cache": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if sample:
            kwargs.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                kwargs["top_k"] = top_k
        with torch.no_grad():
            output = self._model.generate(**inputs, **kwargs)
        generated = output[0, prompt_len:].tolist()
        return GenerationOutcome(
            text=self._tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=[int(value) for value in generated],
            prompt_tokens=prompt_len,
            completion_tokens=len(generated),
            backend_metrics={"batch_size": batch_size, "returned_rows": int(output.shape[0])},
        )

    def first_token_latency(self, prompt: str, *, max_new_tokens: int, seed: int) -> float:
        """Time until the first token is decodable, using the library's streamer."""
        import threading

        import torch
        from transformers import TextIteratorStreamer

        set_seed(seed)
        inputs = self._encode(prompt, 1)
        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, timeout=120.0)

        def worker() -> None:
            with torch.no_grad():
                self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens, min_new_tokens=max_new_tokens,
                    do_sample=False, use_cache=True, streamer=streamer,
                    pad_token_id=self._tokenizer.pad_token_id,
                )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        thread = threading.Thread(target=worker, daemon=True)
        start = time.perf_counter()
        thread.start()
        for _ in streamer:
            break
        elapsed = time.perf_counter() - start
        thread.join(timeout=300.0)
        return elapsed

    def supports_batch(self, batch_size: int) -> bool:
        return batch_size >= 1

    def unload(self) -> None:
        import gc

        import torch

        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def unsupported(reason: str) -> UnsupportedConfiguration:
    return UnsupportedConfiguration(reason)
