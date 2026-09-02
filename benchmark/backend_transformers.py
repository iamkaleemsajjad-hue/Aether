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
    MixedBatchOutcome,
    UnsupportedConfiguration,
    resolve_dtype,
    set_seed,
    stream_first_token_latency,
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

    def serving_prefill(self, prompt: str) -> Any:
        """Prefill the way generation actually pays for it: final logits only.

        ``prefill`` above asks for logits at every prompt position, which is a fair
        like-for-like comparison of that operation but is *not* what a served
        request needs — generation reads one row. Transformers' own ``generate``
        does not compute the discarded rows either; it passes ``logits_to_keep=1``
        to the model call. This measures that configuration so both backends can be
        compared on the work serving really does.

        Raises :class:`UnsupportedConfiguration` when the installed Transformers
        does not accept the argument, rather than silently measuring the full-logits
        path and labelling it as this one.
        """
        import inspect

        import torch

        signature = inspect.signature(self._model.forward)
        keyword = next(
            (
                name for name in ("logits_to_keep", "num_logits_to_keep")
                if name in signature.parameters
            ),
            None,
        )
        if keyword is None:
            raise UnsupportedConfiguration(
                "this Transformers version's forward() accepts neither "
                "logits_to_keep nor num_logits_to_keep, so a last-position prefill "
                "cannot be requested"
            )
        inputs = self._encode(prompt, 1)
        with torch.no_grad():
            output = self._model(**inputs, **{keyword: 1})
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
        """Time until the first token is decodable, using the library's streamer.

        The timed region is :func:`benchmark.backends.stream_first_token_latency`,
        shared with every other engine measured the same way, so that no row's
        time-to-first-token comes from its own copy of the stopwatch.
        """
        set_seed(seed)
        return stream_first_token_latency(
            self._model, self._tokenizer, self._encode(prompt, 1),
            max_new_tokens=max_new_tokens,
        )

    def generate_mixed(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Any:
        """One batched pass over prompts that genuinely differ in length.

        Left padding with an attention mask, which is what Transformers' own
        batched ``generate`` requires for decoder-only models — and the same
        arrangement Aether uses — so both runtimes are given identical ragged work.

        ``min_new_tokens`` is deliberately *not* pinned here, unlike the uniform
        path: the point of a ragged batch is that rows finish differently, and
        forcing every row to the same length would erase the effect being measured.
        """
        import torch

        tokenizer = self._tokenizer
        previous_side = tokenizer.padding_side
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(list(prompts), return_tensors="pt", padding=True)
        finally:
            tokenizer.padding_side = previous_side
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        padded_length = int(inputs["input_ids"].shape[1])

        sample = temperature > 0.0
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "use_cache": True,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if sample:
            kwargs.update(temperature=temperature, top_p=top_p)
            if top_k > 0:
                kwargs["top_k"] = top_k
        with torch.no_grad():
            output = self._model.generate(**inputs, **kwargs)

        texts: list[str] = []
        completions: list[int] = []
        for index in range(output.shape[0]):
            generated = output[index, padded_length:]
            # Trailing pad is not generated output; counting it would credit the
            # backend with tokens it did not produce.
            kept = [
                int(value) for value in generated.tolist()
                if value != tokenizer.pad_token_id
            ]
            completions.append(len(kept))
            texts.append(tokenizer.decode(kept, skip_special_tokens=True))
        return MixedBatchOutcome(
            texts=texts,
            row_prompt_tokens=[int(value) for value in prompt_lengths],
            row_completion_tokens=completions,
            backend_metrics={
                "padded_length": padded_length,
                "returned_rows": int(output.shape[0]),
            },
        )

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
