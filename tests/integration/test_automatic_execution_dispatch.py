"""Actual CPU source execution after a simulated GPU-driver-only discovery.

The discovery fixture exercises routing only; GPU execution is never simulated.
"""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("family", ["llama", "gpt2"])
def test_raw_local_sources_fall_back_to_real_cpu_and_match_transformers(tmp_path, monkeypatch, family):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    pytest.importorskip("accelerate")
    from aether import Runtime, RuntimeConfig

    source = tmp_path / family
    source.mkdir()
    torch.manual_seed(11)
    if family == "llama":
        config = transformers.LlamaConfig(
            vocab_size=16, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            max_position_embeddings=32, bos_token_id=0, eos_token_id=None, pad_token_id=0,
        )
        reference = transformers.LlamaForCausalLM(config)
    else:
        config = transformers.GPT2Config(
            vocab_size=16, n_embd=16, n_layer=1, n_head=2, n_positions=32,
            bos_token_id=0, eos_token_id=None, pad_token_id=0,
        )
        reference = transformers.GPT2LMHeadModel(config)
    reference.eval().save_pretrained(source)
    vocabulary = {"<unk>": 0, "hello": 1, "world": 2}
    vocabulary.update({f"token{index}": index for index in range(3, 16)})
    tokenizer_impl = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer_impl.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl, unk_token="<unk>", pad_token="<unk>",
        model_input_names=["input_ids", "attention_mask"],
    )
    tokenizer.save_pretrained(source)
    discovery = SimpleNamespace(
        target_id="cuda_sm89", attributes={"available_targets": ["cuda_sm89", "cpu_avx2"]},
        capabilities=[{"target_id": "cuda_sm89", "available": True, "execution_tested": False}],
    )
    monkeypatch.setattr("aether.runtime.runtime.HardwareDetector.detect", lambda _: discovery)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("AETHER_EXECUTION_DEVICES", raising=False)
    runtime = Runtime(RuntimeConfig(
        backend_name="pytorch", hf_offline=True, speculative_decoding=False,
        model_cache_dir=str(tmp_path / "cache"),
        enable_semantic_cache=False, enable_prefix_caching=False,
    ))
    result = runtime.generate(str(source), "hello world", max_tokens=2, temperature=0.0)
    prompt = tokenizer("hello world", return_tensors="pt")
    with torch.inference_mode():
        expected_ids = reference.generate(**prompt, max_new_tokens=2, do_sample=False)[0, 2:]
    assert result.text == tokenizer.decode(expected_ids, skip_special_tokens=True)
    assert result.metrics.kernel_target == "cpu_avx2"
    loaded = runtime._loaded_models[str(source)]
    assert {parameter.device.type for parameter in loaded.parameters()} == {"cpu"}
    assert discovery.capabilities[0]["execution_tested"] is False
    backend = runtime.backend_registry.get_backend("pytorch")
    # Simulate a later model changing load-time placement. Inference still uses
    # this model's real CPU tensors; no GPU execution is simulated or asserted.
    backend._device = "cuda:99"
    backend._devices = ["cuda:99"]
    backend._runtime_family = "cuda"
    repeated = runtime.generate(str(source), "hello world", max_tokens=2, temperature=0.0)
    assert repeated.text == result.text
    assert repeated.metrics.extra["device"] == "cpu"
    streamed = "".join(runtime.generate_stream(str(source), "hello world", max_tokens=2, temperature=0.0))
    assert streamed == result.text
    assert backend.load_model(str(source), execution_devices=["cpu"]) is loaded
    backend.unload_model(str(source))

    if family == "llama":
        from aether import Compiler, CompilerConfig
        from aether.backends.base import GenerationRequest

        artifact = tmp_path / "compiled.aeg"
        Compiler(CompilerConfig(
            targets=["cpu_avx2"], calibration_tokens=16, cache_dir=str(tmp_path / "compiler-cache"),
        )).compile(str(source), output_path=artifact)
        handle = backend.load_model("compiled-cpu", str(artifact), execution_devices=["cpu"])
        request = GenerationRequest(model_id="compiled-cpu", prompt="hello world", max_tokens=2, temperature=0.0)
        baseline = backend.generate(request)
        backend._device = "cuda:99"
        backend._devices = ["cuda:99"]
        backend._runtime_family = "cuda"
        compiled_result = backend.generate(request)
        assert compiled_result.text == baseline.text
        assert compiled_result.metrics["device"] == "cpu"
        assert compiled_result.metrics["runtime_family"] == "cpu"
        assert backend.load_model("compiled-cpu", str(artifact), execution_devices=["cpu"]) is handle
        backend.unload_model("compiled-cpu")
