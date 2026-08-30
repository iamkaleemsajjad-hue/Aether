"""One prompt format, shared by every executor and every model family.

The bug these pin: an artifact carries its checkpoint's ``chat_template``, but the
packaged tokenizer had no way to *render* it, so every instruction-tuned model in every
family received an untemplated prompt and continued it instead of answering.  The two
backends also had separate copies of the formatting rules, and they had drifted — the
same artifact could be formatted two ways depending on which executor loaded it.

Verified end to end against the real Qwen3-0.6B artifact: raw gives 4 prompt tokens and
an unrelated ramble, templated gives 12 and an actual answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aether.backends import prompt_format
from aether.backends.base import GenerationRequest
from aether.backends.native_cpu_backend import NativeCPUBackend, PackagedTokenizer
from aether.backends.torch_backend import TorchBackend

QWEN_TOKENIZER = Path("benchmark/results/aeg-cache/qwen 0.6B.aeg/tokenizer/tokenizer.json")
NEWLINE = chr(10)


# ── stubs ─────────────────────────────────────────────────────────────────────

class Templated:
    """A checkpoint that ships a template and can render it."""

    chat_template = "<<template>>"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ARG002
        body = "".join(
            f"<|im_start|>{m['role']}{NEWLINE}{m['content']}<|im_end|>{NEWLINE}"
            for m in messages
        )
        return "<s>" + body + ("<|im_start|>assistant" + NEWLINE if add_generation_prompt else "")

    def __call__(self, text, return_tensors=None, add_special_tokens=True):  # noqa: ARG002
        self.calls.append({"text": text, "add_special_tokens": add_special_tokens})
        ids = ([1] if add_special_tokens else []) + [2 + (ord(c) % 90) for c in text[:8]]
        return {"input_ids": np.asarray([ids], dtype=np.int64)}


class NoTemplate(Templated):
    """A base or custom-SFT checkpoint: the template data is simply absent."""

    chat_template = None

    def apply_chat_template(self, *_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("no template is packaged, so none may be applied")


class TemplateWithoutRenderer:
    """The exact broken state: template data present, no renderer.

    This is what ``PackagedTokenizer`` was before the fix, and it is why every
    instruction-tuned artifact silently lost its prompt format.
    """

    chat_template = "<<template>>"

    def __call__(self, text, return_tensors=None, add_special_tokens=True):  # noqa: ARG002
        return {"input_ids": np.asarray([[1, 2, 3]], dtype=np.int64)}


class Overridable(Templated):
    """No packaged template, but a renderer that accepts one from the caller."""

    chat_template = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True,  # noqa: ARG002
                            chat_template=None):
        template = chat_template or self.chat_template
        if not template:
            raise RuntimeError("no template available")
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        return environment.from_string(template).render(
            messages=messages, add_generation_prompt=add_generation_prompt
        )


def chat(prompt: str = "Introduce yourself!") -> GenerationRequest:
    return GenerationRequest(
        model_id="m", prompt=prompt,
        extra={prompt_format.CHAT_TEMPLATE_KEY: True},
    )


def completion(prompt: str = "Introduce yourself!") -> GenerationRequest:
    return GenerationRequest(model_id="m", prompt=prompt)


# ── the checkpoint decides ────────────────────────────────────────────────────

def test_template_data_without_a_renderer_does_not_count_as_declaring_one() -> None:
    """The precise bug: a chat_template string nothing could render."""
    assert prompt_format.declares_chat_template(TemplateWithoutRenderer()) is False
    assert prompt_format.declares_chat_template(Templated()) is True
    assert prompt_format.declares_chat_template(NoTemplate()) is False
    assert prompt_format.declares_chat_template(None) is False


def test_a_completion_is_sent_verbatim() -> None:
    assert prompt_format.render_prompt(completion(), Templated()) == "Introduce yourself!"


def test_a_chat_turn_is_templated() -> None:
    rendered = prompt_format.render_prompt(chat(), Templated())
    assert "<|im_start|>user" in rendered
    assert rendered.endswith("<|im_start|>assistant" + NEWLINE)


def test_a_checkpoint_without_a_template_gets_a_turn_boundary_not_a_bare_string() -> None:
    """Its real format is unrecoverable, but a bare string is what made it free-run."""
    rendered = prompt_format.render_prompt(chat(), NoTemplate())
    assert "user: Introduce yourself!" in rendered
    assert rendered.endswith("assistant:")


def test_a_completion_on_a_templateless_checkpoint_is_still_verbatim() -> None:
    assert prompt_format.render_prompt(completion(), NoTemplate()) == "Introduce yourself!"


def test_messages_are_templated_without_being_asked() -> None:
    request = GenerationRequest(model_id="m", messages=[{"role": "user", "content": "hi"}])
    assert "<|im_start|>user" in prompt_format.render_prompt(request, Templated())


def test_a_template_that_fails_to_render_falls_back_instead_of_raising() -> None:
    """A template is untrusted artifact data; it may be malformed or may reject input."""

    class Hostile(Templated):
        def apply_chat_template(self, *_a, **_k):
            raise RuntimeError("template references an unsupported feature")

    rendered = prompt_format.render_prompt(chat(), Hostile())
    assert rendered.endswith("assistant:")


def test_a_request_with_neither_prompt_nor_messages_is_rejected() -> None:
    with pytest.raises(ValueError, match="prompt or messages"):
        prompt_format.render_prompt(GenerationRequest(model_id="m"), Templated())


# ── no doubled opening token ──────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("build", "tokenizer_factory", "expect_special"),
    [
        (completion, Templated, True),
        (chat, Templated, False),
        (chat, NoTemplate, True),
        (completion, NoTemplate, True),
    ],
)
def test_special_tokens_are_added_exactly_once(build, tokenizer_factory, expect_special) -> None:
    """A rendered template already emits BOS; a doubled BOS silently degrades output."""
    tokenizer = tokenizer_factory()
    request = build()
    text = prompt_format.render_prompt(request, tokenizer)
    prompt_format.encode_prompt(text, request, tokenizer)
    assert tokenizer.calls[-1]["add_special_tokens"] is expect_special


def test_a_tokenizer_that_rejects_the_flag_is_still_usable() -> None:
    class Fussy(Templated):
        def __call__(self, text, return_tensors=None):  # noqa: ARG002 - no flag
            self.calls.append({"text": text, "add_special_tokens": "unsupported"})
            return {"input_ids": np.asarray([[1, 2, 3]], dtype=np.int64)}

    tokenizer = Fussy()
    request = chat()
    encoded = prompt_format.encode_prompt(
        prompt_format.render_prompt(request, tokenizer), request, tokenizer
    )
    assert encoded["input_ids"].shape == (1, 3)


# ── both backends agree ───────────────────────────────────────────────────────

@pytest.mark.parametrize("backend_class", [TorchBackend, NativeCPUBackend])
def test_both_backends_format_a_prompt_identically(backend_class) -> None:
    """These were separate implementations that drifted; one artifact, one format."""
    backend = backend_class.__new__(backend_class)
    tokenizer = Templated()
    for request in (completion(), chat()):
        assert backend._request_text(request, tokenizer) == prompt_format.render_prompt(
            request, tokenizer
        )


@pytest.mark.parametrize("backend_class", [TorchBackend, NativeCPUBackend])
def test_both_backends_encode_a_prompt_identically(backend_class) -> None:
    backend = backend_class.__new__(backend_class)
    tokenizer = Templated()
    request = chat()
    text = prompt_format.render_prompt(request, tokenizer)
    backend._encode_prompt(text, request, tokenizer)
    assert tokenizer.calls[-1]["add_special_tokens"] is False


# ── the packaged tokenizer renders real templates ─────────────────────────────

requires_qwen = pytest.mark.skipif(
    not QWEN_TOKENIZER.is_file(), reason="the Qwen artifact is not present"
)


@requires_qwen
def test_the_packaged_tokenizer_renders_the_checkpoints_own_template() -> None:
    """The real Qwen3 ChatML template, rendered by the framework-free tokenizer."""
    tokenizer = PackagedTokenizer(QWEN_TOKENIZER)
    assert tokenizer.chat_template is not None
    assert prompt_format.declares_chat_template(tokenizer) is True
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Introduce yourself!"}]
    )
    assert rendered == (
        "<|im_start|>user" + NEWLINE + "Introduce yourself!<|im_end|>" + NEWLINE
        + "<|im_start|>assistant" + NEWLINE
    )


@requires_qwen
def test_templating_changes_the_token_sequence_the_model_sees() -> None:
    """4 tokens of bare string against 12 that open with the turn marker."""
    tokenizer = PackagedTokenizer(QWEN_TOKENIZER)
    raw_request, chat_request = completion(), chat()
    raw = prompt_format.encode_prompt(
        prompt_format.render_prompt(raw_request, tokenizer), raw_request, tokenizer
    )["input_ids"][0]
    templated = prompt_format.encode_prompt(
        prompt_format.render_prompt(chat_request, tokenizer), chat_request, tokenizer
    )["input_ids"][0]
    assert len(raw) == 4
    assert len(templated) > len(raw)
    assert int(templated[0]) != int(raw[0]), "the template must open the sequence"


@requires_qwen
def test_a_template_needing_undefined_names_still_renders() -> None:
    """Real templates branch on ``tools``; an undefined name must be falsy, not fatal."""
    tokenizer = PackagedTokenizer(QWEN_TOKENIZER)
    assert "<|im_start|>" in tokenizer.apply_chat_template(
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    )


@requires_qwen
def test_tokenize_true_is_refused_rather_than_silently_ignored() -> None:
    from aether.core.exceptions import BackendError

    tokenizer = PackagedTokenizer(QWEN_TOKENIZER)
    with pytest.raises(BackendError, match="renders text only"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True)


@requires_qwen
def test_the_packaged_tokenizer_honours_add_special_tokens() -> None:
    tokenizer = PackagedTokenizer(QWEN_TOKENIZER)
    with_special = tokenizer("hello", return_tensors="np")["input_ids"][0]
    without = tokenizer("hello", return_tensors="np", add_special_tokens=False)["input_ids"][0]
    assert len(without) <= len(with_special)


def test_a_missing_template_is_reported_not_guessed(tmp_path: Path) -> None:
    """An artifact with no template must say so, so the caller can fall back."""
    from aether.core.exceptions import BackendError

    source = (
        json.loads(QWEN_TOKENIZER.read_text(encoding="utf-8"))
        if QWEN_TOKENIZER.is_file() else None
    )
    if source is None:
        pytest.skip("the Qwen artifact is not present")
    target = tmp_path / "tokenizer.json"
    target.write_text(json.dumps(source), encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    tokenizer = PackagedTokenizer(target)
    assert tokenizer.chat_template is None
    with pytest.raises(BackendError, match="no chat template"):
        tokenizer.apply_chat_template([{"role": "user", "content": "hi"}])

# ── a checkpoint that packages no template ────────────────────────────────────
#
# GPTNeo350M-Instruct-SFT is the case: its turn markers (`<kinrel>`, `<user>`) are
# ordinary BPE text, not added tokens, and its tokenizer_config declares no
# chat_template. Nothing in the artifact records the format, so it cannot be recovered —
# only supplied.


def test_a_missing_template_warns_with_the_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence was the worst part: bad output and nothing saying why.

    The logger is captured directly rather than through ``caplog`` because the project
    logs through structlog, and the assertion is about what Aether chose to say.
    """
    said: list[str] = []
    monkeypatch.setattr(
        prompt_format.logger, "warning",
        lambda message, *_a, **_k: said.append(str(message)),
    )
    prompt_format._WARNED.clear()
    prompt_format.render_prompt(chat(), NoTemplate())
    assert said, "a checkpoint with no template must say so"
    assert "no chat template" in said[0]
    assert "--chat-template" in said[0], "the warning must name the remedy"


def test_the_warning_is_emitted_once_per_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-request warning would become a log flood on a served model."""
    said: list[str] = []
    monkeypatch.setattr(
        prompt_format.logger, "warning",
        lambda message, *_a, **_k: said.append(str(message)),
    )
    prompt_format._WARNED.clear()
    tokenizer = NoTemplate()
    for _ in range(5):
        prompt_format.render_prompt(chat(), tokenizer)
    assert len(said) == 1


def test_an_operator_supplied_template_is_used_verbatim() -> None:
    """The only correct answer when the artifact cannot know: let whoever knows say."""
    template = "{% for m in messages %}<user>{{ m['content'] }}<kinrel>{% endfor %}"
    request = GenerationRequest(
        model_id="m", prompt="Introduce yourself!",
        extra={
            prompt_format.CHAT_TEMPLATE_KEY: True,
            prompt_format.TEMPLATE_OVERRIDE_KEY: template,
        },
    )
    rendered = prompt_format.render_prompt(request, Overridable())
    assert rendered == "<user>Introduce yourself!<kinrel>"


def test_an_override_works_even_when_the_checkpoint_has_its_own_template() -> None:
    """An operator reproducing a specific format must win over the packaged one."""
    template = "{% for m in messages %}<<{{ m['content'] }}>>{% endfor %}"
    request = GenerationRequest(
        model_id="m", prompt="hi",
        extra={
            prompt_format.CHAT_TEMPLATE_KEY: True,
            prompt_format.TEMPLATE_OVERRIDE_KEY: template,
        },
    )
    assert prompt_format.render_prompt(request, Overridable()) == "<<hi>>"


def test_the_override_is_ignored_for_a_raw_completion() -> None:
    request = GenerationRequest(
        model_id="m", prompt="hi",
        extra={prompt_format.TEMPLATE_OVERRIDE_KEY: "{{ 1 }}"},
    )
    assert prompt_format.render_prompt(request, Overridable()) == "hi"


# ── the fallback format declares its own turn boundary ────────────────────────

def test_the_fallback_format_supplies_its_own_stop_sequences() -> None:
    """Aether chose the format, so Aether knows where a turn ends."""
    request = chat()
    prompt_format.augment_stops(request, NoTemplate())
    assert request.stop == [NEWLINE + "user:", NEWLINE + "assistant:"]


def test_a_checkpoint_with_its_own_template_gets_no_invented_stops() -> None:
    """It declares its own stop tokens and the engine already receives them."""
    request = chat()
    prompt_format.augment_stops(request, Templated())
    assert not request.stop


def test_a_raw_completion_gets_no_invented_stops() -> None:
    request = completion()
    prompt_format.augment_stops(request, NoTemplate())
    assert not request.stop


def test_caller_supplied_stops_are_kept() -> None:
    request = chat()
    request.stop = ["</done>"]
    prompt_format.augment_stops(request, NoTemplate())
    assert request.stop[0] == "</done>"
    assert NEWLINE + "user:" in request.stop


def test_augmenting_twice_does_not_duplicate() -> None:
    request = chat()
    prompt_format.augment_stops(request, NoTemplate())
    prompt_format.augment_stops(request, NoTemplate())
    assert len(request.stop) == len(set(request.stop))


@pytest.mark.parametrize("backend_class", [TorchBackend, NativeCPUBackend])
def test_both_backends_derive_the_same_stops(backend_class) -> None:
    backend = backend_class.__new__(backend_class)
    request = chat()
    backend._augment_stops(request, NoTemplate())
    assert request.stop == [NEWLINE + "user:", NEWLINE + "assistant:"]


@requires_qwen
def test_an_override_also_works_through_the_packaged_tokenizer() -> None:
    tokenizer = PackagedTokenizer(QWEN_TOKENIZER)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        chat_template="{% for m in messages %}[[{{ m['content'] }}]]{% endfor %}",
    )
    assert rendered == "[[hi]]"
