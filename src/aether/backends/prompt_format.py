"""How a request becomes the exact token sequence a checkpoint was trained on.

This lived twice — once in the PyTorch backend, once in the native CPU backend — and
the copies had drifted.  One of them applied a chat template, the other did not, so the
same artifact answered differently depending on which executor happened to load it.
That is why it is one module now: prompt formatting is a property of the *checkpoint*,
not of the executor that runs it.

Three rules, and each of them is the difference between an answer and fluent nonsense.

**The checkpoint decides whether it wants a template.**  An instruction-tuned artifact
packages a ``chat_template``; a base artifact does not.  That single signal generalises
to every family Aether supports — Qwen's ChatML, Llama's header format, Gemma's turn
markers, Mistral's ``[INST]``, and whatever a future checkpoint ships — where a list of
model names never could.

**A chat turn is templated; a completion is not.**  Given ``"Introduce yourself!"``
with no turn delimiters, an instruction-tuned model does not answer the question, it
*continues the string*, because that is what untemplated text meant during training.
But a completion API must send exactly the bytes it was handed, and a base model has no
template to apply.  So chat messages are always rendered, a bare prompt is rendered
only when the caller asks, and a checkpoint without a template is always sent verbatim.

**A rendered template must not be given a second opening token.**  Templates emit the
checkpoint's own BOS.  Letting the tokenizer add another shifts every position by one
against training, and the only symptom is a worse answer — there is no error to see.
"""

from __future__ import annotations

from typing import Any

from aether.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CHAT_TEMPLATE_KEY",
    "TEMPLATE_OVERRIDE_KEY",
    "augment_stops",
    "can_render",
    "declares_chat_template",
    "encode_prompt",
    "render_prompt",
    "stop_sequences",
    "wants_chat_turn",
]

CHAT_TEMPLATE_KEY = "apply_chat_template"
"""Request key that turns a bare ``prompt`` into a templated chat turn.

Absent means "raw completion", which is what a completion API has to stay. The
interactive surfaces set it; the benchmark and the completion API do not, so their
byte-for-byte comparison is unaffected."""

TEMPLATE_OVERRIDE_KEY = "chat_template"
"""Request key carrying a Jinja chat template supplied by the operator.

Some checkpoints are instruction-tuned but package no template — their fine-tuning
markers exist only in the weights and in the dataset card, not in any artifact
metadata.  Nothing can recover the format from the artifact, so the only correct answer
is to let whoever *does* know supply it.  Every serious runtime provides this lever for
the same reason; the alternative is a table of model names, which is exactly the kind of
per-family special case this project rejects."""

#: Role prefixes used by the neutral fallback transcript.  Kept as data because the
#: fallback's own turn marker doubles as its stop sequence: if Aether supplied the
#: format, Aether knows where a turn ends.
_FALLBACK_USER = "user:"
_FALLBACK_ASSISTANT = "assistant:"

#: Artifacts already warned about, so a per-request warning does not become a log flood.
_WARNED: "set[int]" = set()


def _warn_missing_template_once(tokenizer: Any) -> None:
    """Say once, per tokenizer, that instruction following will be unreliable.

    This is the part that was missing when a checkpoint packaged no template: the output
    was poor and nothing said why.  A warning that names the remedy turns an inexplicable
    answer into an actionable one.
    """
    key = id(tokenizer)
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(
        "this artifact packages no chat template, so its instruction format is not "
        "recoverable from the checkpoint. A neutral 'user:/assistant:' transcript is "
        "being used; an instruction-tuned model that was trained on different markers "
        "may ignore it and continue the text instead of answering. Supply the real "
        "format with --chat-template (or chat_template=...) if you know it."
    )


def _override(request: Any) -> str | None:
    """The operator-supplied template for this request, if any."""
    value = (getattr(request, "extra", None) or {}).get(TEMPLATE_OVERRIDE_KEY)
    return value if isinstance(value, str) and value.strip() else None


def declares_chat_template(tokenizer: Any) -> bool:
    """Whether this checkpoint was trained to see a turn-delimited prompt.

    Both halves are required.  A ``chat_template`` string with no renderer is what the
    packaged tokenizer used to have, and it silently disabled templating for every
    artifact — the data was present and nothing could use it.
    """
    if tokenizer is None:
        return False
    if getattr(tokenizer, "chat_template", None) is None:
        return False
    return callable(getattr(tokenizer, "apply_chat_template", None))


def can_render(request: Any, tokenizer: Any) -> bool:
    """Whether a template is available at all — the checkpoint's or the operator's."""
    if tokenizer is None or not callable(
        getattr(tokenizer, "apply_chat_template", None)
    ):
        return False
    return _override(request) is not None or declares_chat_template(tokenizer)


def wants_chat_turn(request: Any, tokenizer: Any) -> bool:
    """Whether this request will be rendered through a template.

    Used by :func:`encode_prompt` to decide whether the opening token was already
    supplied, so rendering and encoding cannot disagree about it.
    """
    if not can_render(request, tokenizer):
        return False
    if getattr(request, "messages", None) is not None:
        return True
    return bool((getattr(request, "extra", None) or {}).get(CHAT_TEMPLATE_KEY))


def _render(
    tokenizer: Any, messages: "list[dict[str, Any]]", template: str | None = None
) -> str | None:
    """Render through a chat template, or ``None`` if it cannot be rendered.

    A template comes out of an artifact or from an operator, so it is untrusted input
    and may be malformed, may reference a feature the renderer lacks, or may reject the
    messages it was given.  None of those may take down a request: the caller falls back
    to a legible transcript and the reason is logged once, where an operator can act on
    it.
    """
    try:
        keywords: "dict[str, Any]" = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if template is not None:
            keywords["chat_template"] = template
        return str(tokenizer.apply_chat_template(messages, **keywords))
    except Exception as exc:  # noqa: BLE001 - artifact data must not break a request
        logger.warning(
            "the chat template could not be rendered (%s); falling back to a "
            "role-delimited transcript, which an instruction-tuned model may follow "
            "less closely",
            exc,
        )
        return None


def _transcript(messages: "list[dict[str, Any]]") -> str:
    """A neutral, legible turn format for a checkpoint that ships no template.

    Not a guess at the checkpoint's real format — there is no way to recover that — but
    it does mark turn boundaries and hand the model the assistant's turn, which is
    strictly more signal than concatenated content.
    """
    body = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in messages
    )
    return body + "\n" + _FALLBACK_ASSISTANT


def render_prompt(request: Any, tokenizer: Any | None = None) -> str:
    """Return the text to encode for this request."""
    template = _override(request)
    messages = getattr(request, "messages", None)
    if messages is not None:
        if can_render(request, tokenizer):
            rendered = _render(tokenizer, messages, template)
            if rendered is not None:
                return rendered
        elif tokenizer is not None:
            _warn_missing_template_once(tokenizer)
        return _transcript(messages)
    prompt = getattr(request, "prompt", None)
    if prompt is None:
        raise ValueError("either prompt or messages must be provided")
    if not (getattr(request, "extra", None) or {}).get(CHAT_TEMPLATE_KEY):
        return prompt
    if can_render(request, tokenizer):
        rendered = _render(tokenizer, [{"role": "user", "content": prompt}], template)
        if rendered is not None:
            return rendered
    elif tokenizer is not None:
        _warn_missing_template_once(tokenizer)
    # No template is available, or the available one would not render. The checkpoint's
    # real fine-tuning format is not recoverable from the artifact — this model's markers
    # are ordinary text, not added tokens, so nothing in the metadata records them — and
    # inventing one would be a guess dressed as a fact. The neutral transcript at least
    # marks a turn boundary and hands over the assistant's turn; a bare string is what
    # made these models free-run.
    return _transcript([{"role": "user", "content": prompt}])


def stop_sequences(request: Any, tokenizer: Any | None = None) -> "list[str]":
    """Stop strings implied by the prompt format that was actually used.

    This is the part that keeps an unstoppable model from burning the whole token
    budget.  When a checkpoint declares its own template it also declares its own stop
    tokens, and the engine already receives those.  When *Aether* supplied the format,
    Aether knows where a turn ends: the fallback transcript's next-turn marker.  So the
    stop sequence is derived from the format in force rather than guessed, and it is
    added only on the path that needs it.
    """
    if getattr(request, "prompt", None) is None and getattr(request, "messages", None) is None:
        return []
    asked_for_chat = bool(
        getattr(request, "messages", None) is not None
        or (getattr(request, "extra", None) or {}).get(CHAT_TEMPLATE_KEY)
    )
    if not asked_for_chat or can_render(request, tokenizer):
        return []
    # The fallback format is ours, so its turn boundary is known exactly.
    return ["\n" + _FALLBACK_USER, "\n" + _FALLBACK_ASSISTANT]


def augment_stops(request: Any, tokenizer: Any | None = None) -> None:
    """Add the format's own turn boundary to ``request.stop``, in place.

    Only for the fallback format, and only additively.  A checkpoint that declares a
    template also declares its stop tokens and the engine already receives those; the
    fallback format is Aether's, so its boundary is Aether's to declare.  Without this an
    instruction-tuned model with an unrecoverable format never reaches a turn end and
    spends the entire token budget writing a fresh conversation with itself — which is
    exactly what a 1024-token run of GPTNeo-SFT produced.
    """
    derived = stop_sequences(request, tokenizer)
    if not derived:
        return
    existing = list(getattr(request, "stop", None) or [])
    for sequence in derived:
        if sequence not in existing:
            existing.append(sequence)
    request.stop = existing


def encode_prompt(
    text: str, request: Any, tokenizer: Any, *, return_tensors: str = "np"
) -> Any:
    """Tokenize ``text``, adding special tokens only when nothing else did.

    Whether the template already supplied the opening token is decided by the same
    predicate that decided to render it, so the two can never disagree.
    """
    if wants_chat_turn(request, tokenizer):
        try:
            return tokenizer(
                text, return_tensors=return_tensors, add_special_tokens=False
            )
        except TypeError:
            # A tokenizer that does not accept the flag never added one either.
            return tokenizer(text, return_tensors=return_tensors)
    return tokenizer(text, return_tensors=return_tensors)
