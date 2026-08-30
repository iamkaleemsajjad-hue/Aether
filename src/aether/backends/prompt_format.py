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
    "declares_chat_template",
    "render_prompt",
    "encode_prompt",
    "wants_chat_turn",
]

CHAT_TEMPLATE_KEY = "apply_chat_template"
"""Request key that turns a bare ``prompt`` into a templated chat turn.

Absent means "raw completion", which is what a completion API has to stay. The
interactive surfaces set it; the benchmark and the completion API do not, so their
byte-for-byte comparison is unaffected."""


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


def wants_chat_turn(request: Any, tokenizer: Any) -> bool:
    """Whether this request should be rendered through the checkpoint's template."""
    if not declares_chat_template(tokenizer):
        return False
    if getattr(request, "messages", None) is not None:
        return True
    return bool((getattr(request, "extra", None) or {}).get(CHAT_TEMPLATE_KEY))


def _render(tokenizer: Any, messages: "list[dict[str, Any]]") -> str | None:
    """Render through the checkpoint's template, or ``None`` if it cannot be rendered.

    A template comes out of an artifact, so it is untrusted input and may be malformed,
    may reference a feature the renderer lacks, or may reject the messages it was given.
    None of those may take down a request: the caller falls back to a legible transcript
    and the reason is logged once, where an operator can act on it.
    """
    try:
        return str(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    except Exception as exc:  # noqa: BLE001 - artifact data must not break a request
        logger.warning(
            "the packaged chat template could not be rendered (%s); falling back to a "
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
    return body + "\nassistant:"


def render_prompt(request: Any, tokenizer: Any | None = None) -> str:
    """Return the text to encode for this request."""
    messages = getattr(request, "messages", None)
    if messages is not None:
        if declares_chat_template(tokenizer):
            rendered = _render(tokenizer, messages)
            if rendered is not None:
                return rendered
        return _transcript(messages)
    prompt = getattr(request, "prompt", None)
    if prompt is None:
        raise ValueError("either prompt or messages must be provided")
    if not (getattr(request, "extra", None) or {}).get(CHAT_TEMPLATE_KEY):
        return prompt
    if declares_chat_template(tokenizer):
        rendered = _render(tokenizer, [{"role": "user", "content": prompt}])
        if rendered is not None:
            return rendered
    # Either no template is packaged, or the packaged one would not render. The
    # checkpoint's real fine-tuning format is not recoverable — an SFT checkpoint may
    # have used any markers at all — so the neutral transcript is used rather than
    # guessing one. It is not the trained format, but it marks a turn boundary and hands
    # over the assistant's turn, and a bare string is what made these models free-run.
    return _transcript([{"role": "user", "content": prompt}])


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
