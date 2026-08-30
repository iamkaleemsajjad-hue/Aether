"""The load path must survive every executor, and a prompt must reach the model.

Two failures are pinned here, both found on a real multi-GPU load.

**Inherited state.**  ``TorchTensorParallelAEGEngine`` deliberately does not call
``super().__init__`` — doing so would upload the unsharded weights to one device
before splitting them.  Every attribute the parent assigns only in its constructor is
therefore absent on that subclass, and an *inherited* method that reads one raises
``AttributeError`` during model load.  That is how a Phi-3.5 load died on
``host_weights_released``, and every attribute added to the parent since widened the
gap silently.  The parent now declares class-level defaults; these tests keep it that
way and check the property generally rather than one attribute at a time.

**Prompt delivery.**  An instruction-tuned checkpoint given a bare string does not
answer it, it continues it — fluently and irrelevantly.  The checkpoint says which it
is by packaging a chat template, so that is what decides, for every family rather than
for a list of names.
"""

from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aether.backends.base import GenerationRequest  # noqa: E402
from aether.backends.torch_backend import TorchBackend  # noqa: E402
from aether.runtime.cpu_engine import (  # noqa: E402
    CPUExecutionEngine, LayerWeights, ModelWeights,
)
from aether.runtime.torch_engine import TorchAEGEngine  # noqa: E402
from aether.runtime.torch_tensor_parallel import (  # noqa: E402
    TorchTensorParallelAEGEngine,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Methods the backend calls on whatever engine a load produced, before any request.
#: An executor that cannot answer these cannot be loaded at all.
LOAD_PATH_METHODS = (
    "device_tensors_alias_host",
    "release_host_weights",
    "projection_report",
)


def weights(hidden: int = 64, heads: int = 4, kv_heads: int = 2,
            inter: int = 128, vocab: int = 128, layers: int = 2) -> ModelWeights:
    rng = np.random.default_rng(0)
    head_dim = hidden // heads

    def w(out: int, inp: int) -> np.ndarray:
        return rng.standard_normal((out, inp)).astype(np.float32) * 0.05

    stack = [
        LayerWeights(
            attention_norm=np.ones(hidden, dtype=np.float32),
            q_proj=w(heads * head_dim, hidden), k_proj=w(kv_heads * head_dim, hidden),
            v_proj=w(kv_heads * head_dim, hidden), o_proj=w(hidden, heads * head_dim),
            ffn_norm=np.ones(hidden, dtype=np.float32),
            gate_proj=w(inter, hidden), up_proj=w(inter, hidden),
            down_proj=w(hidden, inter),
        )
        for _ in range(layers)
    ]
    return ModelWeights(
        embedding=w(vocab, hidden), layers=stack,
        final_norm=np.ones(hidden, dtype=np.float32), lm_head=w(vocab, hidden),
        context_length=512,
    )


def cpu_engine() -> CPUExecutionEngine:
    return CPUExecutionEngine(weights(), num_heads=4, num_kv_heads=2)


def tensor_parallel_engine() -> TorchTensorParallelAEGEngine:
    """A two-way sharded executor on one CPU — the structure, without the hardware."""
    return TorchTensorParallelAEGEngine(cpu_engine(), ["cpu", "cpu"])


# ── inherited state ───────────────────────────────────────────────────────────

def subclasses_skipping_super() -> "list[str]":
    """Engine subclasses whose ``__init__`` never calls the parent's.

    Found by reading the tree rather than hard-coded, so a new executor written the
    same way is covered the day it lands.
    """
    found: list[str] = []
    for path in sorted((REPO_ROOT / "src" / "aether" / "runtime").glob("torch*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.id if isinstance(base, ast.Name) else getattr(base, "attr", "")
                for base in node.bases
            }
            if "TorchAEGEngine" not in bases:
                continue
            init = next(
                (f for f in node.body
                 if isinstance(f, ast.FunctionDef) and f.name == "__init__"),
                None,
            )
            if init is None:
                continue
            calls_super = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "__init__"
                for n in ast.walk(init)
            )
            if not calls_super:
                found.append(node.name)
    return found


def test_the_subclass_that_skips_the_parent_constructor_is_still_covered() -> None:
    """If this list changes, the guarantees below need to cover the newcomer too."""
    assert subclasses_skipping_super() == ["TorchTensorParallelAEGEngine"]


def constructor_assigned(cls_name: str, path: pathlib.Path) -> "set[str]":
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls_name):
            continue
        init = next(
            (f for f in node.body
             if isinstance(f, ast.FunctionDef) and f.name == "__init__"),
            None,
        )
        if init is None:
            continue
        for statement in ast.walk(init):
            targets: list[ast.expr] = []
            if isinstance(statement, ast.Assign):
                targets = list(statement.targets)
            elif isinstance(statement, ast.AnnAssign):
                targets = [statement.target]
            stack = list(targets)
            while stack:
                current = stack.pop()
                if isinstance(current, (ast.Tuple, ast.List)):
                    stack.extend(current.elts)
                elif (
                    isinstance(current, ast.Attribute)
                    and isinstance(current.value, ast.Name)
                    and current.value.id == "self"
                ):
                    names.add(current.attr)
    return names


def test_every_attribute_the_parent_constructor_adds_is_resolvable_on_the_subclass() -> None:
    """The general form of the crash: a parent attribute a subclass never assigns.

    Anything the parent sets only in ``__init__`` must also exist as a class-level
    default, or the subclass must set it. Otherwise an inherited method reading it
    fails at load time — which is exactly what happened, and what would happen again
    the next time the parent grows a field.
    """
    parent = constructor_assigned(
        "TorchAEGEngine", REPO_ROOT / "src" / "aether" / "runtime" / "torch_engine.py"
    )
    subclass = constructor_assigned(
        "TorchTensorParallelAEGEngine",
        REPO_ROOT / "src" / "aether" / "runtime" / "torch_tensor_parallel.py",
    )
    unresolvable = sorted(
        name for name in parent - subclass
        if not hasattr(TorchAEGEngine, name)
    )
    assert not unresolvable, (
        "TorchTensorParallelAEGEngine skips the parent constructor, so these "
        "attributes are unreachable on it. Give TorchAEGEngine a class-level default "
        f"for each: {unresolvable}"
    )


def test_the_sharded_executor_answers_every_load_path_method() -> None:
    """The load sequence the backend runs, against the executor that used to crash."""
    engine = tensor_parallel_engine()
    for name in LOAD_PATH_METHODS:
        method = getattr(engine, name, None)
        assert callable(method), f"{name} is missing on the sharded executor"
        method()


def test_the_sharded_executor_reports_host_weights_not_yet_released() -> None:
    engine = tensor_parallel_engine()
    assert engine.host_weights_released is False
    engine.release_host_weights()
    assert engine.host_weights_released is True


def test_a_sharded_embedding_is_not_mistaken_for_an_aliasing_tensor() -> None:
    """``self.embedding`` is a list of shards here, and a list has no data pointer."""
    engine = tensor_parallel_engine()
    assert isinstance(engine.embedding, list)
    assert engine.device_tensors_alias_host() is False


def test_the_sharded_executor_still_runs_after_the_load_sequence() -> None:
    """Releasing host weights must not disturb execution on the sharded path."""
    engine = tensor_parallel_engine()
    engine.release_host_weights()
    ids = np.arange(6, dtype=np.int64) % 128
    logits, _ = engine.forward(ids)
    assert np.isfinite(logits).all()
    produced = engine.generate(ids, max_tokens=4, temperature=0.0)
    assert len(list(produced)) == 4


def test_the_projection_falls_back_to_the_reference_without_a_calibration() -> None:
    """A subclass that never built a calibrator must still be able to project."""
    engine = tensor_parallel_engine()
    assert engine._projection is None
    x = torch.randn(2, 8)
    weight = torch.randn(4, 8)
    produced = engine._matmul(x, weight)
    assert torch.allclose(produced, torch.nn.functional.linear(x, weight), atol=1e-5)

# ── prompt delivery ───────────────────────────────────────────────────────────


class ChatTokenizer:
    """Records what it was asked to encode, and renders a ChatML-style template."""

    chat_template = "{% for m in messages %}...{% endfor %}"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):  # noqa: ARG002
        newline = chr(10)
        body = "".join(
            f"<|im_start|>{m['role']}{newline}{m['content']}<|im_end|>{newline}"
            for m in messages
        )
        tail = f"<|im_start|>assistant{newline}" if add_generation_prompt else ""
        # Real templates emit the checkpoint's opening token themselves.
        return "<s>" + body + tail

    def __call__(self, text, return_tensors=None, add_special_tokens=True):  # noqa: ARG002
        self.calls.append(
            {"text": text, "add_special_tokens": add_special_tokens}
        )
        ids = [1] if add_special_tokens else []
        ids += [2 + (ord(ch) % 90) for ch in text[:8]]
        return {"input_ids": np.asarray([ids], dtype=np.int64)}


class BaseTokenizer(ChatTokenizer):
    """A base checkpoint: no template, so nothing to apply."""

    chat_template = None

    def apply_chat_template(self, *_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("a base checkpoint has no chat template to apply")


def backend() -> TorchBackend:
    return TorchBackend.__new__(TorchBackend)


def test_a_completion_request_is_sent_verbatim() -> None:
    """A completion API must deliver exactly the bytes it was given."""
    request = GenerationRequest(model_id="m", prompt="Introduce yourself!")
    assert backend()._request_text(request, ChatTokenizer()) == "Introduce yourself!"


def test_a_bare_prompt_becomes_a_chat_turn_when_the_caller_asks() -> None:
    """This is the fix: an instruct model must see its own turn delimiters."""
    request = GenerationRequest(
        model_id="m", prompt="Introduce yourself!",
        extra={"apply_chat_template": True},
    )
    rendered = backend()._request_text(request, ChatTokenizer())
    assert "<|im_start|>user" in rendered
    assert "Introduce yourself!" in rendered
    assert rendered.endswith("<|im_start|>assistant" + chr(10)), (
        "the model must be handed the assistant turn to continue"
    )


def test_a_base_checkpoint_is_never_templated() -> None:
    """The artifact decides. No template declared means none applied."""
    request = GenerationRequest(
        model_id="m", prompt="Introduce yourself!",
        extra={"apply_chat_template": True},
    )
    assert backend()._request_text(request, BaseTokenizer()) == "Introduce yourself!"


def test_chat_messages_are_templated_without_being_asked() -> None:
    request = GenerationRequest(
        model_id="m", messages=[{"role": "user", "content": "hi"}]
    )
    assert "<|im_start|>user" in backend()._request_text(request, ChatTokenizer())


def test_messages_without_a_template_still_get_role_delimiters() -> None:
    """A base checkpoint asked to chat gets a legible transcript, not raw content."""
    request = GenerationRequest(
        model_id="m", messages=[{"role": "user", "content": "hi"}]
    )
    rendered = backend()._request_text(request, BaseTokenizer())
    assert "user: hi" in rendered
    assert rendered.endswith("assistant:")


@pytest.mark.parametrize(
    ("request_kwargs", "expect_special"),
    [
        ({"prompt": "hi"}, True),
        ({"prompt": "hi", "extra": {"apply_chat_template": True}}, False),
        ({"messages": [{"role": "user", "content": "hi"}]}, False),
    ],
)
def test_a_rendered_template_is_not_given_a_second_opening_token(
    request_kwargs: dict, expect_special: bool
) -> None:
    """A doubled BOS shifts every position against training and is silent.

    The template already emits the opening token, so the tokenizer must not add one.
    Raw completions keep it, because there nothing else supplied it.
    """
    tokenizer = ChatTokenizer()
    request = GenerationRequest(model_id="m", **request_kwargs)
    text = backend()._request_text(request, tokenizer)
    backend()._encode_prompt(text, request, tokenizer)
    assert tokenizer.calls[-1]["add_special_tokens"] is expect_special


def test_a_tokenizer_that_rejects_the_flag_is_still_usable() -> None:
    """Not every tokenizer accepts add_special_tokens; none may crash a request."""

    class Fussy(ChatTokenizer):
        def __call__(self, text, return_tensors=None):  # noqa: ARG002 - no flag
            self.calls.append({"text": text, "add_special_tokens": "unsupported"})
            return {"input_ids": np.asarray([[1, 2, 3]], dtype=np.int64)}

    tokenizer = Fussy()
    request = GenerationRequest(
        model_id="m", prompt="hi", extra={"apply_chat_template": True}
    )
    encoded = backend()._encode_prompt(
        backend()._request_text(request, tokenizer), request, tokenizer
    )
    assert encoded["input_ids"].shape == (1, 3)


def test_a_missing_tokenizer_never_raises() -> None:
    request = GenerationRequest(
        model_id="m", prompt="hi", extra={"apply_chat_template": True}
    )
    assert backend()._request_text(request, None) == "hi"


def test_the_run_command_asks_for_a_chat_turn_by_default() -> None:
    """The CLI is where a human types a question, so that is where it defaults on."""
    import inspect

    from aether import cli

    # `run` is a click Command; the function it wraps is what carries the source.
    source = inspect.getsource(cli.run.callback)
    assert "apply_chat_template=not raw" in source
    assert any(
        parameter.name == "raw" for parameter in cli.run.params
    ), "the escape hatch must exist for base models and raw continuation"
