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


# ── prompt delivery: the CLI default ─────────────────────────────────────────
#
# The formatting policy itself is covered once, for both backends and against the real
# packaged Qwen template, in ``test_prompt_format.py``. Asserting it twice is how the
# two backend implementations drifted apart in the first place.


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
