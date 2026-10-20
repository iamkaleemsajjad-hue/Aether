"""Cold-cache compilation and staged merge publication in a real fresh process."""

import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest


def test_cold_compile_then_runtime_merge_is_verified_and_executable(tmp_path):
    """Fresh compiler output can be merged and run without exposing partial output."""
    project = Path(__file__).resolve().parents[2]
    script = r"""
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from aether import Compiler, CompilerConfig, Runtime, RuntimeConfig
from aether.backends.hardware_validation import _write_probe_checkpoint
from aether.core.aeg_format import AEGPackage

root = Path(sys.argv[1])
cache = root / "cold-native-cache"
cache_initially_absent = not cache.exists()
assert cache_initially_absent
os.environ["AETHER_CACHE_DIR"] = str(cache)
source = root / "source"
_write_probe_checkpoint(source)
config = CompilerConfig(quality_mode="diagnostic", targets=["cpu_generic"],
                        precision="FP32", optimization_level=0,
                        cache_dir=str(root / "compiler-cache"))
for item in fields(config):
    if item.name.startswith("enable_"):
        setattr(config, item.name, item.name == "enable_precision_assignment")
base = Compiler(config).compile(str(source), output_path=root / "base.aeg")
weights = base.weight_store().dequantize_all()
task = root / "task.safetensors"
save_file({"lm_head": weights["lm_head"] + np.float32(.03)}, str(task))
with Runtime(RuntimeConfig(allow_uncertified=True, backend_name="aether_cpu",
                           execution_devices=["cpu"],
                           speculative_decoding=False, dynamic_precision=False,
                           enable_semantic_cache=False,
                           model_cache_dir=str(root / "models"),
                           kv_cache_path=str(root / "sessions"))) as runtime:
    merged = runtime.merge(str(base.root),
                           [{"name": "changed", "path": str(task), "coefficient": 1.0}],
                           method="task_arithmetic")
    package = AEGPackage(merged["output_model"]).load()
    package.verify_integrity()
    result = runtime.generate(str(package.root), "hello world", max_tokens=1, temperature=0)
    report = {"cache_initially_absent": cache_initially_absent,
              "tensor_count": len(package.weight_store().entries),
              "generated_tokens": result.usage["completion_tokens"],
              "native_payloads": [str(path.relative_to(package.root))
                                  for path in package.root.rglob("*")
                                  if path.suffix.lower() in {".dll", ".so", ".dylib"}],
              "output": str(package.root)}
(root / "publication-evidence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
"""
    env = dict(os.environ)
    env.update(
        PYTHONIOENCODING="utf-8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=project,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    (tmp_path / "cold-compile-merge.log").write_text(completed.stdout, encoding="utf-8")
    assert completed.returncode == 0, completed.stdout
    evidence = json.loads((tmp_path / "publication-evidence.json").read_text(encoding="utf-8"))
    assert evidence["cache_initially_absent"]
    assert evidence["tensor_count"] == 12
    assert evidence["generated_tokens"] == 1
    assert Path(evidence["output"]).is_dir()


def test_public_aeg_merge_rejects_existing_dangling_output_link(tmp_path):
    """Resolving an output link must not create an artifact at its missing target."""
    from aether import Compiler, CompilerConfig
    from aether.backends.hardware_validation import _write_probe_checkpoint
    from aether.compiler.aeg_merging import merge_aeg_packages

    source = tmp_path / "source"
    _write_probe_checkpoint(source)
    config = CompilerConfig(
        quality_mode="diagnostic",
        targets=["cpu_generic"],
        precision="FP32",
        optimization_level=0,
        cache_dir=str(tmp_path / "compiler-cache"),
    )
    for item in fields(config):
        if item.name.startswith("enable_"):
            setattr(config, item.name, item.name == "enable_precision_assignment")
    base = Compiler(config).compile(str(source), output_path=tmp_path / "base.aeg")
    output, target = tmp_path / "output.aeg", tmp_path / "absent-target.aeg"
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"OS does not permit unprivileged symlink creation: {exc}")
    with pytest.raises(FileExistsError, match="already exists"):
        merge_aeg_packages([base.root, base.root], output)
    assert output.is_symlink()
    assert os.path.lexists(output)
    assert not target.exists()
