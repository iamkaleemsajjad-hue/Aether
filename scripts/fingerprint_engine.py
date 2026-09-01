"""Capture a deterministic fingerprint of an AEG engine's arithmetic.

Written to a JSON file so the same script can be run before and after a change and
the two compared exactly. Greedy decode plus raw prefill logits together cover both
"does it pick the same tokens" and "does it compute the same numbers", which is what
"unchanged response quality" has to mean for a memory change.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["AETHER_PLAN_BOOTSTRAP"] = os.environ.get("AETHER_PLAN_BOOTSTRAP", "0")
os.environ["AETHER_TORCH_DTYPE"] = os.environ.get("AETHER_TORCH_DTYPE", "fp16")
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from aether.runtime.aeg_loader import load_engine_from_path  # noqa: E402
from aether.runtime.torch_engine import TorchAEGEngine  # noqa: E402

AEG = Path(sys.argv[1])
OUT = Path(sys.argv[2])
PROMPT_LEN = int(os.environ.get("FP_PROMPT_LEN", "24"))
NEW_TOKENS = int(os.environ.get("FP_NEW_TOKENS", "24"))

engine = TorchAEGEngine(load_engine_from_path(AEG), device="cpu")
vocab = int(engine.weights.embedding.shape[0])
rng = np.random.default_rng(20260901)
prompt = rng.integers(0, min(vocab, 30000), size=PROMPT_LEN, dtype=np.int64)

logits, _ = engine._forward_device(prompt, None, validate_ids=True, logits="last")
tail = logits.detach().float().reshape(-1).cpu().numpy()

t0 = time.perf_counter()
greedy = list(engine.generate_iter(prompt, max_tokens=NEW_TOKENS, temperature=0.0))
elapsed = time.perf_counter() - t0

fingerprint = {
    "artifact": AEG.name,
    "dtype": str(engine.compute_dtype),
    "prompt": prompt.tolist(),
    "greedy_tokens": [int(t) for t in greedy],
    "tail_logits_sha256": hashlib.sha256(tail.tobytes()).hexdigest(),
    "tail_logits_head": [round(float(x), 6) for x in tail[:8]],
    "tail_argmax": int(np.argmax(tail)),
    "tail_sum": round(float(tail.sum()), 4),
    "decode_s": round(elapsed, 4),
    "tokens_per_s": round(len(greedy) / elapsed, 3) if elapsed else None,
}
OUT.write_text(json.dumps(fingerprint, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in fingerprint.items() if k != "prompt"}, indent=2))
