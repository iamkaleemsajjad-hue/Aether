import os, sys, gc, time
os.environ["AETHER_PLAN_BOOTSTRAP"] = "0"
os.environ["AETHER_TORCH_DTYPE"] = os.environ.get("AETHER_TORCH_DTYPE", "fp16")
sys.path.insert(0, "src")
import numpy as np, torch
from pathlib import Path

G = 1024 ** 3
AEG = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmark/results/aeg-cache/qwen 0.6B.aeg")


def host_bytes(w):
    tot, seen = 0, set()

    def v(x):
        nonlocal tot
        if isinstance(x, np.ndarray) and id(x) not in seen:
            seen.add(id(x))
            tot += x.nbytes

    def walk(o):
        for n in dir(o):
            if n.startswith("_"):
                continue
            try:
                v(getattr(o, n))
            except Exception:
                pass

    walk(w)
    for L in getattr(w, "layers", None) or []:
        walk(L)
        for e in getattr(L, "experts", None) or []:
            walk(e)
    return tot


def device_bytes(eng):
    tot, seen = 0, set()

    def v(x):
        nonlocal tot
        if isinstance(x, torch.Tensor):
            st = x.untyped_storage()
            if st.data_ptr() not in seen:
                seen.add(st.data_ptr())
                tot += st.nbytes()

    for n in ("embedding", "lm_head", "final_norm", "final_norm_bias", "position_embedding"):
        v(getattr(eng, n, None))
    for L in getattr(eng, "layers", None) or []:
        if isinstance(L, dict):
            for x in L.values():
                if isinstance(x, list):
                    for e in x:
                        if isinstance(e, dict):
                            for y in e.values():
                                v(y)
                else:
                    v(x)
    return tot, len(seen)


from aether.runtime.aeg_loader import load_engine_from_path
from aether.runtime.torch_engine import TorchAEGEngine

t0 = time.perf_counter()
cpu = load_engine_from_path(AEG)
t1 = time.perf_counter()
hb = host_bytes(cpu.weights)
eng = TorchAEGEngine(cpu, device="cpu")
t2 = time.perf_counter()
db, n = device_bytes(eng)
resident = host_bytes(cpu.weights)
print(f"artifact                : {AEG.name}")
print(f"loader time             : {t1 - t0:6.2f} s")
print(f"engine build time       : {t2 - t1:6.2f} s")
print(f"host weight bytes       : {hb / G:6.3f} GiB")
print(f"device weight bytes     : {db / G:6.3f} GiB  ({n} storages, dtype={eng.compute_dtype})")
print(f"lm_head shares embedding: {eng.lm_head.untyped_storage().data_ptr() == eng.embedding.untyped_storage().data_ptr()}")
print(f"streamed during upload  : {eng.host_bytes_streamed / G:6.3f} GiB")
print(f"host still resident     : {resident / G:6.3f} GiB  (load peak holds this, not {hb / G:.3f})")
freed = eng.release_host_weights()
print(f"host released           : {freed / G:6.3f} GiB -> {host_bytes(cpu.weights) / G:.3f} GiB")
