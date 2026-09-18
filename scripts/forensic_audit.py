"""
Forensic audit script - run from repo root.
"""
from pathlib import Path
import re

root = Path(r'c:\Users\pc\Desktop\Aether Runtime\src\aether')

files_of_interest = {
    'cpu_engine': 'runtime/cpu_engine.py',
    'torch_engine': 'runtime/torch_engine.py',
    'native_cpu': 'kernels/native_cpu.py',
    'native_cuda': 'kernels/native_cuda.py',
    'torch_backend': 'backends/torch_backend.py',
    'backends_base': 'backends/base.py',
    'registry': 'backends/registry.py',
}

for name, path in files_of_interest.items():
    p = root / path
    if not p.exists():
        print(f'MISSING: {path}')
        continue
    txt = p.read_text(errors='ignore')
    lines = txt.splitlines()
    np_refs = txt.count('np.') + len(re.findall(r'\bnumpy\b', txt))
    torch_refs = txt.count('torch.')
    triton_refs = len(re.findall(r'\btriton\b', txt, re.IGNORECASE))
    cuda_kernel_refs = len(re.findall(r'__global__|cudaMalloc|ctypes|cffi|cdll', txt))
    has_not_impl = 'NotImplementedError' in txt or 'raise NotImplemented' in txt
    print(f"\n=== {name} ({len(lines)} lines) ===")
    print(f"  numpy: {np_refs}  torch: {torch_refs}  triton: {triton_refs}  native_c: {cuda_kernel_refs}")
    print(f"  NotImplementedError: {has_not_impl}")
    imports = [l.strip() for l in lines[:50] if l.strip().startswith('import') or l.strip().startswith('from')]
    for i in imports[:12]:
        print(f"  IMPORT: {i}")
