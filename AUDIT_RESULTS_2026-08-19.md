# Aether PRD audit — 2026-08-19

This is an evidence report, not a restatement of the existing acceptance
documents. The repository, source code, tests, and the supplied local Qwen3
checkpoint were inspected directly. Existing reports were not used as proof.

## Verified

- Full test suite: **2673 passed, 21 skipped, 3 warnings** in 484.61 seconds.
- Focused compiler/runtime/CLI/Qwen-path suite: **253 passed, 2 skipped**.
- `python -m compileall -q src tests`: passed.
- The supplied `qwen 0.6B` SafeTensors checkpoint was compiled from source into
  `scratch/qwen_clean.aeg` for `cpu_avx2`.
- The generated artifact records the source geometry: 28 layers, hidden size
  1024, 16/8 attention heads, head dimension 128, `rms_norm_eps=1e-6`,
  `rope_theta=1e6`, and Q/K head normalization.
- The artifact contains all 56 Qwen3 Q/K norm vectors and the terminal
  `model.norm` tensor. Greedy CLI inference returns ` Paris.` for
  `The capital of France is`.
- Against the locally loaded Hugging Face reference, the clean artifact matched
  the next-token top-1, top-5, top-10, and top-50 sets; logits cosine was 1.0.
- Benchmark measurements are recorded in
  [`benchmarks/results/qwen3_0.6b_cpu_avx2_2026-08-19.json`](benchmarks/results/qwen3_0.6b_cpu_avx2_2026-08-19.json).

## Correctness fixes made

1. Architecture detection now takes a local `config.json` as authoritative and
   preserves non-derived geometry and numerical constants.
2. Qwen3 Q/K norms are represented as parameter-bearing graph nodes, retained
   by quantization, loaded into the CPU engine, and applied before RoPE.
3. Bare terminal norm names such as `model.norm.weight` now bind to
   `final_norm`; missing it previously caused a silent all-ones substitution.
4. Operator fusion no longer creates cycles when Q/K normalization interrupts
   the QKV-to-RoPE path.
5. Sparse-attention plans are activated only at their recorded long-context
   threshold; short prompts retain dense-reference behavior.
6. Uncertified auto precision now stays BF16. Lossy Q4/Q3 and pruning masks are
   not silently applied by the default clean compile path.
7. Bit-packed 2/4-bit unpacking now fails closed when the input buffer is too
   short.
8. The CLI now handles non-CP1252 generated text and dynamically bound Click
   output streams on Windows.

## Not proven by this workspace

The PRDs describe a much larger platform than the locally executable CPU path.
The following are not honestly certified here:

- CUDA/ROCm/Metal/NPU/RISC-V/FPGA hardware execution, GPU speed claims, or
  multi-GPU topology plans.
- Network Hub/fleet integrations and network-enabled tests; those tests were
  skipped because network tests are disabled in this environment.
- Windows distributed process-IPC validation; one test was skipped because the
  environment denied process IPC.
- Physical TEE attestation, CXL hardware, Rubin hardware, and external MCP
  deployments.
- The source still contains explicit unsupported/stub paths for some collective
  backends, GPU kernel generation on a host without the target hardware, ZK
  ownership proofs, and selected future hardware/features.
- The full PRD feature set cannot be called “perfectly complete” merely because
  its classes or metadata files exist. Opt-in v4/v5 features requiring trained
  drafter bundles, schemas, accelerators, services, or attestation hardware
  need separate acceptance runs.

## Static quality status

The full runtime test suite is green, but repository-wide static hygiene is not:

- `ruff check src/aether`: failed with **3567** existing violations.
- `ruff format --check src tests`: failed with Ruff exit **101**.
- `mypy src/aether`: failed with **1223** reported errors.

Therefore the repository is not yet eligible for an unconditional “all PRD
claims are complete and production-perfect” sign-off. The tested CPU compiler
and runtime path is now evidence-backed; the remaining platform and static
quality work must be completed before making that broader claim.
