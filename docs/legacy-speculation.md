# Legacy tree speculation

`aether.runtime.TreeSpeculativeEngine` remains importable with its existing
constructor and methods. It has no draft executor. `build_draft_tree` now raises
`SpeculativeDecodingError` rather than generating candidates from token IDs;
an empty prefix raises `ValueError` rather than inventing root token zero.
Supplying a draft model identifier to this utility does not load that model.

For executing speculation, use the explicit compatible draft-model path in
[runtime feature integration](runtime-feature-integration.md), or
`aether.runtime.speculative_engine.DraftTargetSpecDecoder` with actual engines.
The existing real draft/target path has separate execution and cache ownership.

`DraftTreeNode`, pruning, and manual greedy verification remain available.
`verify_tree` requires finite, equal-width target-logit rows for the supplied
candidate path. Missing or exhausted rows, malformed evidence, invalid candidate
IDs or probabilities, and cycles raise before acceptance counters are committed.
The result includes the original root as context; that root is not counted as a
generated token. No observations means `should_use_speculation()` is false.

This utility does not run tree-masked attention or implement stochastic tree
sampling. Supplying a candidate tree is distinct from evaluating a target model
at its prefixes. The executable decoder's distribution guarantee cannot be
inferred from legacy tree inspection.

Before the correction, the new regression file produced 35 failures against the
original module. After the correction, the focused legacy/runtime/real-draft/
Torch/persistence selection passed all 131 tests. The full-suite result is
tracked in [the mission validation record](mission-validation.md).
