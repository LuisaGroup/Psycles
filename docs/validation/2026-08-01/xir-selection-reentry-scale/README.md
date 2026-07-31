# XIR selection-reentry scale

This checkpoint fixes the Vulkan cold-JIT bottleneck found while compiling the
production heterogeneous-volume path kernels. The implementation and regression
were committed directly to Luisa `next` as
`21612b45b` (`XIR: query selection reentry owners by dominance`), based on
`9516ee301`.

No verifier was disabled or moved out of the pass transaction, and no
Psycles-specific kernel shape is recognized. The change replaces a redundant
graph-wide relation query with an equivalent dominator-tree query.

## Formal relation

For a selection with header `H` and merge `M`, an edge from predecessor `P` to
destination `E` is a post-merge re-entry exactly when:

1. `H` dominates `P`;
2. `M` dominates `P`;
3. `H` dominates `E`; and
4. `M` does not dominate `E`.

The third predicate means precisely that `H` is an ancestor of `E` in the
dominator tree. Therefore, walking `E`'s ancestors from deepest to shallowest
and taking the first candidate satisfying the other predicates chooses exactly
the same deepest owning selection as the previous all-block scan.

Loop-boundary selection membership is also a relation of one immutable CFG
version. It is now materialized once before the edge queries rather than
recomputed by region traversal for every candidate. A node split terminates the
current query; the surrounding fixed point rebuilds dominance and loop-boundary
membership before any subsequent query, so the optimization never consumes
stale analysis.

The pass report exposes boundary-analysis, edge-query, and owner-query counters.
The augmented fallback-proxy regression adds 128 reachable structured
selections in a sibling root arm. Those nodes cannot dominate the re-entry
destination, and the regression asserts that owner work remains bounded by the
destination's ancestor chain rather than unrelated graph width. The fixture
continues to check exact re-entry elimination, affine ray-query alloca cloning,
verifier validity, graph-size bounds, and pass idempotence.

## Isolated cold-JIT A/B

Both runs used the same Psycles volume-path executable and generated kernels,
the same RADV GFX1201 device, an isolated empty runtime cache, and:

```sh
LUISA_LOG_LEVEL=verbose \
LUISA_XIR_TRACE_PASSES=1 \
LUISA_SPIRV_OPT_LEVEL=2 \
build/bin/psycles_luisa_volume_path_tests vk
```

| measurement | before `9516ee301` | after `21612b45b` | change |
| --- | ---: | ---: | ---: |
| whole process | `40.49 s` | `9.86 s` | `4.11x` faster |
| six `restructure_cfg` module runs | `36,533.716 ms` | `6,269.162 ms` | `5.83x` faster |
| selection-reentry query | `30,399.418 ms` | `168.987 ms` | `179.89x` faster |
| maximum selection-reentry query | `9,148.684 ms` | `42.115 ms` | `217.23x` faster |
| maximum resident memory | `542,352 KiB` | `543,744 KiB` | `+0.26%` |

After the query fix, the remaining restructure time is principally the existing
fixed point (`4,436.663 ms`), selection-exit draining (`2,005.966 ms`), and
construct-exit fixup (`1,263.686 ms`). These are measured separately and are
not attributed to the solved width-scaling defect. The machine-readable
snapshot is in [profile.json](profile.json).

## Validation

- Luisa configured build: `625/625` build steps with `--parallel 32` in
  `112.58 s`.
- Luisa XIR unit label: `48/48` passed in `0.46 s`.
- Dedicated restructure suite: 57 tests and 1,076 assertions passed.
- Luisa `test_complex_kernel`: fallback passed 920 assertions in `2.93 s`,
  HIP passed in `2.28 s`, and Vulkan passed in `0.61 s`.
- Psycles final integrated build: `--parallel 32` completed in `12.02 s`.
- Psycles final integrated CTest: `111/111` passed, including fallback, HIP,
  Vulkan, OpenEXR, and source-size gates.

The wider Luisa `unit` label passed `115/116`. The sole failure,
`test_eastl_allocation`, reproduces standalone as eight `fixed_vector`
assumption failures against the unchanged pinned
`EASTL@d9d9a86560f5fe23d1eb559b20ae89e9e3676f5f`. It is an existing dependency
baseline issue and is deliberately not reported as passing validation for this
XIR change.

This checkpoint changes shader compilation, not the path estimator. It
therefore has no Cycles/Psycles image or triptych; visual evidence remains tied
to scene-rendering checkpoints.
