# Sparse XIR restructure analyses

Date: 2026-08-10

## Outcome

LuisaCompute `next@62737f27d` removes three dense or repeatedly recomputed
relations from `restructure_cfg` without weakening its fixed-point or verifier
contracts:

1. Loop-boundary selection membership is materialized once for each immutable
   CFG version and reused by all construct queries.
2. The physical construct hierarchy is derived by one event walk over the
   sparse immediate-dominator tree instead of comparing every construct with
   every other construct.
3. The final post-merge selection-reentry audit queries each merge's sparse
   dominance frontier instead of scanning every block for every selection.

The unchanged Lone Monk production kernel generates the same
`1,431,985`-word raw and `1,116,158`-word optimized SPIR-V modules. Its
cache-cold Vulkan shader JIT falls from `180.533 s` to `162.519 s`, saving
`18.014 s` (`9.98%`). `restructure_cfg` itself falls from `53.138 s` to
`37.088 s`, a `30.2%` reduction. The compile-smoke image is byte-identical to
the preceding checkpoint.

This checkpoint does not claim that the pass is finished. The largest
remaining transform cost is `drain_selection_exits` at `14.502 s`; the driver
then spends `85.384 s` creating the monolithic RADV pipeline. The next
compiler repair must address the selection-exit fixed point and its
invalidation rules rather than skip correctness checks.

## Formal transformations

### CFG-versioned boundary value

For a fixed CFG, loop-boundary membership is the relation

```text
boundary(x) iff exists loop L:
    x is structurally reachable inside L and
    x is an IfInst whose arms cross only L's physical boundary.
```

The old construct-entry and construct-exit scans evaluated that existential
predicate independently for each candidate header, repeatedly walking all
loop regions. The replacement walks the loop regions once, materializes the
set of satisfying headers, and performs constant-time membership queries.
Any CFG rewrite abandons the scan, invalidates the set, and rebuilds it once
for the next version. This is value numbering at the analysis level: equal
`(CFG version, boundary predicate)` inputs share exactly one computed value.

### Sparse construct hierarchy

For a physical construct with header `H`, merge `M`, and optional continue
target `C`, the headers it may enclose are exactly

```text
subtree(H) - subtree(M) - {C}
```

in the immediate-dominator tree. The `subtree(M)` term is absent when `M` is
unreachable or outside `H`'s subtree. If `M` dominates `H`, the set is empty.

The old implementation tested this relation for every ordered construct pair
and selected the deepest enclosing header, requiring `O(C^2)` dominance
queries for `C` constructs. The replacement performs an iterative dominator-
tree DFS:

- entering `H` activates its construct after resolving `H`'s own parent;
- entering a dominated `M` suspends that construct for the complete merge
  subtree;
- leaving `M` restores it for sibling subtrees;
- the deepest active construct other than a matching continue target is the
  unique parent.

An ordered active set makes the traversal
`O((V + C) log C + Q)`, where `Q` counts only actual continue-target skips,
with `O(V + C)` auxiliary storage. The pass also iterates the already-built
construct-region block set directly when collecting exits, rather than
rescanning every function block and testing set membership.

### Dominance-frontier re-entry audit

For selection `(H, M)`, the old final audit reported a re-entry when an
executable edge `P -> B` satisfied

```text
H dominates B
M does not dominate B
M dominates P.
```

The last two predicates are precisely the definition that `B` lies in the
dominance frontier of `M`. The new audit therefore visits only `DF(M)`, keeps
the `H dominates B` test, and validates an actual executable predecessor to
exclude XIR ownership-only use-list edges. This is the same graph predicate,
not a heuristic or sampled audit. Its work is proportional to the stored
sparse frontier candidates and their checked predecessors instead of the
selection-by-block Cartesian product.

## Regression coverage

The 256-diamond scale regression now makes all three complexity contracts
observable:

- one construct-entry dominator tree per immutable CFG version;
- one boundary-membership analysis for entry enforcement and one for exit
  repair, independent of the 256 constructs;
- zero construct-parent candidate queries because every completed diamond is
  suspended for its merge subtree;
- 256 selection audit queries but zero frontier/block queries because the
  sequential diamonds have empty merge dominance frontiers.

Focused `test_xir_pass_restructure_cfg` passes `62/62` tests and `1,275`
assertions. After a complete all-thread rebuild, the `unit_xir` label passes
`48/48` tests. `test_vk_spirv_codegen_path vk` passes `92/92` runtime tests
and `2,096` assertions on the RX 9070 XT. The pass remains verified only at
its complete input and output boundaries by default; this change adds no
per-iteration verifier invocation.

## Lone Monk cold-JIT measurement

Command:

```sh
PSYCLES_DISABLE_SHADER_CACHE=1 \
LUISA_LOG_LEVEL=verbose \
LUISA_VULKAN_PROFILE_COMPILATION=1 \
LUISA_XIR_TRACE_PASSES=1 \
build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-transmission-dbdcb17/export \
  /var/tmp/psycles-sparse-cfg-20260810/vk.ppm \
  vk 1 1 1 1
```

As in the preceding checkpoint, the 1x1/1-spp launch is a compile smoke. It
records and compiles the complete 37-material path kernel; it is not a visual
quality render and does not reduce shader complexity.

| Boundary | Sparse-verifier baseline | Sparse-restructure run | Change |
| --- | ---: | ---: | ---: |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |
| `restructure_cfg` module pass | 53.138 s | 37.088 s | -30.2% |
| Post-restructure fixed point | 32.493 s | 23.195 s | -28.6% |
| `enforce_unique_construct_entries` | 4.331 s | 1.240 s | 3.49x |
| `fixup_construct_exits` | 10.735 s | 0.978 s | 10.98x |
| Final selection-reentry audit | 3.817 s | 0.331 s | 11.53x |
| Native AST-to-SPIR-V | 93.597 s | 76.979 s | -17.8% |
| RADV pipeline creation | 86.780 s | 85.384 s | effectively unchanged |
| Complete shader JIT | 180.533 s | 162.519 s | -9.98% |
| Complete process wall time | 182.71 s | 164.68 s | -9.87% |
| Peak RSS | 9,416,224 KiB | 9,416,496 KiB | effectively unchanged |

The old and new PPM files both hash to
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`
and compare byte-for-byte equal. The complete new log is
`/var/tmp/psycles-sparse-cfg-20260810/vk.log`.

## Remaining hotspot

The sparse analyses expose rather than hide the next bottleneck. In the large
definition, `drain_selection_exits` still takes `13.972 s` in its first post
iteration (`14.502 s` over all definitions and iterations). It repeatedly
rebuilds global site/loop relations, rewrites one site, and starts again.
A correct worklist must state which mutation invalidates which selection-exit
facts and must preserve the phase's well-founded progress measure. Merely
omitting the final unchanged scan, weakening verification, or recognizing
Lone Monk-specific shapes would not be a valid repair.
