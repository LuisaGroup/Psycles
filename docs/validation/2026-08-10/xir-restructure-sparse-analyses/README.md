# Sparse XIR restructure analyses

Date: 2026-08-10

## Outcome

LuisaCompute `next@c340b1841` removes five dense or repeatedly recomputed
relations from `restructure_cfg` without weakening its fixed-point or verifier
contracts. The first three shipped at `62737f27d`; the final two are the
profile-guided follow-up:

1. Loop-boundary selection membership is materialized once for each immutable
   CFG version and reused by all construct queries.
2. The physical construct hierarchy is derived by one event walk over the
   sparse immediate-dominator tree instead of comparing every construct with
   every other construct.
3. The final post-merge selection-reentry audit queries each merge's sparse
   dominance frontier instead of scanning every block for every selection.
4. Enclosing loops are persistent linked contexts on the sparse dominator
   walk, rather than one copied loop-exit set for every dominated block.
5. Loop-boundary arm classification uses block value numbering, a sparse
   reverse-CFG CSR, and one monotone dataflow solution per loop, rather than a
   fresh graph search and hash set for every `IfInst` arm.

The unchanged Lone Monk production kernel generates the same
`1,431,985`-word raw and `1,116,158`-word optimized SPIR-V modules. Its
cache-cold Vulkan shader JIT falls from `180.533 s` to `162.519 s`, saving
`18.014 s` (`9.98%`). `restructure_cfg` itself falls from `53.138 s` to
`37.088 s`, a `30.2%` reduction. The compile-smoke image is byte-identical to
the preceding checkpoint.

The follow-up production run preserves the same raw/optimized SPIR-V sizes
and byte-identical output. Against the immediately preceding persistent-loop-
context run, `restructure_cfg` falls from `35.226 s` to `17.945 s`, XIR
legalization from `62.053 s` to `44.126 s`, and native AST-to-SPIR-V from
`75.920 s` to `57.766 s`. Peak RSS falls from `9,415,608 KiB` to
`1,654,768 KiB`. The driver pipeline hit its disk cache in the follow-up, so
the observed `57.933 s` total JIT is recorded but is not attributed to this
compiler change.

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

Focused `test_xir_pass_restructure_cfg` passes `63/63` tests and `1,283`
assertions. The added 128-selection loop regression requires one persistent
loop context and one sparse dataflow solution per observed CFG version, while
exactly two constant-time classifications per selection are allowed to scale.
After a complete all-thread rebuild, the `unit_xir` label passes
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

## Value-numbered loop-boundary follow-up

Linux `perf` sampled the first large `drain_selection_exits` invocation. Of
5,269 samples with none lost, the dominant self costs were fresh pointer-set
insertion and hashing plus `classify_loop_boundary_path`: unordered-set
insertion paths accounted for more than 22%, `BasicBlock::is_terminated` for
9.99%, classification for 9.66%, and pointer hashing for 7.36%. The call graph
placed these costs under `collect_loop_boundary_selection_entries`; this was
evidence of repeated per-arm graph traversal, not verifier time.

For a fixed loop context, each block now has a fact in the finite lattice

```text
P({BREAK, CONTINUE, INVALID})
```

The loop merge and valid `BreakInst` seed `BREAK`; loop entry/update and valid
`ContinueInst` seed `CONTINUE`; malformed or escaping terminal paths seed
`INVALID`. Nested loops are atomic and contribute only their merge successor,
exactly as in the previous path classifier. Facts propagate backwards over a
sparse CSR until the least fixed point. Union is monotone, and a block can
gain each of the three bits at most once, so one loop costs `O(V + E)` and an
arm query is `O(1)`. `INVALID` still dominates the result, and simultaneous
break/continue reachability remains `MIXED`; no legality rule was weakened.

The storage is not a dense dominance relation. Blocks receive consecutive
value numbers so parent/fact/epoch tables can use arrays, while the immediate-
dominator tree retains one parent per reachable block and reverse CFG edges
retain exactly `O(E)` IDs in CSR form.

The production command is the same as above with output/log rooted at
`/var/tmp/psycles-sparse-vn-dataflow-20260810/`. Comparison is against the
immediately preceding complete run in
`/var/tmp/psycles-sparse-loop-context-20260810/vk.log`:

| Boundary | Persistent loop contexts | Value-numbered sparse dataflow | Change |
| --- | ---: | ---: | ---: |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |
| `drain_selection_exits`, aggregate | 13.229 s | 0.723 s | 18.30x |
| `drain_selection_exits`, maximum | 12.826 s | 0.613 s | 20.91x |
| Post-restructure fixed point | 21.781 s | 6.140 s | 3.55x |
| `restructure_cfg` module pass | 35.226 s | 17.945 s | -49.0% |
| SPIR-V XIR legalization | 62.053 s | 44.126 s | -28.9% |
| Native AST-to-SPIR-V | 75.920 s | 57.766 s | -23.9% |
| Peak RSS | 9,415,608 KiB | 1,654,768 KiB | -82.4% |
| RADV pipeline creation | 84.200 s | 0.015 s | driver-cache hit; not attributable |
| Observed complete shader JIT | 160.274 s | 57.933 s | mixed cache state; informational |

Both outputs hash to
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`
and compare byte-for-byte equal. The complete follow-up log is
`/var/tmp/psycles-sparse-vn-dataflow-20260810/vk.log`.

## Remaining hotspot

The sparse dataflow exposes the next independent transform hotspot:
`canonicalize_loop_boundary_selection_merges` still takes `5.558 s`
aggregate (`2.202 s` maximum). Unlike the now-read-only membership scan, that
phase mutates arm targets while walking them. Its repair must therefore use a
formally defined immutable snapshot plus transactional rewrite batch, or an
explicit invalidation/worklist rule; reusing stale classifications across a
mutation would be incorrect. The complete pass still verifies only its input
and output boundaries by default.

## Reverse-use DCE fixed-point follow-up

After the sparse restructure changes, a compile-interval `perf` recording
captured 46,885 samples with none lost. The next independent cost was no
longer dominance or loop-boundary classification: the five DCE invocations
took `14.182 s` in aggregate, and the recursive whole-function dead-code
scan alone accounted for `10.22%` of compile samples. The implementation
revisited every reachable instruction, recomputed memory effects, and
rescanned every use until a dead-value fixed point stopped changing. A dead
SSA chain therefore admitted quadratic work.

For removable instructions `R`, let `U(i)` be the multiset of instruction
users of `i`. The old algorithm computed the least fixed point

```text
D = { i in R | U(i) is a subset of D }
```

from `D = empty`. Luisa `next@f0018499d` solves the same equation by
reverse-use peeling: value-number removable instructions once, count live
instruction uses, seed zero-count values, and decrement operand-definition
counts when a user enters `D`. Duplicate operand uses are counted and
decremented separately. Each instruction and use is visited `O(1)` times, so
the solve is `O(I + U)`. A removable cycle without a dead sink is deliberately
retained, preserving the old least-fixed-point semantics rather than silently
switching to a greatest fixed point.

The 64-instruction dead-chain regression now checks the complexity contract,
not merely the final IR: the worklist pops exactly 64 values, while the two
outer CFG versions scan exactly `65 + 1` instructions. The complete
all-thread rebuild passes `unit_xir` `48/48`; the RX 9070 XT native Vulkan
route passes `92/92` tests and `2,096` assertions.

The unchanged production command writes to
`/var/tmp/psycles-dce-worklist-20260810/`. Comparison is against the
immediately preceding sparse-dataflow run:

| Boundary | Repeated DCE scan | Reverse-use worklist | Change |
| --- | ---: | ---: | ---: |
| Aggregate DCE | 14.182 s | 3.004 s | 4.72x |
| SPIR-V XIR legalization | 44.126 s | 33.540 s | -24.0% |
| Native AST-to-SPIR-V | 57.766 s | 47.533 s | -17.7% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

Both PPM files hash to
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`
and compare byte-for-byte equal. This run rebuilt the Vulkan backend and
retriggered `84.119 s` of RADV pipeline compilation, so its `131.808 s` total
JIT and `9,414,528 KiB` process peak are not compared with the preceding
driver-cache-hit run. The compiler-boundary timings above exclude that driver
stage. The complete log is
`/var/tmp/psycles-dce-worklist-20260810/vk.log`.

## Versioned loop-boundary merge canonicalization

The post-DCE profile captured 97K user-cycle samples with none lost. It
confirmed that `canonicalize_loop_boundary_selection_merges` still performed
the retired per-arm graph search: 48 invocations took `5.706 s`, while the
flat profile again exposed pointer-set allocation, hashing, recursive block
traversal, and `classify_loop_boundary_path`. The four expensive invocations
were `2.272 s`, `2.244 s`, `0.592 s`, and `0.597 s`; only one post iteration
actually changed this phase.

Luisa `next@8085ee23b` now treats every classification as a value of one
immutable CFG version. Blocks are numbered once, the structured reverse CFG
is stored sparsely in CSR, and the finite `{BREAK, CONTINUE, INVALID}` lattice
is solved once per visited loop. Every IfInst arm is then an `O(1)` fact
lookup. Rewrites for one loop context are planned against the snapshot:

- replacing a forwarding break arm with `Break(loop_merge)` preserves its
  `BREAK` fact;
- replacing an obsolete selection merge with a transparent false-arm proxy
  retains both executable arms and can only remove spurious successor facts.

Thus every planned rewrite remains sound under the batch. A batch may expose
additional canonicalization opportunities, but it cannot invalidate a fact
that justified an existing action. After each batch, the entire numbering,
CSR, and fact solution is discarded and rebuilt before another loop context
is inspected. This is an explicit versioned-analysis rule, not reuse of stale
facts across mutation.

The 128-selection regression observes one dataflow solution per loop and CFG
version, exactly two constant-time classifications per selection, and zero
rewrite invalidations. The existing physical loop-guard regression requires
a nonzero rewrite-batch count. The complete `unit_xir` label passes `48/48`;
the RX 9070 XT native Vulkan route passes `92/92` tests and `2,096`
assertions.

The unchanged Lone Monk command writes to
`/var/tmp/psycles-boundary-merge-dataflow-20260810/`:

| Boundary | Per-arm graph search | Versioned sparse dataflow | Change |
| --- | ---: | ---: | ---: |
| Merge canonicalizer | 5.706 s | 0.084 s | 67.8x |
| `restructure_cfg` | 18.343 s | 12.737 s | -30.6% |
| SPIR-V XIR legalization | 33.540 s | 27.515 s | -18.0% |
| Native AST-to-SPIR-V | 47.533 s | 41.046 s | -13.6% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The production module used 26 immutable analyses, 496 per-loop dataflow
solutions, 33,984 constant-time arm classifications, and only 4 invalidating
rewrite batches. Both PPM files remain byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
The RADV disk cache hit in `22.149 ms`; the resulting `41.220 s` total JIT and
`1,661,008 KiB` process peak are recorded but not used to inflate the
compiler-boundary speedups. The complete log is
`/var/tmp/psycles-boundary-merge-dataflow-20260810/vk.log`.

The next measured transform hotspot is main structurization:
`try_restructure_if_batch` takes `7.930 s` of the `8.977 s`
`main_loop_iteration` aggregate. `simplify_cfg` follows at `3.495 s`, while
the five DCE calls now total `2.989 s`. Further work should first decompose the
if-batch cost by immutable analysis, candidate ordering, and post-dominator
invalidation rather than weaken legality checks.
