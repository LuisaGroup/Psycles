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

## Immutable-dominance if batching

The decomposed profile showed that merge discovery was not the dominant cost
inside `try_restructure_if_batch`. Across 1,723 candidates, the initial and
dynamic merge queries took `0.877 s` and `1.157 s`, while 1,723 complete
dominator-tree rebuilds took `5.229 s`. Allowed-target construction, scope
collection, and edge-retarget walks together took less than `14 ms`. Rebuilding
global dominance after every local structural edge subdivision was therefore
the scaling error.

Luisa `next@6a03ec2f7` uses an exact quotient-graph invariant instead. One
batch starts with an immutable dominator tree. A successful rewrite replaces a
raw conditional with a structured If and introduces only a transparent merge
subdivision. Contracting all such new blocks recovers the input CFG, so
dominance between any two pre-existing blocks is unchanged. Each new merge
stores a dominance anchor: the nearest common dominator, in the immutable tree,
of all dynamic predecessor anchors. For every old block `d` and new merge `m`,
`d` dominates `m` exactly when `d` dominates `anchor(m)`. Dynamic merge
inference and later enclosing-scope retargeting can therefore query inserted
blocks without rebuilding global dominance.

The lexical merge discovered on the immutable graph is retained only as the
exact contracted-graph fallback. A preliminary implementation that reused it
unconditionally failed the existing nested loop/switch break/continue
round-trip regression by changing the structured block count; that
approximation was discarded rather than weakening the regression. The final
implementation performs dynamic merge inference through the overlay and uses
the fallback only when transparent subdivisions hide all normal paths.

The 64-diamond complexity regression now proves one immutable analysis per CFG
version, exactly one candidate query per eligible diamond and analysis, and no
overlay query for sibling diamonds. The nested loop/switch regression exercises
the mutation-sensitive case. The complete `unit_xir` label passes `48/48`; the
RX 9070 XT Vulkan native-codegen route passes `92/92` tests with `2,096`
assertions.

The unchanged Lone Monk command writes to
`/var/tmp/psycles-if-batch-overlay-20260810/`:

| Boundary | Per-rewrite dom rebuild | Dominance overlay | Change |
| --- | ---: | ---: | ---: |
| Dominator rebuilds | 1,723 | 0 | eliminated |
| `try_restructure_if_batch` | 7.930 s | 2.091 s | 3.79x |
| `restructure_cfg` | 12.737 s | 6.826 s | -46.4% |
| SPIR-V XIR legalization | 27.515 s | 21.587 s | -21.5% |
| Native AST-to-SPIR-V | 41.046 s | 35.714 s | -13.0% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The production module used 11 immutable if-batch analyses, 1,723 candidate
queries, and 378 overlay-block queries. Both PPM files remain byte-identical
with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
The cache-cold RADV pipeline took `85.652 s` and process peak RSS was about
`9.419 GB`; neither is attributed to this compiler-boundary change. The
remaining `2.091 s` if-batch cost is almost entirely the two merge-inference
passes (`0.922 s` plus `1.142 s` in the final detailed run), which is the next
independent optimization target.

## Value-numbered selection-merge inference

The remaining implementation performed the same allocation-heavy query twice
per raw If: once while collecting the immutable candidate set and again while
processing the overlay graph. Every query rescanned all structured loops,
created one `unordered_map<BasicBlock *, distance>` per arm, traversed the
reachable subgraph, and rescanned every owned block up to three times. Even
canonicalizing a chain of transparent branches allocated a fresh visited set.
The measured pair of query phases consumed about `2.064 s` despite reaching
only a small portion of the 6,145-block function from a typical arm.

Luisa `next@411322cd2` moves the query into a 582-line standalone
`SelectionMergeBatchAnalysis` component. One immutable batch now:

- value-numbers owned blocks once and retains definition order, preserving the
  old candidate and tie-break order exactly;
- derives a persistent enclosing-loop context along the sparse dominator tree,
  so each header obtains its boundary set by walking only its active context;
- reuses dense distance, support, minimum, maximum, and total arrays with epoch
  markers instead of constructing pointer maps per arm;
- registers each transparent overlay block and evaluates its exact immutable
  dominance anchor; and
- canonicalizes transparent branch chains with Floyd cycle detection, returning
  the same terminal or first cycle-entry block with `O(1)` scratch storage.

For each query, the finite state is the product of `(entry, block)` shortest
reachability within the header-dominated region, plus the historical one-edge
normal-boundary candidates. Each state is discovered once and every traversed
edge is inspected once. Aggregate merge scores are reduced in stable block-ID
order, so the result is equivalent to the former maps and whole-definition
scans. Overlay mutation remains visible to the second query; this optimization
does not revive the invalid unconditional lexical-merge cache.

The 64-diamond regression requires exactly two dense merge queries per accepted
candidate and nonzero block/edge visits. The nested loop/switch
break/continue regression now also requires a nonempty persistent loop-context
tree. The complete `unit_xir` label passes `48/48`; the RX 9070 XT native
Vulkan route passes `92/92` tests with `2,096` assertions.

The unchanged Lone Monk run is recorded under
`/var/tmp/psycles-merge-dense-20260810/`:

| Boundary | Hash-map queries | Dense epoch workspace | Change |
| --- | ---: | ---: | ---: |
| `try_restructure_if_batch` | 2.091 s | 0.299 s | 6.99x |
| `restructure_cfg` | 6.826 s | 5.191 s | -24.0% |
| SPIR-V XIR legalization | 21.587 s | 20.334 s | -5.8% |
| Native AST-to-SPIR-V | 35.714 s | 33.752 s | -5.5% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

Across all definitions, 3,448 queries reused 357 loop-context nodes and visited
584,856 block states plus 734,477 successor edges. The RADV cache hit reduced
pipeline creation to `15.873 ms`; end-to-end shader JIT was `33.926 s` with
`1,661,244 KiB` peak RSS. The output remains byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.

## Versioned loop-continue normalization

The next restructure profile isolated `normalize_structured_loop_continues` at
`1.531 s`. Its loop-region walk was not itself large. The implementation
materialized the complete owned-block set and rebuilt the full dominator tree
before every loop site, even when all preceding sites were read-only. This
made an immutable CFG version pay `O(L * (V + E))` analysis cost for `L` loop
sites.

Luisa `next@d17763dc4` gives ownership and dominance an explicit version. One
normalization invocation consumes the already-current dominator tree and builds
the owned-block set once. Consecutive site queries share both analyses. A site
first computes its complete loop region, then applies the historical rewrite;
if it mutates the CFG, both analyses are immediately invalidated and rebuilt
before the next site is inspected. Thus query observations are identical to
the former per-site rebuild, while a read-only run costs `O(V + E + L)`.

A new 64-loop regression constructs canonical sequential loops, requires zero
invalidations, and proves exactly one ownership/dominance analysis per observed
CFG version rather than one per loop. Mutation-sensitive loop, nested-loop,
proxy-chain, break, and continue regressions remain unchanged. The complete
`unit_xir` label passes `48/48`; the RX 9070 XT native Vulkan route passes
`92/92` tests with `2,096` assertions.

The unchanged Lone Monk run is recorded under
`/var/tmp/psycles-loop-continue-versioned-20260810/`:

| Boundary | Per-site analysis | Versioned analysis | Change |
| --- | ---: | ---: | ---: |
| Continue normalization | 1.531 s | 0.552 s | 2.77x |
| Post-restructure fixed point | 2.884 s | 1.867 s | -35.3% |
| `restructure_cfg` | 5.191 s | 4.053 s | -21.9% |
| SPIR-V XIR legalization | 20.334 s | 18.883 s | -7.1% |
| Native AST-to-SPIR-V | 33.752 s | 32.691 s | -3.1% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The production module issued 361 loop-site queries against 157 CFG-version
analyses; 129 site rewrites caused and justified the 129 invalidations. The
RADV cache hit produced a `15.287 ms` pipeline and `32.860 s` shader JIT with
`1,661,080 KiB` peak RSS. The PPM remains byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.

## Sparse selection-exit follow-up

The next profile separated three costs that had previously appeared under one
selection-exit timer. First, every successful rewrite rebuilt both dominance
and post-dominance even though the inner drain observes only dominance.
Second, each loop-boundary dataflow value-numbered the CFG but still cleared,
seeded, scanned, and propagated over the complete function for every loop.
Third, every rewrite restarted the selection-site scan from the beginning.

Luisa `next@5c3ce4334` applies three invariant-preserving changes:

- a successful rewrite immediately refreshes dominance, while post-dominance
  is refreshed once after the drain reaches its fixed point and before its
  next observer;
- loop-boundary classification solves its finite lattice only on the
  structurally reachable, successor-closed induced region plus explicit
  break/continue boundary seeds; by closure, this is exactly the restriction
  of the whole-CFG least fixed point; and
- one-target quotient-CFG funnels invalidate only the selected site, affected
  enclosing constructs, changed transparent headers, and structural
  references to bypassed proxies. Multi-target state dispatch still performs
  a conservative global invalidation because its correlated reachability is
  not representable as one local funnel.

The dataflow implementation is now a real standalone `.h`/`.cpp` component,
not an included implementation fragment. A 64-loop scale regression requires
block and edge visits to remain linear in the active loop regions. A
65-selection funnel regression bounds queries by `3N` and requires every
rewrite to use local invalidation. The complete `unit_xir` label passes
`48/48`; the RX 9070 XT native Vulkan route passes `92/92` tests with `2,096`
assertions.

The unchanged Lone Monk cold-cache run is recorded under
`/var/tmp/psycles-selection-worklist-split-20260810/`:

| Boundary | Versioned loop analysis | Sparse selection-exit analysis | Change |
| --- | ---: | ---: | ---: |
| Selection-exit drain | 0.726 s | 0.408 s | 1.78x |
| Loop-boundary relation construction | about 0.282 s | 0.069 s | 4.09x |
| Loop-boundary merge analysis | about 0.079 s | 0.021 s | 3.76x |
| Aggregate post-dominator construction | 0.714 s / 239 calls | 0.649 s / 229 calls | -9.1% |
| `restructure_cfg` | 4.053 s | 3.555 s | -12.3% |

Production issued 1,599 loop dataflows over 222,912 active block visits and
605,194 active edge visits. Selection-exit analysis performed 14,813 site
queries: three rewrites used dependency-local invalidation and nine
multi-target state dispatches conservatively invalidated the site set. The
post-dominator refresh count fell from 12 per-rewrite refreshes to two
fixed-point-boundary refreshes. Raw/optimized SPIR-V remain exactly
1,431,985/1,116,158 words, and the output is byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.

The next measured transform hotspot is therefore the 229 private
post-dominator constructions. Although their old implementation stores the
fixed-point state in dense vectors, it still hashes block pointers for every
query, builds predecessor hash maps, allocates temporary successor vectors per
block and iteration, and repeatedly rediscovers sinks. The next step is block
value numbering plus a sparse reverse-CFG edge representation and a fully
dense immediate-dominator fixed point, with pointer maps only at the API
boundary.

## Dense post-dominator fixed point

Luisa `next@f16af7348` implements that boundary as a standalone 277-line
component. Each immutable analysis value-numbers owned blocks exactly once,
numbers every owned edge once, and builds the original and reverse CFG as two
sparse CSR views. An iterative DFS from one synthetic exit produces reverse
postorder over precisely the historical sink-reachable subgraph. A dense
Cooper-Harvey-Kennedy immediate-dominator solve then runs on the reversed CFG;
pointer hashing is confined to the initial numbering and final compatibility
map.

This intentionally does not reuse the public post-dominator pass: that pass
models DFS backedges as virtual exits, whereas restructuring historically
connects only real return, unreachable, raster-discard, unterminated, or
successor-free sinks to its virtual exit. Preserving that domain is part of
the transform semantics, not an optimization detail.

A 128-diamond regression constrains the algorithm through the public
restructure entry point. For every immutable analysis it requires exactly
`1 + 3N` numbered blocks, `4N` numbered edges, two RPO fixed-point passes, and
linear block/edge visits. This rules out a hidden pointer-valued Cartesian
relation while also checking the structured no-op result. The system-STL
focused gates pass 66/66 restructure tests with 1,325 assertions and 350/350
pass tests with 1,978 assertions. The complete `unit_xir` label passes 48/48;
the RX 9070 XT native Vulkan route passes 92/92 tests with 2,096 assertions.

The exact cache-cold Lone Monk comparison is recorded under
`/var/tmp/psycles-dense-postdom-20260810/`:

| Boundary | Sparse selection-exit baseline | Dense post-dominance | Change |
| --- | ---: | ---: | ---: |
| Post-dominator construction | 0.649 s / 229 calls | 0.174 s / 229 calls | 3.73x |
| `restructure_cfg` | 3.555 s | 3.036 s | -14.6% |
| SPIR-V XIR legalization | 18.358 s | 17.722 s | -3.5% |
| Native AST-to-SPIR-V | 31.847 s | 31.228 s | -1.9% |
| Process wall time | 34.01 s | 33.42 s | -1.7% |
| Peak RSS | 1,661,564 KiB | 1,660,856 KiB | effectively unchanged |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The production run performed 229 analyses over 666,268 numbered blocks and
872,735 numbered edges. The dense fixed point made 1,975,437 block visits,
2,589,358 edge visits, and 2,065,041 parent-intersection steps. The PPM is
byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`, and
both raw and optimized SPIR-V word counts are unchanged.

After this change the post-dominator solver disappears from the primary perf
hotspots. The largest measured restructure leaf is now loop-continue
normalization at about `0.520 s`, followed by selection-exit drain at about
`0.394 s` and the if batch at about `0.286 s`. Perf also exposes redundant
pointer-hash lookups in `DomTree::dominates`; the next stage therefore targets
versioned dense loop-region discovery and direct dominance-node queries while
retaining immediate invalidation after every CFG mutation.

## Deferred dominance frontiers for loop rewrites

The first decomposed profile showed that the region walk was not the remaining
loop-continue bottleneck. Of `526.492 ms`, mutation-triggered dominator rebuilds
consumed `475.692 ms` (90.4%), ownership-set reconstruction consumed
`46.323 ms`, and region discovery plus rewrite scanning consumed only
`1.463 ms`. All 129 CFG mutations require a fresh ancestry relation before the
next loop site, but none of those intermediate versions observes a dominance
frontier.

Luisa `next@23bc5b064` therefore separates the two mathematical products of a
dominator tree. Immediate dominators and DFS ancestry intervals are rebuilt
after every mutation exactly as before. The derived frontier relation is
omitted from intermediate versions and materialized once on the final tree
retained by each mutating normalization batch. Frontier construction is
idempotent by clearing and recomputing node-local frontier vectors; no state is
added to the exported `DomTree` layout. The original one-argument
`compute_dom_tree(Function *)` symbol remains the default full-tree entry point,
while an explicit options overload selects ancestry-only construction.

The production Lone Monk run still performs 129 dominance invalidations and
129 ancestry rebuilds, but only eight final frontier materializations. A new
16-loop mutation regression requires rebuild count to equal invalidation count
and frontier materializations to be strictly fewer. Direct node-interval
queries also remove the former `contains()+node()` double lookup while retaining
foreign/null-block behavior. Full 32-thread rebuilding is part of the gate so
mixed-layout incremental artifacts cannot hide an ABI regression.

The final matched run is recorded under
`/var/tmp/psycles-lazy-dom-frontier-final-20260810/`:

| Boundary | Dense post-dominance | Deferred frontier | Change |
| --- | ---: | ---: | ---: |
| Loop-continue normalization | 0.520 s | 0.356 s | -31.6% |
| `restructure_cfg` | 3.036 s | 2.877 s | -5.2% |
| SPIR-V XIR legalization | 17.722 s | 17.675 s | -0.3% |
| Native AST-to-SPIR-V | 31.228 s | 30.921 s | -1.0% |
| Peak RSS | 1,660,856 KiB | 1,652,776 KiB | -0.5% |

All 48 XIR tests pass after a complete rebuild, the system-STL focused gates
pass 8/8 dominance, 67/67 restructure, and 350/350 pass tests, and the RX 9070
XT native Vulkan route passes 92/92 tests with 2,096 assertions. Raw and
optimized SPIR-V sizes remain 1,431,985 and 1,116,158 words. The PPM is
byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.

The residual `308.926 ms` dominance-rebuild cost is now the primary component
inside loop-continue normalization. Its constructor still hashes block
pointers throughout the immediate-dominator fixed point and walks linked
predecessors repeatedly. The next step is a single pointer-to-ID boundary,
sparse predecessor CSR, and a fully dense RPO fixed point, while preserving
the public tree and frontier API.

## Dense RPO dominance for mutation versions

Luisa `next@13fadb811` implements that next step without changing the analyzed
CFG domain or the public pointer-based tree. Reachable blocks retain the
historical DFS postorder and receive consecutive reverse-postorder IDs once.
Predecessors are then stored as one sparse CSR. Cooper-Harvey-Kennedy solves
the immediate-dominator relation entirely over IDs: an intersection only
climbs the parent of the larger RPO ID, and the pointer-valued `DomTree` is
constructed after the fixed point converges. The hot solve therefore performs
no pointer hashing, predecessor allocation, or repeated block-to-index lookup.

For entry ID zero, `idom[0] = 0`. Each non-entry block joins only already
resolved predecessor chains, and every parent climb moves to a smaller RPO ID.
This preserves the standard CHK fixed point and proves termination; it is not
a graph-shape special case. Unreachable predecessors remain excluded at the
single numbering boundary. The pre-existing one- and two-argument exported
functions and `DomTreeBuildOptions` layout are unchanged; a new three-argument
diagnostic overload reports work without creating a silent ABI break.

A raw 128-diamond CFG regression requires exactly `1 + 3N` numbered blocks,
`4N` numbered edges, two RPO passes, `2(V - 1)` block visits, and `2E` edge
visits. The production report exposes the same counts. The matched cold-cache
Lone Monk comparison is recorded under
`/var/tmp/psycles-dense-dom-20260810/`, against
`/var/tmp/psycles-dom-components-baseline-20260810/`:

| Boundary | Pointer-valued fixed point | Dense RPO/CSR fixed point | Change |
| --- | ---: | ---: | ---: |
| Loop-continue dominance ancestry | 319.951 ms | 230.078 ms | -28.1% |
| Loop-continue normalization | 372.243 ms | 281.445 ms | -24.4% |
| `restructure_cfg` | 3.056 s | 2.745 s | -10.2% |
| SPIR-V XIR legalization | 18.169 s | 17.796 s | -2.1% |
| Native AST-to-SPIR-V | 31.873 s | 31.371 s | -1.6% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The 129 mutation-triggered rebuilds numbered 645,720 blocks and 807,853
edges. They made exactly 258 fixed-point passes, 1,291,182 block visits,
1,615,706 edge visits, and 1,274,837 parent climbs. The output remains
byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
Peak RSS is `1,652,660 KiB`.

After a complete all-thread rebuild, all 48 XIR tests pass. The system-STL
gates pass 9/9 dominance tests (56 assertions), 67/67 restructure tests
(1,340 assertions), and 350/350 pass tests (1,986 assertions). The RX 9070 XT
native Vulkan route passes 92/92 tests with 2,096 assertions. The initial
partial target build deliberately was not accepted as a gate: tests retaining
the older public report-struct size detected the mixed object ABI, after which
the complete rebuild passed.

The dense solve moves loop-continue ancestry below the next independent
hotspot. The current largest restructure leaf is selection-exit site scanning
at `307.857 ms` inside a `392.109 ms` drain, followed by the `290.665 ms` if
batch. Input/output verification remains exactly at the two public pass
boundaries. The next profile therefore targets selection-exit query and
invalidation work rather than weakening verification or approximating
dominance.

## Batched selection-exit SSA repair

Luisa `next@58984c670` moves selection-exit SSA repair to the exact drain
boundary. A 65,213-cycle-sample `perf` capture with no lost samples showed
that the apparent site-scan hotspot was mostly eager whole-function work:
within `drain_selection_exits`, exact reg2mem repair accounted for 43.2% of
samples, relation construction for 23.8%, and dominance construction for
18.0%. Every multi-target state-dispatch rewrite was scanning the complete
function and rebuilding dominance even though no intervening drain query
observes instruction operands.

The drain now records that SSA transport is required, reaches its final CFG
fixed point, and performs one exact repair over the resulting graph. This is
trace-equivalent to eager repair: a state-dispatch selector preserves the
original dynamic successor; later drain decisions inspect only CFG edges,
structured targets, and dominance; and no rewrite moves or evaluates the
path-local value. Therefore every feasible final use is still preceded on its
original dynamic path by the same definition. The final reg2mem pass
transports exactly those remaining syntactic cross-path uses. Other
restructure clients retain eager repair because they do not share this drain
boundary.

The regression `restructure_batches_state_dispatch_ssa_repair_at_drain_boundary`
uses two sequential three-way selections with distinct path-local return
values. It requires more logical repair requests than physical repairs,
exactly two cross-block spills, loaded return operands after repair, and a
verified output module. Production counters report nine requests and one
physical repair:

| Boundary | Dense-dominator baseline | Batched SSA repair | Change |
| --- | ---: | ---: | ---: |
| Repair requests / physical repairs | 9 / 9 | 9 / 1 | 8 scans removed |
| Final isolated SSA repair | included per site | 25.388 ms | measured boundary |
| Selection-exit site scan | 307.857 ms | 58.156 ms | -81.1% |
| `drain_selection_exits` | 392.109 ms | 163.168 ms | -58.4% |
| `restructure_cfg` | 2.745 s | 2.428 s | -11.6% |
| SPIR-V XIR legalization | 17.796 s | 17.201 s | -3.3% |
| Native AST-to-SPIR-V | 31.371 s | 30.399 s | -3.1% |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The production output remains byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
After a complete all-thread rebuild, all 48 XIR tests pass. The system-STL
gates pass 9/9 dominance tests (56 assertions), 68/68 restructure tests
(1,351 assertions), and 350/350 pass tests (1,989 assertions). The RX 9070 XT
native Vulkan route passes 92/92 tests with 2,096 assertions.

The largest remaining restructure leaves are `try_restructure_if_batch` at
271.223 ms and loop-continue normalization at 264.942 ms, including
216.460 ms of exact dominance ancestry. Verification remains exactly at the
two public pass boundaries (about 626 ms combined). The next profile targets
the if-batch computation and loop mutation strategy; it does not weaken those
boundary checks.

## Sparse selection-merge scoring

Luisa `next@8c6951520` removes the whole-function scans identified by the
if-batch profile. Of 152 samples below `try_restructure_if_batch`, 146 (96.1%)
were inside selection-merge inference; batch construction and CFG rewriting
were not material contributors. Each query had already value-numbered its
graph walk, but merge scoring still filtered the complete physical block table
up to three times.

The query now records the exact support set when an aggregate epoch is first
created and scores only those IDs. This is set-equivalent to the old scan
because every rejected ID had `_has_aggregate(id) == false`. Fallback to an
enclosing structured selection walks the header's immediate-dominator
ancestors: for reachable blocks, that chain is exactly the set satisfying
`dominates(candidate, header)`. Both sparse walks explicitly compare the
original dense block ID after equal support and distance scores, preserving
the old first-in-physical-order tie result rather than changing merge choice.

The 64-diamond regression includes 256 physically present but unreachable
blocks. It requires sparse scoring work to remain bounded by per-query support
instead of the definition block table, while retaining all 64 structured
selections and a verified module. On Lone Monk, 3,448 queries visit 584,856
entry-relative blocks and 734,477 edges, but score only 344,125 aggregate IDs
and inspect 7,397 dominator ancestors:

| Boundary | Batched-SSA baseline | Sparse merge scoring | Change |
| --- | ---: | ---: | ---: |
| `try_restructure_if_batch` | 271.223 ms | 101.746 ms | -62.5% |
| `restructure_cfg` | 2.428 s | 2.271 s | -6.4% |
| SPIR-V XIR legalization | 17.201 s | 17.112 s | -0.5% |
| Native AST-to-SPIR-V | 30.399 s | 30.464 s | within run noise |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The PPM is byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
A complete all-thread standard build and all 48 XIR tests pass. The focused
system-STL gates pass 9/9 dominance tests (56 assertions), 68/68 restructure
tests (1,354 assertions), and 350/350 pass tests (1,991 assertions). The RX
9070 XT native Vulkan route passes 92/92 tests with 2,096 assertions.

Loop-continue normalization is now the largest mutable transform at
269.093 ms, of which 218.621 ms is 129 exact dominance rebuilds. The next
optimization must therefore reduce CFG versions or incrementally maintain an
exact dominance relation; weakening dominance or moving the two public
verifier boundaries is not an acceptable substitute.

## Immutable-version loop-continue batches

Luisa `next@1db5b30a3` removes those per-site CFG versions. Each normalization
invocation first discovers every loop region and records its raw-edge rewrites
against one immutable CFG and dominance version. Only then are the guarded
actions applied, followed by one exact dense dominance rebuild and one
frontier materialization for the resulting version. The outer restructure
fixed point handles the next version.

The batching invariant is stronger than retaining a source-block pointer. A
planned action exists only when its supported raw terminator targets the old
block in the immutable input graph. Application checks that exact edge again.
Therefore an earlier action may consume a later action's precondition, making
it fail closed, but it cannot create a newly eligible planned action. Raw
branches converted to BREAK/CONTINUE retain their structured destination;
conditional and indexed edges are subdivided by a proxy; and canonical
backedges are factored through their declared continue target. None introduces
an incoming path from outside the entry-dominated loop region used to justify
another action. Thus non-conflicting actions preserve the membership facts
observed by the immutable analysis, while conflicting actions are reconsidered
on the next exact CFG version.

This edge condition was caught during the pre-commit proof audit: before it
was made explicit, the production planner recorded 17,463 syntactic
possibilities and relied on application-time rejection. With the condition it
records exactly the 134 edges that existed, and all 134 apply. A 16-loop
disjoint regression requires `planned == applied`; it would fail if an action
could become eligible only because an earlier action created its source edge.

The matched Lone Monk cold-JIT run is under
`/var/tmp/psycles-loop-continue-batch-guarded-20260810/`, against
`/var/tmp/psycles-if-merge-sparse-20260810/`:

| Boundary | Per-site dominance versions | Immutable-version batch | Change |
| --- | ---: | ---: | ---: |
| Loop-continue normalization | 269.093 ms | 12.865 ms | -95.2% |
| Loop-continue dominance rebuild | 218.621 ms / 129 | 4.823 ms / 8 | -97.8% |
| `restructure_cfg` | 2.271 s | 1.964 s | -13.5% |
| SPIR-V XIR legalization | 17.112 s | 16.752 s | -2.1% |
| Native AST-to-SPIR-V | 30.464 s | 29.984 s | -1.6% |
| End-to-end cold smoke | 32.65 s | 32.14 s | -1.6% |
| Peak RSS | 1,655,344 KiB | 1,656,720 KiB | within run noise |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The eight retained dominance versions number 14,064 blocks and 17,596 edges,
requiring 16 CHK passes, 28,112 block visits, 35,192 edge visits, and 27,542
parent climbs. All 48 XIR tests pass after a complete all-thread rebuild. The
system-STL focused gates pass 9/9 post-dominance tests (56 assertions), 68/68
restructure tests (1,358 assertions), and 350/350 pass tests (1,995
assertions). The RX 9070 XT native Vulkan route passes 92/92 tests with 2,096
assertions. The PPM is byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.

The largest remaining mutable leaves are now 229 exact post-dominator
constructions at 171.274 ms, selection-exit draining at 159.605 ms,
`try_restructure_loop` at 105.357 ms, and if batching at 104.430 ms. Input and
output verification remain the two public boundaries and account for about
614 ms; they are deliberately not candidates for removal. The next profile
must determine which post-dominator constructions share a CFG version before
changing that exact analysis.

## Batched straight-line CFG contraction

The whole-pipeline profile changed the priority. `simplify-cfg` was the largest
independent XIR leaf at `3.465 s`, well above any remaining restructure
component. Its straight-line transform found the first eligible edge, removed
exactly one successor block, and restarted the complete pass fixed point.
Lone Monk contained 892 such edges, so the pass repeatedly traversed the
function and rebuilt the structural-target set about once per contraction.

Luisa `next@6022a8f04` contracts every currently eligible maximal chain in one
physical-order scan. An edge `A -> B` is accepted only when A branches
unconditionally, B is neither the entry nor a declared structural target, B
has exactly A as its predecessor, B contains no Phi, and replacing B's outgoing
predecessor identity cannot affect a successor Phi. Under those conditions
every execution of A continues through B, no other path enters B, and
concatenating B's instructions onto A is the standard semantics-preserving
single-predecessor edge contraction.

After contracting `A -> B`, no unrelated source changes predecessor count or
structural role. Only A can become newly eligible through B's former
terminator, so the implementation immediately rechecks A and consumes that
maximal chain. Independent contractions commute; overlapping contractions are
the same chain. This is therefore the same least fixed point as the former
one-edge loop, in the same physical source order, without interleaving a
quadratic number of read-only scans. Detached blocks remain owned until the
snapshot worklist is exhausted, so skipped raw pointers cannot dangle.

A 256-edge regression requires exactly two scans: one maximal-chain mutation
and one fixed-point confirmation. It also bounds live block visits by twice
the input block count. On the production module, all 892 contractions require
30 module-wide per-function scans and 23,412 live source visits:

| Boundary | One edge per fixed-point scan | Maximal-chain batch | Change |
| --- | ---: | ---: | ---: |
| `simplify-cfg` | 3,465.08 ms | 53.44 ms | 64.8x |
| SPIR-V XIR legalization | 16.752 s | 13.346 s | -20.3% |
| Native AST-to-SPIR-V | 29.984 s | 26.570 s | -11.4% |
| End-to-end cold smoke | 32.14 s | 28.74 s | -10.6% |
| Peak RSS | 1,656,720 KiB | 1,658,920 KiB | within run noise |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The matched run is under
`/var/tmp/psycles-simplify-chain-batch-20260810/`. The PPM is byte-identical
with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
The simplify regression passes 21/21 tests with 79 assertions under both
container configurations. A complete all-thread build and all 48 XIR tests
pass; the system-STL pass suite passes 350/350 tests with 1,997 assertions;
and the RX 9070 XT native Vulkan route passes 92/92 tests with 2,096
assertions.

After removing this quadratic scan, the largest independent XIR leaves are
the five DCE invocations at 2.990 s total, SPIR-V pointer-argument inlining at
2.374 s, CFG destructuring at 2.186 s, and restructure at about 2.005 s. The
next profile should split DCE's initial discovery scans from its mutation
worklist before changing another pass.

## Sparse product fixed point for DCE

Luisa `next@4e81dcc0f` replaces DCE's candidate numbering and repeated
whole-function confirmation with two event-driven worklists. For ordinary
dead values, unlinking the last live user is the exact event that can make a
removable definition dead. The solver therefore seeds zero-user definitions
once, detaches them while retaining ownership, and follows only operand
use-lists that just became empty. Repeated operands in one instruction are
deduplicated before testing that transition.

Ordinary values and write-only allocas cannot be solved as unrelated phases.
For example, deleting a write-only sink alloca can make a load from a source
alloca unused; deleting that load then makes the source alloca write-only. The
first implementation audit detected this formally relevant case because the
production pass removed 148 fewer instructions even though later passes
produced identical SPIR-V. The final solver treats the two rules as a product
fixed point. Every alloca is considered initially. A rejected alloca is
requeued only when an actual removed user of its exact `Alloca -> GEP*`
pointer chain changes the write-only predicate. No other mutation can change
that predicate.

Both deletion rules are monotone and the instruction set is finite. Ordinary
work is exhausted before the next alloca query; each successful action
strictly removes at least one linked instruction; and a failed alloca is
revisited only after one of its blockers disappears. The worklists therefore
terminate at the same least fixed point as the former alternating global
scans. Production `removed_inst`, `removed_block`, and dead-value worklist
counts match the old implementation for all five invocations, including the
148-node cascade found by the audit.

The regressions cover a 64-node dead chain, a repeated shared operand, value
propagation out of an explicitly removed alloca graph, and a two-alloca
`sink -> load -> source` cascade. The cascade must finish in one invocation,
scan each of its six instructions once, and make a second invocation a strict
no-op. The matched cold Lone Monk run is under
`/var/tmp/psycles-dce-product-worklists-20260810/`, against
`/var/tmp/psycles-simplify-chain-batch-20260810/`:

| Boundary | Numbered/repeated DCE | Sparse product worklists | Change |
| --- | ---: | ---: | ---: |
| DCE instruction scans, five calls | 5,279,368 | 2,454,040 | -53.5% |
| DCE time, five calls | 2,989.58 ms | 1,079.94 ms | -63.9% |
| Post-inline cleanup group | 3,770.64 ms | 1,924.46 ms | -49.0% |
| SPIR-V XIR legalization | 13.346 s | 11.271 s | -15.5% |
| Native AST-to-SPIR-V | 26.570 s | 24.468 s | -7.9% |
| End-to-end cold smoke | 28.74 s | 26.65 s | -7.3% |
| Peak RSS | 1,658,920 KiB | 1,656,896 KiB | within run noise |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The PPM remains byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
A complete all-thread build and all 48 XIR tests pass. The pass suite passes
353/353 tests with 2,018 assertions under both the default and system-STL
configurations. The RX 9070 XT native Vulkan route passes 92/92 tests with
2,096 assertions.

The post-change 26,000-cycle-sample profile lost no samples. DCE's remaining
instruction-set hash insertion accounts for only about 0.55% self time, so a
larger alloca-container rewrite is no longer the next priority. The largest
independent leaves are now SPIR-V pointer-argument inlining at 2.318 s, CFG
destructuring at 2.119 s, and restructuring at 1.960 s. Exact XIR verification
is also visible in the profile; restructure retains exactly its input and
output boundary checks (297.944 ms and 317.966 ms). The next analysis targets
pointer-argument planning/mutation and verifier data structures, not verifier
frequency or semantics.

## Sparse SPIR-V argument-usage propagation

Luisa `next@7636699b9` replaces the SPIR-V callable argument-usage global
fixed point with an explicit reverse call dependency graph. The previous
solver rescanned every instruction in every structurally owned function after
any argument changed. On a call chain, function-list order could therefore
turn one new usage bit per edge into one complete module traversal per level.

The new solver still scans each exact structural closure once per immutable
module version. During that scan it records only transfers whose actual value
is an argument of the caller, since every other actual is a no-op under the
existing transfer function. Local resource operations initialize the finite
argument lattice. When a formal's `Usage` bits or one of its five SPIR-V
feature requirements grows, only the recorded caller actuals depending on
that formal are scheduled. Queue coalescing reads the latest lattice value,
so propagation order does not affect the result.

Formally, every dependency implements the same monotone transfer
`state(callee formal) -> state(caller actual)`. The lattice is a finite product
of two usage bits and five booleans. Chaotic worklist iteration from the same
local seeds therefore reaches the same least fixed point as repeated global
scans. Each slot can grow only finitely many times, while unrelated
instructions are never revisited.

The regression deliberately creates a 128-callable chain in the adverse
physical order. It requires all 129 structural closures and all 258
instructions to be scanned exactly once, with 128 recorded dependencies, 129
worklist pops, and exactly 128 dependency visits. The terminal buffer read
must still propagate to the kernel buffer. A separate legalization assertion
requires one pointer-call batch plus its empty confirmation to perform exactly
two argument analyses.

Lone Monk requires two specialization batches plus the final empty
confirmation. Across those three immutable module versions, the production
report records 87 structural closures, 3,982,983 one-time instruction visits,
3,798 call dependencies, 171 worklist pops, and only 199 dependency visits.
The matched cold run is under
`/var/tmp/psycles-spirv-arg-usage-worklist-20260810/`, against
`/var/tmp/psycles-dce-product-worklists-20260810/`:

| Boundary | Global rescans | Sparse dependencies | Change |
| --- | ---: | ---: | ---: |
| Argument-usage self samples | 2.50% | 0.83% | -66.8% |
| `inline-spirv-pointer-args` | 2,317.93 ms | 1,930.77 ms | -16.7% |
| SPIR-V XIR legalization | 11.271 s | 11.018 s | -2.2% |
| Native AST-to-SPIR-V | 24.468 s | 24.175 s | -1.2% |
| End-to-end cold smoke | 26.65 s | 26.41 s | -0.9% |
| Peak RSS | 1,656,896 KiB | 1,658,144 KiB | within run noise |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The PPM remains byte-identical with SHA-256
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b`.
The complete default-STL build passes, as do all 21 SPIR-V tests. The focused
pointer-legalization suite passes 21/21 tests with 188 assertions, and the RX
9070 XT native Vulkan route passes 92/92 tests with 2,096 assertions. The
Psycles production build itself uses system STL; its complete rebuild and the
matched render above pass with that configuration.

The new no-loss 26,000-sample profile moves argument propagation below the
next costs. Inline call-site barrier preflight is now the largest named
pointer-legalization subproblem at 1.46% self time, followed by readonly
resource-origin analysis at 0.82%. The next change should cache immutable
per-function inline properties across call sites within one plan, while still
invalidating a summary if an earlier planned inline mutates that function.

## Versioned inline-call preflight reuse

Luisa `next@9df5171e1` makes that invalidation rule explicit. The selected-call
pass first summarizes each function definition at most once while the module
is immutable. A summary contains the single-block strategy, return-shape and
metadata legality, and both caller-barrier predicates. Call-specific argument
and metadata checks are still performed for every selected call before any IR
is changed.

Let `S(f, v)` be the summary of function `f` at definition version `v`. An
inline operation changes only the caller definition; it does not change its
callee. Therefore, after applying a prepared call `caller -> callee`, every
cached summary remains valid except the one for `caller`. The implementation
maintains the monotone set of functions already changed as callers:

```text
M(0) = empty
M(i + 1) = M(i) union {caller(i)}
```

When `callee` is not in `M`, application uses the already-selected
single-block or multi-block strategy without repeating whole-definition
validation. When `callee` is in `M`, the pass takes the original generic path
and completely revalidates its current definition. This is an exact version
test, not a heuristic cache timeout. Multiple independent calls in one caller
do not invalidate their shared callee; a nested call chain does invalidate the
middle function when it changes from caller to later callee.

The scale regression creates 32 selected calls to one 65-instruction callable
and requires exactly one function summary, 65 summary instruction visits, 32
cached applications, and zero revalidations. A separate
`inner -> middle -> kernel` regression applies the inner call first and then
uses the mutated `middle` as a callee; it requires one cached application and
one full revalidation. Both end with no reachable calls and verified IR.

The matched Lone Monk cold run is under
`/var/tmp/psycles-inline-prevalidated-apply-20260810/`, against the immediately
preceding summary-only checkpoint in
`/var/tmp/psycles-inline-summary-cache-20260810/`:

| Boundary | Summary-only application | Versioned prevalidated application | Change |
| --- | ---: | ---: | ---: |
| Selected pointer calls | 7 | 7 | unchanged |
| Cached / revalidated applications | not reported | 7 / 0 | all immutable |
| Summarized functions / instructions | not aggregated | 5 / 563,685 | one scan per function/version |
| `inline-spirv-pointer-args` | 1,845.33 ms | 1,620.18 ms | -12.2% |
| SPIR-V XIR legalization | 10,975.70 ms | 10,766.86 ms | -1.9% |
| Native AST-to-SPIR-V | 24,231.55 ms | 24,038.64 ms | -0.8% |
| Complete shader JIT | 24.4005 s | 24.2129 s | -0.8% |
| Process wall time | 26.39 s | 26.22 s | -0.6% |
| Peak RSS | 1,657,900 KiB | 1,656,976 KiB | effectively unchanged |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The PPM and linear Combined output are byte-identical to the preceding
checkpoint, with SHA-256 respectively
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b` and
`4f93bceff43a46a086454e9a50745a497b39cd27e8a694f617a5c934fe3ed3eb`.
A complete all-thread standard build passes, as do `unit_xir` 48/48, all 21
SPIR-V tests, the pointer legalization suite (21/21 tests, 188 assertions),
and the RX 9070 XT native Vulkan route (92/92 tests, 2,096 assertions). The
system-STL pass suite independently passes 355/355 tests with 2,051
assertions, and the production Psycles build and render also use system STL.

The remaining 1.62-second pointer pass is no longer dominated by repeated
immutable legality scans. Its next profile must separate actual instruction
cloning, resolver insertion, use-list rewrites, and callable cleanup. Boundary
verification and malformed-input validation remain intact.

## Use-list call graph and sparse recursion SCCs

The next Luisa checkpoint replaces the inline pass's per-start reachability
search with one exact strongly connected component decomposition. More
importantly, it no longer discovers call edges by scanning every instruction
in every callable. A linked call has exactly one callee operand use, so
enumerating each callable's use list and accepting only the exact callee
operand is set-equivalent to a full instruction scan over owned calls:

```text
E = {(parent_function(c), f) |
     u is in uses(f), user(u) = c,
     c is CallInst, u = c.callee_operand, c.callee = f}
```

The operand-identity predicate is necessary: a function value used as an
ordinary call argument is not a call-graph edge. `parent_function()` retains
calls in disconnected but owned blocks, matching the inline pass's candidate
domain. Removing an instruction unlinks its operands, so detached calls do not
remain in the use lists.

The implementation assigns dense function IDs, materializes forward and
reverse CSR graphs, and runs iterative Kosaraju traversal. Each function is
visited once in each direction and each edge is inspected once in each
direction, giving `O(F + U + E)` time and `O(F + E)` auxiliary storage, where
`U` is the number of function uses inspected. A multi-vertex SCC is recursive;
a singleton SCC is recursive exactly when it has an explicit self edge. No
recursion-stack depth depends on shader graph depth.

Three regressions make these invariants observable. A two-function cycle must
classify both vertices as recursive with exactly `2F` vertex and `2E` edge
visits. A 128-function chain must inspect 128 function uses (127 internal calls
plus the selected kernel call), recover 127 callable edges, and perform exactly
256 vertex and 254 edge visits. A deliberately malformed ordinary argument use
must be counted as a use but must not invent an edge or recursive SCC. The
existing disconnected-owned-block recursion regression continues to pass.

The matched cold Lone Monk run is under
`/var/tmp/psycles-inline-use-scc-20260810/`, against
`/var/tmp/psycles-inline-prevalidated-apply-20260810/`:

| Boundary | Repeated reachability | Use-list sparse SCC | Change |
| --- | ---: | ---: | ---: |
| Production callable uses inspected | not reported | 995 | sparse domain |
| Production graph vertices / edges | not reported | 57 / 881 | exact graph |
| SCC vertex / edge visits | not reported | 114 / 1,762 | exactly `2F` / `2E` |
| `inline-spirv-pointer-args` | 1,620.18 ms | 1,515.62 ms | -6.5% |
| Ordinary selected-call inlining | 764.42 ms | 689.71 ms | -9.8% |
| SPIR-V XIR legalization | 10,766.86 ms | 10,511.51 ms | -2.4% |
| Native AST-to-SPIR-V | 24,038.64 ms | 23,696.24 ms | -1.4% |
| Complete shader JIT | 24.2129 s | 23.8657 s | -1.4% |
| Process wall time | 26.22 s | 25.86 s | -1.4% |
| Peak RSS | 1,656,976 KiB | 1,658,696 KiB | within run noise |
| Raw SPIR-V | 1,431,985 words | 1,431,985 words | identical size |
| Optimized SPIR-V | 1,116,158 words | 1,116,158 words | identical size |

The PPM and linear Combined hashes remain respectively
`3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b` and
`4f93bceff43a46a086454e9a50745a497b39cd27e8a694f617a5c934fe3ed3eb`.
The complete all-thread standard build passes, as do `unit_xir` 48/48, all 21
SPIR-V tests, the pointer suite (21/21 tests, 188 assertions), the system-STL
pass suite (358/358 tests, 2,101 assertions), and the RX 9070 XT native Vulkan
route (92/92 tests, 2,096 assertions).

A fresh 51,058-sample profile lost no samples. Recursion discovery no longer
appears as an independent inline leaf. Selected-call inlining is now 1.86%
inclusive, dominated by actual multi-block cloning (1.49%): metadata-aware
instruction cloning is 0.71%, resolver hash-map insertion is 0.45%, and
callable destruction is 0.27%. The full pointer legalizer has only 0.58%
unattributed self time, so the next target is clone-time value resolution,
not another speculative rewrite of its local preflight.
