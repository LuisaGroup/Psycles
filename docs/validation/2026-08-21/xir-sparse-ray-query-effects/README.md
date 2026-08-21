# Sparse XIR ray-query scratch effects

## Scope and result

The compact surface program makes Barbershop's staged path practical at run
time, but exposed a host-compiler complexity defect in Luisa's coroutine
ray-query normalization. On the same Blender 5.2 Barbershop export, RX 9070 XT,
8x8 image, 256 fixed spp, and one 64-sample dispatch, the complete shader-JIT
interval fell from `99.7033--100.821 s` to `7.50286--7.77417 s`. The final
warm shader-JIT interval is now `5.83266--5.96071 s`; coroutine compilation is
`4.71652--4.83497 s`, pre-distill optimization is `1.17985--1.29784 s`, CFG
distillation is `0.675203--0.710138 s`, and ray-query normalization is
`0.357776--0.392472 s` instead of the original dominant host-compiler path.

The optimization changes host analysis only. The generated coroutine graph is
unchanged at 9 subroutines, 212 frame fields, and 3,184 frame bytes. Render-only
time remained in the same narrow range (`0.359723 s` before and
`0.328770--0.329317 s` after). Across the final 15-pass EXR comparison, the
maximum RMS is `3.58471e-8`, maximum relative RMS is `1.33829e-7`, and maximum
absolute error is `2.38419e-7`; this is the normal non-deterministic
floating-point accumulation envelope, not a structured image change.

## Root cause and formal model

For each candidate handler-local aggregate `A`, the original analysis attached
two dense `AggregateFieldBitmask(A)` values to every XIR instruction, including
instructions unrelated to `A`. With `L` primitive leaves and `N` instructions,
the identity effects alone therefore performed `Theta(NL/64)` allocation,
zeroing, copying, and destruction. Barbershop's profile placed most host time
in aggregate-mask construction/size operations and
`RayQueryHandlerScratchAnalyzer::instruction_effect`; this work did not depend
on the number of actual accesses.

The dataflow domain for a path is the pair `(need, define)`:

```text
need(A; B)   = need(A) union (need(B) - define(A))
define(A; B) = define(A) union define(B)

need(join)   = union of predecessor needs
define(join) = intersection of predecessor defines
```

`need` means a leaf may be read before a must-definition, while `define` means
the leaf is defined on every represented path. The lattice identity is
`(empty, empty)`. Luisa now represents that identity with two disengaged
optionals and materializes a dense mask only at a related load, store, or
reference-call boundary. The equations and fixed point are unchanged; only the
physical representation of the bottom element is sparse. This changes the
unrelated-instruction cost from `Theta(NL/64)` to `Theta(N)`, while relevant
mask operations retain their exact aggregate-leaf semantics.

Pointer provenance was the next measured hotspot. Resolution now uses one
memo table with the conventional white/gray/black DFS states: absence is
white, a pessimistic invalid cache entry is gray, and the final summary is
black. A back edge through a malformed cyclic GEP remains conservatively
invalid. Pure SSA operands are rejected before provenance recursion, and the
analysis validates only the operand roles admitted by each XIR instruction
kind. This removes one temporary active-set allocation per lookup without
weakening the accepted language.

`AggregateFieldBitmask` copy and move construction were also corrected to
reuse the already interned field tree. Copy allocates and copies exactly once;
move steals the large backing allocation rather than allocating, clearing, and
swapping a second one.

## Measured decomposition

| Revision | Complete shader JIT | Ray-query normalization | Pre-distill | CFG distill | Render-only |
|---|---:|---:|---:|---:|---:|
| dense effects | `99.7033--100.821 s` | not separately instrumented | not separately instrumented | not separately instrumented | `0.359723 s` |
| sparse identity masks | `32.9784 s` | `20.9157 s` | `6.56713 s` | `1.64452 s` | `0.360227 s` |
| cached provenance, pre-rebase | `28.7626 s` | `16.3784 s` | `6.59192 s` | `1.66676 s` | `0.364793 s` |
| sparse effects `b1a969e23`, rebased | `29.1316 s` | `16.8576 s` | `6.49122 s` | `1.69597 s` | `0.364140 s` |
| dead selection proof removed, pre-rebase | `20.3212 s` | `8.20558 s` | `6.39471 s` | `1.68025 s` | `0.358903 s` |
| selection proof demand `933811bae` | `20.3864 s` | `8.24251 s` | `6.53486 s` | `1.66747 s` | `0.360216 s` |
| final sparse pointer support `ef07fb841` | `15.6040 s` | `3.62191 s` | `6.45809 s` | `1.65430 s` | `0.360176 s` |
| demand-sparse coroutine lifetimes `e213b46b5` | `11.1738 s` | `3.55809 s` | `1.98082 s` | `1.66059 s` | `0.329078 s` |
| sparse handler event domain `d32a3f93b` | `8.7816 s` | `0.500144 s` | `2.24290 s` | `1.48968 s` | `0.332590 s` |
| demand-driven restructure frontiers `3d2c322f1` | `7.50286--7.77417 s` | `0.371753--0.395625 s` | `1.96247--2.19189 s` | `1.64413--1.66490 s` | `0.328332--0.329465 s` |
| projected scope dataflow and RPO `b0f235f0f` | `6.51264--7.19622 s` | `0.363771--0.381739 s` | `1.94524--2.08270 s` | `0.691603--0.729620 s` | `0.328954--0.329396 s` |
| demand-projected reaching values `ab63c926b` | `6.03530--6.24901 s` | `0.390374--0.395700 s` | `1.46225--1.50582 s` | `0.689384--0.703952 s` | `0.329001--0.329528 s` |
| snapshot instruction order `76be1a09f` | `5.83266--5.96071 s` | `0.357776--0.392472 s` | `1.17985--1.29784 s` | `0.675203--0.710138 s` | `0.328770--0.329317 s` |

The final whole-JIT reduction relative to the two dense observations is
`16.73--17.29x` (`94.0--94.2%`). Frame layout takes approximately `0.1 ms`, so
frame layout is not being mistaken for this host-analysis bottleneck.

## Selection proof demand

The remaining profile exposed a second, independent duplicate: handler-local
scratch was proved once to estimate the callback capture count during loop
selection, then proved again to perform the selected loop's ABI rewrite. Let
`R` be raw input captures, `O` output captures, `L` the inputs localization can
remove, and `B` the capture budget. Selection asks only

```text
R - L + O <= B, where 0 <= L <= R.
```

There are exactly three cases:

```text
O > B       : false for every L; the proof cannot change the result
R + O <= B  : true for every L; the proof cannot change the result
otherwise  : the result may depend on L; run the proof
```

Luisa commit `933811bae` implements this demand rule with overflow-free
comparisons. The unbounded default is the second case, so the selection scan is
removed while actual lowering still runs the complete definite-initialization
proof once before changing the callback ABI. Relative to `b1a969e23`, this
halves ray-query normalization (`16.8576 -> 8.24251 s`) and reduces whole JIT by
30.0% (`29.1316 -> 20.3864 s`). A pass-report counter makes the control
dependency testable: it is zero for an unbounded already-eligible candidate and
one when a finite zero budget can be satisfied only by localizing scratch.

## Sparse pointer support

After duplicate selection work was removed, `perf` still placed the largest
normalization symbol in `RayQueryHandlerScratchAnalyzer::resolve_pointer`.
Each candidate alloca was resolving every other lvalue in the handler even
though the earlier ownership proof had already computed the candidate's exact
pointer support.

For a root `p`, let `support(p)` be the transitive use-list closure through GEP
base edges. XIR has no implicit pointer casts or hidden alias-producing
instructions: a use outside load/store, GEP-base, or an explicit reference
call rejects localization. Therefore, after the ownership proof succeeds, a
parent-function instruction whose lvalue operands are outside `support(p)` has
the identity effect in `p`'s product-lattice coordinate. A related reference
call is the boundary: its callable is analyzed without the parent support
filter after binding the formal reference to the actual pointer view.

Commit `ef07fb841` carries the proven use-list closure into the field-sensitive
dataflow and filters unsupported lvalues before provenance lookup. It also
shares one immutable provenance memo across block visits and fixed-point
iterations; provenance is a function of the environment and value graph, not
of the current need/define fact. Relative to `933811bae`, normalization falls
56.1% (`8.24251 -> 3.62191 s`) and whole JIT falls 23.5%
(`20.3864 -> 15.6040 s`).

The compact staged path is also a meaningful runtime boundary. Against the
same compact-populate megakernel (`2.08996 s` render-only), the staged path at
`0.359723 s` is `5.81x` faster. A prior one-sample dispatch took `6.743 s`
because it issued 256 host dispatch chunks; the accepted result uses
`samples_per_dispatch=64`, as required by the per-(pixel, sample) topology.

## Demand-sparse coroutine lifetime proofs

Once ray-query normalization ceased to dominate, a pass trace showed that
`coro-alloca-scope` consumed `5.70 s` of a `6.67 s` pre-distill pipeline. The
real Barbershop module has 68,341 post-SROA locals. Of these, 60,637 use the
single-definition motion rule, 7,682 pass the general Must proof, and only 12
are rejected. The former implementation nevertheless explored 2,530,940
guarded abstract states for those 12 rejected candidates.

The guarded domain is a forward May partition of predicate cubes paired with
a Must fact vector. State merging intersects facts, and predicate widening
admits paths but never adds a definite fact. Therefore, once a reached read
lacks a fact, no later fixed-point update can make that same abstract state
definitely initialized. Commit `e213b46b5` checks reads during propagation and
conservatively rejects at this monotone failure point. It retains the
SCCP-like correlated-predicate proof: both correlated success regressions
still pass. Guarded evaluations on the real module fall from 2,530,940 to
133,483 and widenings fall from 9,702 to 178.

The next profile split the remaining proof construction time:

```text
problem construction  2.72 s
  reverse slice       0.026 s
  fact layout         0.009 s
  event transfer      2.68 s
unconditional solve   0.083 s
guarded solve         0.15 s
```

Event construction now projects directly onto the candidate's ordered pointer
users. A once-per-function instruction-location index supplies deterministic
block and instruction order; unrelated instructions and empty CFG blocks do
not enter the per-candidate event domain. This is exact because the lifetime
transfer function changes only at explicit uses in the proven pointer closure.

The `2.68 s` event transfer was then traced to one reference summary for
`path_volume_segment`: 21,666 discovered pointer views, 47 requested reference
coordinates, and 2,248 reachable blocks. The underlying pointer-usage product
is separable by pointer root. Two views rooted at distinct allocas or formal
references cannot overlap in XIR; possible aliasing between call-site actuals
is handled by the caller's read-before-all-Must-writes rule. Pointer discovery
and malformed-use validation remain whole-function, while access projection
now indexes only result coordinates with the same root.

Finally, pointer-usage transfer storage and scheduling were corrected without
changing the lattice equations. Two type-shaped scratch states are reused
instead of allocating a map and one heap-owned `PointerUsage` per coordinate
at every block evaluation. Forward changes enqueue only successors; backward
changes enqueue only predecessors. With the same top/bottom initialization,
meet operators, and monotone transfer functions, this chaotic worklist reaches
the same fixed points as round-robin iteration. On `path_volume_segment`:

| Pointer-usage phase | Round-robin | Sparse worklist |
|---|---:|---:|
| Forward evaluations | 101,160 (45 full rounds) | 7,369 |
| Forward time | `561.104 ms` | `40.721 ms` |
| Backward evaluations | 92,168 (41 full rounds) | 21,293 |
| Backward time | `355.527 ms` | `78.910 ms` |
| Complete reference summary | `~2.74 s` before storage reuse | `158.510 ms` |

The complete Barbershop shader JIT is 28.4% lower than `ef07fb841`
(`15.6040 -> 11.1738 s`), while pre-distill is 69.3% lower
(`6.45809 -> 1.98082 s`). Coroutine structure is unchanged at 9 subroutines,
212 frame fields, and 3,184 bytes. The comparison report
`/var/tmp/psycles-compact-populate-20260821/barbershop-pointer-worklist-comparison.json`
covers all 15 film passes: maximum absolute error is `4.76837e-7`, maximum RMS
is `5.01295e-8`, and maximum relative RMS is `1.36393e-7`. This is the existing
floating atomic-order noise, not a structured image difference.

## Sparse handler event domain

The next whole-process `perf` capture showed that the remaining
`3.55809 s` ray-query phase was still evaluating every handler instruction for
every candidate root. The earlier pointer-support filter made unrelated
lvalues cheap, but the implementation still called `instruction_effect` and
queried that support for all scalar and unrelated-pointer operands. On the
profiled workload, `instruction_effect` was 11.18% inclusive process time and
general pointer hashing alone was 3.59% self time.

Let `S(p)` be the already-proven GEP-closed pointer support for root `p`, and
let

```text
R(p, b) = [i in instructions(b) | operands(i) intersects S(p)]
```

retain source order. For an instruction outside `R(p, b)`, every operand is
outside `S(p)`. The provenance language has only GEP as an address-preserving
instruction, so the instruction's effect in root coordinate `p` is exactly
the path-effect identity `(empty, empty)`. Removing those identity elements
from sequential composition is therefore semantics preserving:

```text
fold(effect, instructions(b)) = fold(effect, R(p, b)).
```

The implementation now constructs `R` from XIR use lists and scans only blocks
that contain a relevant use to recover instruction order. Pointer-keyed
temporary tables use address identity rather than the general byte-content
hash. The CFG equations are unchanged. In addition, a block's ordered effect
is a pure function of the immutable pointer environment, so it is computed
lazily on first reach and memoized; subsequent fixed-point visits apply the
same block transfer instead of reinterpreting its instructions. Lazy creation
preserves the prior treatment of malformed instructions in unreachable
blocks.

The complexity regression exposes the number of semantic instruction-effect
evaluations through `PassReport`. A handler containing 8,192 unrelated scalar
operations and 8,192 unrelated GEP/load chains must evaluate exactly the two
events on its real scratch root. This is an algorithmic assertion, not a wall
clock threshold.

On the same Barbershop command, ray-query normalization falls 85.9%
(`3.55809 -> 0.500144 s`), complete coroutine compilation falls 28.2%
(`10.08268 -> 7.23847 s`), and shader JIT falls 21.4%
(`11.1738 -> 8.7816 s`). The follow-up `perf` capture contains no sample for
`instruction_effect` above the 0.05% reporting threshold; the remaining
scratch-summary construction is 0.80% self time. Coroutine structure remains
9 subroutines, 212 frame fields, and 3,184 bytes.

The all-pass comparison report is
`/var/tmp/psycles-compact-populate-20260821/barbershop-ray-sparse-events-comparison.json`.
It compares the new EXR against the immediately preceding compiler-only
checkpoint across Combined, Normal, Albedo/color, all direct/indirect light,
emission, environment, transmission, and volume passes. Maximum absolute
error is `2.38419e-7`, maximum RMS is `4.34352e-8`, and maximum relative RMS is
`1.50384e-7`.

## Demand-driven restructure dominance frontiers

The next profile put `restructure_cfg_on_definition_in_place` at 6.77% of the
whole process and `compute_dom_tree` at 2.30%. Dominance-frontier construction
alone was 1.02%, even though a complete consumer audit found exactly one
semantic client in `restructure_cfg`: the sparse post-merge selection re-entry
witness search. Loop recovery, if recovery, construct-entry repair, selection
exit repair, and all intermediate CFG refreshes consume only immediate
dominators, ancestry, or depth.

For an immutable CFG version `G`, let `D(G)` be its immediate-dominator tree
and `DF(D(G))` its dominance-frontier relation. `DF` is a pure derivative of
`D`; it does not participate in computing `D`. If an analysis observes only
`dominates`, `immediate_dominator`, or dominator depth, replacing

```text
build D(G); build DF(D(G)); run ancestry consumer
```

with

```text
build D(G); run ancestry consumer
```

is observationally exact. The re-entry theorem still explicitly materializes
`DF` immediately before it enumerates `DF(merge)`, so its candidate set and
all CFG fixed-point equations are unchanged. Public `compute_dom_tree` defaults
are unchanged; only the restructure pass uses an ancestry-only construction
helper.

Commit `3d2c322f1` applies this demand contract to every dominance refresh in
the pass and moves frontier materialization to the re-entry analyzer. The
regression counter must equal one materialization per transform re-entry query
plus the final audit; a loop-continue rewrite can no longer manufacture an
eager frontier as a side effect. Across three Barbershop canaries, including a
final run rebased onto upstream `89f9da8e2`, continuation restructuring is
stable at `1409.643--1435.250 ms`, down 23.1--24.5% from `1865.846 ms`.
Complete coroutine compilation falls to `6.38855--6.65823 s`, and shader JIT
falls to `7.50286--7.77417 s`.

The generated coroutine remains 9 subroutines, 212 frame fields, and 3,184
bytes. The 15-pass comparison at
`/var/tmp/psycles-compact-populate-20260821/barbershop-demand-frontiers-rebased-comparison.json`
has maximum absolute error `4.76837e-7`, maximum RMS `5.02054e-8`, and maximum
relative RMS `1.27865e-7`. Raw runs are
`barbershop-demand-frontiers.log` and
`barbershop-demand-frontiers-repeat.log`, with the pushed-upstream validation
in `barbershop-demand-frontiers-rebased.log`, in the same directory.

## Projected scope dataflow and RPO scheduling

The next profile showed that CFG distillation still solved every coroutine
scope over the complete global atom domain. Barbershop has 255,078 global
atoms and 9 scopes, although most atoms are absent from most scopes. For block
`B`, define three local generators: `K_B` is its must-kill set, `T_B` its
may-touch set, and `E_B` its use-before-local-kill set. The exact active
universe of scope `s` is

```text
U_s = union over B in s of (K_B union T_B union E_B).
```

The forward Must/May equations are coordinate-wise over a product lattice.
For an atom outside `U_s`, every local generator is zero. Scope construction
includes only root-reachable blocks, the entry boundary is empty, and
therefore induction on path length gives zero for that coordinate at every
block. Solving on `U_s` and zero-extending the result is consequently
isomorphic to solving the global product; this is an exact domain projection,
not a scene heuristic. Cross-scope relationships are expanded once before
projection into immutable global atom numbers.

Sparse block-effect discovery uses three marker bits per global atom plus
compact vectors, avoiding one hash allocation per block. The fixed-point
solver visits blocks in DFS reverse postorder. RPO changes no equation and no
fixed point; on an acyclic scope every predecessor is visited before its
consumer, so each block is evaluated exactly once. Cyclic scopes retain the
same monotone worklist and revisit only blocks affected through backedges.

On Barbershop, projection reduces the scope product from 2,295,702 atom
coordinates to 274,190 (`11.94%`, or `8.37x` smaller). Total Must/May block
evaluations fall from 126,083 to 28,394 (`77.5%`). The largest material scope
still contains 237,456 atoms, so the measurement does not hide the genuinely
dense case. CFG distillation falls from `1.64413--1.66490 s` to
`0.691603--0.729620 s` (`55.6--58.5%`), while the fixed-point solver's self
time falls from 3.11% to 0.54% of the complete process. Projection collection
itself accounts for 0.05%. The new leading host-compiler opportunity is
`resolve_reaching_values` in pre-distillation, at approximately 1.95%.

This work also moves the dataflow model and solver into
`coro_cfg_dataflow.h/.cpp`: the pass driver is now 1,652 lines and the new
implementation unit is 892 lines after formatting. This is a real C++ API and
implementation split rather than textual inclusion. Optional public stats
make the domain sizes and iteration counts observable without introducing a
timing threshold.

Commit `b0f235f0f` preserves the generated program at 9 subroutines, 212 frame
fields, and 3,184 bytes. The final 15-pass comparison is
`/var/tmp/psycles-compact-populate-20260821/barbershop-scope-rpo-final-comparison.json`;
its maximum absolute error is `4.7683716e-7`, maximum RMS is `4.1643165e-8`,
and maximum relative RMS is `1.0743231e-7`. Raw runs are
`barbershop-scope-rpo-final-a.log`, `barbershop-scope-rpo-final-b.log`, and
`barbershop-scope-rpo-profile.log` in the same directory.

## Demand-projected reaching values

The next profile identified `resolve_reaching_values` at 1.95% self time. The
cause was again a product-domain mismatch, but for a different analysis:
Barbershop has 2,676 multi-store local-state candidates and a 4,747-block
semantic CFG. The pass independently allocated and solved the complete CFG for
every candidate, performing 25,041,416 block transfers even though most
candidate loads are separated from most blocks by an overwriting store.

For one candidate, let `last(B)` be the last whole-object store in block `B`.
Its original equations are

```text
in(entry) = undefined
in(B)     = meet(edge(P, B, out(P))) for every predecessor P
out(B)    = unique(value(last(B))) if last(B) exists, otherwise in(B).
```

Only a load before the first store in its block observes `in(B)`; later loads
are resolved directly by the preceding local store. Let `D` be those
input-demanding blocks, and let `A` be the least set containing `D` such that,
for every non-entry `B` in `A`, each predecessor without a store is also in
`A`. A predecessor with a store is an exact boundary because its transfer is a
constant function of no incoming state. Thus every dependency of every
demanded `in(B)` is either another coordinate in `A` or a known boundary
constant. Restricting the equations to `A` and substituting those constants is
therefore isomorphic to the demanded projection of the full least fixed point.
No path or value is approximated. Suspend-edge state is applied at the same
boundary edge, and cycles inside `A` retain the monotone worklist fixed point.

The implementation discovers `A` with a reverse walk that stops at
overwriting-store blocks, then solves the projected graph in semantic-CFG
reverse postorder. A workspace shared across candidates retains sparse event
storage and resets only blocks touched by the previous coordinate; it no
longer constructs 4,747 empty event vectors 2,676 times.

On Barbershop, the projected domains contain 166,638 active blocks in total,
an average of 62.27 per candidate and 1.31% of the full
`2,676 * 4,747 = 12,702,972` product. Worklist evaluations fall 99.28%, from
25,041,416 to 180,101 (`139.0x`). Pre-distill optimization falls
22.6--29.8%, from `1.94524--2.08270 s` to `1.46225--1.50582 s`. The old
resolver disappears below the flat profile reporting threshold; the complete
rematerialization pass has only about 0.03% self time. Warm coroutine
compilation is `4.91388--5.11661 s`, and shader JIT is
`6.03530--6.24901 s`.

The optional dense product solver remains available as a semantic oracle
through `CoroRematerializeOptions::verify_dense_reaching_values`; the normal
coroutine pipeline maps
`LUISA_CORO_VERIFY_DENSE_REACHING_VALUES=1` to that option. It compares the
unresolved count, reaching SSA value, and crossed-suspend bit for every load.
Unlike the pointer-set oracle discussed below, this compact lattice oracle
completed successfully on the full Barbershop shader as well as the unit
suite.

Commit `ab63c926b` preserves the generated coroutine at 9 subroutines, 212
frame fields, and 3,184 bytes. The final report is
`/var/tmp/psycles-compact-populate-20260821/barbershop-reaching-projection-final-comparison.json`;
across all 15 passes its maximum absolute error is `4.7683716e-7`, maximum RMS
is `4.0016332e-8`, and maximum relative RMS is `1.5557180e-7`. The warm logs
are `barbershop-reaching-projection-profile.log` and
`barbershop-reaching-projection-final.log` in the same directory.

## Snapshot instruction-order queries

The next flat profile attributed approximately 0.62% of the complete process
to `find_latest_insertion_point` and another 0.61% to
`instruction_strictly_precedes`. Both helpers repeatedly walked a basic
block's intrusive instruction list. This is a structural mismatch for
Barbershop: the alloca-scope pass considers 68,341 local allocas, while most
queries need the relative order of only an alloca, its unique whole-object
definition, and its actual observations.

The replacement is based on an explicit mutation invariant. Before the first
contraction, the pass freezes both its candidate set and an immutable
`(block_id, ordinal)` location for every instruction. Processing a candidate
moves only that candidate's alloca and, when the single-definition proof
applies, its unique whole-object store. Neither moved instruction can be an
observation or definition of a different valid candidate. Therefore earlier
contractions cannot change the relative order of any observation, definition,
SSA value, terminator, or insertion instruction queried for an unprocessed
candidate. Snapshot ordinal comparison is exactly equivalent to walking the
current list for those strict-order questions. Immediate adjacency is the one
property that can change when an unrelated moved node enters a gap, so it is
kept as a live intrusive-list `next()` query.

The latest insertion point is likewise the minimum snapshot ordinal among the
candidate's users in the target block and the terminator. Its cost is now
proportional to the candidate's actual use set, not to every unrelated
instruction in the block. Public counters report 186,042 order queries and
120,277 user inspections on the full shader. Relative to the preceding build,
first-definition planning falls from `311.574 ms` to
`30.046--35.266 ms` (`88.7--90.4%`) and general placement falls from
`31.017 ms` to `2.890--3.361 ms` (`89.2--90.7%`). Pre-distill optimization
falls another `11.2--21.7%`, from `1.46225--1.50582 s` to
`1.17985--1.29784 s`. The two formerly visible linear-search symbols disappear
from the flat profile; the complete alloca-scope pass accounts for
approximately 0.16% self time.

`CoroAllocaScopeOptions::verify_instruction_order` retains the original linear
walk as a semantic oracle, and the coroutine pipeline maps
`LUISA_CORO_VERIFY_ALLOCA_ORDER=1` to it. Every snapshot result is compared
with the current intrusive list. The oracle passed on the complete Barbershop
shader after all 68,329 accepted contractions, not only on a reduced fixture.

Commit `76be1a09f`, rebased over upstream's independent unity-build fix,
preserves the generated coroutine at 9 subroutines, 212 frame fields, and
3,184 bytes. The final 15-pass report is
`/var/tmp/psycles-compact-populate-20260821/barbershop-alloca-order-rebased-comparison.json`;
its maximum absolute error is `2.3841858e-7`, maximum RMS is
`3.5847062e-8`, and maximum relative RMS is `1.3382906e-7`. The measured logs
are `barbershop-alloca-order-final-a.log`,
`barbershop-alloca-order-profile.log`,
`barbershop-alloca-order-rebased-b.log`, and
`barbershop-alloca-order-oracle.log` in the same directory.

## Rejected register-bank experiment

A diagnostic scalarized the compact value banks behind generated switch
accessors. It was semantically correct but structurally worse: Barbershop HIP
JIT grew from `18.8898 s` to `95.4506 s`, render-only grew from `2.08996 s` to
`2.89015 s`, generated AMD code grew to `1,961,268 B`, and peak host RSS was
`7,283,544 KiB`. Strict native Vulkan spent more than three minutes before
emitting its first SPIR-V module. The experiment was fully reverted and was
not committed.

This result is consistent with Cycles 5.2 rather than contrary to it. Cycles'
`intern/cycles/kernel/svm/svm.h` deliberately uses a 255-float local SVM stack
and relies on same-shader wave coherence and separate, sorted shade-surface
wavefront kernels. Replacing indexed storage with a large control-flow mux is
not the right abstraction. Psycles should continue to improve scheduling,
storage lifetime, and fixed interpreter structure instead.

## Regression coverage

Luisa commit `b1a969e23` adds a complexity regression with a 4,096-leaf
aggregate and 8,192 unrelated arithmetic instructions before DCE. The full
lowering must still localize the one real scratch aggregate, making sparse
effect complexity observable without timing thresholds.

Commit `933811bae` extends that regression to require zero speculative
selection-localization analyses under the default unbounded budget. A second
finite-budget regression requires exactly one selection proof and confirms
that the localized alloca does not consume the callback capture budget. These
cover both sides of the demand condition rather than encoding a renderer or
scene special case.

Commit `ef07fb841` adds 8,192 unrelated shared-root GEP/load chains to the same
pre-DCE fixture. The real scratch must still localize, making sparse pointer
support part of the regression rather than only a timing observation.

Commit `e213b46b5` adds two further structural regressions. A predicate-rich
suffix after an already failing read must be rejected after exactly one
guarded state evaluation, while the correlated-predicate success cases remain
accepted. A projected pointer analysis with 48 requested reference roots and
4,096 unrelated local roots must discover and validate all 4,144 pointers,
materialize exactly 48 coordinates, and retain each reference's live-in fact.
Existing cyclic-CFG, branch-intersection, full-product equivalence, malformed
projection, aliasing, reference-call, and ray-query lifetime tests continue to
pass.

Commit `d32a3f93b` additionally requires the pre-DCE
16,384-instruction unrelated suffix to leave the semantic event count at
exactly two. All existing field-sensitive aggregate, conditional Must-write,
cross-handler duplication, callable-reference, cyclic/malformed pointer, and
transactional rejection cases continue to pass.

Commit `3d2c322f1` requires a mutating 16-loop continuation fixture to report
exactly one dominance-frontier materialization for each post-merge re-entry
transform query plus its final audit. The same fixture retains its dense CHK
idom accounting, proving that the optimization removes an unobserved derived
relation rather than skipping dominance maintenance.

Commit `b0f235f0f` adds a separate scope-dataflow regression. It constructs
two acyclic scopes containing a diamond, a resume-only 256-block chain, and
local frame state. The test runs the independent dense pointer-set oracle,
requires the projected domain to be strictly smaller than the global product,
and requires each RPO Must and May solver to evaluate every acyclic block
exactly once. Invalid input must also reset the optional statistics rather
than exposing stale observations. The complete existing CFG-distillation
suite passes under `LUISA_CORO_VERIFY_DENSE_DATAFLOW=1` (380 assertions in 49
tests), and the new regression contributes 17 assertions in 2 tests.

The independent oracle deliberately stores a pointer set per block and is not
intended to scale to production modules. Enabling it on the complete
Barbershop shader reached approximately 57 GB RSS after two minutes, so that
diagnostic run was terminated before host OOM; it is not reported as a pass.
Equivalence on the production module is instead checked by the unchanged
9/212/3,184 coroutine ABI and the complete 15-pass EXR comparison above. The
bounded CFG oracle remains the direct semantic differential test.

Commit `ab63c926b` extends the rematerialization suite with a 512-block
root-reachable arm unrelated to the only demanded load. Two stores force the
general solver, but the structural counters must report exactly one active
block and one evaluation. Branch-conflict, loop-backedge, phased suspension,
and projection fixtures invoke the dense oracle directly through the public
options object, so CI exercises both solvers without relying on process
environment mutation. The suite now passes 163 assertions in 24 tests.

Commit `76be1a09f` adds a same-block complexity regression with 128 independent
alloca/store/load candidates and 4,096 unrelated arithmetic instructions.
With the linear oracle enabled, all 128 candidates must move and verification
must succeed after every prior mutation. Structural counters must remain
exactly 256 order queries and 128 user inspections, proving that the normal
algorithm depends on candidate uses rather than the 4,096-instruction block
body. The complete alloca-scope suite now passes 304 assertions in 25 tests.

The following checks passed after rebasing onto and pushing current Luisa
`next`:

```text
test_xir_aggregate_field_bitmask
  42 assertions in 8 tests

test_xir_pass_lower_ray_query_to_pipeline
  311 assertions in 31 tests

test_xir_pass_pointer_usage
  221 assertions in 12 tests

test_xir_pass_coro_alloca_scope
  304 assertions in 25 tests

test_xir_pass_coro_rematerialize
  163 assertions in 24 tests

test_xir_coro_cfg_distill
  380 assertions in 49 tests

test_xir_coro_cfg_dataflow
  17 assertions in 2 tests

test_coro_compile_trigger
  passed

test_xir_pass_restructure_cfg
  1466 assertions in 79 tests

test_xir_pass_post_dom_tree
  58 assertions in 9 tests

test_xir_pass_mutation_safety
  185 assertions in 28 tests

test_xir_passes
  2452 assertions in 392 tests

cmake --build build-luisa-tests --parallel
cmake --build build --parallel

./build/bin/psycles_luisa_scene_traversal_tests fallback
./build/bin/psycles_luisa_scene_traversal_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
./build/bin/psycles_luisa_scene_traversal_tests vk
```

All three traversal runs exited successfully. The Vulkan log contains native
SPIR-V optimization/codegen records and does not load DXC.

Reproduction of the final Barbershop compiler workload:

```text
LUISA_CORO_PROFILE_COMPILATION=1 \
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
./build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 \
  output.exr hip 8 8 256 64 - 0 0 0 0 256 - 1 0 \
  wavefront-staged 32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```

Raw logs, EXRs, and `perf` captures remain under
`/var/tmp/psycles-compact-populate-20260821` and
`/var/tmp/psycles-barbershop-coro-host*.perf.data`. The final profile is
`/var/tmp/psycles-barbershop-coro-host-alloca-order.perf.data`. This is a
compiler-focused 8x8 canary, so no enlarged visual triptych is presented as a
scene-quality claim. The previously committed full-resolution scene triptychs
remain the quality oracle; the numerical all-pass comparison here is stronger
evidence for this analysis-only change.
