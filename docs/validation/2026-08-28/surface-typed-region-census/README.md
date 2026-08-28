# Typed straight-line surface regions

## Outcome

The compact surface compiler now constructs an exact, deterministic partition
of each serialized value program into maximal straight-line regions. A region
contains every adjacent producer/consumer edge for which the producer's final
semantic use is the consumer. Its symbolic operands refer to immutable
parameters, canonical live inputs, or region-relative definition epochs; they
never use colored slot numbers as value identity.

This checkpoint changes no device evaluator and makes no render-speed claim.
The existing dynamic SVM remains the production route. The planner and the
diagnostic-only v6 histogram establish whether a typed-region JIT can remove
the interpreter backedge without repeating the code-size/private-memory
regressions of the rejected bank wrappers and pair overlay.

On the Blender 5.2 Barbershop export, 298,009,579 ordinary value-instruction
executions form 83,460,904 regions. The average region has 3.57 instructions,
and 214,548,675 internal edges can remove a producer bank write and successor
bank read. Non-singleton regions cover 91.11% of ordinary instruction
executions. The opportunity is therefore real and broadly distributed.

The code-size constraint is equally real. The active scene has 524 exact
region AST shapes containing 2,795 static handler sites, versus 67 shared value
handler variants. Expanding every maximal region would replicate handler call
sites about 41.7 times before backend inlining. The top 32 shapes cover only
60.90% of instruction executions. The next implementation must use a measured
code budget and shared handlers; a material-by-material or all-shapes expansion
is rejected by this census.

## Formal construction

Within one normal-transaction segment, ordinary instructions are a finite path
`0 .. n-1`. Let `D_i` be the definition epoch produced by instruction `i`, and
let `last(D_i)` be its exact final serialized observer. Slot reuse does not
identify definitions: the analysis state maps each encoded typed address to
its currently active defining offset, with operands read before the result
redefines its address. A normal commit consumes its selected epoch and clears
that map; closure operands observe the distinguished terminal offset `n`.

Define the legal forwarding edge relation

```text
F = { (i, i + 1) |
      i and i + 1 are ordinary,
      i + 1 reads D_i,
      last(D_i) = i + 1 }.
```

`F` is a subgraph of a path. Its maximal connected components are therefore
the unique non-overlapping partition that contains every legal edge without
duplicating an instruction. Normal commits are not vertices and remain hard
boundaries. Singleton components preserve every ordinary instruction not
incident to a legal edge.

For region `R = [b, e]`, the boundary projection is

```text
live_in(R)  = { D_j | j < b and some instruction in R reads D_j }
live_out(R) = { D_j | b <= j <= e and last(D_j) > e }.
```

Both sets contain definition offsets, not addresses. Live inputs are numbered
by first use. Each operand is then mapped injectively to one of:

- a runtime parameter address;
- a canonical live-input index; or
- a region-relative instruction-result index.

The planner independently derives the symbolic data-flow graph and adjacent
forwarding masks, then requires exact equality between the two representations
on every internal edge. Undefined reads, backward/self references, a missing
ordinary-instruction region, malformed stream extents, and a forwarding edge
crossing a normal commit all fail closed.

Induction over the region's topological order gives the intended lowering
theorem: load each `live_in` once, evaluate handlers in order, substitute the
typed SSA result for every symbolic internal use, and store only `live_out`.
Every region operand then equals the canonical SVM operand at the same offset;
the final bank state differs only in dead definitions, so the following region,
normal commit, and closure program observe identical values.

## Exact AST identity

The v6 histogram interns shapes by complete lexicographic equality, not hash
alone. Identity contains:

- the scene-wide exact handler-variant sequence;
- every successor forwarding mask;
- per-instruction operand boundaries;
- parameter/live-input/instruction-result source kinds and canonical indices;
- the typed banks of live inputs; and
- the region-relative results that must be stored.

Parameter indices normalize to zero because their addresses and values remain
runtime bytecode/material data. Local colored-slot indices are absent. Thus two
equal records generate the same proposed typed-region AST, while any control,
type, wiring, boundary ABI, or result-store difference remains distinguishable.

## Barbershop census

The HIP run used the RX 9070 XT, compact populate-once execution, fixed samples,
and the staged wavefront scheduler:

```sh
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  /var/tmp/psycles-surface-regions.XqoGBo/barbershop.exr \
  hip 320 240 16 16 - 0 0 0 0 16 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  - /var/tmp/psycles-surface-regions.XqoGBo/barbershop-histogram.json
```

The render completed in 0.414648 s, but this is a correctness/census run, not a
timing comparison. Histogram projection happens on the host after rendering
and does not enter shader AST or cache identity.

| Metric | Exact value |
| --- | ---: |
| Surface populations | 3,366,093 |
| Ordinary instruction executions | 298,009,579 |
| Region invocations | 83,460,904 |
| Forwarded internal edges | 214,548,675 |
| Average region length | 3.57065 |
| Average live inputs / outputs | 1.36395 / 1.00000 |
| Non-singleton instruction share | 91.1053% |
| Unique active AST shapes | 524 |
| Static sites across unique shapes | 2,795 |
| Maximum region length | 18 |

| Specialized shapes, hottest first | Instruction-execution coverage |
| ---: | ---: |
| 1 | 4.40% |
| 4 | 14.95% |
| 8 | 25.35% |
| 16 | 42.69% |
| 32 | 60.90% |
| 64 | 77.01% |
| 128 | 91.29% |
| 256 | 98.97% |

The complete compact summary is retained in [metrics.json](metrics.json).
The raw v6 histogram is 968 KiB and remains in the command's temporary output
rather than being checked into the repository.

## Regression and backend validation

Permanent compiler regressions cover:

- a maximal in-place chain and its exact symbolic parameter/result wiring;
- a definition remaining live across two singleton regions;
- canonical live-input numbering independent of its colored slot;
- duplicate successor operands in one forwarding mask;
- an undefined local read failing closed;
- a closure-terminal live-out; and
- a normal commit consuming the prefix result without joining a region.

The all-pass integration test validates shape extents, topological operand
sources, normalized parameter identity, typed boundaries, maximal internal
edges, and exact equality between region totals and the independent handler-
transition census. It also retains split-dispatch equality for Combined,
Normal, Albedo, all light passes, and volume passes.

```sh
cmake --build build --parallel 32 --target \
  psycles_surface_program_metadata_tests \
  psycles_luisa_sample_dispatch_film_tests \
  psycles_render_blender_scene

build/psycles_surface_program_metadata_tests
build/bin/psycles_luisa_sample_dispatch_film_tests fallback
build/bin/psycles_luisa_sample_dispatch_film_tests hip

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
build/bin/psycles_luisa_sample_dispatch_film_tests vk
```

All configurations passed. Vulkan used the strict native XIR-to-SPIR-V canary
with DXC disabled. Because this checkpoint changes host analysis and optional
diagnostics only, no renderer image changed and a new triptych would contain
three identical copies; visual validation remains attached to the preceding
runtime-candidate reports.

## Next lowering constraint

The retained planner supplies the proof and exact interning key, but the census
rejects unconditional expansion. The next candidate should tile maximal
regions into a bounded dictionary of typed segments, keep evaluator handlers
shared, and measure actual AST/object/private-storage deltas before any timing
claim. A segment is useful only if it removes the corresponding bank
store/load; merely replacing the dispatch pair is the already rejected
superinstruction overlay.
