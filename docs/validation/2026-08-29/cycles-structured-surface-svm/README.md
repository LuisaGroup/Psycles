# Cycles-Structured Surface SVM Replacement

## Current outcome

The Cycles-structured stream is now the device execution path for compact
surface population, emission, preparation, and BSSRDF-normal reconstruction.
One Luisa interpreter owns the program counter, one physical 32-bit lane
stack, guards, `SetNormal`, closure decode, and `End`; the old split
value/closure device interpreter and its 1,729-line implementation file have
been physically removed. Production runtime construction no longer builds the
old executable as an interning or migration oracle.

This is now both a device-path correctness milestone and a first complex-scene
execution checkpoint. The official Blender 5.2 Barbershop export builds and
renders through HIP with the replacement SVM. It is not yet a matched
multi-sample Cycles image or performance milestone: Lone Monk, Classroom,
Monster, strict native-XIR Vulkan complex scenes, inspected triptychs, and
1080p performance claims remain pending.

The current implementation produces a typed semantic stream containing

```text
Value | MixClosure | AddClosureWeight | JumpIfOne | JumpIfZero |
ClosureLeaf | End
```

and an exact local-storage allocation for that stream. Shared value nodes are
evaluated once before the lowest closure region that dominates all uses;
branch-private texture/math nodes remain after the corresponding Mix guard.
Closure operand ordering is now defined by one ABI function shared with the
established closure lowering, rather than duplicated by the new scheduler.
Closure leaves shared by multiple closure paths are emitted once with the
same ordered weight accumulation that Cycles builds in
`transform_multi_closure`.

## Cycles 5.2 source model

The reference is Blender's `blender-v5.2-release` source at
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`:

- `intern/cycles/scene/svm.cpp::generate_multi_closure` computes the Mix
  factor and shared dependencies before branch-private dependencies, then
  emits `NODE_JUMP_IF_ONE` and `NODE_JUMP_IF_ZERO` around the two subtrees;
- `intern/cycles/kernel/svm/svm.h::svm_eval_nodes` executes those records in
  the same interpreter as value and closure records;
- `intern/cycles/kernel/svm/closure.h::svm_node_mix_closure` saturates the
  factor before producing child weights; and
- the guards skip A for `factor >= 1` and B for `factor <= 0`. They are not
  exact-equality tests.

Psycles previously evaluated the complete endpoint dependency union and only
then traversed a separate linear closure-weight stream. That architecture
cannot skip branch-private texture/math work and retains all closure operands
until the value stream ends. The replacement removes that structural split.

## Formal placement model

For one endpoint projection, expand the reachable closure relation into a
rooted occurrence tree `T`. An Add contributes two ordinary children. A
dynamically reachable Mix contributes two conditionally guarded children.
Each region `r` has a unique prefix point which dominates its subtree.

For value `v`, let `D(v)` be closure regions that directly consume `v`, and
let `users(v)` be value instructions that consume it. In reverse value-topology
order define

```text
R(v) = LCA(D(v) union {R(u) | u in users(v)})
```

over `T`. Parameters receive a region but emit no record. Every computed value
is emitted exactly once in the prefix of `R(v)`.

This recurrence has two required properties:

1. For every edge `v -> u`, `R(v)` is an ancestor of `R(u)`; therefore the
   definition of `v` dominates every execution of `u`.
2. Any placement below `R(v)` fails to dominate at least one member of the
   joined use set; therefore `R(v)` is the lowest legal structured placement.

The implementation validates strict value and closure DAG order, operand
dominance, unique computed-value emission, forward jump targets, endpoint
projection, and a unique final `End` transactionally.

## Closure-weight SSA

The invalid weight id denotes the root constant one. For an occurrence with
incoming weight `w`, a dynamic Mix with factor `f` defines one transaction

```text
s  = saturate(f)
wL = w * (1 - s)
wR = w * s
```

and Add propagates `w` unchanged to both children. The occurrence tree is
visited left before right, matching `ShaderGraph::transform_multi_closure`.
For every unique closure leaf `c`, its occurrence weights are folded in that
stable order:

```text
W(c) = (((w0 + w1) + w2) + ...)
```

The leaf and its final weight are placed at the LCA of all occurrences, which
is exactly where Cycles emits a shared closure before branch guards. Weight
definitions are themselves strict SSA. Reverse use propagation places every
Mix/Add at the lowest region dominating its consumers, and paired Mix outputs
share one instruction and one factor evaluation.

No algebraic identity is applied without an input-domain proof. In
particular, complementary outputs are not rewritten from
`w*(1-s) + w*s` to `w`: a linked IEEE factor may be NaN, for which Cycles
produces NaN and the rewrite would incorrectly produce `w`. The permanent
regression exercises both finite and NaN factors.

## Unified bytecode image

`lower_surface_svm_program` now serializes the proven schedule as one 16-byte
instruction stream. Existing value instructions retain their exact control,
result, packed-operand and metadata fields. Six opcodes above the closed
`ValueOperation` range encode `MixClosure`, `AddClosureWeight`, the two guards,
`ClosureLeaf` and `End`; opcode `0xff` remains reserved by the established
automatic-normal transaction.

The three payload words are statically typed by opcode:

```text
Mix:  factor address | parent weight slot | packed left/right slots
Add:  input weight A | input weight B     | result weight slot
Jump: factor address | absolute target PC | invalid
Leaf: operand begin  | incoming weight    | Principled feature mask
End:  invalid        | invalid            | invalid
```

The leaf's existing closure control word is shifted above the unified opcode,
so closure identity, endpoint projection, BSSRDF method, anisotropy, thin-film
and tangent capabilities are preserved without another dispatch side table.
Value overflow operands remain packed pairs of 16-bit addresses, while
closure operands remain full 32-bit addresses in a distinct side stream. This
is an intentional physical distinction rather than a weakly typed common
array.

Passthrough quotient members emit no bytecode. Lowering first constructs a
semantic-PC to bytecode-PC boundary map, then rewrites every forward guard
target through that map. Permanent regression places two erased aliases under
a guard and proves the target lands exactly on the following guard.

The serialized validator projects ordinary records through the single
established value-image validator and closure leaves through the single
established closure-image validator. It additionally solves forward
must-initialization on the real unified CFG. The stack is physically untyped:
a scalar or weight occupies one lane, a vector occupies three consecutive
lanes, and a uint64 occupies two. Every definition invalidates all logical
values whose lane intervals overlap its write. This rejects cross-bank
clobbering as well as treating a value lane as a closure weight or a weight
lane as a Mix factor, while still permitting proven read-before-write lane
donation.
Closure operands and side tables must be dense, jumps strictly forward, Mix
outputs non-aliased, aggregate masks exact, and `End` unique and final.

## SetNormal transaction and scene image

Cycles' authored automatic-normal evaluation is a sequencing boundary, not an
ordinary graph dependency. The unified stream therefore retains `0xff` as an
explicit `SetNormal` record:

```text
normal value prefix; SetNormal(vector); structured closure body; End
```

The prefix and body are independently colored. `SetNormal` reads its vector
before killing every local definition, so the two allocations may overlap but
the body cannot observe stale prefix storage. The verifier models this as a
new must-initialization epoch; a permanent regression deliberately redirects a
body read to a physically populated prefix slot and requires rejection.

`build_surface_svm_scene_image` concatenates programs in runtime-tag order.
Guard targets, overflow value-operand ranges, metadata indices, metadata
static-table ranges and closure-operand begins become absolute scene offsets.
Physical local-lane addresses and material parameter ids remain
invocation-relative. The compact 32-byte device descriptor contains only the
instruction range, typed-color census, flags, endpoint projection, and stack
lane extent. Exact side-stream partitions live in a parallel host-only table
and are not uploaded per invocation.

The public scene verifier proves every descriptor and side range is a dense
partition, de-relocates each program, and runs the complete program verifier
again. It also recomputes the scene's maximum typed capacities and closure
capability masks. This rejects missing relocation, cross-program references,
unowned suffixes and stale aggregate metadata transactionally.

## Runtime scene assembly and exact evaluator provenance

`build_surface_value_runtime` now constructs the replacement preparation and
emission programs for every real material topology, composes the applicable
automatic-normal prefix, and aggregates the resulting programs in exactly the
runtime tag order. The image is retained in `SurfaceValueRuntime::svm_scene`;
it is no longer only a compiler-test artifact.

Evaluator dispatch is not reconstructed from an opcode or a material tag. A
host-only relation records the original `ValueExpressionId` at each new value
record. The unified executable builder now validates that relation directly on
the forward CFG and interns the exact semantic variant from the source value;
the production dispatcher no longer obtains variants from the established
split-stream executable.

For every PC, a forward must-definition state maps each typed local slot to its
exact quotient representative. Branch joins retain a source only when all
predecessors agree. Value operands must therefore name the exact source value,
not merely a type-compatible initialized slot. Mix/Add weight definitions kill
overlapping scalar value identities, and SetNormal proves its exact local or
parameter source before beginning a new lifetime epoch. Metadata, static-table
payloads, signed zero and NaN payloads are compared bit-for-bit; metadata and
static data must form canonical one-use/dense partitions.

The former migration proof was:

```text
ValueExpressionId -> old proven instruction -> exact evaluator variant
        |                                        |
        +-> structured SVM value record --------+
```

That old-stream commuting oracle has been removed from production evaluator
selection. One source value may occur in both normal and root epochs, but both
occurrences select the same exact variant. Every Mix, guard, closure leaf,
`SetNormal`, and `End` record carries the invalid variant sentinel. Permanent
regressions cover independently constructed variant merging, opcode-immediate
domains, local/parameter route joins, signed-zero distinction, metadata
substitution, parameter-backed SetNormal, and a type-correct but semantically
permuted local operand.

The cross-backend compact-surface fixture additionally compares the unique
variant domain of each new program tag with the corresponding established
transaction. It requires a real structured guard and closure leaf, matching
endpoint projection and flags, matching `SetNormal` count, one final `End`, and
the complete public scene verifier. All three available backends pass:

```sh
ctest --test-dir build --output-on-failure \
  -R '^psycles\.luisa_compact_surface_preparation_(fallback|hip|vk)$'
```

This was the runtime-assembly checkpoint. It proved scene construction and
evaluator selection before the device route was switched; the single-PC
execution results below supersede its old execution-status statement.

## Single-PC device interpreter and exact closure domains

`SurfaceSvmInterpreter` executes one loop over the scene-absolute unified
records. Value records dispatch through the exact semantic evaluator relation;
Mix and Add define weight SSA in the same typed scalar bank; guards apply the
raw Cycles `>= 1` and `<= 0` predicates; leaves retain Cycles' exact
`!(weight <= 0)` NaN behavior; and `SetNormal` begins a new local-lifetime
epoch before closure evaluation.

Closure dispatch is specialized by the exact pair

```text
(static closure variant, per-leaf Principled feature mask)
```

rather than a scene-wide union. Runtime construction canonicalizes the exact
pair domains separately for preparation, emission, and BSSRDF consumers. A
permanent regression independently scans every unified record, reconstructs
the preparation/emission domains, and requires equality with the JIT domains;
the BSSRDF domain must be canonical and a subset of preparation.

The four consumers share the same interpreter and differ only by host-side
Luisa AST visitors. Transparent closures preserve Cycles' source ordering:
the first retained transparent closure owns the allocation slot and later
transparent setup calls fold additively into it. Production collectors use one
pass with explicit finalization. Diagnostic collectors without that contract
perform a whole-program second pass only after the first transparent leaf;
restarting at a suffix would be unsound because branch-local typed slots may
already have been overwritten.

The former forwarding shim was removed: the public compact-surface factories
are implemented directly by the SVM consumers, closure decoding is a separate
744-line `.h + .cpp` module, and no surface implementation file exceeds 1,000
lines.

The cross-backend fixture compares all four consumers against the expanded
Cycles-derived reference, including transparent capacity/exhaustion behavior,
merged weight/sample weight, shading normal, emission, and BSSRDF normal. It
also checks the exact value and closure JIT domains:

```sh
cmake --build build \
  --target psycles_luisa_compact_surface_preparation_tests \
  -j"$(nproc)"

ctest --test-dir build --output-on-failure -j"$(nproc)" \
  -R '^psycles\.luisa_compact_surface_preparation_(fallback|hip|vk)$'
```

Result: 3/3 passed in 3.30 s (fallback 2.08 s, HIP 3.28 s, native-XIR Vulkan
3.30 s). These are test wall times with concurrent execution, not renderer
performance measurements.

## CFG liveness and Cycles-sized lane-stack storage

The emitted CFG is acyclic because every ordinary successor and jump target
has a greater instruction index. Liveness therefore needs one reverse pass,
not a fixed point:

```text
live_out[i] = union(live_in[s] for s in successors(i))
live_in[i]  = uses[i] union (live_out[i] - definition[i])
```

`uses[i]` includes value operands, Mix factors at the weight and guard
records, weight-SSA operands, and exact active operands at each closure leaf.
A program point may define both outputs of one Mix transaction. A forward
must-defined analysis uses predecessor intersection and rejects any path on
which a local definition does not dominate its use.

Same-bank `Passthrough` values are contracted as the congruence
`p = identity(x)` before liveness. Under the read-before-write contract, a
result may share a slot with an operand precisely when the operand is absent
from `live_out` at that definition.

For each semantic bank, a definition interferes with every same-bank value in
its `live_out`. Strict SSA dominance makes the resulting interference graph
chordal. The allocator does not merely assume this theorem: maximum-cardinality
search constructs a candidate perfect-elimination ordering and the compiler
checks that every vertex's later neighbors form a clique. Reverse-PEO greedy
coloring must then use exactly the maximum-clique number; otherwise lowering
is rejected.

The three optimal colorings are then packed into one Cycles-compatible stack:

```text
scalar/weight base = 0
vector base        = scalar colors
uint64 base         = scalar colors + 3 * vector colors
stack lanes         = scalar colors + 3 * vector colors + 2 * uint64 colors
```

The semantic upper bound is exactly Cycles 5.2
`intern/cycles/kernel/svm/types.h::SVM_STACK_SIZE`: 255 32-bit lanes, with
legal offsets 0 through 254. Material parameters remain in immutable SoA
buffers and do not consume stack lanes. Component color bounds only prevent an
individual coloring from exceeding what can fit in 255 lanes; the final
physical extent is the authoritative capacity proof.

## Permanent regressions

`psycles.surface_svm_schedule` covers:

- one shared computed input hoisted before a Mix and two private computed
  inputs retained behind their respective guards;
- `factor <= 0`, interior factor, and `factor >= 1` paths;
- endpoint projection moving formerly shared work inside the only live guard;
- nested Mix target boundaries and Add siblings outside Mix control;
- static 0/1 Mix pruning;
- a shared closure DAG emitted once with ordered Add-weight accumulation;
- a closure shared by both sides of one Mix, including NaN preservation;
- a leaf shared between an unconditional Add child and a Mix child, proving
  weight hoisting, exact factor 0/interior/1 behavior, and one remaining
  private guard;
- definite-assignment rejection when a mutated jump skips a required value;
- rejection of a Mix-weight expression bound to the wrong source factor;
- same-bank Passthrough quotienting and component-wise capacity failure;
- chordal clique-optimal coloring of value and weight SSA with
  read-before-write slot donation;
- a nine-scalar live clique matching the Barbershop failure shape, proving the
  removed eight-scalar ABI rejects it while the 255-lane SVM accepts it in
  exactly nine lanes; and
- exact rejection when the same graph is given only eight physical lanes.

`psycles.surface_svm_scene` additionally covers the SetNormal lifetime epoch,
all five relocation classes, a static 4x4 table, two-program tag order, and
malformed cross-program guard/closure references. It also proves that a
three-lane vector write invalidates an overlapping live scalar even though the
two addresses have different semantic bank tags, accepts the exact 255-lane
bound, and rejects 256 lanes.

`psycles.luisa_surface_mix_svm_{fallback,hip,vk}` uses the production stack
views inside a real Luisa callable. It writes a scalar, a three-lane vector,
and the exact 64-bit pattern `0x0102030405060708` into disjoint lanes, then
requires bit-exact recovery on all three backends. This permanently covers the
float2-bitcast representation of uint64 stack values rather than checking only
their low byte.

One independent-branch fixture requires at least two vector slots under the
old split value/closure plan but exactly one slot after closure uses are
interleaved in the structured stream. This is a compiler storage result, not a
renderer performance result.

```sh
cmake --build build --target \
  psycles_surface_svm_schedule_tests \
  psycles_surface_svm_scene_tests \
  psycles_surface_closure_execution_plan_tests \
  -j"$(nproc)"

ctest --test-dir build --output-on-failure \
  -R 'psycles\.(surface_svm_(schedule|scene)|surface_closure_execution_plan)'
```

All focused tests pass. Virtual closure weights are assigned and colored in
the scalar bank together with ordinary scalar values, and the semantic CFG is
lowered transactionally into the final compact bytecode, aggregated into a
scene-wide image, uploaded, and executed by the single-PC interpreter for the
actual runtime topology set.

The latest all-thread repository run after the lane-stack replacement and a
complete relink executed all 318 registered tests in 30.81 s:
312 passed and the same six pre-existing exact numeric fixtures failed
(`luisa_stacked_volume_fallback`, `luisa_homogeneous_volume_fallback`,
`luisa_area_light_forward_vk`, `luisa_volume_path_fallback`,
`luisa_volume_path_vk`, and `luisa_volume_triangle_fallback`). The new compiler,
exact-domain, and device-interpreter tests passed. The six failures are
unchanged from the pre-switch baseline and remain small existing numeric
differences in volume/Vulkan fixtures rather than new surface-SVM failures.

## Removal of the split runtime

The static execution histogram now projects measured topology populations
directly over the unified PC stream. It classifies Value, SetNormal, closure,
weight and guard records without inventing dynamic guard-visit counts. The old
linear-region ABI fields remain empty because no canonical linear partition
exists across structured guards.

The production runtime no longer constructs `SurfaceValueExecutableScene`.
Closure reachability, endpoint-specific value domains and exact
`(static_variant, Principled features)` closure domains are derived from the
unified scene and its total evaluator side stream. The builder checks that the
preparation closure domain reproduces the scene aggregate operation/feature
masks before creating any device resource.

The following obsolete implementation was physically removed:

- the split value/closure executable stored in `SurfaceValueRuntime`;
- region-specialization planning, inline tags and diagnostic side stream;
- eleven legacy host arrays, buffers, bindless slots and upload commands;
- the old whole-program value callable and AO callable; and
- the region callable lowering and duplicate bytecode interpreter.

The replacement dispatcher regression records a real Luisa callable around
one unified Value record, requires one `SurfacePointCall&`, no aggregate return
state, and one named semantic handler boundary per exact active evaluator. The
host-image regression independently verifies every unified instruction against
its evaluator, reconstructs the preparation/emission value and closure domains,
and scans Bump records without consulting the removed executable.

Validation after the deletion:

```sh
cmake --build build \
  --target psycles_luisa_compact_surface_preparation_tests \
  -j"$(nproc)"

ctest --test-dir build --output-on-failure \
  -R '^psycles\.luisa_compact_surface_(preparation|tail)_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure \
  -R '^psycles\.(surface_program_metadata|surface_svm_scene|luisa_sample_dispatch_film_fallback)$'
```

Results: all six cross-backend compact-surface tests passed in 15.04 s
(preparation fallback 2.80 s, HIP 4.19 s, native-XIR Vulkan 7.05 s; tail
fallback 0.18 s, HIP 0.24 s, Vulkan 0.56 s). The histogram and two focused
compiler tests passed 3/3 in 8.11 s. These are correctness wall times, not
renderer performance results.

## Blender 5.2 Barbershop lane-stack checkpoint

The first real-scene attempt exposed the structural defect in the removed ABI:
topology 59 (`armchair_cushions.001`, shared with `armchair_cushions`) has 73
parameters, 42 nodes, 105 values, and five closures. Its exact interference
coloring needs nine scalar colors, so the old eight-scalar bank rejected the
scene before JIT. Raising that bank alone was rejected as an ad hoc fix. The
single lane-stack model above replaces the arbitrary 8/12/1 limits.

After the replacement, the unchanged official Blender 5.2 Barbershop export
completed this HIP smoke command:

```sh
env PSYCLES_COMPACT_SURFACE_VALUES=1 PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 \
  /var/tmp/psycles-surface-svm-barbershop-20260829/hip-smoke.ppm \
  hip 320 180 1 1 - 0 0 0 0 1 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 auto 0 0 0 1 1048576 \
  - /var/tmp/psycles-surface-svm-barbershop-20260829/hip-smoke-svm-histogram.json
```

Observed production census and timings:

```text
1055 geometries, 1109 instances, 564 materials
380 SVM programs, 10177 records, 67 exact value evaluators
maximum typed colors: scalar 9, vector 8, uint64 0
maximum physical stack: 33 lanes (132 bytes of semantic payload)
scene compile: 17.1764 s
reported complete shader JIT: 18.441 s
render-only 320x180x1: 0.0633394 s
```

The run produced Combined/Normal/Albedo PFM passes, the surface histogram, and
a multilayer EXR. Visual inspection of the display PPM found the expected
camera, salon silhouette, ceiling lights, and major furniture, with no obvious
structural corruption from lane remapping. One sample is intentionally too
noisy and dark for material or Cycles-fidelity judgment. The exporter also
reported ten unavailable source images (`agent_face_*`,
`guilder_ornament.png`, and `generic_scratches.png` variants), so this smoke is
not used as a texture-fidelity claim or triptych. A fixed-sample re-export,
Cycles 5.2 reference, numerical comparison, and inspected triptych remain a
completion gate.

The remaining completion gate is fallback, HIP, and strict native
XIR-to-SPIR-V Vulkan complex-scene rendering, triptychs, code-object and
compile-time measurements, followed by Cycles-aligned per-kernel and
end-to-end performance profiling. No replacement-complete or speed claim is
made before that evidence exists.

## Coroutine-local SVM stack lifetime

The first production coroutine dump after the 255-lane replacement found a
structural regression which the isolated callable tests could not expose. The
single stack appeared as the following logical frame value:

```text
name='_reg_5835' type=array<float,255> size=1020 align=4
scope_external=[5] scope_touched=[5] scope_live_in=[5] scope_live_out=[5]
edge_live=[0..28] edge_store=[21,22,23]
```

The resulting Barbershop wavefront frame was 1,856 bytes with 177 physical
fields. Scope 5 is `shade_surface`; edges 21--23 leave it for direct-light and
path continuations. Thus this was not 1,020 bytes of material state which
semantically crossed a cut. It was the undefined entry contents of a
scratch allocation whose fresh lifetime had not been expressed in the new
unified interpreter.

The formal distinction is definition-sensitive. For memory atom `a` in scope
`s`, the scope exposed-use generator contains a use only when no must-definition
of `a` reaches it. A whole-aggregate definition at interpreter entry kills
all 255 lane atoms before any handler reference escape. Since no successor
observes the stack after `shade_surface`, backward inter-scope liveness then
contains neither `a` nor an edge store. It would be unsound to remove every
allocation merely because it is touched in one continuation: an aggregate
defined before suspension and first read after resume must still cross the
edge.

`SurfaceValueLocals` now establishes the whole-stack undefined definition in
its constructor. The host bytecode verifier is the proof obligation which
makes this fresh lifetime valid: every scheduled read is dominated by a write
in the same SVM invocation. Encoding the witness as a constructor invariant
prevents a future interpreter call site from forgetting it. The undefined
assignment is not a zero fill or observable material operation and is erased
by native lowering.

Luisa's permanent CFG regression contains both sides of the theorem:

- a root-scope `array<float,255>` fully defined after the surface resume and
  conservatively passed by reference remains absent from every frame/edge;
- an aggregate defined before suspension and observed after resume remains a
  frame value.

With `LUISA_CORO_VERIFY_DENSE_DATAFLOW=1`, the Luisa test passes all 493
assertions in 56 cases. The Psycles fresh-lifetime AST test now verifies that
construction itself emits exactly one argument-free aggregate definition.
The fallback/HIP/native-XIR Vulkan compact-preparation matrix passes 3/3.

The same production Barbershop command after the fix reports:

```text
subroutines=9 frame_fields=176 frame_bytes=848
array<float,255> frame values: 0
```

This is a 1,008-byte frame reduction without changing the 255-lane semantic
limit, scene topology, cache identity, or bytecode. A new HIP performance
claim still requires an interleaved render/profile run; frame size alone is
not treated as elapsed-time evidence.

## Scene-bounded physical stack specialization

The 255-lane Cycles limit is an address-domain bound, not a requirement that
every scene materialize 1,020 bytes of private storage. For every verified
program `p`, the bytecode/storage validators establish

```text
local_index + operand_width <= p.stack_lanes
```

and the scene descriptor establishes

```text
M = max(p.stack_lanes).
```

Production now chooses the smallest host/JIT-only callable ABI bucket
`B in {32, 64, 128, 255}` with `M <= B`. Therefore every legal local access
fits `B`; no device-side capacity branch or address remapping is introduced.
The 32-lane floor bounds the number of generated callable types. Above that
floor, `B < 2M`; the terminal 255 bucket preserves the complete Cycles domain.
Boundary regressions cover `0/1/32/33/64/65/128/129/255/256`, and every bucket
must retain exactly one whole-aggregate fresh-lifetime definition.

Barbershop has `M = 33`, so it selects 64 physical lanes. A cold LLVM dump
confirmed that both `[255 x float]` private allocations in `shade_surface`
became `[64 x float]`, while the coroutine frame remained 176 fields / 848 B.
The generated surface code object remained effectively unchanged
(`390788 -> 390784` bytes), so this change is not presented as a code-size
solution.

Two warm HIP `rocprofv3 --kernel-trace --scratch-memory-trace --stats` runs at
640x480, 64 fixed spp, staged wavefront, block size 64 gave:

| measurement | 255 lanes | 64 lanes | change |
|---|---:|---:|---:|
| `shade_surface` static scratch | 4896 B | 3364 B | -31.3% |
| `shade_surface` VGPR | 256 | 256 | unchanged |
| `shade_surface` ns/item, mean of 2 | 25.7597 | 25.7386 | -0.08% |
| render-only, mean of 2 | 2.43553 s | 2.44169 s | +0.25% |

The elapsed differences are measurement noise: private storage fell
substantially, but the unchanged 256-VGPR limit prevented a measurable
occupancy or render-time gain. No speedup is claimed.

The profile command shape was:

```sh
env PSYCLES_COMPACT_SURFACE_VALUES=1 \
    PSYCLES_POPULATE_SURFACE_ONCE=1 \
    LUISA_CORO_SHADER_MAP=1 \
rocprofv3 --kernel-trace --scratch-memory-trace --stats -f rocpd -- \
  build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 \
  out.exr hip 640 480 64 64 - 320 240 0 0 64 - 1 0 \
  wavefront-staged 64 32768 32 1 1 0 4 2 auto 0 0 0 1 1048576
```

A naive full-image A/B initially appeared to lose a rare glossy contribution.
The raw single-path trace showed that the camera RNG and ray were bit-exact and
that the first divergence was the HIPRT closest hit, before surface execution:
primitive `6451795` versus `6438118`. Repeating the same 255-lane executable
with the same ray alternated between those two hits, proving this is
non-deterministic BVH/equidistant fine-geometry selection rather than a stack
specialization effect. This is why production-image validation must locate the
first pre-surface trace divergence before attributing an isolated firefly to a
material change.

The final all-thread build passed. The focused compiler/fallback, HIP, and
strict native-XIR Vulkan matrices passed `8/8`, `3/3`, and `3/3`. Full CTest
passed `312/318`; the unchanged failures are the six existing numerical
volume/area-light regressions:

```text
psycles.luisa_stacked_volume_fallback
psycles.luisa_homogeneous_volume_fallback
psycles.luisa_area_light_forward_vk
psycles.luisa_volume_path_fallback
psycles.luisa_volume_path_vk
psycles.luisa_volume_triangle_fallback
```

## Cross-backend Barbershop smoke matrix

The same 320x180, one-sample image and SVM histogram were produced on all
three Luisa routes. Vulkan ran with `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`; its log contains successful native SPIR-V
optimization/compilation after DXC was made unavailable.

| backend | scene compile | shader JIT | render-only |
|---|---:|---:|---:|
| HIP | 17.1764 s | 18.441 s | 0.0633394 s |
| fallback | 9.13115 s | 9.90685 s | 0.12632 s |
| native-XIR Vulkan | 9.34201 s | 18.3884 s | 0.418543 s |

These are cold/smoke observations, not a performance benchmark. Across the
one-sample outputs, Environment and both Volume passes are exact, while
Emission and direct Transmission differ only around `1e-7`. Normal/Albedo
whole-frame RMS is `0.00155..0.00228`; isolated silhouette pixels choose
different primitive/normal results across traversal implementations. Combined
RMS is dominated by high-energy paths landing on different pixels after such
one-sample path divergence, so it is not used as a material-fidelity metric.
Higher-spp comparison against the exact Cycles 5.2 export remains required.
