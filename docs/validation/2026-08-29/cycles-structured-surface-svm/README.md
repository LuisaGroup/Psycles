# Cycles-Structured Surface SVM Replacement

## Current outcome

This milestone establishes the compiler foundation for replacing Psycles'
split value/closure surface evaluator with one Cycles-structured instruction
stream. It is intentionally not wired into the renderer yet and therefore
makes no render-time, code-object-size, or image-equivalence claim.

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
must-initialization on the real unified CFG. Scalar slots have two disjoint
logical states, value and weight: defining one kills the other. This rejects
both treating a value slot as a closure weight and treating a weight slot as a
Mix factor, while still permitting proven read-before-write slot donation.
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
Typed local addresses and material parameter ids remain invocation-relative.
The compact 32-byte device descriptor contains only instruction range, typed
slot bounds, flags and endpoint projection. Exact side-stream partitions live
in a parallel host-only table and are not uploaded per invocation.

The public scene verifier proves every descriptor and side range is a dense
partition, de-relocates each program, and runs the complete program verifier
again. It also recomputes the scene's maximum typed capacities and closure
capability masks. This rejects missing relocation, cross-program references,
unowned suffixes and stale aggregate metadata transactionally.

## CFG liveness and optimal typed storage

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

For each physical bank, a definition interferes with every same-bank value in
its `live_out`. Strict SSA dominance makes the resulting interference graph
chordal. The allocator does not merely assume this theorem: maximum-cardinality
search constructs a candidate perfect-elimination ordering and the compiler
checks that every vertex's later neighbors form a clique. Reverse-PEO greedy
coloring must then use exactly the maximum-clique number; otherwise lowering
is rejected. Component-wise scalar/vector/uint64 capacity remains explicit.

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
- same-bank Passthrough quotienting and component-wise capacity failure; and
- chordal clique-optimal coloring of value and weight SSA with
  read-before-write slot donation.

`psycles.surface_svm_scene` additionally covers the SetNormal lifetime epoch,
all five relocation classes, a static 4x4 table, two-program tag order, and
malformed cross-program guard/closure references.

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
lowered transactionally into the final compact bytecode and aggregated into a
scene-wide image. The next stage is to execute this stream through one Luisa
interpreter, replacing the old value/closure dual-stream runtime.

The all-thread repository run executed all 318 registered tests in 12.54 s:
312 passed and the same six pre-existing exact numeric fixtures failed
(`luisa_stacked_volume_fallback`, `luisa_homogeneous_volume_fallback`,
`luisa_area_light_forward_vk`, `luisa_volume_path_fallback`,
`luisa_volume_path_vk`, and `luisa_volume_triangle_fallback`). The new compiler
test passed, and this milestone is not connected to a runtime path that could
change those images.

Only after the old split runtime has been removed will fallback, HIP, and
strict native XIR-to-SPIR-V Vulkan image gates and complex-scene benchmarks be
meaningful. No triptych is produced for this compiler-only milestone.
