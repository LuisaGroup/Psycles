# Surface closure program: compiler contract

## Purpose

The typed surface-value interpreter removed value-DAG expansion, but its
result was still materialized into one host-specialized closure tree per
surface topology. Barbershop therefore retained 564 copies of closure setup
inside the path kernel. This change introduces the compiler-side closure
program that will replace that final topology switch.

The program stores original closure operands as typed device addresses. It
does not evaluate, bake, or approximate a material on the host. Immutable node
properties such as the BSSRDF method, Beckmann selection, GGX energy policy,
and linked-normal topology are encoded explicitly in the control word.

## Formal lowering

Let `T = (V, E)` be the reachable closure DAG, `r` its root, and `P(v)` the
ordered Mix decisions on a root-to-leaf visit. For a shading point `x`, the
weight of a visited leaf is

```text
w(r, x) = 1
w(a, x) = w(parent, x)                         for Add
w(a, x) = w(parent, x) * (1 - clamp(f(x),0,1)) for Mix left
w(b, x) = w(parent, x) * clamp(f(x),0,1)       for Mix right
```

The lowering performs an iterative left-to-right DFS and emits one leaf record
for every visit, carrying the exact ordered factor path `P(v)`. Pushing the
right child before the left child on the LIFO work list preserves the existing
`GraphSurface::for_each_closure` order. Add introduces no factor. A Mix whose
host proof removes one branch introduces no factor, matching the established
closure-plan semantics. If both graph branches remain reachable, a leaf still
carries the authored factor even when the other branch belongs only to a
different consumer domain.

Every child must be a strict predecessor in `SurfaceProgram`'s topological
order. Thus the work-list measure `id.value` decreases on every edge, proving
termination and rejecting malformed cycles without a heuristic depth bound.
All streams are built transactionally; invalid topology, missing live typed
addresses, undefined flag bits, and 32-bit extent overflow reject the complete
image.

The current path representation deliberately avoids a device traversal stack.
Its instruction data can grow with the sum of visited Mix depths, but generated
shader AST is bounded by the fixed interpreter and the closure families used
by the scene. Before selecting the production representation, the device
integration will measure this path form against a Cycles-style weight-slot
form on the real scene corpus.

## Relation to Cycles 5.2

Cycles first transforms multi-closure topology into explicit
`MixClosureWeightNode` values. `NODE_MIX_CLOSURE` writes left and right weights
to typed SVM stack offsets; later fixed closure handlers read those weights.
Its `svm_eval_nodes` loop interprets uploaded node words and is pruned only by
scene-wide feature constants. Psycles follows the same binding-time boundary:
topology, typed addresses, endpoint membership, and feature masks are scene
data, while Luisa records each used handler once. The exact bytecode layout is
allowed to differ where a Luisa-friendly representation measures better.

The target kernel construction complexity is therefore

```text
O(fixed interpreter + closure families used by the scene)
```

instead of

```text
O(sum of material-topology AST sizes).
```

## Regression matrix

The compiler regression constructs real shader graphs and runs the complete
graph compiler, surface lowering, parameter binding, reachability analysis,
typed storage allocation, value bytecode lowering, and closure-program
lowering. It verifies:

- physical and emission leaves retain exact left-to-right order and endpoint
  partitioning through a live Mix;
- left and right weights reference the same typed factor address with exact
  complement polarity;
- a two-level Mix retains the root-to-leaf multiplication order and exact
  depth for all three leaves;
- a direct zero factor removes the unreachable branch and the redundant Mix
  term;
- a Principled leaf retains its semantic 26-address ABI, endpoint mask,
  feature mask, and BSSRDF method;
- deleting a live Mix factor address rejects the image transactionally.

## Scene aggregation and runtime upload

`SurfaceValueExecutionInput` now optionally names the closure plan belonging
to that exact value-storage plan. The executable-scene builder lowers closure
bytecode from the freshly produced `value_addresses`; it never accepts an
independently generated address image. A regression deliberately attaches a
live closure plan to an empty value schedule and proves that the complete
scene is rejected.

For every valid program, aggregation verifies:

- opcode, control flags, endpoints, feature masks, and BSSRDF method belong to
  their closed enums;
- every semantic operand has the expected scalar/vector bank and every local
  address is within this program's liveness-derived capacity;
- every Mix factor is a valid scalar address and every subrange is in bounds;
- recomputed operation, feature, and maximum-depth summaries equal the stored
  summaries;
- all aggregate stream extents and relocations fit 32-bit device offsets.

The runtime program descriptor is uploaded as
`[value_begin, value_count, closure_begin, closure_count]`. Parallel
`uint4` closure instructions, Principled feature masks, typed operands, and
`uint2` Mix terms are now resident on fallback/HIP/Vulkan through the same
transactional compact-surface runtime. Empty normal/Bump programs receive
empty closure ranges. The existing expanded closure execution remains active
until the fixed semantic handlers and transparent-allocation ordering have
their own backend regressions.

Commands:

```text
cmake --build build --parallel \
  --target psycles_surface_closure_execution_plan_tests
ctest --test-dir build --output-on-failure \
  -R 'psycles.surface_closure_execution_plan|psycles.surface_program_metadata'
```

Both tests pass, and `psycles_render_blender_scene` builds after the runtime
ABI/upload change. No renderer performance or image claim is made until the
fixed device interpreter replaces the topology switch and passes the
fallback, HIP, and strict native-XIR Vulkan matrix.

## Shared physical setup component

The expanded graph route and the upcoming closure-bytecode interpreter now
meet at `expand_physical_surface_closure`. This component accepts exactly one
raw semantic closure and emits its fixed Cycles-compatible physical sequence.
It owns the Principled, glass, glossy, diffuse, translucent, and BSSRDF setup
math, but deliberately does not own graph traversal or cross-leaf transparent
merging. Those are distinct sequence operations and keeping them outside the
leaf handler makes the binding-time boundary explicit.

The existing expanded route was reduced to closure traversal, calls to the
shared component, and the unchanged global transparent fold. This removes the
second copy of the setup implementation before the bytecode route is wired,
so the two execution models cannot silently drift in material math. The
following backend regressions all pass after the extraction:

```text
cmake --build build --parallel \
  --target psycles_luisa_surface_population_tests \
           psycles_luisa_surface_closure_collection_tests
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_surface_(closure_collection|population)_(fallback|hip|vk)$'
```

This is six tests total: closure collection and transactional population on
fallback, HIP, and Vulkan. Vulkan uses the configured strict native
XIR-to-SPIR-V route. This refactor alone makes no performance claim; its role
is to establish one tested semantic handler before replacing the remaining
per-material topology switch.

## Runtime closure execution and preparation fold

The compact preparation route now executes the uploaded closure stream
directly. Its generated AST has one case for each sorted unique static
semantic variant used by the scene; material identity selects only a bytecode
range. Endpoint membership and Mix factors remain device data. The static
variant key contains exactly the fields which select a host/JIT algorithm:
opcode, BSSRDF method, linked Coat Normal topology, GGX energy policy, and
Beckmann selection.

Preparation is a device-stage left fold over the retained physical sequence.
For a raw closure sequence `R`, let `expand(r)` be the shared pure physical
setup component, `allocated(c)` the Cycles weight-cutoff predicate, and `C=12`
the capacity. The observable sequence is:

```text
P = concat(expand(r) for r in R)
T = sum of every transparent contribution in P
Q = P with all transparent elements removed, with T inserted at the position
    of the first allocated transparent element (if one exists)
retained = first C elements of Q satisfying allocated
```

The implementation realizes this without a per-thread closure arena. Its
first pass folds the non-transparent prefix, accumulates `T`, and records the
raw instruction containing the first allocated transparent result. It then
folds `T` and replays physical setup from that instruction only to fold the
non-transparent suffix. Pure setup plus the recorded first-transparent
boundary proves the same source order; the accumulator applies allocation and
capacity as one transaction. No value bytecode or texture input is replayed.

Principled handlers are pruned with the scene-wide feature union, matching the
Cycles kernel-feature binding boundary. Per-instruction operands excluded by a
stronger closure plan decode to the algebraic neutral element for that layer
(zero weight/emission, alpha one, or the corresponding identity parameter).
The now-unused per-instruction feature buffer was therefore removed from the
device ABI.

The backend regression compares the expanded `GraphSurface::prepare` result
with compact bytecode preparation for all output fields. It covers a minimal
Principled under a richer scene feature union, a thin layered Principled with
alpha/sheen/coat/metallic/transmission/subsurface/emission, linked Coat Normal,
Beckmann glass, nested Mix factors, explicit emission, sub-cutoff and allocated
transparent merging, non-transparent suffix replay, back-facing state,
caustics switches, AOV/runtime-flag switches, glossy filtering, and the exact
12-slot capacity boundary.

```text
cmake --build build --parallel \
  --target psycles_luisa_compact_surface_preparation_tests
ctest --test-dir build --output-on-failure \
  -R '^psycles\.luisa_compact_surface_preparation_(fallback|hip|vk)$'
```

All three tests pass. The Vulkan registration explicitly sets
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, so that result cannot silently use
the DXC compatibility route. This validates the preparation boundary; complete
scene image and performance measurements remain a separate next step.
