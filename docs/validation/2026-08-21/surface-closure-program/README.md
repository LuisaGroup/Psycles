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

Commands:

```text
cmake --build build --parallel \
  --target psycles_surface_closure_execution_plan_tests
ctest --test-dir build --output-on-failure \
  -R 'psycles.surface_closure_execution_plan|psycles.surface_program_metadata'
```

Both tests pass. This milestone is compiler-only; no renderer performance or
image claim is made until the fixed device interpreter replaces the topology
switch and passes the fallback, HIP, and strict native-XIR Vulkan matrix.
