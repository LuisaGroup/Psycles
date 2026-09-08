# Architecture

Updated 2026-09-08. Current coverage and measured limitations belong in
[cycles-compatibility.md](cycles-compatibility.md) and
[the validation index](../VALIDATION.md).

## Required execution model

Cycles 5.2.1 defines both the observable material behavior and its SVM
execution model. Psycles implements the original node word stream, typed
payloads, stack addressing, program-counter loop, node dispatch, closure
state, feature masks and surface/volume/displacement control flow in Luisa
DSL. The implementation lock in [DEVELOP.md](../DEVELOP.md) forbids an
alternative material interpreter. Cycles itself is the only rendering oracle;
there is no independently implemented CPU reference renderer.

The data flow is:

```text
Blender evaluated scene and normalized graph
  -> typed graph / immutable scene contract
  -> Cycles-isomorphic SVM words, shader metadata and geometry images
  -> scene-specialized Luisa DSL and device JIT
  -> renderer session, linear passes and original-Cycles differential
```

Native SVM is the default path for ordinary surface, volume, shadow, world
and light material evaluation. A private displacement-prepass bridge still
uses the old evaluator. That dependency and unreachable legacy helpers are
removal work, not a second supported architecture.

## Graph and scene contracts

The pre-SVM Cycles adapter preserves socket topology, defaults, properties
and closure composition. Canonical node/variant keys and explicit socket
mappings make unsupported nodes a coverage diagnostic. A graph already
marked SVM-lowered is not reconstructed into a different closure tree.
Node-group recursion and memoization belong to normalization; host code
does not bake or evaluate shading as a substitute for device execution.

The typed ShaderGraph has surface, volume and displacement roots, with an
internal surface-normal projection for Cycles' bump terminal. Structure and
parameter signatures describe invalidation boundaries, not permission to
change semantics. Values folded into SVM words or host-specialized DSL must
invalidate the corresponding compiled program when changed.

SceneDatabase applies a SceneDelta to a candidate immutable snapshot,
validates references, and commits atomically. Contract identifiers do not
contain Luisa resource handles or Cycles device pointers. Mesh attributes
retain point, corner or face domains; shared vertices are not duplicated to
replace domain-aware interpolation. Generated affine transforms and native
attribute requests control the data needed by admitted shaders.

## Native compiler and material runtime

The native compiler performs the Cycles graph transformations and stack
allocation, then emits typed opcode payloads. The linked image retains a
ShaderJump table and the original surface, volume and displacement entries.
The bump prefix falls through into surface evaluation without an intervening
END. BOTH-displacement evaluation saves P and its derivatives, evaluates
undisplaced geometry and the bump graph, installs the new normal, restores
P/derivatives, then continues into the surface program. Restoring N at that
boundary would change Cycles' state machine.

KernelShader metadata carries material capabilities, closure bounds and
emission information. Used-shader attribute requests determine native
geometry residency. KernelObject, triangle, curve/key and attribute images
supply ShaderData services rather than an alternate per-node geometry model.
Static ribbon curves use containing-curve identity plus packed segment
indices; they are not represented as triangle identities.

The Luisa SVM retains one PC-loop node dispatcher. Unused cases are omitted
during host recording; original node/scene feature masks also remove
unreachable handlers and consumers. Stack bounds are derived from native
compiler allocation facts, closure bounds from the finalized Cycles graph
and scene cap. Neither is inferred from a render/profile or a scene name.
Unknown specialization information retains a conservative bound.

Surface closure initialization, allocation, setup, evaluation and sampling
use native Cycles state. Shared incoming closure weights accumulate with
ADD, as in Cycles' multi-closure transformation. Failed image loading remains
distinct from an unassigned socket and follows native missing-image behavior.
These are independently tested word/state contracts, not visual heuristics.

Ordered volume evaluation retains one closure allocator across the stack.
Per-entry phase merging precedes the final active-prefix copy of up to eight
phases; only allocated closures contribute scattering. Main/shadow volume,
collision, majorant and density consumers use native SVM. Camera-dependent
background importance and density resources are built after finalizing the
session's camera parameters.

## Transport and coroutine scheduling

RendererBackend compiles a scene into an opaque CompiledScene. A
RenderSession accepts sample ranges and writes named pass tiles to an
OutputSink. Host-stage PathKernelPipeline components assemble Luisa device
expressions for event handling, shading, direct lighting, volume and closure
continuation. They are AST-building interfaces, not host shading evaluators.

Megakernel and staged wavefront modes execute the native material path.
Coroutine boundaries preserve Cycles event ownership: closest-event work,
volume handling when needed, surface shading, shadow work and subsurface
continuation. A successful subsurface intersection proceeds to shade the
known surface instead of repeating closest intersection. Stage presence
depends on statically admitted scene features; a scene's stage count alone
does not prove state-machine parity.

Psycles-specific sorting/queue policy is expressed through generic Luisa
Coro Ext/Handler facilities. Scheduler dispatch produces stream commands,
using `stream << scheduler(...).dispatch(...)`. Persistent per-path frame
state is separate from temporary shader-local storage and uniform arguments.
Frames live in scheduler-owned reusable pools, not per-thread malloc paths.
Fallback's LLVM barrier-frame arena can reuse overflow chunks beyond its
4 MiB fast buffer; that fast buffer is not a maximum frame size.

Generic Luisa analyses may eliminate provably unnecessary zero stores or
promote transitively read-only references. Ordinary scalar/vector default
zero initialization remains unchanged. Such compiler work must establish
effect, alias and lifetime safety independently and validate the original
renderer module after the minimal regression. No renderer-specific branch
belongs in Luisa's compiler analysis.

Direct-light components preserve Cycles' RNG dimensions, proposal/resampling
relationships, PDFs, roulette, MIS, closure evaluation and pass routing.
Deferred emission is evaluated only at its corresponding transport predicate.
Structural alignment of every remaining work predicate is still under audit;
component composition or matching aggregate energy is not proof of parity.

## Validation and remaining boundaries

Every migrated family needs original Cycles words or GPU-state fixtures and
whole-program validation. The fallback backend executes the same Luisa DSL;
it is not a separate correctness oracle. Full-scene comparisons use the same
original .blend, frame, camera, settings, seed and fixed samples, with linear
passes, invalid counts and visual inspection. Indexed per-path traces locate
the first RNG/state/visibility divergence without consuming extra samples.

Fast math stays enabled. Discrete words, state, addressing, predicates and
sampling dimensions must align; a harmless last-bit difference does not
justify slow software math, texture filtering or intersections. Inlining is
left to the compiler rather than manually forced noinline boundaries.

The [equal-pass HIP baseline](validation/2026-09-08/matched-pass-hip/README.md)
separates render wall time, scene compilation, session initialization
(JIT plus setup/baking) and coroutine frame size. It does not establish
complete performance or path parity. Remaining
native opcodes, legacy displacement removal, unsupported geometry/motion
configurations and indirect-light differences are listed in the compatibility
status. OSL and unadmitted features are not silently approximated.
