# Psycles development status

Updated 2026-09-08. This page contains current development rules and remaining gates. “Implemented”
does not mean “Cycles compatible”: compatibility requires an exact-revision
Cycles render, linear-pass metrics, and visual comparison. The current
commands, machine, reports, and triptychs are in
[VALIDATION.md](VALIDATION.md).

## Mandatory Cycles SVM implementation lock

Until the project owner gives an explicit superseding instruction, the sole
implementation task is an isomorphic Luisa DSL implementation of Blender
Cycles 5.2.1 SVM. This is a hard development constraint, not a design
preference:

- reproduce Cycles' SVM node stream, typed node payloads, stack addressing,
  program-counter loop, single node-type dispatch, closure state, feature
  masks, and surface/volume/displacement control flow before attempting any
  alternative architecture;
- do not retain or extend Psycles-specific substitutes such as execution-family
  plus semantic-subtype dispatch, independently scheduled value/closure
  programs, or a second closure-leaf decode layer;
- do not redesign, generalize, fuse, split, or otherwise "improve" the Cycles
  execution model. Luisa multistage/JIT facilities may erase node cases and
  closures that Cycles feature masks prove unreachable, but may not change the
  observable state machine or bytecode semantics;
- do not advance unrelated renderer features or optimizations while this SVM
  replacement remains incomplete;
- every migrated node family requires a field/state mapping to the exact
  Cycles 5.2.1 source, a Cycles-oracle regression, and whole-program validation.

No deviation is implied by temporary regressions or by the convenience of the
existing implementation. Only an explicit project-owner instruction can relax
this lock.

## Mandatory structural-parity performance lock

The Cycles lock applies to the observable renderer state machine, not to an
incidental floating-point bit pattern selected by one compiler or device. Until
the project owner explicitly says otherwise, a non-structural one-ULP or
last-bit difference must never justify a runtime performance cost:

- do not disable fast math, emulate native math or texture filtering in
  software, add branches or memory traffic, or replace a hardware instruction
  solely to reproduce the final bit of one Cycles build;
- keep structural facts exact: SVM words and cursor movement, control flow,
  visibility/support, RNG dimensions, sampling probabilities, closure state,
  energy, texture addressing/filter type, and finite/invalid behavior;
- external-oracle regressions must compare structural words exactly and use a
  documented numerical tolerance only for host/backend arithmetic whose small
  representation difference does not alter those structural facts;
- stricter arithmetic is permitted only with a written proof obligation, such
  as maintaining an outward probability bound. Each exception must be narrowly
  allowlisted and regression-tested rather than inferred from an exact-hash
  fixture.

Native texture operations are part of this lock: nearest and linear filtering
must remain one native sample, and Cycles cubic filtering must remain its four
native bilinear samples. A backend-specific software texel loop is not an
acceptable way to match interpolation rounding.

Vector normalization follows the same rule. Cycles' zero, fallback, and
near-zero domain predicates remain explicit because they can change control
flow or finite/invalid behavior; arithmetic inside the accepted domain uses
the shared native reciprocal-square-root implementation in
`native_vector_math.h`. Do not reintroduce scalar `sqrt` plus division merely
to reproduce a CPU/GPU rounding sequence.

## Authoritative-reference policy

The version-pinned Blender Cycles 5.2.1 source and renders from the same
`.blend`, frame, integrator settings, seed, samples, and linear passes are the
only correctness oracle for Psycles rendering and sampling.

- Do not build a CPU renderer, CPU sampling implementation, or host-side
  “reference” evaluator for any Luisa device algorithm.
- Host code may compile immutable scene data for device upload, but must not
  duplicate a Luisa selection, BSDF, light-shape, transport, or MIS algorithm
  for validation.
- Regression tests for renderer behavior must execute the Luisa DSL/JIT path
  on the supported backends and compare against current Cycles outputs or
  explicitly versioned Cycles fixtures.
- Reading Cycles source establishes both SVM semantics and, under the active
  implementation lock above, the required SVM execution model. Luisa DSL is
  the implementation language; it is not permission to substitute a different
  interpreter architecture.

## Current implementation and evidence

Native Cycles 5.2.1 SVM is the default material execution path. Ordinary
surface, volume, shadow, world and light consumers no longer depend on the
legacy material evaluator. A private displacement-prepass bridge remains;
it is removal work, not an approved alternate architecture.

Use [the compatibility status](docs/cycles-compatibility.md) for current
coverage and [the validation index](VALIDATION.md) for completed gates.
The [equal-pass HIP campaign](docs/validation/2026-09-08/matched-pass-hip/README.md)
records 12 paired 256 spp runs at Psycles c5bf9247 / Luisa 9ea3b720f,
with both engines' EXRs validated against the same 15-pass contract.
Historical progress notes and old performance results are not current gates;
their dated reports remain under docs/validation and in Git history.

## Worktree and publication discipline

Verify the designated root worktree and its nested Luisa checkout before
editing. Do not assume the shell's default directory or reuse an independent
Luisa checkout. On the validated workstation, the active worktree is
/home/mike/Projects/Psycles-surface-svm. Inspect both repositories' status,
branch, remotes and diffs; preserve unrelated user changes.

Publish generic Luisa corrections to origin/next before advancing the
Psycles gitlink. Publish reviewed Psycles changes to origin/main without
force-pushing. Stage exact task files, including a gitlink only after its
referenced child commit is published and validated. Commit useful validated
checkpoints promptly; do not accumulate unrelated edits into one change.

Renderer-specific coroutine policies belong in generic Coro Ext/Handler
clients, not Psycles-specific branches in Luisa compiler/scheduler facilities.
Use stream insertion for scheduler dispatch commands. Keep frame storage in
scheduler-owned pools; never introduce a per-thread/per-resume malloc path.
Do not use noinline markers as a renderer optimization strategy.

## Verification order

Compiler/XIR work follows: formal effect/control-flow analysis, a minimal
failing counterexample, a permanent regression, the generic correction, then
validation of the full original module. Passing a reduced sibling is not
evidence that the original renderer compiles or runs.

Build with every hardware thread (32 on the validated workstation). Validate
HIP first, then fallback, then strict native XIR-to-SPIR-V Vulkan canaries.
Vulkan validation must set all three guards:

```bash
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j32 -R '<focused-native-Vulkan-tests>'
```

Report configuration, compilation, focused tests, complete suites, and real
scene canaries separately. Do not call a build successful from configuration
alone, promote an expected failure into a pass, or conceal pre-existing
suite failures. Keep toolchains and generated evidence in local build/temp
directories; do not modify system packages or shell profiles for a probe.

## Static specialization and binding

Preserve the original word stream and runtime predicates. Derive generated
node cases and stack/closure bounds from compile-time graph/emission facts
and used-scene data, never from a pre-render, profile or scene-name table.
Both arms of dynamic branches contribute to the conservative bound. The bump
prefix belongs to the surface ShaderJump entry and must not be discarded.

Keep frequently changing render values in arguments/resources when the
generated program remains valid: seed, sample range, extent, camera,
transforms and integrator parameters. Changes that affect folded words,
reachable cases, static array bounds or binding types require appropriate
program/cache invalidation. Never pin an old shader cache name to hide a
structural change. Ordinary scalar/vector default zero initialization is
unchanged; lifetime-without-initialization is not permission to read
uninitialized storage.

## Measurement and sampling gates

Use the [schema-v2 benchmark runner](docs/scene-benchmark.md). Compare Cycles'
original main-loop wall time with Psycles render-only wall time. Keep scene
compilation, main JIT, whole render calls, process time and profiled GPU sums
separate. State cache policy, exact revision/build, device, seed, frame,
extent, samples and scheduler options. Do not overlap performance runs with
other renders or builds.

Repeated full-scene measurements accompany performance conclusions. Smaller
IR, scratch, or coroutine frames alone do not prove a renderer speedup.
Inspect per-pass absolute error, relative error, reference signal scale and
non-finite counts; aggregate energy agreement is not same-path correctness.

The Cycles 5.2.1 sampler uses the scene's actual scrambling configuration and
path-event dimensions. Verify random words and advancement decisions through
the original Cycles trace, not a second host sampler. A matching subset of
random fields or pixels is not proof of global RNG/path parity.

## Remaining completion gates

- Implement the remaining native semantic opcodes with original word/state
  oracles; unsupported reachable behavior must not silently fall back.
- Remove the private legacy displacement bridge and unreachable old material
  execution code, then validate displacement/bump, geometry and camera state.
- Resolve remaining indirect-light/path structural differences and remove
  unnecessary work at the same predicates as Cycles.
- Keep surface/volume/displacement control flow, closure allocation, sampling,
  scene identity and all exposed render passes faithful to the original.
- Complete original-scene correctness and performance checks after each
  material change, including failures and remaining uncertainty.
- Reach the requested rendering-efficiency target across the tested scenes;
  one faster scene or a successful showcase render does not finish the goal.

OSL and unadmitted geometry/motion configurations are not implicitly supported
by the default SVM switch. Exact coverage and known backend exceptions belong
in the current compatibility/validation records, not an obsolete node count
or roadmap copied from an earlier Cycles version.
