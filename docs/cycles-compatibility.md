# Cycles compatibility status

Updated 2026-09-08. This page describes the current implementation, not the
performance or feature coverage of the legacy material executor.

## Default execution path

Scene compilation selects the native Cycles 5.2.1 SVM surface interpreter.
No environment opt-in is required. `PSYCLES_NATIVE_CYCLES_SVM_SURFACE=0`
does not restore the old route. This applies to megakernel and staged
wavefront rendering, including static triangles and admitted ribbon curves.

The implementation preserves Cycles' node word stream, typed payloads,
stack offsets, program counter, dispatch, closure state, feature masks,
surface/volume/displacement shader jumps, and bump-state transitions.
Blender/Cycles itself supplies the compiler and GPU-state oracles. There is
no independent CPU reference renderer.

The default switch is **not** completion of the legacy code removal:
the displacement prepass still has SurfaceProgram consumers. Its private
compilation input now contains only displacement plus an inert required root;
ordinary surface, volume, light and world admission no longer depends on it.
Native KernelShader metadata supplies all material bindings, while native
attribute requests supply geometry residency. Background, stacked volume,
shadow-volume, collision and majorant evaluation use native SVM. Classroom's
Object Index admission blocker is fixed. Remaining legacy dependencies are
removal work, not supported alternate SVM architectures. The
[default-path checkpoint](validation/2026-09-07/native-default/README.md)
records the actual deletions and remaining dependencies.

## Static specialization and local storage

After scene loading, compiler metadata bounds the JIT program. Neither
profiling, pre-rendering nor scene-name constants determine allocation sizes.

- Unused opcode cases are omitted at host recording time. The same Cycles
  node/scene feature masks also omit unreachable handler bodies and closure
  consumers. This includes BSSRDF exit setup on scenes without subsurface.
- Stack capacity comes from the native compiler's stack-address analysis.
  The recorded array extent is passed to main surface, light, background,
  importance-bake, volume, density-bake and shadow SVM entries. The standalone diagnostic API
  retains a conservative default.
- Closure capacity follows the finalized Cycles graph count and scene cap,
  not the old SurfaceProgram estimator.
- Local scratch is distinct from persistent coroutine frame storage. Generic
  Luisa Local lifetime and coroutine extension/handler mechanisms express
  storage lifetime and scheduling. Ordinary scalar/vector initialization
  remains zero.
- Fallback's separate LLVM barrier frames use a CPU-worker-owned arena with
  reusable, pointer-stable overflow chunks. Its 4 MiB fast buffer is not a
  frame-size limit; capacity reuse avoids per-lane/per-dispatch malloc/free.
- Native fast math remains enabled. No forced noinline boundary or slow
  software floating-point/intersection path is used for bitwise alignment.

Proofs, counterexamples and regression boundaries:

- [Static opcode and feature pruning](validation/2026-09-07/native-static-pruning/README.md)
- [Native closure budget](validation/2026-09-07/native-closure-budget/README.md)
- [Scene-local stack extents](validation/2026-09-07/scene-local-extents/README.md)
- [Published Luisa Local/coroutine integration](validation/2026-09-07/luisa-local-coro-publication/README.md)
- [Coroutine boundary audit and SSS queue correction](validation/2026-09-07/coroutine-boundaries/README.md)

The latter reports are dated checkpoints; their isolated-snapshot or
then-unpublished qualifications describe those runs, not a second current
production route.

## Native SVM coverage and outstanding work

The AST regression checks 99 implemented handlers in Cycles' 110-tag
inventory. Nine semantic opcodes remain unimplemented: Holdout, Radial
Tiling, Bevel, Ambient Occlusion, Raycast, AOV Start/Color/Value and Scene
Time. NONE and PAD1 are not executable shading handlers. A material using
an unsupported reachable operation must not silently fall back to the
legacy executor or substitute a socket default.

Recent independently checked native families include:

| Area | Evidence |
| --- | --- |
| Surface allocation, closure setup/evaluation/sampling, BSSRDF exit and state flags | [Surface state](validation/2026-09-07/native-surface-state/README.md), [zero-BSDF state](validation/2026-09-07/zero-bsdf/README.md) |
| ShaderData geometry, packed object/primitive identity, curve segments and lamp emission | [Default-path checkpoint](validation/2026-09-07/native-default/README.md) |
| Background/NEE ShaderData, native world evaluation and camera-dependent importance baking | [Native background](validation/2026-09-08/native-background/README.md) |
| Volume Absorption/Scatter, Volume Coefficients and Principled Volume node streams and allocation state | [Native volume SVM](validation/2026-09-07/native-volume-svm/README.md) |
| Ordered volume stacks, main/shadow consumers, phase copy, runtime extrema and density baking | [Native volume consumers](validation/2026-09-08/native-volume-consumers/README.md) |
| Native scene admission, used-shader attribute residency, mesh constant emission and volume NEE emission | [Scene admission](validation/2026-09-08/native-scene-admission/README.md) |
| Map Range and analytic Sky node behavior | [Map Range](validation/2026-09-07/map-range/README.md), [analytic Sky](validation/2026-09-07/analytic-sky/README.md) |

The native volume consumer retains one closure allocator across the whole
ordered stack, Cycles' per-entry phase merging and its final eight-phase
active-prefix copy. Sigma_s includes only successfully allocated closures.
Majorant baking uses one/sixteen samples and session-owned resources built
after camera parameters are finalized. This does not claim imported sparse
volume grids or motion-object support.

Geometry setup uses the native KernelObject, packed triangle data,
KernelCurve and curve-key images. Curve acceleration segments map to
Cycles' containing curve plus packed segment index; they are not triangle
indices. The current scene upload admits static ribbon curves. This does not
claim motion geometry, other curve shapes or point-cloud rendering.

## Image parity and performance

A focused oracle pass establishes that tested node/state behavior agrees
with Cycles; it does not establish every scene's image parity or speed.
Historical custom-executor performance gains and five-way promotions have
been removed from this current-status page. Their dated evidence remains
under `docs/validation/`.

Current large HIP checkpoints use fixed samples and native fast math.
The old SurfaceProgram instruction/topology histogram and its CLI/API have
been removed; its counts do not describe native Cycles SVM. Closure-count
histograms and per-path traces remain supported.
The [native scene admission checkpoint](validation/2026-09-08/native-scene-admission/README.md)
records these 256 spp single canaries, with all 46 channels finite:

| Scene | Extent | Cold main JIT s | Render-only s | Frame | Combined / DiffInd rel. RMSE |
| --- | --- | ---: | ---: | ---: | --- |
| Lone Monk | 1440x1080 | 44.3570 | 13.8669 | 220 B | 0.01240942 / 0.12881692 |
| Monster | 1080x1080 | 57.2307 | 15.1202 | 284 B | 0.00547924 / 0.02552799 |
| Classroom | 1920x1080 | 58.5554 | 18.3531 | 264 B | 0.00353344 / 0.17820336 |

Barbershop currently stops at unavailable-image admission; it has no new
successful rendering or performance result. Missing-image identity and
sampling state need an original-Cycles regression. Shader/binding cleanup
shortened the observed JIT canaries but did not remove the indirect residuals.

A fresh, profiler-free Cycles HIP check on 2026-09-08 ran each scene three
times. Main-loop times were 13.4344/13.4429/13.4521 s for Monk and
14.4242/14.4129/14.4294 s for Monster (medians 13.4429/14.4242 s).
Evidence is in `/var/tmp/psycles-cycles-hip-check-9wipGw`; Blender build identity
is `9e2066aef7ef`, with fixed 256 spp, seed 0, no adaptive sampling or denoise,
on the same RX 9070 XT. These are main-loop wall times, not summed kernel
timings or the Python render-call duration. The latest single Psycles canaries
are approximately 3.2%/4.8% slower; this is not a paired current-revision
benchmark. A fresh Classroom Cycles main loop takes 20.7432 s at the same
1920x1080/256 and seed 1. Cycles' precompiled/cache behavior is not equivalent
to Psycles' cold main-path compilation. The older Monk reference was
captured under rocprofv3 and is not the timing baseline for this comparison.

The [same-sample Monk diagnosis](validation/2026-09-07/lone-monk-residual/README.md)
identifies a concrete visibility divergence at coincident leaf geometry.
The original scene and duplicate primitives are retained. The observation
does not justify deduplication, extra shading, a global RNG mismatch claim,
or a slow bit-matching intersection path. The
[sampler contract](validation/2026-09-07/sampler-contract/README.md) separately
pins the actual Cycles automatic-scrambling property.

Single canary times are not a paired benchmark. The complete current-revision,
multi-scene performance campaign and remaining structural DiffInd/visibility
alignment are still open. Do not infer a speedup from smaller IR, fewer local
lanes or a smaller frame alone.

## Differential policy

Use original Cycles 5.2.1 source and original Cycles GPU execution for oracle
results. Keep fixture inputs and expected outputs separate; do not generate
expected streams with the Psycles compiler.

Compare discrete state, node words, stack addresses, PC transitions,
visibility, object/primitive identity and RNG dimensions structurally.
Continuous values use native-fast-math tolerances. One-ULP differences do
not justify slower algorithms or strict software emulation.

Scene reports must state source bundle, revisions, device/backend, sampler,
seed, dimensions, sample range, scheduler and feature settings. Separate
render-only timing from cold JIT and scene compilation. Verify finite linear
passes, contribution-weighted residuals and same-sample paths; do not use
DiffInd relative RMSE alone as a count of unnecessary paths or shading.

Builds use all 32 available threads. Validate HIP first, then fallback.
Vulkan canaries require native XIR-to-SPIR-V with DXC disabled; a successful
Vulkan render through a different compiler route is not the required gate.
