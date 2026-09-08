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
  Opcode usage and stack high-water marks are tracked per ShaderJump entry;
  bump falls through into surface and shares its maximum allocation bound.
  Linking unions corresponding entries; unproven external images retain a
  conservative whole-image bound.
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
- [Per-entry opcode and stack specialization](validation/2026-09-08/svm-entry-usage/README.md)
- [Published Luisa Local/coroutine integration](validation/2026-09-07/luisa-local-coro-publication/README.md)
- [Coroutine boundary audit and SSS queue correction](validation/2026-09-07/coroutine-boundaries/README.md)
- [Transitive read-only references and uniform frame state](validation/2026-09-08/coro-readonly-forwarding/README.md)

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
| Assigned-but-failed image identity and native sampling before UV wrapping | [Failed-image state](validation/2026-09-08/native-missing-image/README.md) |
| Shared surface/volume closure weight accumulation, full word images and original GPU allocator state | [Shared closure weights](validation/2026-09-08/shared-closure-weights/README.md) |
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

The [equal-pass HIP campaign](validation/2026-09-08/matched-pass-hip/README.md)
completed 12 fresh pairs at Psycles c5bf9247 / Luisa 9ea3b720f, using original
production Cycles 5.2.1 LTS build 9e2066aef7ef on the same RX 9070 XT.
Both engines produce exactly 15 linear passes / 46 channels at 256 fixed
samples. This supersedes earlier unequal-pass ratios and the withdrawn
Classroom speed-lead interpretation; those dated reports remain historical
evidence, not current performance claims.

Times are three-run medians in seconds. Cycles uses its main-loop wall
interval; Psycles uses render-only wall time. Main shader caching is disabled,
auxiliary/downstream/OS caches retain normal policy, native fast math is on,
and no concurrent build/render or GPU profiler overlaps these pairs.

| Scene / extent / authored seed | Cycles HIP | Psycles HIP | Relative time | Session init | Frame |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monk / 1440x1080 / 0 | 13.2425 | 13.6826 | 1.0332 | 18.7327 | 220 B |
| Monster / 1080x1080 / 0 | 14.2865 | 15.1328 | 1.0592 | 22.1471 | 284 B |
| Classroom / 1920x1080 / 1 | 18.0319 | 18.7626 | 1.0405 | 18.1196 | 264 B |
| Barbershop / 2048x858 / 0 | 25.3775 | 39.2753 | 1.5476 | 25.6044 | 416 B |

At that paired checkpoint the efficiency goal is not complete: Barbershop is
54.8% slower, compared with 3.3%–5.9% for the other scenes. Session init is the CLI's
shader_jit_seconds, including JIT plus setup/baking, not compiler-only.
Cycles loads precompiled GPU kernels. Smaller IR, local arrays and frames
do not independently establish end-to-end speedup.

The [native descriptor sampler correction](validation/2026-09-08/bound-image-sampler/README.md)
and [HIP descriptor investigation](validation/2026-09-08/hip-texture-descriptors/README.md)
confirm image/sampler SRDs are already device-resident and sampled directly.
Neither more aggressive descriptor inlining nor explicit index grouping is
adopted after GPU controls. A smaller pointer chain or synthetic sampling
speed is not reported as an end-to-end renderer gain. Their dated follow-up
timings remain in those reports, not mixed with current measurements below.

The earlier [light-endpoint correction](validation/2026-09-08/light-endpoints/README.md)
separates geometric lamp hits from spot/spread evaluation and indirect shader
visibility, matching original Cycles' empty-emission state transition. The
permanent original-GPU regression has 240 endpoint and 2048 visibility checks.
Fresh 64-spp Barbershop work counts now differ from Cycles by only 196 surface
and 70 volume visits out of 332.3M / 85.1M. This repairs a real structural
error, but surface visits were already within 0.04%; it does not explain away
the remaining surface cost per path. Subsequent lamp-routing work below fixes
post-lamp closest scheduling; shadow surplus and residual DiffInd remain.
Its original-GPU tests and dated canaries remain in that report; they are not
substituted for fresh paired Cycles measurements.

The [scheduler trace comparison regression](validation/2026-09-08/dispatch-trace-comparison/README.md)
also resolves the former fallback test failure: it passed 182/182 at that
checkpoint, with exact RNG/discrete state checks and unchanged film tolerances.
No renderer binaries or captured trace bits change with that test-only fix.
The [lamp-routing / CFG / surface investigation](validation/2026-09-08/lamp-routing-and-surface/README.md)
adds the original transparent-lamp traversal boundary, native miss-distance
normalization and loop-epoch CFG repair. The full original native module now
restructures. Hidden input default provenance and Vector Math host folding
have 81 original-Cycles word-image regressions. The
[graph-boundary repair](validation/2026-09-08/svm-graph-boundaries/README.md)
adds 24 exact images for bump edge ownership, retained BUMP displacement
entries and procedural Color/Factor sharing. The latest
[native socket/declaration repair](validation/2026-09-08/svm-socket-declarations/README.md)
adds 53 exact images for native texture POINT inputs, conversion type-pair
identity, closure declaration order and unavailable Voronoi defaults.
Entire raw Barbershop used-shader images matching the oracle increase from
112 to 115; only `bricks` retains a different length (+6 words). Equal-length
differences and resource-ID payload mapping are not claimed fully aligned.
The static stack bound remains 33 floats; the complete surface machine-code
text and its register/private-memory requirements remain unchanged.

Main SVM dispatch and most handlers are already inlined. Keeping more HIP
function boundaries in a controlled A/B/A experiment slows surface time by
35.5%; that intervention is reverted. No inlining policy or device arithmetic
is changed. The current 64-spp surface GPU total is 5.871504 s against the
retained original Cycles 2.962829 s. Surface visits remain effectively
unchanged at 332.3 million. Final machine code retains only three outlined
noise/math helpers, 256 VGPRs and 2496 fixed private bytes. The report records
actual code-object identities and callees rather than equating static spill
sites with dynamic memory traffic or mistaking pre-HIPRTC LLVM definitions
for final out-of-line functions. Per-invocation surface cost remains open.

Current suites pass HIP 182/182 and fallback 184/184; Luisa 155/155 is the
retained validation of the unchanged child checkpoint.
The parallel fallback run exposed a separate production-queue lost-wakeup
race, repaired generically in child `85e5300f1`, with two minimal failures
and 100 green repetitions each. Strict native Vulkan lamp-routing and
bump-state tests pass 2/2 with 31 native SPIR-V compilations and no DXC/DXIL
load. Current host results are 164/164, including the unwaived source-size
gate. Existing oversized tests are separated into cohesive modules without
removing assertions; ConvertNode now has its own ordinary translation unit.

At Psycles `33c7b335` / Luisa `85e5300f1`, six new full-resolution 256-spp
follow-ups complete against retained Cycles references, with exact prior
geometry/images and new source socket metadata. Current render times are
13.4347 / 14.7923 / 18.2949 s for Monk/Monster/Classroom (one each), and
40.2231 s for Barbershop (three-run median; 40.2231 / 40.2419 / 40.1987 s).
Current frames are 220 / 280 / 260 / 416 B. The smaller temporal follow-up
times are not an isolated causal speedup estimate; Barbershop remains 58.5%
slower than the retained Cycles median.
Initialization is reported separately for every run, and is not cold JIT:
the earlier profiler run had already warmed downstream caches. These are
not new paired Cycles timings. The [latest report](validation/2026-09-08/svm-socket-declarations/README.md)
retains the exact six-run metrics and source hashes.

All 46 Psycles channels are finite in every run. The revision-pinned paired
baseline's first-pair relative RMSE is:

| Scene | Combined | DiffCol | DiffInd |
| --- | ---: | ---: | ---: |
| Monk | 0.01241440 | 0.000545920 | 0.12881733 |
| Monster | 0.00547872 | 0.000110133 | 0.02552496 |
| Classroom | 0.00353306 | 0.000102311 | 0.17820333 |
| Barbershop | 0.01079848 | 0.001597578 | 0.07454033 |

The report retains all 15 passes for all repeats, exact commands, observed
ranges, source/export/output/implementation hashes and seed/frame semantics.
First-pair Combined triptychs were inspected at resized viewing resolution.
Original Cycles Classroom has 25 non-finite DiffDir and 27 non-finite
GlossDir pixels per run; metrics explicitly exclude the invalid union.
Nearly empty passes retain absolute errors and reference signal scale.
These residuals are open correctness work, not waived as noise or one ULP.

Barbershop's unavailable-image admission and missing shared transparent
closure are fixed by original-Cycles word/GPU-state regressions. At the
diagnosed pixel, the first four surface events and all 45 sampled random
fields match, but a later NEE event still selects an adjacent emitter triangle.
Its generic read-only-reference frame correction reduces 896 B to 416 B
without changing six-stage control flow. The
[volume work correction](validation/2026-09-08/volume-work/README.md) has no
measurable volume-kernel speedup; surface remains the larger performance
difference. None of these observations proves global path parity.

The [same-sample Monk diagnosis](validation/2026-09-07/lone-monk-residual/README.md)
identifies a visibility divergence at coincident leaf geometry. Original
duplicate primitives are retained. It does not justify deduplication, an
unproven global RNG-mismatch claim or a slow bit-matching intersection path.
The [sampler contract](validation/2026-09-07/sampler-contract/README.md) pins
Cycles' actual automatic-scrambling property.

Remaining structural DiffInd/visibility alignment and cross-scene efficiency
are open; average-energy agreement is not same-path correctness. The old
SurfaceProgram instruction/topology histogram and CLI/API no longer describe
the native path. Closure-count histograms and indexed path traces remain
available to diagnose work without changing sampling dimensions.

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
