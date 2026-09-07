# Native scene admission and multi-scene check

Base: Psycles `867bb4d0`; Luisa remains `fb315d27d`. Original Cycles source:
`cb168525138fecc792cc393f94afc39582b0103c`; installed Blender build:
`9e2066aef7ef`. Evidence: `/var/tmp/psycles-native-volume-svm-06XnDX`.

## Structural change

Classroom failed before JIT because its valid Object Index -> Math graph was
first sent through the old SurfaceProgram compiler. The standalone native
compiler already supported this graph. The permanent whole-scene admission
test reproduces that failure in `native-admission-before-hip.log`.

Ordinary surface/volume/light/world admission now compiles only the original
native SVM graph. Material bindings read the native KernelShader image, not
legacy topology tags or parameter arrays. Attribute residency projects the
native symbolic/standard requests onto geometry upload sources, including
Pointiness and named current/undisplaced tangents and their UV inputs. Its
union includes every Geometry::used_shaders slot and instance override, not
only primitive-selected slots. The host regression checks unused slots,
overrides, tangent/sign dependencies, name precedence and curves.

Native surface population no longer constructs unused legacy preparation,
sampling, emission, BSSRDF-normal or replay callables. The scene admission
test checks that these AST roots do not exist. Mesh-emitter constant emission
also reads the original KernelShader constant field; it no longer accesses
legacy parameter buffers before SHADE_LIGHT_NEE.

Removing those callable roots exposed a remaining production dependency:
heterogeneous-volume mesh-light emission still invoked the old callback
(`native-admission-volume-backtrace.log`). Volume deferred emission now uses
one shared native light evaluator after the receiving-phase test, including
the sampled endpoint, native object/primitive IDs, static volume time and zero
ray differentials, and the original path counters and LCG inputs. This reuses
the surface NEE ShaderData reconstruction; it does not claim that volume NEE
already has an independent coroutine/queue stage. The volume-triangle test
also evaluates Object Index + 10 at the sampled emitter and asserts that the
native shader is nonconstant before comparing its output to the golden.

The retired SurfaceProgram execution histogram, its host instruction census,
public request/sink types and CLI output are deleted. They required the old
topology/value-runtime image and cannot measure native Cycles SVM execution.
The fallback sample-dispatch test exposed this stale dependency; it retains
its film, path-state, sample-chunking and closure-count assertions. Native
closure-count histograms and path tracing remain available. No native
instruction counts are inferred from an obsolete topology.

This is not complete deletion of SurfaceProgram. The remaining displacement
prepass gets a private displacement-only graph, with an inert surface root
required by that old compiler. That root never enters the native word image.
Its implementation and the unreachable legacy helpers still require removal.
Native AO remains unsupported and cannot silently obtain scene state from
that private displacement graph.

## Large HIP canaries

RX 9070 XT / gfx1201, HIP 7.2.53211. Fixed samples [0,256), no adaptive
sampling/denoise, native fast math, staged wavefront, surface sort and separate
NEE queue, frame capacity 1,048,576. Main shader cache disabled; auxiliary
shader caches retain their normal policy. No profiler or concurrent GPU test
ran during these timings. These are single canaries, not paired medians.

| Scene / extent | Scene compile s | Cold main JIT s | Render-only s | Coro frame | Combined rel. RMSE | DiffInd rel. RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lone Monk / 1440x1080 | 4.59664 | 44.3570 | 13.8669 | 220 B, 4 stages | 0.01240942 | 0.12881692 |
| Monster / 1080x1080 | 1.47973 | 57.2307 | 15.1202 | 284 B, 6 stages | 0.00547924 | 0.02552799 |
| Classroom / 1920x1080 | 2.16361 | 58.5554 | 18.3531 | 264 B, 5 stages | 0.00353344 | 0.17820336 |

All 46 output channels are finite in each successful image. Pass comparisons
verify exact Blender build metadata. Monk and Monster use
seed 0, Classroom seed 1. Reference EXRs retain the earlier identical-source
goldens; Monk's profiler-instrumented old golden is an image reference only,
never the performance baseline. Classroom's DiffCol / DiffDir relative RMSE
is 0.00010232 / 0.00265630, and its Combined luminance ratio is 0.9999983.
That does not prove identical paths or remove the remaining indirect residual.

A fresh, profiler-free Cycles HIP Classroom run at the same 1920x1080/256,
seed 1 and TABULATED_SOBOL settings takes **20.7432 s in the main loop**
(`cycles-classroom-fresh.log`), versus a Python render-call duration of 22.183 s.
Do not mix those timing definitions. Together with the earlier three-run
Cycles Monk/Monster medians (13.4429/14.4242 s), these checks show competitive
render throughput but do not establish the full correctness/performance goal.
The preceding Psycles main-JIT canaries were 66.0925/71.9387 s for Monk/Monster;
the new shorter single JIT times are not a controlled attribution or runtime
speedup claim.

Bundles: Monk `/var/tmp/psycles-fullres-hip-20260830/exports/lone-monk-current`;
Monster `monster-export` in the evidence directory; Classroom
`/var/tmp/psycles-multiscene-52-Vw3BkE/classroom/export`.
Canary and metric filenames use `<scene>-native-admission-*`.

## Multi-scene blocker at this checkpoint

The subsequent [failed-image checkpoint](../native-missing-image/README.md)
fixes the resource admission failure below and records the first complete
Barbershop canary, its remaining image/closure discrepancy and frame size.

Barbershop at 2048x858/256 stops before JIT. Its source has unavailable image
datablocks (`guilder_ornament.png`, `generic_scratches.png`); the exporter
omits them and import maps their references to resource zero. Native scene
admission rejects that unavailable resource. Cycles instead retains image
identity and has explicit failed-image sampling state. Fixing that boundary
requires an original-Cycles missing-image oracle, not a scene-specific texture
substitution or a fabricated expected stream. No Barbershop time or successful
image is claimed. Its effective seed **1267069554** matches the original
reference (seed 0, animated seed enabled at frame 1).

## Validation

All-target builds use all 32 threads. The final diagnostic-cleanup revision
passes all **175/175 HIP tests** (`native-admission-cleanup-all-hip.log`,
662.30 s). The five focused admission/volume/normal tests also pass, including
the new nonconstant mesh-light case. Test-suite durations include shader
compilation and are not render-performance comparisons.

Fallback is **176/177** (`native-admission-cleanup-all-fallback.log`,
440.12 s). The sole remaining failure is the pre-existing sample-dispatch
trace difference at slot 30/component 2: light normal z is -0.623230
(`0xbf1f8bfd`) versus -0.623204 (`0xbf1f8a50`). The obsolete histogram failure
is gone. No tolerance or software floating-point path was changed.

Host tests are **149/150** (`native-admission-cleanup-host.log`, 14.48 s).
The unchanged source-size guard still reports four files above 2,000 lines:
`cycles_svm_nodes.cpp`, `test_cycles_svm_compiler.cpp`,
`test_luisa_compact_surface_preparation.cpp`, and `test_luisa_cycles_svm.cpp`.
The limit has not been relaxed.

Strict native Vulkan passes **7/7** (`native-admission-cleanup-strict-vulkan.log`,
379.69 s), including the complete volume path, triangle/environment volume
lighting, admission, native density/stack and surface-normal tests. All runs
set `LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`. The separate uncached density loader audit emits
13,372 SPIR-V words and loads no DXC/DXIL library
(`native-admission-cleanup-vulkan-loader.log`).
