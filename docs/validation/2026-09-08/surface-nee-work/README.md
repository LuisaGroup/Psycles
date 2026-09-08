# Surface NEE work follows the Cycles evaluation predicate

Renderer base: Psycles 266b7f6a / Luisa 9ea3b720f; e988a993 changes only
benchmark qualifications. The checks and four full-scene canaries below
are complete. No repeated performance-gain claim is made for this change.
Raw logs use the nee-work prefix under
/var/tmp/psycles-native-volume-svm-06XnDX.

## Formal cause and scoped correction

Original Cycles 5.2.1 kernel/integrator/shade_surface.h,
integrate_surface_direct_light, returns before PRNG_LIGHT and discrete
emitter selection unless use_direct_light and SD_BSDF_HAS_EVAL are both set.
Psycles previously emitted its light random tuple and flat-distribution
lookup before surface shading, and its light-tree selection before that
BSDF predicate. Megakernel scenes with volume also placed the random state
before resolving the closest event. Correct proposal values on accepted
paths do not justify executing these operations on rejected paths.

The host scene-stage plan already proves direct lighting is enabled and
has a supported emitter population. The production pipeline now records
the complete surface random state, flat/tree selection and NEE body inside
the existing BSDF-evaluation predicate. A small host-recording callback keeps
the state alive only during that consumer. It adds no device call or
inlining policy. A host assertion detects consumers without a live binding.

Volume proposals stay before distance tracking: equiangular sampling needs
them there. Their state is bound inside the nonempty-volume segment in both
megakernel and coroutine modes. A scattered volume path advances its RNG
offset and continues before surface shading; an unscattered segment and
forward lamp events do not advance that offset. Re-evaluating the same pure
Sobol function at the actual consumer preserves sample identity and
dimensions. No stream, sampler arithmetic, closure state or coroutine
transition is redesigned. The old scheduler-dependent random-placement
policy is removed.

Original Cycles diagnostics observe PRNG_LIGHT even on a path later rejected
by NEE. Psycles' trace-only block therefore evaluates the pure random tuple
independently; diagnostics do not require a flat-distribution or tree lookup.
Trace schema and values are unchanged.

## Permanent work-count regression

The independent GPU test executes the real PathBounceRandomStage with an
atomic counter at its flat-distribution callable boundary, plus a separate
counter in the guarded consumer. Other production stages are unbound, not
replaced by a CPU sampler/shader/renderer. It covers 37 paths, all four
BSDF-eval/transmission flag combinations, direct lighting on/off, and both
flat-distribution and light-tree recording modes. The tree-mode counter
checks the gated callback, not the internals of the tree algorithm; existing
original-Cycles light-tree and full-render oracles cover those values.

The old eager emit location was first preserved in the recording helper.
HIP then fails with 56 excess flat-query invocations: all 37 disabled-NEE
paths and the 19 paths without BSDF evaluation when NEE is enabled. The NEE
body counter correctly remains zero on those paths. Moving emit inside the
guard is the semantic correction; an earlier unbound-callable test setup
compilation error is not counted as the red regression. This is renderer
work placement, not evidence of a Luisa compiler defect.

## Completed validation and scene checks

| Gate | Result | Qualification |
| --- | --- | --- |
| Full build | Passed | All 32 threads |
| Focused HIP | 7/7, 265.10 s | Work counter, zero BSDF, NEE, light tree, volume path/triangle/environment |
| Full HIP | 178/178, 331.34 s | No failed or skipped selected tests |
| Full fallback | 179/180, 519.53 s | Existing film light_ng.z mismatch only |
| Full host | 153/154, 14.39 s | Existing four source-size violations only |
| Strict native Vulkan | 5/5, 197.13 s | Work counter, zero BSDF, NEE, volume path and light tree |

Vulkan uses all three native-XIR/require-SPIR-V/disable-DXC guards. The
verbose loader log records native SPIR-V compilation and no loaded
DXC/DXIL library. The selection includes actual volume-path and zero-BSDF
renderer tests, not only arithmetic probes. No numerical tolerance or
backend fallback was changed. The known film mismatch remains expected
0xbf1f8bfd versus actual 0xbf1f8a50. The GPU work test checks 148 scenarios
with two counters each; scalar/vector default-zero semantics are unchanged.

The following are one new Psycles HIP canary per scene at fixed 256 spp,
compared with unchanged original Cycles run-1 images from the preceding
campaign. They are not a repeated or newly paired performance benchmark.
The source/export/reference hashes and six executable/library hashes were
verified. No profiler, build or concurrent render overlapped these canaries.
Main shader caching is disabled and other caches retain normal policy.

| Scene / extent | Render s | Session init s | Frame | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monk / 1440x1080 | 13.6829 | 44.2676 | 220 B | 0.01240038 | 0.12881324 |
| Monster / 1080x1080 | 15.1684 | 57.7604 | 284 B | 0.00547891 | 0.02552520 |
| Classroom / 1920x1080 | 18.7494 | 43.5315 | 264 B | 0.00353327 | 0.17820333 |
| Barbershop / 2048x858 | 39.6390 | 62.7606 | 416 B | 0.01079831 | 0.07450956 |

`Session init` is the CLI's complete create_session interval, including JIT
and setup/baking. These first observations are not isolated or fully cold
compiler timings. All 46 actual channels are finite and all 15 common-pass
comparisons complete. All four Combined triptychs were visually inspected;
existing residuals remain. Original Classroom direct passes retain their
25/27 invalid pixels, explicitly excluded by the comparison's invalid union.
Exact commands, timings, frame layout, all pass metrics and artifact hashes
are in [canaries.json](canaries.json).

The original Cycles references contain extra passes, including AO in
Classroom and Barbershop. Those references remain common-pass image oracles,
but their times are not equal-work baselines. See the
[pass-workload qualification](../four-scene-hip/README.md). No Cycles speedup
ratio is inferred from these canaries. Barbershop's single 39.639 s result
is essentially unchanged from the preceding Psycles median 39.5579 s, so
this surface-work correction does not explain or solve its remaining gap.

Technical-report QA separates work-count correctness, full-suite results,
single-run timing observations and unresolved image error. In response to
the Barbershop diagnosis, the next independent audit targets homogeneous
volume rejection: its original fog material is camera-ray-dependent
Emission only, while the current transport records scattering math even
when sigma_s is zero. Original Cycles returns after emission/attenuation on
that path. That volume correction is not part of this validated change.
