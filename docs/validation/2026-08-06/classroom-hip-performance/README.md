# Classroom HIP performance validation

## Result

Classroom at 640 x 480 and 256 spp now renders on Psycles HIP in a three-run
median of **17.935 s**, down from the corrected uninstrumented baseline of
**57.832 s**. This is a 3.23x speedup and a 69.0% runtime reduction with no
change to any of the 15 linear output passes.

The root cause was algorithmic. Forward-hit MIS mapped every triangle surface
intersection back to the emissive-triangle table with a linear scan over all
emitters. That made ordinary non-emissive surface shading O(E), where E is the
number of emissive triangles, at every path bounce.

The first correction was a formal lexicographic `lower_bound` over
`(instance_index, primitive_index)`, reducing the reverse lookup to O(log E).
The production implementation now removes reverse lookup altogether. Primitive
material resolution carries the exact effective material's authored emission
sampling policy forward with the committed hit, and forward-hit MIS consumes
that value directly. The component interface accepts no scene, emitter table,
instance identity, or primitive identity, making O(E) or O(log E) reverse
mapping structurally impossible. Authored emission sampling, side selection,
geometric PDF, MIS weight, and random-number consumption are unchanged.

Potential endpoint emission and sampled-light membership are deliberately
separate predicates. `EmissionSampling::NONE` remains visibly emissive but has
no competing light PDF, while camera-only instances are excluded from the
sampled-light population. The five-state sampling policy is explicitly packed
into bits 4-6 of the existing device material flag word. Compile-time size,
alignment, offset, enum-capacity, and mask-disjointness checks keep
`MaterialBindingGpu` at its original 24-byte stride.

## Runtime comparison

All timings use the same RX 9070 XT machine, fixed 640 x 480 extent, 256 spp,
Tabulated Sobol sampling, seed 1, disabled adaptive sampling and denoising, and
a warm shader cache. The Cycles reference is Blender 5.3.0 Alpha.

| Renderer/device | Time | Relative to Cycles HIP |
| --- | ---: | ---: |
| Cycles HIP, RX 9070 XT | 4.431 s | 1.00x |
| Cycles CPU, Ryzen 9 9950X3D | 13.487 s | 3.04x |
| Psycles HIP, before | 57.832 s | 13.05x |
| Psycles HIP, ordered lookup | 17.209 s | 3.88x |
| Psycles HIP, ordered lookup plus capability gate | 17.202 s | 3.88x |
| Psycles HIP, direct O(1) PDF plus packed policy | 17.935 s | 4.05x |

The final row is the median of 17.9324, 17.9346, and 18.0173 s. It is 4.3%
slower than the earlier single-run O(log E) measurement, which was collected in
a separate benchmark session. This result does not support a performance claim
for the final O(1) cleanup; its benefit is the stronger complexity and
interface contract. The small timing delta remains part of the subsequent
HIPRT and register-pressure investigation.

At 32 spp, the direct before/final comparison is 7.266 s versus 2.295 s, or
3.17x. The same scaling at 256 spp rules out fixed dispatch or scene-upload
overhead as the explanation.

## How the bottleneck was isolated

The following deliberately non-equivalent diagnostic kernels were used only to
bound costs; none is part of the production change. These are pre-fix 32-spp
measurements, so each still contains the linear emitter lookup unless stated
otherwise.

| Diagnostic | Time | Observation |
| --- | ---: | --- |
| Production baseline | 7.266 s | Reference |
| Skip tie re-traversal | 6.886 s | Exact Cycles tie resolution is about 5.2% here |
| Assume unoccluded shadows | 6.721 s | All shadow work is at most about 7.5% |
| Cheap Lambert NEE evaluation plus unoccluded shadows | 6.342 s | NEE BSDF evaluation is not the dominant gap |
| Disable surface NEE | 6.320 s | Direct-light sampling is not the dominant gap |
| Disable analytic forward-light intersection | 7.222 s | Analytic forward hits are negligible |
| Ordered emitter lookup, intermediate result | 2.210 s | Removes the dominant O(E) work |
| Direct forward PDF, final result | 2.295 s | Removes reverse lookup from the interface |

A forward-path-tracing specialization retaining all ten analytic lights took
0.083 s at 1 spp, the same envelope as the earlier light-free diagnostic. That
excluded scene simplification as the explanation and narrowed the search to
code emitted only for next-event estimation. Inspection then found the linear
`from_intersection` emitter scan.

Dispatch batching was also not the primary issue: batches of 1, 4, 8, 16, and
32 samples measured 7.786, 7.437, 7.322, 7.256, and 7.266 s before the fix.

## Shader compilation and register pressure

The optimization changes runtime complexity rather than hiding work through a
different compiler policy. The ordered-lookup/gated HIP code object used 256
VGPRs, 108 SGPRs, 5376 bytes of private storage per thread, dynamic stack, 1250
VGPR spills, and 151 SGPR spills. Its `.text` section was 4,588,928 bytes. The
pre-fix values were 256 VGPRs, 108 SGPRs, 5376 private bytes, 1252 VGPR spills,
148 SGPR spills, and 4,588,416 bytes of `.text`.

Cold shader JIT for the direct-PDF kernel took 69.471 s, while a warm cache load
took 0.436 s. The raw AMDGPU object is 3,261,588 bytes and its cache entry is
4,714,280 bytes. JIT latency is therefore a separate remaining engineering
problem, not part of the 17.935 s steady-state render time.

A global HIP register cap was rejected. A 96-register cap improved the old
Classroom kernel from 57.832 to 30.243 s with byte-identical output, but made a
controlled Lone Monk 64-spp render regress from 4.979 to 9.076 s. Register
policy must therefore be derived from kernel/scene characteristics rather than
set as a backend-wide default.

## Image and numerical validation

All 15 final direct-PDF 256-spp PFM passes are byte-for-byte identical to the
ordered-lookup render: Combined, Normal, Albedo, Diffuse Direct/Indirect, Glossy
Color/Direct/Indirect, Transmission Color/Direct/Indirect, Emission,
Environment, and Volume Direct/Indirect. `oiiotool --diff` reports `PASS` for
Combined, and the same exact comparison was performed for every pass.

Against Cycles HIP, Combined relative RMSE remains 0.037245 and mean luminance
ratio remains 1.012414. Visual inspection of both the final 256-spp image and
the full-resolution triptych found matching geometry, silhouettes, textures,
material regions, and lighting structure. The amplified difference is
stochastic/highlight noise rather than a coherent transform, UV, material, or
lighting mismatch.

![Cycles HIP, Psycles HIP, and amplified absolute difference](triptychs/combined.png)

The machine-readable comparison is
[Psycles HIP vs Cycles HIP](reports/psycles-hip-vs-cycles-hip.json).

## Regression coverage

The direct forward-PDF regression captures no scene or emitter buffer and
covers front, back, automatic, front-and-back, disabled sampling, density
scaling, flipped sides, and degenerate triangles. Primitive-material regression
coverage exercises base and override materials, endpoint emission versus light
sampling, camera-only visibility, the packed policy bits, and decoded values.
Both tests pass on fallback, HIP, and Vulkan. Existing forward area-light tests
also pass on all three backends, and the full suite passes **207/207**.

The remaining measured steady-state gap is 4.05x versus Cycles HIP and 1.33x
versus Cycles CPU. Since shadow traversal, analytic forward lights, dispatch
batching, and the emitter table no longer explain it, the next profiling pass
should measure material/closure evaluation, per-bounce traversal, and the
effect of the still-extreme spill footprint on the new baseline rather than
carrying forward percentages from the removed O(E) bottleneck.
