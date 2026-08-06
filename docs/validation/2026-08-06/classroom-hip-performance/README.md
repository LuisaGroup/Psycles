# Classroom HIP performance validation

## Result

Classroom at 640 x 480 and 256 spp now renders on Psycles HIP in **17.202 s**,
down from the corrected uninstrumented baseline of **57.832 s**. This is a
3.36x speedup and a 70.3% runtime reduction with no change to any of the 15
linear output passes.

The root cause was algorithmic. Forward-hit MIS mapped every triangle surface
intersection back to the emissive-triangle table with a linear scan over all
emitters. That made ordinary non-emissive surface shading O(E), where E is the
number of emissive triangles, at every path bounce.

The replacement is a formal lexicographic `lower_bound` over
`(instance_index, primitive_index)`, reducing the lookup to O(log E). Scene
upload validates that keys are unique and strictly increasing before the table
crosses the host/device boundary. A conservative material capability bit also
proves when a surface cannot emit and avoids the lookup entirely. Authored
emission sampling, side selection, geometric PDF, MIS weight, and random-number
consumption are unchanged.

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

The capability gate is effectively neutral in this scene (about 0.04%), which
shows that the logarithmic lookup already removed the material table as a
runtime bottleneck. It remains useful as a conservative complexity boundary
for scenes with larger emitter sets.

At 32 spp, the direct before/after comparison is 7.266 s versus 2.210 s, or
3.29x. The same scaling at 256 spp rules out fixed dispatch or scene-upload
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
| Ordered emitter lookup, production result | 2.210 s | Removes the dominant O(E) work |

A forward-path-tracing specialization retaining all ten analytic lights took
0.083 s at 1 spp, the same envelope as the earlier light-free diagnostic. That
excluded scene simplification as the explanation and narrowed the search to
code emitted only for next-event estimation. Inspection then found the linear
`from_intersection` emitter scan.

Dispatch batching was also not the primary issue: batches of 1, 4, 8, 16, and
32 samples measured 7.786, 7.437, 7.322, 7.256, and 7.266 s before the fix.

## Shader compilation and register pressure

The optimization changes runtime complexity rather than hiding work through a
different compiler policy. The final HIP code object still uses 256 VGPRs, 108
SGPRs, 5376 bytes of private storage per thread, dynamic stack, 1250 VGPR
spills, and 151 SGPR spills. Its `.text` section is 4,588,928 bytes. The
pre-fix values were 256 VGPRs, 108 SGPRs, 5376 private bytes, 1252 VGPR spills,
148 SGPR spills, and 4,588,416 bytes of `.text`.

Cold shader JIT for the final kernel took 68.18 s, while a warm cache load took
0.434 s. JIT latency is therefore a separate remaining engineering problem,
not part of the 17.202 s steady-state render time.

A global HIP register cap was rejected. A 96-register cap improved the old
Classroom kernel from 57.832 to 30.243 s with byte-identical output, but made a
controlled Lone Monk 64-spp render regress from 4.979 to 9.076 s. Register
policy must therefore be derived from kernel/scene characteristics rather than
set as a backend-wide default.

## Image and numerical validation

All 15 final 256-spp PFM passes are byte-for-byte identical to the pre-fix
render: Combined, Normal, Albedo, Diffuse Direct/Indirect, Glossy
Color/Direct/Indirect, Transmission Color/Direct/Indirect, Emission,
Environment, and Volume Direct/Indirect. `oiiotool --diff` reports `PASS` for
Combined, and the same exact comparison was performed for every pass.

Against Cycles HIP, Combined relative RMSE remains 0.037245 and mean luminance
ratio remains 1.012414. Visual inspection of the full-resolution triptych found
matching geometry, silhouettes, textures, material regions, and lighting
structure. The amplified difference is stochastic/highlight noise rather than
a coherent transform, UV, material, or lighting mismatch.

![Cycles HIP, Psycles HIP, and amplified absolute difference](triptychs/combined.png)

The machine-readable comparison is
[Psycles HIP vs Cycles HIP](reports/psycles-hip-vs-cycles-hip.json).

## Regression coverage

The ordered device lookup regression covers the first, middle, and last table
entries plus missing keys before, between, within, and after valid key ranges.
It passes on fallback, HIP, and Vulkan. Existing forward area-light tests also
pass on all three backends, and the full suite passes **207/207**.

The remaining steady-state gap is 3.88x versus Cycles HIP and 1.28x versus
Cycles CPU. Since shadow traversal, analytic forward lights, dispatch batching,
and the emitter table no longer explain it, the next profiling pass should
measure material/closure evaluation, per-bounce traversal, and the effect of
the still-extreme spill footprint on the new baseline rather than carrying
forward percentages from the removed O(E) bottleneck.
