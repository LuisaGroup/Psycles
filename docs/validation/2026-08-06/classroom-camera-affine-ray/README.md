# Classroom camera affine-ray validation

## Result

The camera-to-world transform now uses the same explicit affine FMA tree as
Cycles geometry transforms. This restores structural path agreement at thin
and coincident geometry boundaries without changing ray epsilon or adding a
primitive-specific exception.

At pixel `(566, 389)`, sample `119`, both Cycles CPU and HIP trace the surface
sequence `464957 -> 475316 -> 481199` (emission). Before this change Psycles
traced `464957 -> 475321 -> 475316` and lost the glossy-indirect contribution.
After the change Psycles traces `464957 -> 475316 -> 481199`; its sample-level
Glossy Indirect value is `(26.615324, 26.609171, 32.561451)`.

The invariant is formal: every world-space ray origin and direction that can
participate in a geometric predicate must be constructed with the Cycles
affine contraction order. Backend-native matrix lowering is not equivalent at
primitive boundaries because it may round a product before adding the
translation. The regression test uses the real boundary input and checks the
expected `origin.x` bit pattern, `0x3f24212b`, on fallback, HIP, and Vulkan.

## Full-image comparison

Classroom was rendered at 640 x 480, 256 spp. The candidate is Psycles HIP;
the references are Cycles CPU and Cycles HIP using the same scene and seed.

| Pass | rel. RMSE vs Cycles CPU | mean luminance ratio | rel. RMSE vs Cycles HIP |
| --- | ---: | ---: | ---: |
| Combined | 0.037190 | 1.012389 | 0.037245 |
| Diffuse Color | 0.003880 | 0.999235 | 0.003880 |
| Glossy Color | 0.000382 | 0.999988 | 0.000398 |
| Diffuse Direct | 0.405453 | 0.998261 | 0.405447 |
| Diffuse Indirect | 0.250487 | 1.055182 | 0.252157 |
| Glossy Direct | 0.075579 | 0.999739 | 0.077358 |
| Glossy Indirect | 0.360722 | 1.076381 | 0.372426 |

The direct/indirect pass RMSE remains dominated by finite-sample path
assignment and high-energy outliers. Combined appearance, geometry,
silhouettes, UV textures, and broad lighting support align visually. Inspection
found no transform, material, or texture-coordinate scrambling. Remaining
visible differences are noise, highlights, contact regions, and window-light
variance rather than a structured scene difference.

Representative triptychs (reference, Psycles, amplified absolute difference):

- [Combined vs Cycles CPU](triptychs/cpu/combined.png)
- [Glossy Indirect vs Cycles CPU](triptychs/cpu/glossind.png)
- [Combined vs Cycles HIP](triptychs/hip/combined.png)
- [Before vs after, Combined](triptychs/before/combined.png)
- [Before vs after, Glossy Indirect](triptychs/before/glossind.png)

Machine-readable reports contain all 15 requested passes:

- [Psycles HIP vs Cycles CPU](reports/psycles-hip-vs-cycles-cpu.json)
- [Psycles HIP vs Cycles HIP](reports/psycles-hip-vs-cycles-hip.json)
- [Psycles before vs after](reports/before-vs-after.json)

## Runtime observation

With a warm shader cache, the corrected 256-spp Psycles HIP render took
76.281 s. Cycles HIP took 4.431 s and Cycles CPU took 13.487 s on the same
machine. The steady-state gaps are therefore 17.2x and 5.66x respectively.
The previous 59 s Psycles timing was artificially low because the divergent
paths terminated early, so it is not a valid performance baseline.

Cold Psycles shader compilation is separate from that runtime gap: LLVM
code generation took about 23.7 s, bitcode linking about 44.0 s, and total JIT
74.7 s. A warm cache load took 0.492 s. Scene compilation/upload took about
2.09 s. Further profiling should therefore focus first on steady-state kernel
execution rather than JIT latency.

## Regression coverage

- `luisa_camera_sampling_fallback`
- `luisa_camera_sampling_hip`
- `luisa_camera_sampling_vk`
- full test suite: 204/204 passed
