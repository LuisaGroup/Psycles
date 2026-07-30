# Luisa per-path trace bring-up

This checkpoint connects the version-1 Cycles path oracle to the same Luisa
path-tracing kernel used by fallback, HIP, and Vulkan.

Configuration:

- Cycles reference: official main `ff404d072bb4`, CPU kernel
- Luisa devices: fallback, HIP on AMD Radeon RX 9070 XT, Vulkan/RADV
- scene: `build/diagnostics/minimal-point/point_light.blend`
- exported bundle: `build/diagnostics/minimal-point/export`
- full film: 32×32
- traced pixel: `(17, 16)`, Cycles lower-left coordinates
- sample: 0 of 1
- sampler: Tabulated Sobol, scrambling distance 1

The first comparison exposed a camera-ray representation mismatch:

- Cycles advances the origin to the near clipping plane;
- Cycles then uses `tmin = 0`;
- `tmax` is `(far - near)` and is divided by the camera-space direction
  cosine for perspective rays;
- the static camera ray time is `0.5`.

Psycles previously kept the unadvanced camera origin and put `near/far`
directly in the traversal interval. This produced a 0.1-unit ray-origin and
intersection-distance difference in the probe. The Luisa camera construction
now uses the Cycles representation, with a fallback GPU regression covering
orthographic and oblique perspective clipping.

After the correction:

- 13 currently populated sampled-random fields are exact against Cycles;
- 22 currently populated discrete fields are exact;
- 32 currently populated continuous fields pass the float32 bound;
- maximum absolute error is `4.76837158203125e-7`;
- fallback versus HIP: zero failures, maximum absolute error
  `2.384185791015625e-7`;
- fallback versus Vulkan: zero failures, maximum absolute error
  `2.384185791015625e-7`.

The Cycles comparison still reports 30 gates. All are explicit unpopulated
records for direct-light sampling, raw closure inventory/selection, BSDF
sampling, post-bounce state, and Cycles shader identity. They remain failures
until implemented; the comparator does not waive them. This checkpoint is an
infrastructure and camera-contract milestone, not a claim that the material
or lighting algorithms are already aligned.

Observed fresh trace-enabled JIT times were about 0.71 seconds for HIP and
0.72 seconds for Vulkan. Vulkan's logged compute SPIR-V optimization reduced
77,253 words to 69,543 words before pipeline creation; this small probe did not
reproduce the earlier complex-scene Vulkan compile stall.
