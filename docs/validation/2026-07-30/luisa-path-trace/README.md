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

After the camera correction:

- 13 currently populated sampled-random fields are exact against Cycles;
- 22 currently populated discrete fields are exact;
- 32 currently populated continuous fields pass the float32 bound;
- maximum absolute error is `4.76837158203125e-7`;
- fallback versus HIP: zero failures, maximum absolute error
  `2.384185791015625e-7`;
- fallback versus Vulkan: zero failures, maximum absolute error
  `2.384185791015625e-7`.

The next differential gate exposed an important point-light representation
difference. Psycles divided point radiance by distance squared and used a
conditional PDF of one. Cycles keeps the light's `eval_fac` independent of
distance and represents the area-to-solid-angle Jacobian in the light PDF:

```text
delta point conditional PDF = distance²
normalized delta point eval_fac = 1 / (4π)
```

Those forms have the same quotient in this isolated estimator, but only the
Cycles form composes correctly with selection PDF, light identity, forward
sampling, and MIS. The implementation now shares one Luisa DSL invariant for
delta and finite oriented-disk point lights. A device regression exercises the
formula and the measure-level rule that a zero-radius point has no competing
BSDF technique even when Blender's use-MIS property is enabled. The regression
passes on fallback, HIP, and Vulkan.

The point-path trace now also records emitter/type/primitive identity,
selection and conditional PDF, `eval_fac`, sampled direction/position/normal,
distance, BSDF PDF, and MIS weight. Against Cycles CPU, every newly populated
light field passes:

- `pdf = 2.09093976020813`;
- `selection_pdf = 1`;
- `eval_fac = 0.07957746833562851`;
- `distance = 1.4460082054138184`;
- `bsdf_pdf = 0`, `mis_weight = 1`;
- continuous direction differences remain below the existing float32 bound.

The comparison has therefore moved from 30 to 24 failing gates without
weakening the comparator. The remaining gates are raw closure
inventory/selection, BSDF sampling, post-bounce state, source object/light
group identity, and Cycles shader identity. Internal source identities are not
guessed from array lengths: they remain explicitly unwritten until the Blender
export preserves the corresponding Cycles sync identity.

The estimator image is unchanged by the point-light refactor (`oiiotool
--diff` passes), confirming that this checkpoint changes the sampling
representation rather than the intended energy.

Observed fresh trace-enabled JIT times were about 0.71 seconds for HIP and
0.72 seconds for Vulkan. Vulkan's logged compute SPIR-V optimization reduced
77,253 words to 69,543 words before pipeline creation; this small probe did not
reproduce the earlier complex-scene Vulkan compile stall.
