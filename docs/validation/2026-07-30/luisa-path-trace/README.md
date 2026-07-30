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

## Cosine-hemisphere mapping checkpoint

The next formal comparison separated a measure-level match from a
sample-by-sample match. Psycles' diffuse closure previously used
`sqrt(u), 2πv` polar disk sampling and selected an arbitrary orthonormal frame.
That estimator has the correct cosine-weighted PDF, but the same two Cycles
random dimensions produce a different world-space ray. This breaks
correlated-path comparison and changes which microgeometry is visited on later
bounces.

Psycles now uses one shared Luisa DSL mapping for camera and closure sampling:

- the Shirley-Chiu concentric square-to-disk map, including Cycles' strict
  branch at equal squared coordinates;
- the fixed Cycles algebraic tangent/bitangent orientation, including its
  equal-normal-component branch;
- `sqrt(max(1 - |disk|², 0))` and no post-construction renormalization;
- `pdf = cos(theta) / π`.

The actual first-bounce diffuse oracle record is locked by the device test:

```text
N      = (0, 0, 1)
random = (0.82080835, 0.67639267)
wo     = (0.601930976, -0.222150967, 0.767025411)
pdf    = 0.244151756
```

The new regression passes independently on fallback, HIP, and Vulkan, and all
31 project tests pass with 32-way CTest scheduling. Fresh trace-enabled
renders agree between fallback/HIP and fallback/Vulkan with zero failures and
maximum absolute error `2.384185791015625e-7`. The Cycles CPU comparison still
shows the same 24 deliberately unwritten closure/BSDF/post-state gates; none
were waived. The minimal point-light EXR is unchanged under `oiiotool --diff`
because its current visible contribution is direct lighting, while this
checkpoint changes the subsequently sampled diffuse path.

## Raw closure and BSDF checkpoint

The differential surface interface now exposes two trace-only operations:

- a runtime-indexed view of the post-shader closure array, including count,
  Cycles type, sample weight, spectral weight, and normal;
- the selected closure alongside the real `SurfaceSample`, including the
  rescaled selection dimension.

The normal render kernel continues to call the compact production sampler.
The trace kernel calls the extended sampler, so closure enumeration and
selection have one implementation rather than a second diagnostic model.
Principled remains visibly represented as an aggregate virtual closure at this
checkpoint; it is not relabeled as one of Cycles' physical closures and is not
considered aligned until physical expansion is implemented.

On the diffuse point-path oracle, the following groups now pass in full:

- raw closure `count=1`, `index=0`, `type=2`,
  `sample_weight≈0.42`, weight `(0.62, 0.41, 0.23)`, and `N=(0,0,1)`;
- closure pick and exact rescaled random
  `(0.82080835, 0.67639267, 0.37341177)`;
- BSDF `pdf=unguided_pdf=0.244151756`, label `6`;
- `wo=(0.601930976, -0.222150967, 0.767025411)`;
- weighted evaluation
  `(0.151374087, 0.100102216, 0.056154907)`;
- sampled roughness `(1,1)` and eta `1`.

The strict Cycles CPU comparison drops from 24 to 12 failures without changing
the schema, tolerances, or written-state rules. Remaining failures are source
shader/light identities, post-bounce state, surface runtime flags, visibility,
and light shader identity. Fallback versus HIP and fallback versus Vulkan both
pass with zero failures. The dedicated GraphSurface device regression passes
on all three Luisa backends, and the complete suite passes 32/32 with
`ctest -j32`.

Fresh trace JIT timing on this probe was about `0.93 s` on HIP and `2.27 s` on
Vulkan. Vulkan's compute SPIR-V optimization reduced 83,078 words to 74,829;
the optimization interval accounted for about `1.29 s` of the trace compile.
The corresponding non-trace Vulkan kernel compiled in `0.51 s`, with its
SPIR-V optimization taking about `0.39 s` (75,552 to 67,995 words). This
localizes the new cost to deliberately expanded trace instrumentation rather
than the production sampler. HIP's fresh non-trace kernel compiled in
`0.71 s`. Non-trace HIP and Vulkan EXRs remain identical to the preceding
checkpoint under `oiiotool --diff`.

## Formal post-bounce state checkpoint

Post-bounce state is advanced by one Luisa DSL state-transition function whose
inputs and outputs follow the Cycles path-state ABI. It owns bounce counters,
absolute RNG offset, path flags, path visibility, lobe-specific limits, and
transparent-path preservation as one invariant. The renderer maps the
resulting Cycles visibility to the scene contract instead of maintaining a
second, independently updated state machine.

The dedicated device regression covers diffuse, transparent, and singular
glossy transitions on fallback, HIP, and Vulkan. It also locks the Cycles
closure allocation cutoff: a diffuse weight below `1e-5` is not allocated and
one at the boundary is allocated.

The point-path comparison now passes all of:

- post depth `(bounce=1, transparent=0, rng_offset=32)`;
- throughput `(0.62, 0.41, 0.23)`;
- secondary ray position and normalized direction;
- exact path flag `266513` and visibility `4`;
- MIS PDF, minimum PDF, and continuation probability;
- exact surface runtime flag `12`.

The strict Cycles CPU failure count drops from 12 to 4. Those four gates are
the source surface shader ID halves, light object/group identity, and light
shader identity; they remain unwritten rather than inferred. Cross-backend
fallback/HIP and fallback/Vulkan traces pass with zero failures and maximum
absolute difference `2.384185791015625e-7`. The full suite passes 33/33 with
32-way scheduling. The fallback, HIP, and Vulkan multilayer EXRs are unchanged
from the preceding closure checkpoint under `oiiotool --diff`.
