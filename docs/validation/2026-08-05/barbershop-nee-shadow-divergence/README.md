# Barbershop NEE shadow divergence

This checkpoint isolates the lower-left floor sample at pixel `(550, 99)`,
absolute sample `0`, in the `1152x480` Barbershop material-isolation render.
The oracle is current Cycles CPU at Blender commit `29ccd5e2`; the candidate is
Psycles HIP with LuisaCompute `next@73bfe5e9e0fe`.

The diagnostic trace records direct lighting at seven semantic boundaries:
raw BSDF sum, diffuse and glossy partitions, BSDF after light PDF/MIS weighting,
light-shader evaluation, path-throughput-weighted unshadowed radiance, and the
final shadowed contribution. Light roulette runs before shadow traversal in
both implementations, matching Cycles' phase order.

| phase | Cycles CPU RGB | Psycles HIP RGB |
|---|---:|---:|
| raw BSDF | `(0.02202219, 0.01584192, 0.01320305)` | `(0.02204796, 0.01586002, 0.01321790)` |
| weighted BSDF | `(0.00202379, 0.00145584, 0.00121333)` | `(0.00202615, 0.00145750, 0.00121469)` |
| light shader | `(117.80952, 81.38113, 34.75935)` | `(117.80953, 81.38114, 34.75935)` |
| unshadowed | `(0.23842150, 0.11847758, 0.04217454)` | `(0.23870036, 0.11861289, 0.04222197)` |
| final contribution | `(0.23842150, 0.11847758, 0.04217454)` | *not written* |

The first categorical divergence was therefore shadow visibility, not texture
color, light sampling, or a missing material closure. Cycles kept the
unshadowed contribution, while Psycles terminated it after its scene query.
The remaining small pre-shadow residual is tracked separately and must not be
hidden by changing true displacement into bump mapping.

## Exact shadow contract

Schema version 2 appends 32 records without changing any version-1 index. For
each of four path events it records the exact shadow ray, closed interval,
source and light identities, first eligible Psycles hit, and accumulated
transmittance. The Cycles wavefront kernel does not expose a symmetric
standalone hit/transmittance value, so those two fields are diagnostic-only;
its ray and final contribution remain the visibility oracle.

The two implementations cast the same first-bounce light ray to float32
precision:

| field | Cycles CPU | Psycles HIP before fix |
|---|---:|---:|
| origin | `(1.67305946, 4.73234129, -0.0050326395)` | `(1.67305851, 4.73234034, -0.0050326362)` |
| direction | `(0.51509523, 0.66049135, 0.54628569)` | `(0.51509535, 0.66049129, 0.54628563)` |
| interval | `[0, 3.60355449]` | `[0, 3.60355544]` |
| source | object `66`, primitive `871527` | object `66`, primitive `871527` |

Before the correction, HIP reported object `11`, primitive `173299`, at
distance `8.3885615e-10`, with barycentrics
`(0.08999910, 0.88957107)`. That object and the source object are two
near-overlapping instances of the same local floor triangle `15417`; their
transforms differ only by rotations on the order of `1e-7`. The false hit
reduced transmittance to zero.

## Formal correction

Hardware triangle traversal is now only a broad phase. Every triangle
candidate on every backend is resolved with the same Cycles/Embree Plücker
predicate, world-to-object transform contract, closed ray interval,
barycentrics, and stable Cycles identity order. No scene-specific epsilon or
Barbershop object exception was introduced.

The regression reproduces the full causal chain: it casts the captured camera
ray through both near-overlapping instances, reconstructs the surface from the
resolved barycentrics, then casts the captured light ray while excluding the
resolved source identity. It locks object `5066`, primitive `700000`, primary
distance `4.4699316`, and a shadow miss. The test passes on fallback, HIP, and
Vulkan.

After the correction, Psycles records no shadow hit, unit transmittance, and
the final contribution is present:

| phase | Cycles CPU RGB | Psycles HIP after fix RGB |
|---|---:|---:|
| unshadowed | `(0.23842150, 0.11847758, 0.04217454)` | `(0.23872066, 0.11862265, 0.04222541)` |
| final contribution | `(0.23842150, 0.11847758, 0.04217454)` | `(0.23872066, 0.11862265, 0.04222541)` |

This removes the categorical visibility divergence; it does not declare the
remaining normal/closure residual aligned. True displacement remains geometry
displacement, bump remains a differential normal perturbation, and normal
mapping remains its own tangent/object/world-space operation. None may be
downgraded to another to make an image comparison pass.

The trace schema, decoder, comparison policies, Cycles diagnostic patch, and
fallback/HIP/Vulkan traversal and point-light regressions are versioned in the
repository. The relevant nine tests pass: schema, decoder, comparison,
traversal on three backends, and point-light transport on three backends.
The raw Barbershop EXR/JSON traces remain under
`/var/tmp/barbershop-three-material-original-20260805/` because they are large
diagnostic artifacts rather than source fixtures.
