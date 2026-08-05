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

The first categorical divergence is therefore shadow visibility, not texture
color, light sampling, or a missing material closure. Cycles has unit shadow
transmittance for this sample, while Psycles terminates the contribution after
its scene query. The remaining small pre-shadow residual is tracked separately
and must not be hidden by changing true displacement into bump mapping.

True displacement remains geometry displacement. This checkpoint neither
downgrades it to bump nor changes the estimator to compensate for the failed
visibility query. The next diagnostic records the exact shadow ray and the
first eligible hit identity so the correction can be derived from Cycles'
closed ray interval and self/light primitive exclusion contract.

The trace schema, decoder, comparison policies, Cycles diagnostic patch, and
fallback/HIP/Vulkan point-light regressions are versioned in the repository.
The raw Barbershop EXR/JSON traces remain under
`/var/tmp/barbershop-three-material-original-20260805/` because they are large
diagnostic artifacts rather than source fixtures.
