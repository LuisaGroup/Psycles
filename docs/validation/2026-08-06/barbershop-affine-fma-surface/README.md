# Barbershop affine-FMA surface validation

This checkpoint fixes a second CPU/HIP-consensus Barbershop shadow failure
without requiring Cycles CPU and HIP to choose the same member of coincident
geometry. The validation scene is the current 1152 x 480 isolated Barbershop
bundle with 126 triangle geometries, 152 instances, 564 compiled materials,
15 lights, and 128 Tabulated Sobol samples.

## Oracle policy

Cycles CPU and HIP disagree at film pixel `(634, 209)`, sample 6 about which
of two coincident cupboard objects is the primary hit:

| Device | Primary object | Primary primitive | Next object | Next distance |
| --- | ---: | ---: | ---: | ---: |
| Cycles CPU | 112 | 1137366 | 115 | `4.3548123e-8` |
| Cycles HIP | 115 | 1147418 | 112 | `4.3426489e-8` |

That identity choice is accelerator-dependent and is not an alignment target.
Both devices agree on the transport semantics: the other coincident cupboard
surface blocks the direct-light ray, the NEE contribution is absent, and the
128-spp Combined, DiffDir, and GlossDir values at this pixel are all zero.
The fix therefore targets that shared invariant rather than either device's
tie-breaking result.

## First semantic divergence

Psycles selected the same primary object and primitive as Cycles HIP, but its
ordinary instance surface reconstruction produced a different x coordinate:

| Renderer | Surface x | IEEE-754 bits |
| --- | ---: | ---: |
| Cycles CPU | `0.6411311030387878` | `0x3f24212b` |
| Cycles HIP | `0.6411311030387878` | `0x3f24212b` |
| Psycles before | `0.6411311626434326` | `0x3f24212c` |
| Psycles after | `0.6411311030387878` | `0x3f24212b` |

The source coordinate is transformed by `scale * x + translation`. A generic
Luisa matrix multiplication rounded the product before adding the translation;
Cycles contracts the affine expression and rounds once. The one-ULP change put
the old Psycles shadow origin on the outside of both coincident surfaces. This
was not a tolerance problem, so adding a ray epsilon would have hidden the
cause and broken the closed-endpoint geometry relation elsewhere.

## Formal correction

`cycles_transform::point` and `cycles_transform::direction` now define one
shared, explicit nested-FMA operation tree in a real header/source module. The
same primitive is consumed by:

- ordinary triangle vertex and hit-position reconstruction;
- object/world normal transformation;
- the Cycles Pluecker triangle predicate; and
- shadow and continuation self-exclusion coordinate transforms.

This makes the surface representation and the predicate that tests it use the
same floating-point algebra on fallback, HIP, and Vulkan. It does not classify
nearby geometry with an epsilon and does not alter exported geometry, UVs,
materials, or closures.

After the correction, the sample-6 shadow ray hits object 112, primitive
1137366 at `4.3559677e-8`; shadow transmittance changes from one to zero. The
single-sample Combined value changes from
`[3.1857278, 1.7369525, 1.6327000]` to exactly zero. Continuation traversal
also reaches object 112 at `4.2520256e-8`, consistent with both Cycles devices'
coincident-surface behavior.

## Full-image measurements

The final Psycles image is
`isolated-affine-fma-128-psycles-hip.exr`, rendered on the RX 9070 XT at
1152 x 480 and 128 spp. Values below are linear full-frame RMSE; percentages
compare with the exact-source-shadow checkpoint immediately before this fix.

| Pass | vs Cycles CPU before | vs Cycles CPU after | Change | vs Cycles HIP before | vs Cycles HIP after | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined | 0.01669068 | 0.01563279 | -6.34% | 0.01694717 | 0.01590656 | -6.14% |
| DiffDir | 0.19474643 | 0.19003102 | -2.42% | 0.19616793 | 0.19150960 | -2.37% |
| GlossDir | 0.16111262 | 0.15728307 | -2.38% | 0.18026957 | 0.17685422 | -1.89% |
| DiffInd | 0.00612550 | 0.00609469 | -0.50% | 0.00611083 | 0.00608005 | -0.50% |
| GlossInd | 0.00863579 | 0.00858660 | -0.57% | 0.00752287 | 0.00746838 | -0.72% |
| DiffCol | 0.00051512 | 0.00051540 | +0.05% | 0.00277073 | 0.00276964 | -0.04% |
| GlossCol | 0.00009480 | 0.00009515 | +0.36% | 0.00051649 | 0.00051635 | -0.03% |
| Normal | 0.00160001 | 0.00160046 | +0.03% | 0.00263849 | 0.00263831 | -0.01% |

DiffCol, GlossCol, and Normal remain effectively stable while direct and
combined lighting improve on both oracles. This isolates the result to path
geometry/visibility instead of a material or texture change.

The final CPU-relative Combined luminance ratio is 1.24996 and the DiffDir
ratio is 1.25366. Those residuals are still too large and are not claimed as
alignment. The next investigation should continue ranking spatially coherent
pixels where Cycles CPU and HIP agree, with particular attention to the
remaining floor direct-light excess and the cupboard normal/closure chain.

## Visual inspection

The three panels in each image are reference, Psycles, and absolute
difference with one shared linear-to-sRGB display scale.

- [Combined versus Cycles CPU](triptychs/combined-cycles-cpu.png)
- [Combined versus Cycles HIP](triptychs/combined-cycles-hip.png)
- [Combined before versus after](triptychs/combined-before-after.png)
- [Diffuse direct versus Cycles CPU](triptychs/diffdir-cycles-cpu.png)
- [Diffuse color versus Cycles CPU](triptychs/diffcol-cycles-cpu.png)
- [Normal versus Cycles CPU](triptychs/normal-cycles-cpu.png)

Manual inspection confirms that the cupboard's dark internal separators and
several floor-panel gaps reappear in the same locations as Cycles. The
before/after difference is concentrated in direct illumination, while floor
board direction, cupboard texture placement, DiffCol, and Normal stay fixed.
The remaining Psycles floor and cupboard illumination is visibly brighter
than both references, consistent with the numeric residual above.

## Timing and regression matrix

The measured HIP run reported 3.178 s for scene/HIPRT construction, 141.467 s
for the cold changed-kernel JIT, and 2.174 s for the 1152 x 480 x 128 render.
This is a diagnostic component scene, not the canonical five-way benchmark,
so no Cycles speedup claim is made here.

The regression embeds the real cupboard transform, triangle vertices,
barycentrics, and expected Cycles x bits. It fails on the old two-rounding
matrix lowering and passes on all three Luisa backends:

- `psycles.luisa_transform_applied_surface_fallback`
- `psycles.luisa_transform_applied_surface_hip`
- `psycles.luisa_transform_applied_surface_vk`

Scene traversal and source-size checks pass on fallback, HIP, and Vulkan. The
complete build used `--parallel 32`; all 195 tests passed in 172.52 s. No CPU
reference renderer or material baking is involved: the only transport oracles
are the instrumented current Cycles CPU and HIP kernels.

Machine-readable reports are stored in [reports](reports/).
