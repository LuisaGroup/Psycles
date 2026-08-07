# Monster BSSRDF exit-normal validation

## Outcome

Psycles `main@fb69b15` now constructs the synthetic diffuse closure at a
BSSRDF exit with the same closure-weighted normal as current Cycles. The old
path reused the ordinary exit `ShaderData::N`. That was geometrically valid,
but it was not the normal Cycles obtains by re-evaluating a bump-capable
material and reducing its retained BSSRDF closures.

The repair is a material-independent closure reduction, not a Monster-shaped
patch. At film coordinate `(534, 374)`, sample zero of a 128-sample sequence,
the first semantic BSSRDF-exit event now agrees with Cycles HIP through the
closure normal, cosine-hemisphere sample, and following surface identity.
At 960x960 and 512 fixed samples, Combined relative RMSE falls from
`0.08272117` to `0.04890461` and the Combined mean-luminance ratio moves from
`1.00726634` to `1.00355265`.

This is a substantial Monster transport checkpoint, not a blanket claim that
all remaining indirect Monte Carlo differences are solved.

## Oracle and scene identity

The only rendering oracle is Blender/Cycles 5.3 Alpha at
`82186b01ad2e79435e67a02de93b178bfbe0f6c4`. LuisaCompute is
`next@3e63df0c6619641b91265d940136d7041dfc262e`.

The official source scene is
`monster_under_the_bed_sss_demo_by_metin_seven.blend`, SHA-256
`9463082f8d365ad8ae19c1ba86429deb3b532c5a2d0db0b9492c3c759f64d28d`.
The immutable raw-closure export has `scene.json` SHA-256
`d530583a73f8e0a310aa19d84e83e157ca95c93c5f56d1be789e291f6202ac12`
and `geometry.bin` SHA-256
`dcf25e4e63bb8e11ace4e639fbca0cf6eec6f862a8095b4e771a4b5e51274b15`.
No material, BSSRDF, normal, or closure result is baked by Blender/Cycles.

## Formal exit-closure relation

Current Cycles shades a bump-capable BSSRDF exit, retains the normal closure
allocation order and capacity, and reduces only BSSRDF closures. For retained
closure normal `N_i` and spectral weight `w_i`, the relation is

```text
S = sum(N_i * abs(average(w_i))) for retained BSSRDF closures i
N_exit = ShaderData::N             when S is exactly zero
N_exit = normalize(S)              otherwise
```

Cycles then discards the graph-produced closure array and creates exactly one
unit-weight Lambert closure with `N_exit`. The exact-zero predicate matters:
Psycles does not introduce an epsilon or a scene-dependent threshold.
Non-BSSRDF closures never contribute, but still participate in the shared
retained-closure allocation law.

Psycles represents this as `SurfaceBssrdfNormalVisitor`, which consumes the
original Luisa graph expressions while the material branch is recorded. The
result travels separately from `SurfacePoint::shading_normal` in
`SurfaceQuery::subsurface_normal`; the latter still owns intersection and ray
offset semantics. Sampling, direct-light evaluation, trace diagnostics, and
the synthetic closure normal all consume the same reduced normal.

The fallback/HIP/Vulkan regression builds two differently weighted BSSRDF
closures plus a strong non-BSSRDF closure, verifies the exact reduction and
zero-sum fallback, and checks the resulting unit Lambert evaluation, sample,
closure trace, and sample trace.

## Exact path result

Cycles records the exit in semantic event 2 because its separate subsurface
kernel leaves event 1 unused. Psycles performs the same local traversal in
its fused path loop and records the exit in event 1. Comparing those semantic
events gives:

| Quantity | Cycles HIP | Psycles before | Psycles after |
| --- | ---: | ---: | ---: |
| RNG offset | 48 | 48 | 48 |
| Direction random `u` | 0.6814339161 | 0.6814339161 | 0.6814339161 |
| Direction random `v` | 0.7971861959 | 0.7971861959 | 0.7971861959 |
| Exit normal `x` | 0.0699615106 | 0.0059145540 | 0.0699632838 |
| Exit normal `y` | -0.4471633434 | -0.5253249407 | -0.4471623302 |
| Exit normal `z` | 0.8917120695 | 0.8508811593 | 0.8917124271 |
| Outgoing `x` | 0.5856168866 | 0.53631353 | 0.5856180787 |
| Outgoing `y` | -0.1036300734 | -0.19453557 | -0.1036286056 |
| Outgoing `z` | 0.8039364815 | 0.82129395 | 0.8039357662 |
| Following object | 34 | different path | 34 |
| Following primitive | 952675 | different path | 952675 |

The post-fix differences are continuous HIP/HIPRT float differences. The
following hit distance is `0.69208217` in Cycles and `0.69208282` in Psycles;
there is no remaining path-topology divergence at this event.

The decoded machine evidence is retained as
[Cycles HIP](reports/exact-path-cycles-hip.json),
[Psycles before](reports/exact-path-psycles-before.json), and
[Psycles after](reports/exact-path-psycles-after.json). The generic trace
comparator intentionally does not hide the renderer-specific event-index
gap, so the table above aligns events by their explicit semantic state,
intersection, RNG offset, and closure operation.

## 960x960, 512-spp image result

Both renderers use seed zero, fixed Tabulated Sobol samples, no adaptive
sampling, and no denoising on the same Radeon RX 9070 XT. The before and after
runs use the same exported raw graph and geometry.

| Pass / metric | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| Combined relative RMSE | 0.08272117 | 0.04890461 | 1.69x lower |
| Combined RMSE | 0.01168779 | 0.00690980 | 1.69x lower |
| Combined luminance ratio | 1.00726634 | 1.00355265 | closer to one |
| Diffuse Direct relative RMSE | 0.09158052 | 0.04844536 | 1.89x lower |
| Diffuse Indirect relative RMSE | 0.20816032 | 0.14674033 | 1.42x lower |
| Glossy Direct relative RMSE | 0.04789864 | 0.04789858 | unchanged control |
| Glossy Indirect relative RMSE | 0.35305310 | 0.26694744 | 1.32x lower |
| Diffuse Color relative RMSE | 0.00054742 | 0.00054742 | unchanged control |
| Normal relative RMSE | 0.00041160 | 0.00041160 | unchanged control |

All 921,600 pixels are valid in every compared pass. The unchanged material
color, normal, and direct-glossy controls support the diagnosis: this repair
changes the post-BSSRDF scattering frame, not exported textures, UVs,
geometry, or general surface shading.

The full report is [hip-512-vs-cycles-hip.json](reports/hip-512-vs-cycles-hip.json)
and the complete command/timing manifest is
[hip-512-benchmark.json](reports/hip-512-benchmark.json).

## Visual inspection

All four triptychs were inspected at their original `2880x720` resolution.
The Combined Cycles and Psycles panels agree in silhouette, eye and tooth
placement, bed and blanket geometry, floor boards, material boundaries,
subsurface color distribution, and highlight placement. No coherent UV,
transform, texture, missing-geometry, or closure-family displacement remains.
The amplified difference is predominantly granular Monte Carlo residual,
strongest on the subsurface monster and high-energy indirect highlights.

![Combined: Cycles HIP, Psycles HIP, absolute difference](triptychs/combined.png)

Diffuse Direct is structurally aligned across the monster, child, bed, floor,
and background. Its remaining residual has sampling-noise texture rather than
a displaced lighting or normal pattern.

![Diffuse Direct: Cycles HIP, Psycles HIP, absolute difference](triptychs/diffuse-direct.png)

Diffuse Indirect shows the expected largest remaining low-frequency transport
noise on the BSSRDF object, but the reference and actual panels agree in
large-scale color and occlusion structure.

![Diffuse Indirect: Cycles HIP, Psycles HIP, absolute difference](triptychs/diffuse-indirect.png)

Glossy Indirect also remains noisy at 512 spp. Highlight families and object
boundaries coincide; the difference panel does not expose a second displaced
normal frame.

![Glossy Indirect: Cycles HIP, Psycles HIP, absolute difference](triptychs/glossy-indirect.png)

## Performance and verification

Render-only time is `26.6229 s` for Cycles HIP and `135.852 s` for Psycles
HIP, so this checkpoint is `5.103x` slower on the same GPU. Before the repair,
the matched times were `26.5232 s` and `129.579 s` (`4.886x` slower). The
additional raw-graph evaluation occurs only at BSSRDF exits, but currently
costs about 4.8% in this SSS-heavy scene.

The cold production shader also exposes a code-generation boundary that must
be tightened: LLVM code generation takes `26.454 s`, then HIP bitcode linking
takes `100.737 s` for a 6.98 MB code object. Total shader JIT is `134.160 s`,
up from `30.556 s` before the new independent material-dispatch callable.
This is recorded as an active engineering issue, not normalized away as
render time. The next repair will specialize the exit-normal dispatch to
statically BSSRDF-capable material programs so unrelated graphs are absent
from that callable without baking any material values.

Verification completed with:

```text
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure --parallel 32
```

All `218/218` tests passed. The focused exit-normal test passes on Luisa
fallback, HIP, and Vulkan.
