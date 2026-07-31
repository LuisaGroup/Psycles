# Volume-majorant tracking RNG handoff

This checkpoint makes the Cycles heterogeneous tracking RNG state an explicit
input to volume-majorant traversal. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` is the only renderer oracle.
Psycles continues to evaluate the original Volume closure with Luisa DSL; no
Psycles CPU reference renderer, material pre-bake, or Cycles preprocessing is
used.

## Formal state contract

Cycles evaluates Light Path-dependent heterogeneous extrema with
`PRNG_VOLUME_SHADE_OFFSET` from an explicit `RNGState`. The exact ordering has
two domains:

1. initial `volume_octree_setup` runs before tracking scramble and therefore
   uses the enclosing path RNG offset;
2. Cycles samples the reservoir from that same enclosing state, then scrambles
   a copied offset with seed `0xe35fad82`;
3. every later octree-boundary advance uses the current copied tracking
   offset, and all roots reconstructed by that one ordered-overlap operation
   share its shade sample;
4. after finding a candidate, Cycles advances the copied offset by
   `PRNG_BOUNCE_NUM`, currently 16; and
5. subsequent boundary advances use the newly addressed shade sample.

The previous production provider captured one value when it was constructed.
That representation could express neither the post-scramble transition nor
step 5 and would silently reuse the initial extrema samples after a boundary
or collision. The provider is now stateless with respect to tracking RNG.
`VolumeMajorantOverlapTraversal` accepts the pre-scramble initial shade offset
and requires an explicit shade offset for every `advance`. Its internal setup
passes that one operation's value through the ordered overlap reduction, so
provider call count and root ordering cannot alter the random dimension.

Homogeneous roots retain Cycles' deterministic midpoint (`0.5`) policy.
Heterogeneous roots evaluate four points using the supplied offset and retain
the existing 1.5 safety expansion. This change does not add a new random
sequence; it exposes the state boundary required for the production
null-collision integrator.

## Regression

The overlap regression records the actual offset observed by the entry
provider and pins an explicit constructor/advance sequence to `0.25`, `0.75`,
and `0.125`. The production graph regression evaluates the same raw
Light Path-dependent Volume graph twice:

| shade offset | minimum | maximum |
| ---: | ---: | ---: |
| `0.25` | `1.54166667` | `3.0625` |
| `0.75` | `1.625` | `3.1875` |

Both regressions pass on fallback, HIP, and Vulkan:

| regression | fallback | HIP | Vulkan |
| --- | ---: | ---: | ---: |
| overlap traversal | `0.26 s` | `0.49 s` | `1.16 s` |
| production provider | `0.15 s` | `0.28 s` | `0.18 s` |

The complete incremental build used `--parallel 32`. Serial CTest passed
`105/105` tests in `9.04 s`, including the source-size gate and all three
volume backends.

This is an internal random-state regression and does not introduce a distinct
pixel estimator, so no new image is presented as evidence. The immediately
preceding production-scene checkpoint contains the generated EXRs and
visually inspected Cycles/fallback, Cycles/HIP, and Cycles/Vulkan triptychs.

## Reproduction

```sh
cmake --build build \
  --target psycles_luisa_volume_majorant_overlap_tests \
           psycles_luisa_volume_majorant_scene_tests \
  --parallel 32
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_volume_majorant_(overlap|scene)_(fallback|hip|vk)$' \
  --parallel 3
```
