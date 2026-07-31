# Volume-majorant tracking RNG handoff

This checkpoint makes the Cycles heterogeneous tracking RNG state an explicit
input to volume-majorant traversal. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` is the only renderer oracle.
Psycles continues to evaluate the original Volume closure with Luisa DSL; no
Psycles CPU reference renderer, material pre-bake, or Cycles preprocessing is
used.

## Formal state contract

Cycles evaluates Light Path-dependent heterogeneous extrema with
`PRNG_VOLUME_SHADE_OFFSET` from the current local tracking `RNGState`. The
state transition is attached to a candidate, not to an octree leaf:

1. sample one shade offset from the current tracking RNG offset;
2. use that same value for every root and leaf visited while searching for the
   candidate, including overlap reconstruction and residual free-flight
   continuation;
3. after finding the candidate, advance the local RNG offset by
   `PRNG_BOUNCE_NUM`, currently 16; and
4. sample and pass the new shade offset to the next traversal operation.

The previous production provider captured one value when it was constructed.
That representation could not express step 4 and would silently reuse the
first candidate's extrema samples after a collision. The provider is now
stateless with respect to tracking RNG. `VolumeMajorantOverlapTraversal`
accepts the initial shade offset and requires an explicit shade offset for
every `advance`. Its internal setup passes that single value through the
ordered overlap reduction, so provider call count and root ordering cannot
alter the random dimension.

Homogeneous roots retain Cycles' deterministic midpoint (`0.5`) policy.
Heterogeneous roots evaluate four points using the supplied offset and retain
the existing 1.5 safety expansion. This change does not add a new random
sequence; it exposes the state boundary required for the production
null-collision integrator.

## Regression

The overlap regression records the actual offset observed by the entry
provider and pins the constructor/advance sequence to `0.25`, `0.75`, and
`0.125`. The production graph regression evaluates the same raw
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
