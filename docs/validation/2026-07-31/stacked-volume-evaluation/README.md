# Stacked volume evaluation

This checkpoint validates coefficient and raw phase aggregation across the
runtime volume stack. It is a Luisa shader-state regression, not yet a volume
render or visual-parity claim.

## Reference and contract

- Official Blender/Cycles reference:
  `b82c3f0da6c1813dabedc563d64e536f4d83e868`.
- Reference code:
  `intern/cycles/kernel/integrator/volume_shader.h`,
  `intern/cycles/kernel/integrator/shade_volume.h`, and
  `intern/cycles/kernel/closure/alloc.h`.
- Each stack entry evaluates its original volume graph at the same world
  sample with its own object coordinates, parameter block, and density scale.
- Extinction, scattering, and emission coefficients add across entries.
- One raw phase collector spans the full stack.
- Cycles does not call `volume_shader_merge_closures()` after entry zero. It
  calls it after every later entry, then copies the first eight phase closures
  in stable order.
- Extinction-only evaluation suppresses emission without changing the source
  graph or coefficient path.

`StackedVolumeEvaluator` owns only these aggregation rules.
`VolumeStackEntryPointProvider` is a host-stage virtual boundary for exact
world/object/grid coordinate construction. Its virtual dispatch runs while
Luisa records the AST; no virtual call or pre-baked medium appears in device
code.

## Regression

The fixture compiles one raw `GraphSurface` containing two HG scatter
closures with exactly equal phase parameters, an absorption closure, and an
emissive Volume Coefficients closure. It then evaluates:

1. one world entry using the first parameter block and density scale one;
2. the world plus an object entry using a second parameter block, a changed
   primary density, and object-density scale two;
3. the same overlapping stack with emission disabled; and
4. an empty stack.

For the single entry the two equal HG closures must remain distinct and in
source order. Adding the second entry produces four raw closures; only then
must exact merging reduce them to one closure whose RGB weight is the summed
`sigma_s`. This distinguishes Cycles' stack-entry merge timing from both
eager per-material merging and a pre-baked averaged phase.

The focused CTest group is:

```text
psycles.luisa_stacked_volume_fallback
psycles.luisa_stacked_volume_hip
psycles.luisa_stacked_volume_vk
```

All three pass. The fifteen records pin coefficient flags and values, raw
phase count/order/parameters/weights, the distinct parameter blocks and
object-density scaling, emission suppression, and zero-state behavior.

## Remaining render gate

The evaluator intentionally accepts a point provider instead of guessing
volume coordinates from a surface hit. Production world and object point
construction is now implemented and validated in
[`../volume-shading-points`](../volume-shading-points/README.md).
Closest-event free-flight integration, phase continuation, volume NEE, and
heterogeneous grid attributes remain required before scene volume materials
are enabled. EXR comparisons and triptychs begin only after those transport
stages operate together.
