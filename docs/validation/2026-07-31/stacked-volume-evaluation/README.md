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
- Closure allocation and live closure count are separate state. Every
  successful `closure_alloc()` permanently consumes `num_closure_left` for
  that evaluation; merging compacts `num_closure` but never refunds the
  allocation budget.
- Direct phase evaluation uses Cycles' scalar `sample_weight` mixture.
  Indirect continuation reservoir-selects one closure and samples that
  closure directly; it does not RGB-reweight or re-evaluate the full mixture.
  RGB scattering coefficients have already entered the free-flight
  throughput estimator.
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

All three pass. The twenty-two stacked-volume records pin coefficient flags
and values, raw phase count/order/parameters/weights, the distinct parameter
blocks and object-density scaling, emission suppression, and zero-state
behavior. Seven of those records exercise the host-polymorphic
`HomogeneousVolumeSegmentComponent` from the original graph through
coefficient aggregation, distance sampling, and phase continuation. Its
reservoir input selects the second raw HG closure and rescales `0.034` to the
official oracle input `0.17`; the resulting direction and PDF match Cycles
main exactly on fallback, HIP, and Vulkan. This catches propagation-axis,
reservoir-rescaling, or composition-order regressions that isolated component
tests cannot catch. The companion coefficient/phase-set fixture contains a
merge-budget regression: two equal closures consume two of three allocations,
merge to one live entry, and leave room for exactly one later closure. A
fourth attempted allocation must remain rejected even though compacted
storage has a free slot.

## Remaining render gate

The evaluator intentionally accepts a point provider instead of guessing
volume coordinates from a surface hit. Production world and object point
construction is now implemented and validated in
[`../volume-shading-points`](../volume-shading-points/README.md).
The reusable free-flight and phase continuation component is complete, but
its closest-event path-state stage, volume NEE, and heterogeneous grid
attributes remain required before scene volume materials are enabled. EXR
comparisons and triptychs begin only after those transport stages operate
together.
