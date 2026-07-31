# Production heterogeneous-volume segment

This checkpoint connects the non-guided heterogeneous weighted-delta-tracking
segment to the production Luisa path-kernel composer. Blender/Cycles
`main@0ae970969e3f37eb63dc5be5701dbc93885fbfae` is the only renderer oracle;
there were no `intern/cycles` changes after the previously inspected
`b82c3f0d` revision. Psycles does not add a CPU reference renderer and does not
bake, fit, or preprocess Blender material closures.

At this historical checkpoint the production scene capability gate remained
closed because direct lighting and VSPG were not yet connected. Those camera-
path estimators are now complete in the
[VSPG/direct-light checkpoint](../heterogeneous-volume-vspg-direct/README.md);
the gate still remains closed until heterogeneous residual-ratio shadow
transport is connected.

## Host-stage composition

`PathHeterogeneousVolumeComponent` is a host-stage OOP adapter that records one
fused Luisa device AST from independently testable components:

- `VolumeMajorantOverlapTraversal` provides the ordered, overlapping object and
  World majorant intervals over the original absolute ray domain;
- `PathVolumeTrackingRandomSource` maps the copied Cycles path RNG state to
  scatter-distance and shade-offset dimensions;
- `StackedHeterogeneousVolumeCollisionProvider` evaluates the original retained
  `GraphSurface` Volume closures at every candidate point and again at the
  selected phase point; and
- `HeterogeneousVolumeSegmentComponent` owns the non-guided indirect
  weighted-delta recursion, reservoir event selection, emission, termination,
  and final raw-phase recovery.

The adapter is implemented in real `.h + .cpp` translation units rather than
textual `.inl` partitioning. The component boundaries are host-language
polymorphism used to generate the dynamic Luisa shader AST; no device-side
virtual dispatch is introduced. The largest touched source file remains well
below the repository's 2,000-line gate.

Scene buffers exposed as `BufferVar<T>` kernel arguments and captured
production `Buffer<T>` resources now share one ownership-safe interface:
majorant traversals retain `Expr<Buffer<T>>` by value. This removes the former
temporary-reference lifetime problem without adding separate production and
test implementations.

## Cycles state and measure

The implementation preserves the current Cycles control-flow domains rather
than replacing them with case-specific approximations:

1. octree setup samples the initial shade offset from the unscrumbled copied
   path RNG offset;
2. tracking then scrambles that copied offset with `0xe35fad82`;
3. candidate free flight uses scatter-distance dimension 5;
4. hierarchy continuation uses the `.y` component of the 2D shade-offset
   dimension 7 and advances one 16-dimension bounce block only after a
   successful candidate;
5. closure coefficients are evaluated at
   `ray_origin + ray_direction * absolute_candidate_t`;
6. the scalar null coefficient is
   `sigma_n = max(sampled_majorant, max(sigma_t)) - sigma_t`;
7. ordinary candidates normalize by the actual scalar majorant, while a
   violated bound takes Cycles' defensive exponential correction and raises
   the observable `majorant_exceeded` diagnostic;
8. absorption-only candidates deterministically recurse through the null event,
   while scattering candidates preserve Cycles' spectral channel measure and
   one-uniform reservoir update; and
9. the selected point re-evaluates the original closure graph to recover raw
   phase closures before phase selection around the ray direction.

The path-kernel adapter keeps traversal distances absolute through hierarchy
lookup and closure evaluation. Only the selected collision distance is
converted back to the segment-relative path distance expected by the existing
ray-update stage. This prevents the nonzero-`tmin` coordinate mismatch that a
locally rebased traversal would otherwise introduce.

`PATH_RAY_TERMINATE` retains absorption and emission while suppressing
scattering closure allocation, matching Cycles' volume shader setup. The
low-throughput roulette also stays after successful free flight and before
candidate closure evaluation.

## Regression evidence

Three device-level boundaries execute on fallback, HIP, and Vulkan:

- the generic segment fixture pins absorption/null recursion, a selected real
  event, emission, phase recovery, RNG offset, optical depth, and all terminal
  flags against current Cycles values;
- the stacked-volume fixture proves that the same original raw `GraphSurface`
  closure supplies candidate coefficients and final phase closures; and
- the production-scene fixture traverses actual retained scene majorant
  buffers, takes the heterogeneous adapter's true branch, and pins the absolute
  ray interval, copied/scrambled RNG state, raw spatial closure emission, and
  exhaustion state.

Warm focused results:

| boundary | fallback | HIP | Vulkan |
| --- | ---: | ---: | ---: |
| generic heterogeneous segment | `0.06 s` | `0.13 s` | `0.09 s` |
| raw stacked GraphSurface | `0.06 s` | `0.10 s` | `0.06 s` |
| production scene adapter | `0.11 s` | `0.15 s` | `0.10 s` |
| production volume path | `0.29 s` | `0.47 s` | `1.17 s` |

The focused set passed `12/12` tests in `1.17 s`. The full build used
`--parallel 32` and completed in `6.39 s`. Parallel CTest passed all
`111/111` tests in `106.03 s`, including source-size, OpenEXR, fallback, HIP,
and Vulkan coverage.

This checkpoint changes an internal stochastic estimator whose full scene
composition is still gated. Consequently there is no honest Cycles/Psycles
image or triptych for this checkpoint; visual evidence will be generated only
after direct volume lighting and VSPG complete the pixel estimator.

## Vulkan cold-JIT diagnosis

A strict native Vulkan cache miss was profiled with verbose Luisa stage timing
and the production SPIR-V optimization level. The original diagnostic process
took `39.32 s`; two large production path-kernel variants spent a combined
`36.0685 s`, or `91.73%` of process wall time, in XIR
`restructure-cfg`. Full XIR legalization accounted for `36.7539 s`
(`93.47%`). AST-to-XIR conversion, SPIR-V optimization/validation, and RADV
pipeline creation were not the dominant stages.

A warm validated-cache rerun took `1.00 s`. The detailed bounded measurements
are retained in
[vulkan-cold-jit.json](vulkan-cold-jit.json). This identified a formal
CFG-restructuring scalability problem in Luisa rather than CMake compilation or
Psycles rendering. The width-dependent query was subsequently fixed and pushed
to Luisa `next` as `21612b45b`; an isolated before/after run reduced process
wall time from `40.49 s` to `9.86 s`. The proof, bounded measurements, and
structural regression are recorded in the
[XIR selection-reentry scale checkpoint](../xir-selection-reentry-scale/README.md).

## Reproduction

```sh
cmake --build build --parallel 32
ctest --test-dir build --parallel 8 --output-on-failure
ctest --test-dir build --parallel 6 --output-on-failure \
  -R 'psycles\.luisa_(heterogeneous_volume_segment|stacked_volume|volume_majorant_scene|volume_path)_(fallback|hip|vk)$'
```
