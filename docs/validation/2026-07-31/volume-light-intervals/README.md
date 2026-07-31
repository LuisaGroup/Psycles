# Finite-emitter volume ray intervals

This checkpoint ports the geometric support of Cycles'
`volume_valid_direct_ray_segment` into an independent host-stage Luisa AST
component. It does not add a second renderer or a CPU reference model. The
expected records were obtained by compiling the official Cycles headers at
`b82c3f0da6c1813dabedc563d64e536f4d83e868`; the production implementation is
Luisa DSL in `VolumeLightInterval`.

## Formal contract

Every operation maps an input interval `I = [a, b]` to the intersection of
`I` and one closed geometric support. An interval is usable exactly when its
lower endpoint is strictly smaller than its upper endpoint, matching Cycles'
NaN-safe `Interval::is_empty`.

- Quadratic roots use Cycles' stable larger-magnitude root, product identity,
  ordered endpoints, and `-1e-5` discriminant tolerance.
- A single-sided cone first intersects the double cone, classifies both roots
  by the positive-axis hemisphere, and converts the three possible root
  states to `[t0, t1]`, `[0, t0]`, or `[t1, FLT_MAX]`.
- A spot light transforms the ray by the complete inverse object transform.
  Its cone apex is expanded by
  `radius * sqrt(1 + |w|² / (tan²(angle/2) * min(|u|², |v|²)))`, including
  non-uniform scale.
- A zero-spread rectangular area light intersects three independent slabs. A
  zero direction component is handled by the formal inside/outside slab
  predicate, avoiding backend-dependent `0 * infinity` while remaining
  equivalent to Cycles' reciprocal form on its defined domain.
- A zero-spread elliptical area light uses the same shifted quadratic as
  Cycles. An exactly axial ray remains invalid because the source quadratic
  has no finite roots; this behavior is pinned rather than silently extended.
- A finite-spread area light uses the smallest conservative cone. Its maximal
  extent is `max(len_u, len_v) / 2` for an ellipse and
  `length(float2(len_u, len_v)) / 2` for a rectangle. An exact `pi` spread
  takes Cycles' `FLT_MAX` tangent and plane limit.
- The area result is finally intersected with the emitter's positive-side
  plane. The implementation retains the source arithmetic for parallel-plane
  IEEE behavior instead of introducing a different special case.

The component has ordinary `.h` and `.cpp` ownership and is independent of
light proposal, radiometric evaluation, and free-flight sampling. Those
pieces can compose it while building one fused runtime kernel AST.

## Regression matrix

Eleven new records extend the homogeneous-volume fixture from 46 to 57
records. They cover:

- zero- and finite-radius spot cones;
- finite-radius apex expansion under non-uniform scale;
- rotated spot transforms and the rejected back hemisphere;
- zero-spread rectangle and ellipse intervals;
- the exact axial-cylinder invalid case;
- distinct rectangle/ellipse finite-spread maximal extents; and
- the exact full-spread plane limit.

Run the fixture on all enabled backends:

```text
./build/bin/psycles_luisa_homogeneous_volume_tests fallback
./build/bin/psycles_luisa_homogeneous_volume_tests hip
./build/bin/psycles_luisa_homogeneous_volume_tests vk
```

All 57 records pass on fallback, HIP, and Vulkan on the RX 9070 XT. During the
first HIP run, the vector reciprocal AABB form incorrectly rejected a
parallel-inside slab that passed on fallback. Rewriting the operation as the
composition of three scalar slab intersections removed the undefined
`0 * infinity` path and made all three backends agree with the Cycles
interval.

This checkpoint is purely geometric and therefore has no meaningful image
triptych. The next integrated spot-light checkpoint must include raw
scene-linear OpenEXR comparisons and visually inspected Cycles/Psycles/diff
triptychs before it is accepted.
