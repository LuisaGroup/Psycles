# Volume majorant hierarchy and traversal

This checkpoint reproduces the current Cycles heterogeneous-volume hierarchy
reduction and its single-root device traversal. Blender/Cycles main
`b82c3f0d` is the sole oracle. No Psycles CPU renderer or material
approximation is involved.

## Cycles construction contract

The host builder consumes one minimum/maximum pair for every cell of a
`128^3` grid. `VolumeMajorantPrepass` now produces those values in a Luisa
kernel by evaluating the original volume graph; the builder only reduces
acceleration metadata.

For each node it merges all covered cell extrema and splits exactly when

```text
depth < 7
and
(sigma_max - sigma_min) * length(node_bounds) * volume_scale > 1.442
```

Each internal node owns eight contiguous child records. Every child stores its
parent, and the root stores the affine transform from object bounds to
`[1, 2)`. Invalid grid size, non-finite or unordered extrema, fully collapsed
bounds, and negative density scale are rejected before construction. Matching
current Cycles, a bound with only one or two collapsed axes is retained and
uses the root extrema during traversal; a zero density scale is valid and
prevents subdivision.

Cycles obtains each cell extrema from sixteen Sobol-Burley positions in a
voxel padded by twenty percent on every side. This is a finite sampled
estimate, not a mathematical guarantee that no shader value exceeds the
stored maximum. The formal transport contract therefore remains conditional
on the majorant inequality, and the local collision component reports the
same runtime violation handled defensively by Cycles.

## Device traversal contract

`VolumeMajorantTraversal` owns one octree and records the Cycles hierarchical
DDA directly in Luisa DSL:

- positive ray axes are mirrored so all local directions are non-positive;
- IEEE-754 mantissa bits select octants and identify the nearest common
  ancestor of adjacent leaves;
- parent links replace a device traversal stack;
- leaf exit uses only the three back faces and Cycles' half-smallest-leaf
  offset;
- leaving the root switches to the root extrema until the active ray segment
  ends, except for the `5e-4` numerical-overlap termination.

Overlapping volume roots are deliberately not folded into this class. Their
smallest-endpoint selection and extrema sum form a separate production
component.

## Regression

The host fixture covers a uniform grid, a one-axis split with exact child
ordering and extrema, the root transform, and rejected malformed inputs. The
Luisa fixture walks the split hierarchy in both ray directions, crosses the
parent boundary, checks the root-extrema tail, and checks the initial
outside-root path.

Run:

```sh
cmake --build build \
  --target psycles_luisa_volume_majorant_tests \
  --parallel 32
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_volume_majorant_(fallback|hip|vk)' \
  --parallel 3
```

All three backends pass. This checkpoint validates internal acceleration
metadata and ray intervals, so an image triptych is not applicable. The
required Cycles/Psycles EXR triptychs begin when heterogeneous transport is
connected to the production scene path.

## Remaining connection

The raw Luisa `128^3 x 16` shader-evaluation prepass and scene-side resource
construction are complete and recorded in
[`../volume-majorant-prepass`](../volume-majorant-prepass/README.md) and
[`../volume-majorant-scene-resources`](../volume-majorant-scene-resources/README.md).
Multi-root stack reduction and the collision/phase/direct-light path are the
next connection before official Cycles image differentials can begin.
