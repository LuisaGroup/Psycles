# Volume majorant scene resources

This checkpoint composes the raw Luisa density prepass and Cycles hierarchy
builder into retained scene resources. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` is the sole oracle. There is no
Psycles CPU reference renderer, material approximation, or closure pre-bake.

## Scene mapping contract

The reference construction is anchored in
`intern/cycles/scene/volume.cpp::VolumeManager::initialize_octree`,
`intern/cycles/bvh/octree.cpp::Octree::evaluate_volume_density`, and
`VolumeManager::flatten_octree`. The Luisa scene component preserves the same
semantic identities:

- one root for every volume-bearing effective shader on every internal object
  instance;
- structurally homogeneous shaders use Cycles' `1^3` density grid, while
  spatially varying shaders use `128^3`;
- instance material overrides replace their corresponding geometry slots,
  and duplicate effective shaders create only one root;
- mesh-space bounds are used without transforming or pre-baking the closure;
- one or two collapsed bound axes are retained, while a fully collapsed mesh
  bound is skipped;
- mesh instance density scale is
  `length(transform_direction(object_to_world, normalize(1, 1, 1)))`;
- the World has its own final range, Cycles background-object identity, and
  practical `[-10000, 10000]^3` bounds; and
- root shader identities contain only Cycles' low shader index, with
  per-hit high flags masked away.

Each instance and the World owns an explicit contiguous root range. These
ranges form an exact ordered partition of the flattened root array; runtime
lookup therefore does not depend on host map ordering or pointers.

## Formal hierarchy boundary

Before relocation and upload, every local hierarchy must satisfy all of the
following invariants:

1. exactly one declared root, whose parent is the `-1` sentinel;
2. every internal node owns exactly eight consecutive children;
3. every child points back to that parent;
4. every node is reachable from the root exactly once, excluding cycles,
   shared children, and disconnected records; and
5. every scalar extrema pair is finite, nonnegative, and ordered.

The range partition and every root's declared range identity are validated
independently. Only after these proofs pass are parent, child, and root indices
relocated into the shared array. Node cardinality remains within Cycles'
signed 32-bit index contract.

This is acceleration metadata, not a density or radiance bake. The exact same
raw `GraphSurface` closure is evaluated again by runtime transport.

## End-to-end backend regression

The backend fixture builds two raw graphs:

```text
Texture Coordinate.Generated -> Volume Coefficients.EmissionCoefficients
constant RGB(0.35, 0.2, 0.1) -> Volume Coefficients.EmissionCoefficients
```

It constructs a translated object, real BLAS/TLAS resources, production
`BufferShaderServices`, the production volume point provider, and the actual
scene component. Every backend evaluates
`(128^3 + 1^3) * 16 = 33,554,448` closure samples, performs host hierarchy
reduction, uploads the three resource buffers, reads them back, and verifies:

- one heterogeneous and one homogeneous object root plus an empty final World
  range;
- a heterogeneous root plus eight children and a homogeneous leaf root;
- exact root/node/range relocation and shader masking;
- object bounds mapped to `[1, 2)`; and
- conservative root extrema spanning the authored Generated field.

The host regression separately covers material overrides, homogeneous
inclusion, surface-only exclusion, duplicate shaders, object and World identity,
partially and fully collapsed bounds, non-finite transforms, range partition
failures, incomplete octrees, parent mismatches, unreachable records, and
invalid extrema.

Focused manual fixture wall times on the local RX 9070 XT workstation were:

| backend | wall time | result |
| --- | ---: | --- |
| fallback (LLVM/Embree, 32 threads) | `0.08 s` | pass |
| HIP (cached shader) | `0.10 s` | pass |
| Vulkan/RADV | `0.07 s` | pass |

These are small fixture timings, not full-scene rendering benchmarks.

Run:

```sh
cmake --build build \
  --target psycles_luisa_volume_majorant_scene_tests \
  --parallel 32
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_volume_majorant_scene_(fallback|hip|vk)' \
  --parallel 1
```

The combined hierarchy, prepass, and scene-resource selection completed
`9/9` focused tests on fallback, HIP, and Vulkan. A subsequent full
`cmake --build build --parallel 32` and serial CTest run completed `102/102`
tests in `8.91 s`.

## Visual status and next connection

This checkpoint produces internal scalar acceleration data and no pixels, so
there is no meaningful image to inspect and no honest triptych to publish.
Visual inspection and Cycles/Psycles multilayer-EXR triptychs remain mandatory
as soon as overlapping-root traversal is connected to production transport.

The scene capability gate intentionally remains in place until that transport
is complete; accepting a heterogeneous scene earlier would render it with the
wrong estimator. The next slice is the Cycles smallest-endpoint reduction
across overlapping roots, followed by composition with weighted delta
tracking, phase sampling, direct-light MIS, and full Blender scene
differentials.
