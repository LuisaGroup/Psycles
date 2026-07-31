# Volume majorant complete root domain

This correction closes the input-domain invariant required by Cycles'
overlapping-volume reducer. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` remains the only behavioral
oracle; no Psycles CPU renderer or pre-baked closure representation is used.

## Reference invariant

The decisive reference path is
`intern/cycles/scene/volume.cpp::VolumeManager::initialize_octree`,
`VolumeManager::is_homogeneous_volume`, and
`intern/cycles/bvh/octree.cpp::Octree::evaluate_volume_density`.

Every volume-bearing object/shader pair that can appear in the device volume
stack has an octree root. Cycles changes only the sampling-grid cardinality:

- structurally homogeneous volume: `1^3` cells;
- spatially varying volume: `128^3` cells;
- every cell: sixteen padded Sobol-Burley evaluations of the original shader.

Omitting homogeneous roots would make a mixed stack incomplete: the overlap
reducer could neither reproduce Cycles' summed extrema nor advance the
smallest endpoint without silently dropping a medium. The Luisa scene planner
therefore now includes both classes. Surface-only materials remain excluded.

The hierarchy builder accepts exactly the two current Cycles resolutions. The
resolution is carried through the device prepass instead of being inferred
from dispatch size, and the shared low shader identity uses one public
`0x203fffff` Cycles ABI constant checked against the scene-side flag
definition.

## Regression coverage

Host tests prove:

1. homogeneous effective materials survive ordinary slots and instance
   override planning;
2. surface-only materials remain absent;
3. ordered instance and World ranges still form a complete partition;
4. the `1^3` hierarchy contains exactly one leaf with the sampled extrema;
5. resolutions other than current Cycles `1` and `128` are rejected.

The end-to-end scene fixture evaluates a spatial Generated-coordinate graph
and a constant homogeneous raw graph in one object. On fallback, HIP, and
Vulkan it reads back two roots and ten nodes: a nine-node heterogeneous tree
followed by one homogeneous leaf whose exact extrema are `0.35`.

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_volume_majorant_tests \
           psycles_luisa_volume_majorant_prepass_tests \
           psycles_luisa_volume_majorant_scene_tests
ctest --test-dir build --output-on-failure --parallel 3 \
  -R 'psycles\.luisa_volume_majorant(_prepass|_scene)?_(fallback|hip|vk)$'
```

Result: `9/9` focused tests passed. Cached end-to-end scene wall times were
`0.08 s` fallback, `0.10 s` HIP, and `0.07 s` Vulkan on the local RX 9070 XT
workstation. A full 32-thread build and serial CTest completed `102/102` tests
in `8.91 s`, including the source-size gate. These are resource-fixture
timings, not render benchmarks.

## Visual status

This correction changes only scalar acceleration metadata and emits no
pixels, so an image triptych would be misleading. Cycles/Psycles multilayer
EXR triptychs and direct visual inspection remain required at the first
production heterogeneous transport render.

The ordered overlapping-root interval reduction is now complete and recorded
in [`../volume-majorant-overlap`](../volume-majorant-overlap/README.md). Its
lookup rejects missing or malformed coverage rather than treating the
corresponding medium as zero.
