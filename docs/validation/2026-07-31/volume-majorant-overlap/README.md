# Overlapping volume majorant traversal

This checkpoint implements the ordered multi-root reduction used by current
Cycles volume integration. Blender/Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868` is the only behavioral oracle.
The implementation is Luisa DSL generated from host-stage `.h`/`.cpp`
components; it introduces no Psycles CPU renderer and does not pre-bake
closures.

## Reference transition

The decisive reference is
`intern/cycles/kernel/integrator/shade_volume.h`:
`volume_find_octree_root`, `volume_octree_setup`, and
`volume_octree_advance`.

At a common interval minimum, Cycles visits the volume stack in order. The
currently active `(object, shader)` root is skipped because its traversal
state persists; every other root is reconstructed at that minimum. Leaf
extrema are added in the same stack order. A candidate replaces the active
root when its endpoint is `<=` the selected endpoint, deliberately making the
last root active when endpoints tie. After advancing that root, the reduction
repeats. Exactly one stack entry enables the `no_overlap` shortcut.

`VolumeMajorantOverlapTraversal` records that state transition directly. A
host-stage-polymorphic `VolumeMajorantEntryProvider` supplies each entry's
object-space ray and object density independently of the interval algebra.
Its default extrema policy is the baked leaf extrema times object density;
the production subclass can override it with Cycles' dynamic Light Path
interval evaluation without modifying traversal.

## Root-domain safety

Object entries use their internal instance range, while the World entry uses
the distinct final range. Each range is searched backward and compares the
stack shader through the shared Cycles `SHADER_MASK`. The first matching
identity is authoritative, matching Cycles' last-root convention.

The production scene builder proves hierarchy topology before upload. At the
device boundary, declared range arithmetic and the selected root-node index
are checked before use. Missing shaders, out-of-domain ranges, or invalid
root nodes invalidate the complete segment rather than silently dropping a
medium. A malformed newer identity cannot fall back to an older duplicate,
and rejected entries invoke neither coordinate conversion nor extrema
evaluation.

## Exact regression

The main fixture overlaps two roots along one ray. Root A has two adjacent
leaves with extrema `0.2` and `0.8`. Root B has one `0.4` leaf and object
density `2`, so its runtime extrema are `0.8`. The expected ordered result is:

| interval | summed extrema | active root | node |
| --- | --- | --- | --- |
| `[0.00, 0.75]` | `[1.0, 1.0]` | A | 7 |
| `[0.75, 1.25]` | `[1.6, 1.6]` | B | 9 |
| `[1.25, 1.75]` | `[1.6, 1.6]` | A | 8 |
| `[1.75, 2.00]` | `[1.0, 1.6]` | B | 9 |

The final A/B endpoints are equal; active B proves the exact `<=` tie rule.
The one-root fixture separately pins A's two leaves, the root-extrema tail,
and `no_overlap`. Additional cases prove backward lookup with masked high
shader flags, a missing shader, malformed range arithmetic, an out-of-range
newer duplicate root ahead of a valid older duplicate, no advancement after
failure, and zero provider callbacks for every rejected entry.

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_volume_majorant_overlap_tests
ctest --test-dir build --output-on-failure --parallel 3 \
  -R 'psycles\.luisa_volume_majorant_overlap_(fallback|hip|vk)$'
```

All `3/3` backend tests pass. Cached fixture wall times are `0.02 s`
fallback, `0.07 s` HIP, and `0.05 s` Vulkan on the local RX 9070 XT.
A full 32-thread build completed in `6.60 s`; serial CTest passed `105/105`
tests in `8.99 s`, including the source-size gate.

## Visual status and next boundary

This component emits interval identities and scalar extrema, not pixels, so
an EXR triptych or visual judgement would not test it. Production
heterogeneous collision, phase sampling, and direct lighting must first
consume the scene resources through a scene-aware provider. That first pixel
render will require official Cycles/Psycles multilayer EXRs, numerical
comparison, and a documented visually inspected triptych.
