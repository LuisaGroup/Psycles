# Lone Monk Cycles geometry identity

This checkpoint aligns Blender export and Luisa intersection/light records
with Cycles' source-scene identity spaces. It does not derive identities from
Psycles map order, TLAS instance order, or per-mesh primitive indices.

## Oracle and invariant

The only oracle was the local Cycles checkout. Temporary diagnostics were
inserted at `GeometryManager::device_update_mesh` and
`ObjectManager::device_update_transforms` to report the final device arrays.
The formal geometry invariant is a monotone collection of non-overlapping
half-open intervals

```text
[geometry.primitive_offset,
 geometry.primitive_offset + geometry.triangle_count)
```

in Cycles' 32-bit global primitive address space. Blender bundles preserve the
reported offsets. Renderer-authored scenes use the same resolver with an
implicit deterministic prefix. Overlap and address-space overflow are scene
compilation errors.

Blender object identity follows `BlenderSync::object_is_geometry`, including
its evaluated object-data type test. Legacy Curve/Surface/Font entries whose
data remains non-Mesh are not counted merely because `Object.to_mesh()` can
return a payload.

## Lone Monk result

The scene was re-exported from
`/home/mike/Downloads/lone-monk_cycles_and_exposure-node_demo.blend` to
`/var/tmp/psycles-lone-monk-identity.ikalcb`.

| Identity | Cycles oracle | Psycles bundle |
|---|---:|---:|
| Non-world objects | 87,534 | 87,534 |
| Background object | 87,534 | 87,534 |
| `sky-3535393_1920` objects | 87,531, 87,532 | 87,531, 87,532 |
| `treeline` object | 87,533 | 87,533 |
| `grass_blade.006` primitive offset | 1,571,979 | 1,571,979 |
| `leaf.003` primitive offset | 1,572,029 | 1,572,029 |
| `leaf.001` primitive offset | 1,572,039 | 1,572,039 |
| `leaf.002` primitive offset | 1,572,055 | 1,572,055 |
| `sky-3535393_1920` primitive offset | 1,572,057 | 1,572,057 |
| `treeline` primitive offset | 1,572,059 | 1,572,059 |

The previously inspected Cycles primitive `1,572,058` is therefore exactly
the second sky triangle in both systems; it is not a local primitive `+4`
special case.

Mesh emitters now also retain the exact Cycles object index, global primitive
index, packed shader flags, light group, shadow-catcher bit, and smooth-normal
bit. Surface `isect_id` records use the same global primitive identity.

## Regression and build matrix

`ctest --test-dir build --output-on-failure --parallel 32` passed all 123
tests. Relevant coverage includes:

- Blender export of a convertible legacy Curve which must not consume a
  Cycles object index while its evaluated data remains Curve;
- exported geometry-prefix continuity and world/light/shadow-catcher policy;
- implicit/explicit primitive intervals, gaps, overlap rejection, and the
  32-bit boundary;
- global triangle primitive identity on fallback, HIP, and Vulkan; and
- the first-party 2,000-line source-size gate.

This checkpoint changes semantic trace identity rather than image transport,
so it has no standalone image triptych. The next direct-light checkpoint uses
these identities in the Cycles/Psycles path-trace comparison and will record
the resulting image triptych with the render validation.
