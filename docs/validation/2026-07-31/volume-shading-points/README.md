# Volume shading points

This checkpoint validates production world/object `ShaderData` construction
for raw volume-graph evaluation. It is a Luisa device-state regression, not
yet a volume-render or visual-parity claim.

## Reference and contract

- Official Blender/Cycles reference:
  `b82c3f0da6c1813dabedc563d64e536f4d83e868`.
- Reference code:
  `intern/cycles/kernel/geom/shader_data.h::shader_setup_from_volume`,
  `intern/cycles/kernel/integrator/volume_shader.h`,
  `intern/cycles/kernel/svm/tex_coord.h`,
  `intern/cycles/kernel/geom/volume.h`,
  `intern/cycles/blender/util.h::mesh_texture_space`, and
  `intern/cycles/blender/mesh.cpp`.
- The world shading position is the free-flight point and `N`, `Ng`, and
  `wi` are the negative ray direction supplied by the transport stage.
- Cycles' density-bake ray direction is exactly zero. Its
  `safe_normalize()` leaves a zero vector unchanged, so object-space normal
  transformation must also produce zero rather than NaN.
- Object coordinates use the inverse TLAS instance transform. Object
  location, random value, and particle index come from that same instance.
- Volume Generated coordinates are not interpolated from a boundary
  triangle. They apply Cycles' `ATTR_STD_GENERATED_TRANSFORM` to
  object-space `P`.
- World Object and Generated coordinates remain in world space; world Object
  Info location/random/particle values are zero.
- A volume point has `PRIM_NONE`, zero UV/barycentric/differential fields,
  and no triangle geometry index. This prevents named triangle-attribute code
  from dereferencing an unrelated boundary primitive.
- Mesh-boundary volumes use Cycles' object-density factor one. Native
  object-space VDB geometry will carry its scale correction with that future
  geometry kind rather than changing mesh semantics.

Blender scene schema v2 now stores the full column-major Generated affine
transform beside the existing point-domain Generated stream. The exporter
uses Blender's live texture-space location and size, the adapter retains the
matrix, and the Luisa geometry record makes it available to volume shading.
Programmatic scenes which omit the transform receive the same bounds mapping
already used for their fallback surface Generated values, including a
constant `0.5` on degenerate axes.

## Regression

The Blender attribute-domain regression checks the exported matrix against
the live mesh texspace and checks every point-domain Generated value against
that matrix. Its C++ inspector then reloads the bundle and repeats the
matrix-to-point consistency check, covering export and import rather than
only JSON text. A host mapping regression additionally pins automatic bounds,
the constant `0.5` degenerate-axis rule, and a general authored affine
transform.

The Luisa fixture constructs an object with negative X scale, non-uniform Y/Z
scale, and translation. It validates:

1. inverse-transform object position;
2. Cycles' transposed object normal transform;
3. inverse-transpose normal-to-world columns;
4. explicit Generated affine mapping;
5. object location/random/particle and path-state forwarding;
6. `PRIM_NONE` plus all zero primitive-coordinate fields; and
7. a world entry whose `~0u` instance identity never enters a TLAS transform
   query; and
8. zero-direction object and world entries whose `N`, `Ng`, `wi`, and
   object-space normal remain exactly zero.

The focused CTest group is:

```text
psycles.generated_coordinates
psycles.blender_export_attribute_domains
psycles.luisa_volume_point_fallback
psycles.luisa_volume_point_hip
psycles.luisa_volume_point_vk
```

All five pass. The device fixture produces the same 26 records on fallback,
HIP, and Vulkan.

## Remaining render gate

This checkpoint supplies `StackedVolumeEvaluator` with production points but
does not yet consume a closest-event segment. Closest-event free flight,
phase continuation, volume emission accumulation, and volume NEE remain
required before the material release gate is removed. There is therefore no
honest EXR triptych for this isolated state layer; visual Cycles comparisons
start when those stages form a complete estimator.
