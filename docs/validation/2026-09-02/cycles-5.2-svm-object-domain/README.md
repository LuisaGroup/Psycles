# Cycles 5.2 SVM object-domain validation

This checkpoint establishes the scene-global object identity domain required
by the production Luisa `KernelGlobals` adapter. It does not yet claim that the
full Cycles `KernelObject`, particle, or attribute tables are uploaded.

The reference is Blender Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`:

- `intern/cycles/blender/object.cpp::BlenderSync::sync_objects` creates
  geometry and analytic-light objects in dependency-graph order and creates
  the background-light object afterwards;
- `intern/cycles/scene/object.cpp::ObjectManager::device_update_object_transform`
  writes records at `Object::index` without compacting unsupported objects;
- `intern/cycles/kernel/geom/object.h` treats `ShaderData::object` as an index
  into that single table for geometry, light, Object Info, Texture Coordinate,
  and particle lookup.

## Formal model

Let `D = [0, N)` be the source `Scene::objects` domain and `R` the objects
represented by Psycles: renderable instances, analytic lights, and an optional
background object. The plan constructs `f : R -> D` with these invariants:

1. `f` is total for every represented object.
2. `f` is injective across all three object kinds.
3. When the exporter declares `N`, every value of `f` is source-authored and
   belongs to `D`; no missing identity is guessed.
4. `D - image(f)` remains a hole. Psycles never renumbers later source
   objects around an unsupported object.
5. When a background object exists in a declared Blender/Cycles domain, its
   index is `N - 1`, matching `sync_background_light` after the object loop.

For a renderer-authored scene with no declared `N`, explicit identities are
reserved first. Missing identities are assigned the least unused index in
stable instance, light, then background order. This is a deterministic total
extension of the partial explicit mapping; it is not presented as a Blender
source identity.

The occupied set is sparse rather than an `N`-element bitmap, so validating a
bad or very sparse source bundle does not allocate memory proportional to an
untrusted maximum index. `UINT32_MAX` is rejected when it would require the
unrepresentable dense extent `2^32`.

## Regression coverage

`tests/test_cycles_svm_object_scene.cpp` covers:

- a nine-entry source domain with four represented objects and five retained
  holes;
- duplicate identities across instance and light kinds;
- an out-of-domain identity;
- a missing identity in a declared source domain;
- a non-final background object;
- deterministic mixed explicit/implicit allocation; and
- the `UINT32_MAX` extent boundary.

Validation commands:

```text
cmake --build build --parallel 32 --target psycles_cycles_svm_object_scene_tests psycles_luisa_runtime
ctest --test-dir build --output-on-failure -R '^psycles\\.cycles_svm_(object_scene|scene)$'
```

Both registered tests passed. The planner is also invoked transactionally by
`build_cycles_svm_runtime` before any SVM device buffer is created.

## Deliberate next boundary

The existing Blender bundle's scalar `particle_index` is not sufficient for
the exact device record: Cycles first groups qualifying duplis by
`ParticleSystemKey`, stores a local append ordinal on each object, then adds a
scene-global particle-system prefix (after dummy entry zero). The next object
table increment must preserve that raw grouping and perform the same packing;
it must not mistake Blender's persistent parent index for the final
`KernelObject::particle_index`.
