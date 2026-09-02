# Cycles 5.2 SVM object-domain validation

This checkpoint establishes the scene-global object identity domain required
by the production Luisa `KernelGlobals` adapter. It also establishes the exact
host-side Cycles particle-table image and its typed Luisa device projection.
It does not yet claim that the full `KernelObject` or attribute tables are
uploaded.

The reference is Blender Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`:

- `intern/cycles/blender/object.cpp::BlenderSync::sync_objects` creates
  geometry and analytic-light objects in dependency-graph order and creates
  the background-light object afterwards;
- `intern/cycles/scene/object.cpp::ObjectManager::device_update_object_transform`
  writes records at `Object::index` without compacting unsupported objects;
- `intern/cycles/kernel/geom/object.h` treats `ShaderData::object` as an index
  into that single table for geometry, light, Object Info, Texture Coordinate,
  and particle lookup;
- `intern/cycles/blender/particles.cpp::sync_dupli_particle` filters by the
  geometry's `ATTR_STD_PARTICLE` demand, rejects child particles, groups by
  `ParticleSystemKey`, and appends a local particle ordinal; and
- `intern/cycles/scene/particles.cpp::device_update_particles` prepends dummy
  entry zero and concatenates particle-system groups in scene order.

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

### Particle prefix algebra

Let qualifying object identities be visited in ascending Cycles object index.
For an object `o`, let `g(o)` be its raw `ParticleSystemKey` equality class and
let `l(o)` be its append ordinal among earlier qualifying objects in that
class. Classes are ordered by the first qualifying object that names them. If
`P(g)` is one plus the sum of the sizes of all preceding classes, the device
index is

```text
particle_index(o) = P(g(o)) + l(o)
```

Non-qualifying objects, rejected child particles, and objects whose shaders do
not demand `NODE_PARTICLE_INFO` map to dummy entry zero. The raw exporter ID
for a class is used only for equality; it is never interpreted as `P(g)`.
Demand remains per shader, including inert holes in the global shader domain,
so an unrelated material containing Particle Info cannot retain particle data
for every object through the scene-wide opcode union.

## Regression coverage

`tests/test_cycles_svm_object_scene.cpp` covers:

- a nine-entry source domain with four represented objects and five retained
  holes;
- duplicate identities across instance and light kinds;
- an out-of-domain identity;
- a missing identity in a declared source domain;
- a non-final background object;
- deterministic mixed explicit/implicit allocation; and
- the `UINT32_MAX` extent boundary;
- particle dummy/group/local-prefix packing from deliberately unsorted object
  inputs, including disabled demand and a missing raw source; and
- rejection of duplicate object identities in particle-table input.

`tests/test_cycles_svm_scene.cpp` additionally proves that per-shader Particle
Info demand survives global shader linking without collapsing into the global
opcode union. Blender exporter/importer regressions preserve raw group IDs,
parent indices, particle payload vectors, and the child-particle sentinel.
`tests/test_luisa_cycles_svm_image.cpp` checks the 80-byte, 16-byte-aligned
`KernelParticle` field projection and all padded `float4` lanes on real Luisa
backends.

Validation commands:

```text
cmake --build build --parallel 32 --target psycles_cycles_svm_object_scene_tests psycles_luisa_runtime
ctest --test-dir build --output-on-failure -R '^psycles\\.cycles_svm_(object_scene|scene)$'
ctest --test-dir build --output-on-failure -R '^psycles\\.luisa_cycles_svm_image_(hip|fallback)$'
```

Both registered tests passed. The identity planner and particle packer are
invoked transactionally by `build_cycles_svm_runtime` before any SVM device
buffer is created.

## Deliberate next boundary

The host image resolves the exact final `KernelObject::particle_index`; its
typed `KernelParticle` array is layout-checked, allocated with dummy entry zero,
and uploaded transactionally with the SVM program. The next increment is the
exact `KernelObject` projection and upload. The legacy scalar bundle field
remains only for the old expanded renderer route and must not be consumed by
the production SVM adapter.
