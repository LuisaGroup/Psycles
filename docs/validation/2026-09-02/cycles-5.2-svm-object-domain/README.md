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
not request `ATTR_STD_PARTICLE` map to dummy entry zero. The raw exporter ID
for a class is used only for equality; it is never interpreted as `P(g)`.
Demand remains per shader, including inert holes in the global shader domain,
so an unrelated material containing Particle Info cannot retain particle data
for every object through the scene-wide opcode union.

### Symbolic shader attribute requests

Cycles invokes `ShaderNode::attributes` and each node-family override over
every node before `ShaderGraph::finalize`; disconnected nodes therefore still
contribute residency requirements even though `clean` later removes their
bytecode. Psycles now performs the same pre-finalize transfer over the
projected Cycles graph. It includes the generic unlinked Generated/UV socket
rules and the represented family-specific rules for info, attribute,
coordinate, geometry, tangent/normal-map, anisotropic closure, and volume
nodes. `DISPLACE_BOTH` adds undisplaced position and normal exactly at the
shader boundary.

The per-shader result remains a canonical set of symbolic `(standard, name)`
requests while local SVM bytecode is emitted. Only after every shader has
compiled are those requests resolved in the scene-wide `AttributeIDMap`.
This ordering is necessary: hidden geometry dependencies such as a named
normal map's base UV must not consume a named ID before the tangent and tangent
sign IDs embedded in SVM bytecode. Formally, request resolution extends the
already-allocated bytecode-ID map; it never changes its existing image.

The canonical symbolic set is part of local-shader structural identity, so
equal bytecode with different out-of-band residency requirements is not
interchangeable. Runtime particle retention now consumes the resolved
per-shader request set rather than inferring demand from `NODE_PARTICLE_INFO`.

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
opcode union. `tests/test_cycles_svm_info.cpp` checks the exact Particle Info
and Hair Info request sets. `tests/test_cycles_svm_attribute_requests.cpp`
checks the complete Cycles 5.2 standard-name domain, request-set transitions,
pre-finalize disconnected-node behavior, every represented family override,
`DISPLACE_BOTH`, and the proof-critical deferred named-ID ordering. Blender
exporter/importer regressions preserve raw group IDs, parent indices, particle
payload vectors, and the child-particle sentinel.
`tests/test_luisa_cycles_svm_image.cpp` checks the 80-byte, 16-byte-aligned
`KernelParticle` field projection and all padded `float4` lanes on real Luisa
backends.

Validation commands:

```text
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure -R '^psycles\\.cycles_svm_'
ctest --test-dir build --output-on-failure -R '^psycles\\.luisa_cycles_svm_image_(hip|fallback)$'
LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 ctest --test-dir build --output-on-failure \
  -R '^psycles\\.luisa_cycles_svm_image_vk$'
```

The full build, all 33 Cycles SVM host tests, the fallback/HIP image tests, and
the strict native XIR-to-SPIR-V Vulkan canary passed. The identity planner and
particle packer are invoked transactionally by `build_cycles_svm_runtime`
before any SVM device buffer is created.

## Deliberate next boundary

The host image resolves the exact final `KernelObject::particle_index`; its
typed `KernelParticle` array is layout-checked, allocated with dummy entry zero,
and uploaded transactionally with the SVM program.

The exact `KernelObject` type is now frozen independently from the Psycles
definition by compiling the pinned Cycles headers. The oracle reports a
256-byte size, 16-byte alignment, and every field offset from `tfm == 0`
through `blocker_shadow_set == 240`; `tests/test_cycles_svm_abi.cpp` encodes
that full result. Explicitly named padding occupies only the ABI holes before
the two 64-bit light-link masks and at the tail, allowing a zero-initialized
host record to have deterministic upload bytes. `ShaderDataObjectFlag` is
also copied as one complete enum domain, including Cycles' deliberate omission
of `SD_OBJECT_HAS_VERTEX_MOTION` from `SD_OBJECT_FLAGS`.

This is an ABI checkpoint, not a fabricated object image. Production upload
remains disabled until a two-stage builder has resolved each reachable
object's geometry attribute-map, position, and normal offsets from the final
typed attribute arrays. The legacy scalar bundle field remains only for the
old expanded renderer route and must not be consumed by the production SVM
adapter.

The type-state boundary is now implemented. `prepare_kernel_object` consumes
only the non-geometry ObjectManager inputs and produces `PendingKernelObject`,
which intentionally contains no uploadable `KernelObject`. It freezes the
Cycles affine inverse, negative-scale predicate, raw uint random identity,
MurmurHash3 Cryptomatte identities, and source-owned flags. The sole conversion
to `KernelObject` is `finalize_kernel_object`, whose geometry descriptor uses
`optional` offsets to distinguish an unresolved value from the legitimate
`ATTR_STD_NOT_FOUND` integer sentinel. Thus the invalid state "uploaded but
not resolved" is not representable through this API.

Finalization copies Cycles' shadow-catcher visibility duplication, 64-set
light/shadow-link clamping, reciprocal shadow-terminator correction, exact
motion/count widths, and geometry-owned flags. The object-scene regression
pins the Cryptomatte oracle words (`Cube == 0xa8fce865`,
`Lone Monk == 0x04c3b823`), transform/inverse rows, raw-random rounding,
padding zeroes, every flag source, and all overflow/rejected-state boundaries.
