# Cycles 5.2 SVM geometry runtime validation

## Scope

This checkpoint installs the exact post-displacement geometry image and the
dense, hole-preserving `KernelObject`/`object_flag` image in the production
Luisa scene runtime. Every Cycles `DeviceScene` array is uploaded as its native
element type. It does not switch the production shade-surface route: exposing
these arrays through `KernelGlobals` and constructing exact `ShaderData` and
closure-pool state remain the next boundary.

The reference is Blender Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`.

## Formal construction boundary

Let `S` be the scene snapshot, `D(S)` the final mesh data after displacement,
`G(S, D)` the native Cycles geometry image, and `O(S, G)` the native object
image derived from that finalized geometry. Scene compilation now enforces the
single transaction

```text
unfinalized -> valid(G, O) -> allocated(G, O) -> installed(G, O)
            -> uploaded(G, O)
```

with these invariants:

1. `G` is built only after displacement has finalized every mesh upload, and
   `O` is built only from that exact `G`.
2. Curve and triangle primitive offsets come from the single production
   interval resolvers; the adapter never reconstructs a second address space.
3. Validation and every geometry/object buffer allocation complete before
   either runtime pointer is installed. A rejected construction therefore
   leaves both pointers null.
4. A runtime can be finalized exactly once, so no consumer can retain object
   records from one image while observing geometry tables from another.
5. Host arrays preserve their exact semantic extent. Luisa's non-zero buffer
   allocation requirement is represented only on the device by one unreachable,
   zero-initialized sentinel for an empty semantic array.

The uploaded arrays retain the native Cycles types: `AttributeMap`, scalar
float, packed float2/3/4, byte RGBA, octahedral packed normal, triangle vertex,
curve key, point, packed triangle index, 32-bit `tri_shader`, `KernelCurve`, the
256-byte `KernelObject`, and its separate 32-bit flag word. The 32-byte,
16-byte-aligned `KernelShader` ABI is frozen alongside them for the immediately
following shader-table transaction. No common float4 payload or
renderer-specific rebaking is introduced.

## Primitive shader image

For the global triangle primitive domain `P`, the geometry transaction now
constructs Cycles' `DeviceScene::tri_shader` beside `tri_vindex`. Let `U_g` be
the effective `Geometry::used_shaders` slot vector for geometry `g`, `m(p)` the
post-displacement material slot of primitive `p`, and `c(g)` the existence of
`ATTR_STD_CORNER_NORMAL`. The packed value is

```text
slot(p)   = min(m(p), |U_g| - 1)
smooth(p) = c(g) or authored_smooth(p)
tri_shader[p] = shader_index(U_g[slot(p)])
                | SHADER_CAST_SHADOW
                | (smooth(p) ? SHADER_SMOOTH_NORMAL : 0)
```

This is the `BlenderSync::sync_mesh` last-slot clamp followed by
`Mesh::pack_shaders`; it is not a hit-time material lookup. Finalized material
and smooth images must agree with their source images when both are present.
Raw shader indices are also proven to lie inside the compiled dense shader
domain and not overlap any decoration bit.

Cycles keys shared Blender geometry by its effective material-slot vector. A
programmatically authored Psycles geometry can still have per-instance
overrides, so native packing first computes each user's complete effective
slot vector. Equal vectors share one `tri_shader` image. Distinct vectors are
rejected with a geometry-split diagnostic because no single global primitive
array can represent both; silently choosing either vector would violate the
source model.

## Dense object image

Let `D_o = [0, N)` be the source `Scene::objects` domain, `R` the represented
instances/lights/background, and `f : R -> D_o` the previously validated
injective identity map. Object construction is the total dense image

```text
O[i] = 0                                                    if i is not in image(f)
O[f(r)] = finalize(prepare(raw_object_state(r)), G[g(r)])   otherwise
```

where `g(r)` is the represented geometry for an instance and an empty geometry
descriptor for a lamp/background object. Thus unsupported source objects remain
byte-zero holes; they are never compacted or fabricated. Raw object state
includes the source random word, particle-table index, asset/name identity,
object color/alpha/pass, transforms, visibility, terminator offsets, and flags.
Geometry-owned counts, primitive type, attribute-map/position/normal offsets,
and static-transform state come only from the finalized geometry image.

The six-bit scene visibility domain maps to Cycles' seven low bits by mapping
`shadow` to both `SHADOW_OPAQUE` and `SHADOW_TRANSPARENT`; the shadow-catcher
high-bit copy is applied only by Cycles object finalization. Legacy all-ones
visibility is normalized to the complete six-bit contract before this map.

For each geometry `g`, `has_volume(g)` is the union of volume roots from every
base material slot and every override of every instance sharing `g`, matching
`Geometry::used_shaders`. With final world-space bounds `B(o)`, the second flag
is defined exactly by

```text
intersects_volume(o) = exists v != o:
    has_volume(v) and intersects(B(o), B(v))
```

Mesh bounds consume the post-displacement accelerator position image. Curve
bounds and procedural AABBs now share one segment enumerator and Cycles 5.2.1's
exact structural rule: float Catmull--Rom positional extrema plus the maximum
radius of the two active segment endpoints. The implementation deliberately
retains Cycles' `cubic_coefficient != 0` derivative-root branch rather than
inventing a separate mathematically tighter quadratic case.

Two source relations are rejected rather than silently erased. A scene with
native light/shadow linking cannot use all-set defaults, and an object-motion
scene cannot use zero motion offsets until those native tables are present.
Both capability gates are source-derived; renderer-authored scenes without the
features remain valid.

## Static-transform image

The geometry transaction now also consumes the exact accelerator
representation plan. For the map-ordered instance domain `O` and plan image
`P`, construction requires `|O| = |P|` and `P[i].instance = O[i].id`; every
stored inverse must equal the inverse of that same instance transform. This
prevents a valid transform bit from being paired with another object's data.

Let `A(g)` denote that the unique mesh user of geometry `g` has
`transform_applied`. The transaction accepts a world-space intersection image
if and only if `A(g)` is true. Its extent and every position must agree with
the accelerator image. Shared geometry, curves, and meshes carrying
true-displacement snapshots cannot enter this state, matching Cycles'
`ObjectManager::apply_static_transforms` gate.

The typed DeviceScene image is then defined by

```text
tri_verts[i] = T * P[i]                    when A(g)
tri_verts[i] = P[i]                        otherwise
normal[i]    = normalize(transpose(T^-1) * N[i]) when A(g)
normal[i]    = N[i]                        otherwise
```

Only Cycles' standard position and normal storage follows this transform.
Generated coordinates, UVs, and UV tangents stay in object space, as required
by the object flags consumed by Cycles texture-coordinate and normal-map code.

## Used-shader domain

An initial production curve regression exposed an important distinction between
legacy render reachability and Cycles `Geometry::used_shaders`. The latter is
not the set of materials selected by observable primitives.

The pinned source establishes the transfer directly:

- `intern/cycles/blender/geometry.cpp::find_used_shaders` records every object
  material slot;
- `intern/cycles/scene/geometry_attributes.cpp::device_update_attributes`
  merges attribute requests from every geometry used shader; and
- `intern/cycles/blender/particles.cpp::sync_dupli_particle` queries the same
  geometry shader domain for particle attributes.

Psycles therefore constructs the SVM shader domain from every mesh and curve
material slot, instance override, analytic-light shader, and world shader. A
slot absent from the legacy primitive-reachable material library is compiled in
a transaction-local supplemental library. This preserves the old route's
reachability contract while making the copied SVM table total.

Compile units are ordered by their source `cycles_shader_index`, not by
`MaterialId`. This is required because Cycles assigns scene-wide named
attribute, image, and IES identities by first insertion in `Scene::shaders`
order. Renderer-authored materials without a source index receive deterministic
indices only after the authored range.

## Regression coverage

`tests/test_luisa_cycles_svm_geometry_runtime.cpp` checks on fallback, HIP, and
strict native-XIR Vulkan that:

- rejected and duplicate finalization cannot expose or replace a partial
  runtime;
- a material slot unused by every primitive is still compiled for the Cycles
  geometry shader domain without entering the legacy material library;
- deliberately reversed `MaterialId` and source shader order assigns named
  attribute IDs in Cycles shader order;
- typed attribute maps, finalized vertices, packed normals, global triangle
  indices, decorated `tri_shader` values, full `KernelObject` records, and
  object flags survive device upload;
- an actual DSL kernel reads nested transform, float, 16-bit, 32-bit, and
  64-bit object fields, preventing a raw-byte-copy-only test from masking a
  reflection or native code-generation failure; and
- every empty semantic array receives a deterministic unreachable zero device
  sentinel.

`tests/test_cycles_instance_support.cpp` independently freezes the complete
used-shader root set, including unused geometry slots, instance overrides,
analytic lights, and the world. Existing host geometry-image tests continue to
cover typed packing and rejected source states.

The host geometry/object regression additionally uses a reflected, non-uniform,
non-diagonal transform. It freezes world-space triangle vertices,
inverse-transpose packed normals, and unchanged object-space tangents, and
rejects missing accelerator images, permuted instance plans, and static
transforms applied to shared geometry.

The same host regression freezes all `Mesh::pack_shaders` branches: authored
smooth shading, corner-normal smooth override, last-slot clamp, an
object-resolved shader array, and rejection of incompatible shared users. The
device regression reads the uploaded sparse primitive entry on fallback, HIP,
and strict native-XIR Vulkan.

Its dense-object fixture freezes sparse holes, raw object state, exact
visibility expansion, particle-prefix lookup, finalized mesh/curve offsets,
lamp/background records, and geometry-level volume flags. A non-degenerate
Catmull--Rom overshoot intersects a post-displacement mesh only outside the
authored key bounds, proving that volume classification consumes the same
curve bounds as production traversal. Separate importer/exporter tests freeze
the world asset identity and detect source light-linking use.

Validation commands:

```sh
cmake --build build --parallel 32

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j3 \
  -R '^psycles\.luisa_cycles_svm_geometry_runtime_(fallback|hip|vk)$'
```

Result:

```text
psycles.luisa_cycles_svm_geometry_runtime_fallback  Passed  0.02 sec
psycles.luisa_cycles_svm_geometry_runtime_hip       Passed  0.08 sec
psycles.luisa_cycles_svm_geometry_runtime_vk        Passed  0.04 sec
100% tests passed, 0 tests failed out of 3
```

The Vulkan test is also registered with the same strict native XIR-to-SPIR-V
environment, so it cannot silently fall back to DXC. All 34 Cycles-SVM host
tests passed independently.

The full 547-test suite then passed under the strict Vulkan environment in
23.20 seconds. Its first run found the pre-existing 2052-line
`tests/test_blender_import.cpp`; the Light Path Portal Depth fixture was moved
to its own compilation unit rather than weakening the 2000-line policy. The
source-size and original import regressions passed after the split, and the
second full run was 547/547.

After adding the static-transform image invariant, a fresh 32-thread build and
the seven directly affected host/device tests passed. The full suite was then
rerun with strict native-XIR Vulkan enabled: 547/547 passed in 17.44 seconds.

After completing the dense object image and device field-read canary, another
32-thread build and strict native-XIR Vulkan full run passed 547/547 tests in
16.78 seconds. This run also includes the Blender 5.2 light-link capability
fixture, the shared curve-bounds regression, source-size enforcement, fallback,
HIP, and Vulkan typed-object execution.

After adding the native `tri_shader` image and freezing the `KernelShader` ABI,
the affected host tests and fallback/HIP/strict-native-XIR Vulkan typed-upload
tests passed. A fresh 32-thread build followed by the full suite under
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1` passed 547/547 tests in 22.38 seconds.

## Next boundary

The geometry and object buffers are deliberately not consumed by the old
expanded surface evaluator. The next structural increment must expose these
typed buffers through the production `KernelGlobals` provider, construct exact
`ShaderData` and closure-pool state from them, and only then switch the
shade-surface route.
