# Cycles 5.2 SVM geometry runtime validation

## Scope

This checkpoint installs the exact post-displacement geometry image in the
production Luisa scene runtime and uploads each Cycles `DeviceScene` array as
its native element type. It does not switch the production shade-surface route:
the exact `KernelObject`, `ShaderData`, and closure-pool adapter remains the
next boundary.

The reference is Blender Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`.

## Formal construction boundary

Let `S` be the scene snapshot, `D(S)` the final mesh data after displacement,
and `I(S, D)` the native Cycles geometry image. Scene compilation now enforces
the state transition

```text
unfinalized -> valid(I) -> installed(I) -> uploaded(I)
```

with these invariants:

1. `I` is built only after displacement has finalized every mesh upload.
2. Curve and triangle primitive offsets come from the single production
   interval resolvers; the adapter never reconstructs a second address space.
3. Validation and all buffer allocations complete before the runtime pointer is
   installed. A rejected construction therefore leaves the runtime null.
4. A runtime can be finalized exactly once, so no consumer can retain buffers
   from one image while observing host tables from another.
5. Host arrays preserve their exact semantic extent. Luisa's non-zero buffer
   allocation requirement is represented only on the device by one unreachable,
   zero-initialized sentinel for an empty semantic array.

The uploaded arrays retain the native Cycles types: `AttributeMap`, scalar
float, packed float2/3/4, byte RGBA, octahedral packed normal, triangle vertex,
curve key, point, packed triangle index, and `KernelCurve`. No common float4
payload or renderer-specific rebaking is introduced.

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
- typed attribute maps, finalized vertices, packed normals, and global triangle
  indices survive device upload; and
- every empty semantic array receives a deterministic unreachable zero device
  sentinel.

`tests/test_cycles_instance_support.cpp` independently freezes the complete
used-shader root set, including unused geometry slots, instance overrides,
analytic lights, and the world. Existing host geometry-image tests continue to
cover typed packing and rejected source states.

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
psycles.luisa_cycles_svm_geometry_runtime_hip       Passed  0.09 sec
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

## Next boundary

The geometry buffers are deliberately not consumed by the old expanded surface
evaluator. The next structural increment must finalize the exact `KernelObject`
records from this installed geometry image, expose these typed buffers through
the production `KernelGlobals` provider, construct exact `ShaderData` and
closure-pool state, and only then switch the shade-surface route.
