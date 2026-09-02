# Cycles 5.2 SVM scene image table

## Scope

This checkpoint projects the scene-wide image handles emitted by the copied
Cycles 5.2.1 SVM compiler into device data. It does not switch the production
surface route yet. The later geometry-runtime checkpoint completes the typed
post-displacement geometry and attribute arrays; exact object, camera,
`ShaderData`, and closure-pool projection must still be complete before that
switch is valid.

Each SVM image handle is represented by the pair

```text
(texture_slot, interpolation | (extension << 2))
```

where interpolation is the canonical finite domain `{closest, linear,
cubic}` (`smart` has Cycles' cubic device behavior) and extension is
`{repeat, clip, extend, mirror}`. This preserves the distinction between an
image resource and its immutable Cycles sampler while using only eight bytes
per handle.

The shader performs one descriptor read and dispatches over the fixed `3 x 4`
sampler product. Therefore generated code is independent of the number of
images and materials in the scene. The shape regression proves the complete
body contains exactly

```text
4 * (1 closest sample + 1 linear sample + 4 cubic samples) = 24
```

native texture sample instructions and zero explicit texel reads. Runtime
execution additionally compares every interpolation/extension combination
through the copied SVM image node against the canonical sampler helper.

Scene construction now rejects an SVM image handle whose resource identity is
absent from the snapshot or cannot fit Luisa's 32-bit bindless address. It
does not silently bind an unrelated texture or add a per-sample bounds check
to valid compiled programs.

## Validation

Build command:

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_cycles_svm_image_tests psycles_luisa_runtime
```

The build completed successfully.

Backend command:

```sh
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j1 \
  -R '^psycles\.luisa_cycles_svm_image_(fallback|hip|vk)$'
```

Result:

```text
psycles.luisa_cycles_svm_image_fallback  Passed  0.20 sec
psycles.luisa_cycles_svm_image_hip       Passed  1.55 sec
psycles.luisa_cycles_svm_image_vk        Passed  0.25 sec
100% tests passed, 0 tests failed out of 3
```

Full regression command:

```sh
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 540
Total Test time (real) = 152.09 sec
```

No full-scene image or performance claim is made at this checkpoint because
the production surface kernel still uses the previous surface population
component.
