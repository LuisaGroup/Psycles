# Reachable-subsurface path-kernel specialization

Date: 2026-08-09

## Invariant

Spatial BSSRDF transport is part of the path kernel if and only if a surface
primitive that can be reached through the scene acceleration structure can
resolve to a material whose original `SurfaceProgram` allocates a BSSRDF.
This is a host/JIT capability predicate; it does not replace, bake, or
approximate any material closure.

For an instance `i`, primitive `p`, and authored material slot `s(p)`, the
resolved material is

```text
R(i, p) = override_i[s(p)]                         if s(p) < |override_i|
          geometry[min(s(p), |geometry| - 1)]      otherwise
```

which is the same override-first, last-slot-clamped rule used by the device
primitive-material resolver. Triangle primitives and valid curve segments are
enumerated exactly. Uninstanced geometry, empty meshes, degenerate one-key
curves, world shaders, and analytic-light shaders cannot enter surface
transport and are excluded.

The resulting scene capability controls construction of
`SubsurfaceTransportStage`. A scene without a reachable BSSRDF therefore does
not record the random-walk, local-intersection, and spatial-exit AST merely so
a runtime condition can remain permanently false. A scene with a reachable
BSSRDF records the unchanged transport implementation.

## Cold Vulkan code-shape comparison

The production linked-normal NEE fixture contains no BSSRDF. It was compiled
twice on the same AMD Radeon RX 9070 XT with the shader cache disabled. The
specialized run uses the scene-derived capability. For the control run only,
the capability was temporarily forced to `true`, reproducing the previous
unconditional path-kernel shape; that diagnostic edit was then reverted.

```sh
/usr/bin/time -f 'wall_seconds=%e max_rss_kib=%M' \
  env PSYCLES_DISABLE_SHADER_CACHE=1 \
      LUISA_VULKAN_PROFILE_COMPILATION=1 \
  build/bin/psycles_luisa_surface_nee_normal_tests vk
```

| Metric | Unconditional transport | Reachability-specialized | Change |
| --- | ---: | ---: | ---: |
| AST to XIR | 29.371 s | 23.364 s | -20.45% |
| XIR/SPIR-V handoff validation | 9.712 s | 8.002 s | -17.60% |
| Raw SPIR-V | 363,915 words | 330,026 words | -9.31% |
| Optimized SPIR-V | 343,130 words | 310,916 words | -9.39% |
| Vulkan pipeline creation | 3.728 s | 2.531 s | -32.11% |
| End-to-end cold process | 44.09 s | 35.05 s | -20.50% |
| Peak RSS | 767,300 KiB | 646,600 KiB | -15.73% |

The reduction appears before backend dead-code elimination and persists in the
optimized module. This demonstrates removal of an unused graph at the
multistage construction boundary rather than an optimizer-specific shortcut.
The fixture is intentionally a code-shape isolation test; these timings are
not presented as a complete-scene render benchmark.

## Regression coverage

`psycles.cycles_instance_support` pins material reachability for triangle and
curve instances, including partial overrides, out-of-range slot clamping,
uninstanced geometry, empty geometry, and invalid one-key curves.

`psycles.luisa_surface_nee_normal` also compiles two capability controls on
every enabled backend: an unreferenced BSSRDF must leave transport disabled,
while replacing the primitive's resolved material with the same BSSRDF must
enable it. The original non-BSSRDF scene then compiles and executes the full
production path kernel.

The focused production path fixture and all existing BSSRDF transport tests
pass after the specialization:

```text
surface_nee_normal:             fallback / HIP / Vulkan
subsurface_exit:                fallback / HIP / Vulkan
random_walk:                    fallback / HIP / Vulkan
surface_closure_collection:     fallback / HIP / Vulkan
principled_thin_wall:           fallback / HIP / Vulkan
```

Seventeen focused CTest entries, including the host reachability and shader
probe contracts, passed in 36.73 seconds. The BSSRDF-positive tests prove that
the specialization does not remove transport when the scene capability is
present.
