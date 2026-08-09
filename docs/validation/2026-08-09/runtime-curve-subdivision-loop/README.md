# Runtime curve-subdivision loop

## Problem

The software Cycles ribbon intersection receives the curve subdivision level
as device scene data, but previously recorded sixteen copies of the interval
body on the C++ host and guarded every copy with `interval < interval_count`.
Level zero and level four therefore paid for the same sixteen-body AST. This
was a real scene-data-driven unroll and contributed the full ribbon
intersection body repeatedly to path-kernel code size.

## Fix and invariant

`CurveRibbonComponent` now clamps the authored level to the existing
Cycles-compatible maximum and executes exactly `1 << level` intervals with a
Luisa device loop. The Catmull-Rom control points, width interpolation and
strict `2r / |D|` self-intersection rule are unchanged.

The regression constructs a kernel with the subdivision level loaded from a
device buffer, translates its complete AST to XIR and requires exactly one
`LoopInst`/`SimpleLoopInst`. This structural check fails if the interval body
is host-unrolled again; the existing numeric and ray-query cases continue to
check hit distance, curve coordinates, object transforms and self exclusion.

## Validation

The target was built with all local build threads:

```text
cmake --build build --target psycles_luisa_curve_ribbon_tests --parallel 32
```

The numeric and XIR-shape regression passed on:

- fallback (Embree, 32 CPU threads);
- HIP on AMD Radeon RX 9070 XT;
- Vulkan on AMD Radeon RX 9070 XT with
  `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`.

The Vulkan run compiled native SPIR-V (`6145 -> 5720` words after the compute
optimization preset) and did not load DXC. A cold HIP test compile generated a
42,224-byte intermediate AMDGPU program and linked a 25,544-byte code object.

This checkpoint changes code shape, not the shading or transport equations,
so it requires no Cycles image oracle or triptych.
