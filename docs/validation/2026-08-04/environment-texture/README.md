# Environment Texture and strict Vulkan acos validation

## Outcome

Psycles now lowers Blender/Cycles Environment Texture as a raw typed graph
node. No world texture, material, closure, or direction lookup is pre-baked.
Both Cycles projection modes and all four exposed interpolation modes execute
inside the Luisa shader and match the current Cycles oracle on fallback, HIP,
and Vulkan.

The authoritative oracle was Blender 5.3.0 Alpha/Cycles commit
`b82c3f0da6c1` (2026-07-31 build). The device was an AMD Radeon RX 9070 XT
(`gfx1201`, RADV for Vulkan), and the CPU golden used an AMD Ryzen 9 9950X3D.
Every image below is a linear-pass Cycles/Psycles/difference triptych generated
from the original Blender node graph at 64×64 and 4 spp.

## Formal implementation contract

- The adapter preserves the Environment Texture node, image resource,
  interpolation, projection, and color-space metadata. It never asks Blender
  or Cycles to bake a texture or material.
- An explicit Vector link remains a typed graph edge. An unlinked Vector uses
  Geometry Position converted to Vector; the world shading point stores the
  ray direction, matching Cycles' `LINK_POSITION` background convention.
- Equirectangular projection uses Cycles' normalized direction formula;
  mirror-ball uses the same shifted-Y divisor construction.
- Environment images always use repeat addressing. Closest, Linear, Cubic,
  and Smart reuse the explicit Cycles texel interpolator; sRGB decode occurs
  after filtering.
- At `+Z` and `-Z`, spherical azimuth is not mathematically unique. Cycles
  CPU/HIP inherit `atan2(0, 0) == 0`, while GLSL.std.450 leaves it undefined
  and RADV returned approximately `3π/4`. The shared Luisa spherical primitive
  now chooses the canonical zero azimuth before any projection consumer uses
  it. This is a domain convention, not a direction-specific patch.

## Regression probes

`environment_texture_projection_modes` is a 4×4 matrix containing both
projections over zero, `±X`, `±Y`, `±Z`, and an oblique `(2,-3,4)` direction.
Its 7×5 Non-Color texture makes every wrong meridian or pole row visible.

| Backend | Combined RMSE | Relative RMSE | Luminance ratio | Max abs error | Render | Cold shader JIT |
|---|---:|---:|---:|---:|---:|---:|
| fallback | 4.770699e-8 | 7.507643e-8 | 1.000000000 | 1.788139e-7 | 4.777 ms | 1.342 s |
| HIP | 4.672728e-8 | 7.353465e-8 | 1.000000000 | 1.788139e-7 | 2.195 ms | 5.563 s |
| Vulkan | 9.414500e-8 | 1.481559e-7 | 0.999999897 | 2.980232e-7 | 4.737 ms | 3.185 s |

Before the pole fix, Vulkan alone had Combined RMSE `4.166216e-2`, relative
RMSE `6.556368e-2`, luminance ratio `0.9672513`, and maximum error `0.195588`.
Per-cell analysis isolated the entire error to the equirectangular `±Z` cells:
Vulkan's undefined `atan2(0,0)` selected `u=0.125`, while Cycles selected the
canonical `u=0.5`.

`environment_texture_sampling_modes` is a 4×2 matrix containing
Equirectangular/Mirror Ball crossed with Closest/Linear/Cubic/Smart. A 5×4
sRGB RGBA texture with nontrivial stored alpha exercises filtering and color
decode without relying on an output Blender does not expose.

| Backend | Combined RMSE | Relative RMSE | Luminance ratio | Max abs error | Render | Cold shader JIT |
|---|---:|---:|---:|---:|---:|---:|
| fallback | 3.940012e-8 | 4.163783e-7 | 0.999999828 | 1.043081e-7 | 4.745 ms | 2.717 s |
| HIP | 2.996332e-8 | 3.166507e-7 | 0.999999743 | 5.960464e-8 | 2.267 ms | 8.569 s |
| Vulkan | 3.047992e-8 | 3.221101e-7 | 0.999999743 | 5.960464e-8 | 4.923 ms | 6.158 s |

`environment_texture_world_default` connects an Environment Texture directly
to a raw Blender World graph and deliberately leaves Vector unlinked. A
perspective camera therefore exercises the complete camera-ray, world-space
direction, implicit `LINK_POSITION`, projection, texture, and Env-pass path.

| Backend | Combined/Env RMSE | Relative RMSE | Luminance ratio | Max abs error | Render | Cold shader JIT |
|---|---:|---:|---:|---:|---:|---:|
| fallback | 9.792797e-8 | 1.971414e-7 | 1.000000000 | 3.516674e-6 | 4.098 ms | 1.033 s |
| HIP | 1.104564e-7 | 2.223629e-7 | 1.000000000 | 3.933907e-6 | 1.977 ms | 2.880 s |
| Vulkan | 4.119836e-7 | 8.293753e-7 | 0.999999937 | 5.483627e-6 | 2.380 ms | 2.246 s |

The runner enforces a `0.99999–1.00001` luminance ratio and relative RMSE at
most `1e-5`. The material probes validate Combined and Emit; the world probe
validates Combined and Env. All nine runs pass. Normal is exact and all other
light-component passes are zero in both renderers.

## Visual inspection

All nine original-resolution triptychs were opened and inspected. The Cycles
and Psycles panels have identical cell boundaries, colors, projection choices,
interpolation results, world-ray orientation, and environment seam placement.
Difference panels show only float-rounding patterns after automatic
amplification between roughly 643 thousand and 15.1 million times; there is no
coherent visual mismatch.

Projection matrices:

- [fallback triptych](projection-fallback-triptych.png)
- [HIP triptych](projection-hip-triptych.png)
- [Vulkan triptych](projection-vk-triptych.png)

Sampling matrices:

- [fallback triptych](sampling-fallback-triptych.png)
- [HIP triptych](sampling-hip-triptych.png)
- [Vulkan triptych](sampling-vk-triptych.png)

Implicit world Vector:

- [fallback triptych](world-fallback-triptych.png)
- [HIP triptych](world-hip-triptych.png)
- [Vulkan triptych](world-vk-triptych.png)

Machine-readable reports are stored beside this document as
`projection-{fallback,hip,vk}-report.json` and
`sampling-{fallback,hip,vk}-report.json`, plus
`world-{fallback,hip,vk}-report.json`.

## Luisa Vulkan backend regression

The Environment pole defect itself was not caused by `acos`; however, the
isolation work exposed a separate strict-math defect. RADV's native
GLSL.std.450 `Acos` missed the C++ float result by roughly `2.7e-5–7e-5` at
representative interior inputs even with fast math disabled. Luisa `next`
commit `0cadf7767` now implements cancellation-free float32 strict `acos` for
both native SPIR-V and HLSL-to-SPIR-V routes by range-reducing through the
existing strict `asin` polynomial. The regression covers nine values from
`+1` through `-1`; disabling the new path makes four interior assertions fail,
while the final implementation passes all 338 device-math assertions on VK.

## Commands

```text
cmake --build build -j32
ctest --test-dir build -R psycles\\.luisa_background_sampling_ -j3 --output-on-failure
python3 tools/run_cycles_shader_probes.py environment_texture_projection_modes ... --backend fallback|hip|vk --width 64 --height 64 --samples 4
python3 tools/run_cycles_shader_probes.py environment_texture_sampling_modes ... --backend fallback|hip|vk --width 64 --height 64 --samples 4
python3 tools/run_cycles_shader_probes.py environment_texture_world_default ... --backend fallback|hip|vk --width 64 --height 64 --samples 4
cmake --build build-luisa-tests -j32 --target test_device_math luisa-compute-backend-vk
(cd build-luisa-tests/bin && ./test_device_math vk)
```
