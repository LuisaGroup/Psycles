# Wave Texture validation

## Outcome

Psycles now imports Blender's original Wave Texture node and evaluates it in
the generated Luisa shader. Nothing is baked by Blender or Cycles. Both Wave
types, every exposed direction and profile, both outputs, the six dynamic
inputs, implicit Generated coordinates, and Cycles' distortion fBm match the
current official Cycles oracle on fallback, HIP, and Vulkan.

This work was selected directly from the official Barbershop Interior asset:
`assets/official-blender-scenes/barbershop-interior/barbershop_interior.blend`
contains 14 Wave nodes. Its most demanding repeated configuration is Bands /
Diagonal / Sin with Scale `12`, Distortion `4.855`, Detail `16`, Detail Scale
`2.1`, Detail Roughness `0.5`, and Phase Offset `pi/2`. That exact case is in
the regression matrix. The asset is the unmodified 287,574,804-byte file from
the URL supplied for this project, with SHA-256
`95972b56180462cac47ec82f3a755bd9111ec18ca37a6196a319c013db994130`.

The source oracle was current Blender/Cycles main `4671e6977336`. Wave and
fractal-noise sources are unchanged from the locally built Blender 5.3 Alpha
`b82c3f0da6c1`, which rendered every golden EXR. No Psycles CPU reference
model exists or was introduced; "Cycles CPU" below means the official Cycles
device and remains the single image oracle.

## Formal implementation contract

- The adapter retains the raw node, links, socket defaults, Wave type, Bands
  direction, Rings direction, profile, and Color/Factor output identity.
- Static properties form a finite configuration encoded in the typed
  `SurfaceProgram`. During Luisa AST construction, the OOP `WaveValueNode`
  expands that configuration into typed branches; no weakly typed runtime
  property switch or `float4` parameter edge is introduced.
- Color and Factor are specialized at graph-lowering time. Only the requested
  typed output instruction is emitted, so the Perlin/fBm AST is not evaluated
  twice for one Blender output edge.
- Vector, Scale, Distortion, Detail, Detail Scale, Detail Roughness, and Phase
  Offset remain dynamic graph expressions. An unlinked Vector uses the same
  Generated-coordinate edge as Cycles.
- The device expression follows Cycles' unit-coordinate precision correction,
  Bands/Rings coordinate reduction, phase addition, normalized signed 3D
  Perlin fBm distortion, and Sin/Saw/Triangle profile equations.
- The distortion implementation is one shared Luisa `Callable`. Material
  branches do not duplicate the full Perlin/fBm AST.
- Cycles defines the fBm loop as `i <= float_to_int(detail)`, with conversion
  truncating toward zero. The implementation derives the general iteration
  count `max(trunc(detail) + 1, 0)` from that definition. A linked Detail of
  `-0.5` with nonzero Detail Scale is a regression case; using `floor` would
  fail it. Normalization also retains Cycles' original division semantics
  instead of adding an unrelated epsilon patch.

## Probe coverage and results

`wave_texture_modes` contains 32 cells: both Wave types, all four directions
for each type, Sin/Saw/Triangle, Color and Factor, signed/zero scale, unit and
large coordinates, phase offsets, and implicit Generated coordinates.

`wave_texture_distortion` contains 16 cells with all six scalar sockets linked
through the original graph. It covers zero/signed/high distortion, fractional
Detail, Detail `16`, the negative-fractional truncation boundary, zero/signed/
large Detail Scale, Detail Roughness endpoints, signed phase, signed Scale,
and alternating Color/Factor outputs.

All runs used 64x64, 4 spp, box filtering, raw Blender graphs, and the same
latest Cycles CPU EXR as oracle. The hard gates are luminance ratio
`0.99999..1.00001`, relative RMSE `1e-5` for modes, and relative RMSE `6e-5`
for distortion. Every recorded pass has zero invalid pixels.

| Probe | Backend | Combined RMSE | Relative RMSE | Luminance ratio | Max error |
|---|---|---:|---:|---:|---:|
| modes | fallback | 1.418827e-6 | 2.499521e-6 | 0.999999870 | 7.450581e-6 |
| modes | HIP | 9.813577e-7 | 1.728839e-6 | 1.000000325 | 5.006790e-6 |
| modes | Vulkan | 1.114092e-6 | 1.962675e-6 | 1.000000065 | 5.006790e-6 |
| distortion | fallback | 3.077049e-5 | 4.709504e-5 | 0.999985898 | 1.220703e-4 |
| distortion | HIP | 8.347943e-6 | 1.277674e-5 | 1.000003789 | 3.051758e-5 |
| distortion | Vulkan | 4.796444e-6 | 7.341082e-6 | 1.000001789 | 1.370907e-5 |

The distortion tolerance is a device floating-point envelope, not permission
for a different algorithm. On this same final graph, official Cycles CPU and
official Cycles HIP differ by RMSE `2.238469e-5`, relative RMSE
`3.426035e-5`, luminance ratio `0.999995475`, and maximum error
`8.559227e-5`. Psycles HIP versus Cycles HIP is within the same envelope
(relative RMSE `3.416173e-5`). The residual is concentrated in high-frequency
Saw/Triangle cells where a few float32 ULPs are amplified by the profile;
there is no missing branch or alternate recurrence.

The final cold mode-matrix shader JITs were 4.44 s for fallback, 4.43 s for
HIP, and 8.46 s for Vulkan. Before requested-output specialization, the same
runs took 4.93 s, 4.55 s, and 10.27 s respectively; Vulkan therefore fell by
17.6% in this direct A/B. Its generated module fell from 254,282 to 247,898
words before optimization and from 219,648 to 213,936 words afterward. The
remaining Vulkan log still identifies the DXC/SPIR-V generation and SPIR-V
optimization chain as the next timing target. These probe numbers are compile
diagnostics, not scene-render throughput claims.

## Visual inspection

I opened all six Cycles/Psycles/difference triptychs and the Cycles CPU/HIP
envelope triptych at original resolution. Cycles and Psycles have identical
cell boundaries, Wave type/direction/profile selection, phase, and distortion
values on every backend. The automatically amplified differences are isolated
float-rounding residuals; there is no shifted band/ring, missing cell,
orientation error, profile discontinuity, or backend-specific structural
mismatch.

Modes:

- [fallback triptych](modes-fallback-triptych.png)
- [HIP triptych](modes-hip-triptych.png)
- [Vulkan triptych](modes-vk-triptych.png)

Distortion and dynamic sockets:

- [fallback triptych](distortion-fallback-triptych.png)
- [HIP triptych](distortion-hip-triptych.png)
- [Vulkan triptych](distortion-vk-triptych.png)

Device envelope:

- [official Cycles CPU/HIP triptych](cycles-cpu-vs-hip-envelope-triptych.png)

Machine-readable reports are stored beside this document as
`modes-{fallback,hip,vk}-report.json`,
`distortion-{fallback,hip,vk}-report.json`, and
`cycles-cpu-vs-hip-envelope-report.json`.

The Release build used all 32 build jobs. The complete project suite passed
`134/134` tests in 7.25 s with three backend lanes, and the source-size gate
passed for 363 first-party files (118,740 lines, 2,000-line per-file limit).
The versioned shader inventory now records 43 `cycles_verified` nodes and 45
complete Cycles nodes including structural roots.

## Commands

```text
cmake --build build -j32
ctest --test-dir build -R '^psycles.contracts$' --output-on-failure
python tools/run_cycles_shader_probes.py wave_texture_modes wave_texture_distortion --blender /path/to/blender --psycles-render build/bin/psycles_render_blender_scene --output-dir /tmp/wave --backend fallback|hip|vk --cycles-device CPU --width 64 --height 64 --samples 4
```

This checkpoint verifies Wave Texture itself. It does not claim complete
Barbershop rendering parity: the full asset still contains other unsupported
or partial node, closure, geometry, and integrator features that remain red in
the scene audit.
