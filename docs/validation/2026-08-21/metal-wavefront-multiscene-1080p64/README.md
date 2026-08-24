# Metal graph-wavefront multi-scene benchmark

## Scope

This checkpoint compares Psycles graph-wavefront against Blender Cycles on
the same Apple M1 Max Metal device. Every formal row is a fresh Blender 5.2.0
LTS export and a 1920x1080, 64-sample render. Psycles dispatches one sample per
launch and runs with:

```text
PSYCLES_COMPACT_SURFACE_VALUES=1
PSYCLES_POPULATE_SURFACE_ONCE=1
```

The Lone Monk and Monster render kernels use `main` at `9c92b24`; the later
`d93c011` change only removes an unconditional displacement debug print and
does not change rendered values. All rows use the LuisaCompute submodule
`next` at `9c5ae75dc`. The benchmark runner records the complete scheduler
command, environment, output hashes, render-only and process-wall timings,
and all-pass comparisons in each scene's `benchmark.json`.

## Scheduler policy

The selected Classroom winner is used unchanged across scenes:

| Setting | Value |
| --- | ---: |
| Scheduler | `wavefront-graph` |
| Execution block | 32 |
| Frame capacity | 1,048,576 |
| Graph workers | 131,072 |
| Selective scheduling | enabled |
| Refill threshold | 0 |
| Counter readback batch / depth | 1 / 1 |
| Tail megakernel threshold | 4,096 |
| Fast math | enabled |

`frame_buffer_compaction` remains a host-side structural specialization: it
changes frame layout and control flow. Luisa's regression suite requires
compact and sparse schedulers to produce different shader hashes. Capacity
and the runtime graph policy above are passed as kernel arguments where they
do not change kernel structure.

## Performance

The primary comparison is renderer-reported render-only time. Psycles scene
compilation, shader JIT, and total process wall are reported separately and
are not folded into throughput.

| Scene | Cycles Metal | Psycles graph | Slowdown | Scene compile | Shader JIT | Psycles wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lone Monk | 15.945 s | 203.364 s | 12.754x | 2.887 s | 134.115 s | 345.109 s |
| Monster Under the Bed | 21.915 s | 203.652 s | 9.293x | 2.413 s | 278.250 s | 487.081 s |

Lone Monk artifacts are under
`build-macos-shader-translation-plan/benchmarks/2026-08-21/lone-monk-1080p64-compact-metal`.
Monster artifacts are under
`build-macos-shader-translation-plan/benchmarks/2026-08-21/monster-under-the-bed-1080p64-compact-metal`.

## Image validation

Lone Monk Combined has no invalid pixels. Its Psycles/Cycles luminance mean
ratio is `1.000015`, RMSE is `0.032046`, relative RMSE is `0.017137`, and the
99th-percentile per-pixel RMSE is `0.124720`. Original-resolution inspection
shows matching scene orientation, materials, lighting, and exposure; the
amplified difference panel is dominated by finite-sample noise and high-
contrast edges.

Monster Under the Bed also has no invalid pixels. Its luminance mean ratio is
`1.003331`, RMSE is `0.021641`, relative RMSE is `0.135429`, and the 99th-
percentile per-pixel RMSE is `0.089521`. The relative metric is amplified by
the deliberately dark scene and high-variance 64-sample BSSRDF/direct-light
paths. Original-resolution inspection shows the same monster skin, child,
bed, geometry, and lighting structure. Diffuse and glossy color passes agree
closely (`0.058%` and `0.084%` relative RMSE respectively), while the noisy
direct and indirect lighting passes dominate the Combined difference.

The remaining formal row is Barbershop Interior. A bounded 1-spp staged
wavefront diagnosis, including the Metal radix visibility fix and sub-kernel
ranking, is recorded in
`docs/validation/2026-08-24/metal-wavefront-barbershop-profile/README.md`.
