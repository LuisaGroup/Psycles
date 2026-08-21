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

The Psycles runs use `main` at `9c92b24` and the LuisaCompute submodule `next`
at `9c5ae75dc`. The benchmark runner records the complete scheduler command,
environment, output hashes, render-only and process-wall timings, and all-pass
comparisons in each scene's `benchmark.json`.

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

Lone Monk artifacts are under
`build-macos-shader-translation-plan/benchmarks/2026-08-21/lone-monk-1080p64-compact-metal`.

## Image validation

Lone Monk Combined has no invalid pixels. Its Psycles/Cycles luminance mean
ratio is `1.000015`, RMSE is `0.032046`, relative RMSE is `0.017137`, and the
99th-percentile per-pixel RMSE is `0.124720`. Original-resolution inspection
shows matching scene orientation, materials, lighting, and exposure; the
amplified difference panel is dominated by finite-sample noise and high-
contrast edges.

The remaining formal rows are Monster Under the Bed and Barbershop Interior.
