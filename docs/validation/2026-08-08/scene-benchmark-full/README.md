# Full-scene benchmarking and performance profiling (2026-08-08)

This pass completes non-Vulkan full-scene runs in the current branch on RX 9070 XT
with Blender 5.3.0-alpha, fixed seed, 640×480+ resolutions, 16/64 spp, fixed
`spp_count`, adaptive off, and no denoising.

Command template:

```bash
TMPDIR=/tmp/psycles-benchmark-tmp python3 tools/run_scene_benchmark.py \
  --blender /home/mike/Projects/blender-install-4fe17ef6/blender \
  --psycles-render /home/mike/Projects/Psycles/build/bin/psycles_render_blender_scene \
  --cycles-hip-device-name "Radeon RX 9070 XT" \
  --width 640 --height 480 --samples 64 \
  --psycles-backends fallback,hip \
  --compiler-tmp /tmp/psycles-benchmark-tmp \
  --output-dir ...
```

## Completed results

`Vulkan` in this batch failed to complete and is excluded from the tables below.
Completed manifests are:

- `/home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp-novk/benchmark.json`
- `/home/mike/Projects/psycles-benchmarks/classroom-480p-64spp-novk/benchmark.json`
- `/home/mike/Projects/psycles-benchmarks/monster-480p-16spp/benchmark.json`
- `/home/mike/Projects/psycles-benchmarks/monster-480p-64spp/benchmark.json`
- `/home/mike/Projects/psycles-benchmarks/monster-640x360-16spp/benchmark.json`

### Scene speed table (render seconds)

| Scene | Resolution | spp | Cycles CPU (s) | Cycles HIP (s) | Psycles fallback render (s) | Psycles hip render (s) | Fallback vs HIP speedup | HIP vs HIP speedup | fallback compile/jit (s) | hip compile/jit (s) | Combined RMSE (F/H) | Luma ratio (F/H) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| barbershop-480p-64spp-novk | 640x480 | 64 | 14.309 | 15.006 | 98.226 | 13.407 | 0.15x | 1.12x | 17.810 / 3.619 | 29.499 / 3.400 | 0.0390/0.0410 | 1.0458/1.0377 |
| classroom-480p-64spp-novk | 640x480 | 64 | 3.598 | 1.356 | 9.508 | 5.240 | 0.14x | 0.26x | 0.961 / 0.525 | 2.320 / 0.584 | 0.0080/0.0079 | 0.9992/0.9992 |
| monster-480p-16spp | 640x480 | 16 | 1.515 | 1.248 | 12.708 | 2.432 | 0.10x | 0.51x | 1.333 / 0.566 | 1.721 / 0.548 | 0.0472/0.0472 | 1.0030/1.0030 |
| monster-480p-64spp | 640x480 | 64 | 4.205 | 2.509 | 51.053 | 9.527 | 0.05x | 0.26x | 1.349 / 0.561 | 1.708 / 0.545 | 0.0225/0.0225 | 1.0028/1.0028 |
| monster-640x360-16spp | 640x360 | 16 | 1.269 | 1.168 | 9.275 | 1.888 | 0.13x | 0.62x | 1.340 / 1183.640 | 1.718 / 0.548 | 0.0453/0.0453 | 1.0034/1.0033 |

### Visual validation

Combined triptychs exist for each completed compare under each run's
`comparisons/<case>/triptychs/combined.png`.
Example: barbershop fallback-vs-cycles-hip combined triptych at:
`/home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp-novk/comparisons/psycles-fallback-vs-cycles-hip/triptychs/combined.png`.

## Vulkan status and profiler notes

The failed Vulkan matrix entries remain unresolved. The most recent known-failing
runs still terminate early in `luisa.backends.vulkan` while creating Vulkan module
and before image write-out. The current recommendation is to keep Vulkan deferred
until the exact-XIR path and alpha/opacity-control flow are stabilized.

No additional profiler traces were collected in this pass; all time figures are
from benchmark wall/render/compile/JIT boundaries emitted by
`tools/run_scene_benchmark.py`.

