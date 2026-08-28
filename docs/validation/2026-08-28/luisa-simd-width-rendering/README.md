# Luisa SIMD width rendering benchmark

Date: 2026-08-28

## Outcome

There is no single best SIMD width on this Apple M1 Max. W4 is the only
width that reliably improves the ordinary Cornell-box path tracer, reaching
1.259x fallback throughput. W8 is best for the arithmetic-heavy SpaceX
shader at 5.135x. The cutout ray-query path remains faster on fallback at
every SIMD width, and W4 only ties fallback on the voxel renderer.

The benchmark exposed and fixed a Luisa SIMD correctness bug at W16. Embree
4.4.1 reports native Ray4 and Ray8 packets on this host but no native Ray16.
Its scalar `N=1` filter callback previously reached a Luisa callback that only
accepted 4/8/16 and aborted. Luisa commit `6af205704` accepts the scalar
state-backed callback and disables packet-layout-dependent direct/output-only
providers when the requested packet width is not native.

## Environment

- Host: Apple M1 Max, 10 CPU cores (8 performance + 2 efficiency), 64 GiB
- OS: macOS 26.6.2 (25G83), arm64
- Build: CMake Release, AppleClang 21, LLVM 22.1.8, Embree 4.4.1
- Luisa branch/commit: `next` / `6af205704`
- Backends: product-default fallback and SIMD with 10 workers
- SIMD widths: `LUISA_SIMD_WARP_WIDTH=1,2,4,8,16`
- SIMD worker count: `LUISA_SIMD_WORKER_COUNT=10`

Each workload used six balanced Williams-order rounds, for 36 independent
processes per workload and 144 measured renders in total. Speedup is paired
within each round and reported as the geometric mean with a log-space
Student-t 95% confidence interval (n=6, df=5). Shader JIT, resource upload,
and output readback are outside each application's reported timing window.

## Results

### Cornell-box path tracing

Built-in executable: `example_path_tracing`, 1024x1024, 32 spp, one synced
32-spp dispatch.

| Backend | Median FPS | Paired speedup | 95% CI | Wins |
|---|---:|---:|---:|---:|
| fallback | 14.618 | 1.000x | - | - |
| SIMD W1 | 13.972 | 1.032x | 0.895-1.190 | 3/6 |
| SIMD W2 | 12.780 | 0.909x | 0.842-0.981 | 0/6 |
| SIMD W4 | 18.135 | **1.259x** | **1.117-1.420** | 6/6 |
| SIMD W8 | 14.597 | 0.993x | 0.773-1.276 | 2/6 |
| SIMD W16 | 12.684 | 0.912x | 0.766-1.086 | 1/6 |

Only W4 has a confidence interval wholly above 1.0. The built-in reference
comparison passed for every width: W1 31.622 dB, W2 36.526 dB, W4 35.558 dB,
W8 34.546 dB, and W16 31.622 dB.

### Cutout ray-query path tracing

Built-in executable: `example_path_tracing_cutout`, 1024x1024, 64 spp,
strictly one spp per dispatch, `cutout-query` trace mode.

| Backend | Median FPS | Paired speedup | 95% CI | Wins |
|---|---:|---:|---:|---:|
| fallback | 13.083 | 1.000x | - | - |
| SIMD W1 | 9.931 | 0.764x | 0.714-0.818 | 0/6 |
| SIMD W2 | 9.270 | 0.713x | 0.671-0.757 | 0/6 |
| SIMD W4 | 10.857 | **0.838x** | 0.754-0.930 | 0/6 |
| SIMD W8 | 7.886 | 0.613x | 0.586-0.641 | 0/6 |
| SIMD W16 | 5.745 | 0.431x | 0.389-0.477 | 0/6 |

W4 is the least slow SIMD choice, but fallback is still 1.194x faster than
W4. W16 pays for the lack of native Ray16 packet support and the required
safe provider path. All comparisons passed: W1 41.467 dB, W2 46.831 dB,
W4 46.183 dB, W8 45.491 dB, and W16 42.486 dB.

### Voxel ray tracer

Built-in executable: `example_voxel_raytracer`, 1024x1024, 64 steady-state
iterations per process.

| Backend | Median ms/frame | Paired speedup | 95% CI | Wins |
|---|---:|---:|---:|---:|
| fallback | 30.038 | 1.000x | - | - |
| SIMD W1 | 43.026 | 0.707x | 0.652-0.767 | 0/6 |
| SIMD W2 | 61.605 | 0.483x | 0.459-0.507 | 0/6 |
| SIMD W4 | 29.494 | **1.005x** | 0.935-1.080 | 3/6 |
| SIMD W8 | 39.304 | 0.770x | 0.736-0.806 | 0/6 |
| SIMD W16 | 54.707 | 0.550x | 0.529-0.571 | 0/6 |

W4 is statistically tied with fallback; no width shows a reliable gain. All
SIMD widths produced the same image and measured 45.388 dB versus fallback.

### SpaceX shader toy

Built-in executable: `example_shader_toy_spacex`, 1280x720, four
steady-state iterations per process.

| Backend | Median ms/frame | Paired speedup | 95% CI | Wins |
|---|---:|---:|---:|---:|
| fallback | 535.580 | 1.000x | - | - |
| SIMD W1 | 504.477 | 1.010x | 0.939-1.087 | 3/6 |
| SIMD W2 | 234.379 | 2.180x | 2.033-2.338 | 6/6 |
| SIMD W4 | 139.256 | 3.808x | 3.671-3.950 | 6/6 |
| SIMD W8 | 99.951 | **5.135x** | **4.818-5.474** | 6/6 |
| SIMD W16 | 102.917 | 5.044x | 4.465-5.698 | 6/6 |

W8 is the best measured width, though W8 and W16 overlap statistically. All
SIMD widths measured 70.207 dB versus fallback.

## Validation

- `test_simd_embree_packet_support`: passed
- `test_simd_accel` and all nine registered accel oracle variants: 10/10
  passed, including W16 sparse empty/direct-output packet cases
- Final W16 cutout render: 1024x1024, 64 spp, built-in comparison passed at
  42.486 dB
- Formatting check: LLVM 22 `clang-format --dry-run --Werror` passed on all
  changed C++ sources
- Generated benchmark PNGs and transient logs were intentionally not staged

Machine-readable paired results are recorded in `benchmark-summary.json`.
