# Psycles

Psycles is a Cycles-compatible renderer built on LuisaCompute's multistage C++ DSL, XIR compiler, and runtime JIT. It implements Blender Cycles 5.2/5.3's Shader Virtual Machine (SVM), closure state machine, geometry/attribute contracts, and sampling/MIS pipeline natively across GPU (`hip`, `dx`, `vk`, `metal`, `metal4`) and CPU (`fallback`, `simd`) backends.

All material and scene evaluation domains — **surface, background, light emission, volume, and true/bump displacement** — compile and execute natively through Cycles SVM (`DirectJitTranslator` + bytecode VM fallback) with no legacy `SurfaceProgram` or separate CPU reference renderer.

## Documentation Map

- **Usage & Build:** [BUILD.md](BUILD.md) — prerequisites, CMake options, render CLIs, shader caching, test suites, troubleshooting
- **System Design:** [docs/architecture.md](docs/architecture.md) — end-to-end data flow, scene/graph contracts, SVM runtime, transport & coroutine scheduling
- **Optimization & Benchmarks:** [docs/optimization.md](docs/optimization.md) — DirectJit SVM, SSA stack elimination, wavefront coroutine splitting, surface sorting, backend tuning, and full 6-backend benchmark tables
- **Cycles Compatibility:** [docs/cycles-compatibility.md](docs/cycles-compatibility.md) — supported domains, specialization invariants, known feature gaps, verification policy
- **Benchmark & Diagnostics:**
  - [docs/scene-benchmark.md](docs/scene-benchmark.md) — equal-pass Blender Cycles vs Psycles multi-backend benchmark runner
  - [docs/cycles-path-trace.md](docs/cycles-path-trace.md) — per-path 344-AOV RNG/state differential oracle
- **Contributor Guide:** [DEVELOP.md](DEVELOP.md) — codebase layout, Cycles semantic/performance policies, CI and regression gates

## Quick Start

### Clone, Build, and Test

```bash
git clone --recurse-submodules https://github.com/LuisaGroup/Psycles.git
cd Psycles
cmake -S . -B build -G Ninja
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure -j32
```

Psycles pins `third_party/LuisaCompute` (`next@db53c6b76`) as a CMake subdirectory. Useful CMake switches:

- `-DPSYCLES_ENABLE_LUISA_SIMD=ON` to build the packet-vectorized CPU SIMD backend alongside `fallback`
- `-DLUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=OFF` to build the DXC-based Vulkan path instead of native XIR-to-SPIR-V
- `-DPSYCLES_ENABLE_LUISA=OFF` for a dependency-free contract/compiler-only build

See [BUILD.md](BUILD.md) for platform packages, GPU SDK setup, and selective CTest targets.

### Render

Built-in smoke demo:

```bash
./build/bin/psycles_luisa_render_demo output.ppm fallback 640 400 256
```

Normalized Blender scene package on any enabled backend (`hip`, `dx`, `vk`, `metal`, `fallback`, `simd`):

```bash
./build/bin/psycles_render_blender_scene /path/to/export-package output.ppm vk 1920 1080 256
```

To force the GPU `wavefront-staged` scheduler directly from the CLI, pass the trailing scheduler selection arguments documented in [BUILD.md](BUILD.md#render-clis). On multi-GPU AMD systems with an integrated GPU at index 0, set `HIP_VISIBLE_DEVICES=1` for `hip`.

## Verified Six-Backend Performance Summary

Measured on `LuisaCompute next@db53c6b76` (AMD Ryzen 9 9950X3D + AMD Radeon RX 9070 XT) against official **Blender Cycles 5.3.0** across all 5 canonical scenes with **0 invalid pixels** on all 30 `(scene, backend)` combinations:

### GPU — 256 spp (`wavefront-staged`, Warm `r1`) vs Cycles 5.3.0 HIPRT

| Scene | Cycles 5.3.0 HIPRT | Psycles `hip` | Psycles `dx` | Psycles `vk-dxc` | Psycles `vk-xir` | Best Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 3.238 s | 1.859 s *(1.74×)* | **1.019 s** *(3.18×)* | 1.229 s *(2.64×)* | 1.037 s *(3.12×)* | **3.18× (`dx`)** |
| `monster` (1024×1024) | 13.348 s | 7.905 s *(1.69×)* | 5.827 s *(2.29×)* | 6.306 s *(2.12×)* | **5.356 s** *(2.49×)* | **2.49× (`vk-xir`)** |
| `lone-monk` (1440×1080) | 13.315 s | 11.586 s *(1.15×)* | 4.826 s *(2.76×)* | 5.384 s *(2.47×)* | **4.665 s** *(2.85×)* | **2.85× (`vk-xir`)** |
| `classroom` (1920×1080) | 17.046 s | 13.273 s *(1.28×)* | 7.538 s *(2.26×)* | 8.004 s *(2.13×)* | **7.137 s** *(2.39×)* | **2.39× (`vk-xir`)** |
| `barbershop` (2048×858) | 24.368 s | 22.366 s *(1.09×)* | **16.912 s** *(1.44×)* | 17.091 s *(1.43×)* | 18.433 s *(1.32×)* | **1.44× (`dx`)** |

### CPU — 64 spp (`megakernel`, Idle-Host Medians) vs Cycles 5.3.0 CPU

| Scene | Cycles 5.3.0 CPU | Psycles `fallback` | Psycles `simd` (W8) | Best Speedup |
| --- | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 2.894 s | 2.277 s *(1.27×)* | **2.127 s** *(1.36×)* | **1.36× (`simd`)** |
| `monster` (1024×1024) | 12.686 s | **10.307 s** *(1.23×)* | 10.956 s *(1.16×)* | **1.23× (`fallback`)** |
| `lone-monk` (1440×1080) | 22.948 s | 16.096 s *(1.43×)* | **15.445 s** *(1.49×)* | **1.49× (`simd`)** |
| `classroom` (1920×1080) | 24.498 s | 20.648 s *(1.19×)* | **19.814 s** *(1.24×)* | **1.24× (`simd`)** |
| `barbershop` (2048×858) | 37.492 s | **29.663 s** *(1.26×)* | 34.093 s *(1.10×)* | **1.26× (`fallback`)** |

Full cold/warm JIT timings, RMSE parity tables, and compiler/scheduler strategy breakdowns are in [docs/optimization.md](docs/optimization.md). Unsupported shader/geometry features fail cleanly with explicit compiler diagnostics rather than silent approximations; see [docs/cycles-compatibility.md](docs/cycles-compatibility.md).
