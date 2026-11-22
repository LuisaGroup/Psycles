# Psycles

Psycles is a renderer implementation that treats Blender Cycles 5.2.1 as its
interface, semantic contract, and current SVM execution-model reference, while
expressing device execution through LuisaCompute's multistage C++ DSL and
runtime JIT.

The active milestone is locked to an isomorphic implementation of Cycles'
SVM node stream, stack, program-counter loop, node dispatch, closure state, and
feature masks. Alternative Psycles-specific SVM architectures are forbidden
unless the project owner explicitly changes that requirement. See the
mandatory implementation lock in [DEVELOP.md](DEVELOP.md).

The surrounding project boundaries include:

- a typed, extensible ShaderGraph contract with explicit surface, volume, and
  displacement roots;
- a pre-SVM Cycles graph adapter with explicit node/socket coverage failures;
- the original Cycles SVM shader jump table, typed word payloads and stack ABI;
- scene-derived static node/feature specialization and allocation bounds;
- transactional scene snapshots with stable resource identifiers;
- a Luisa DSL implementation of the Cycles SVM material execution boundary;
- a Luisa device execution path with camera rays, hardware/fallback RayQuery,
  transformed instances, multi-bounce transport, direct-light sampling, film
  accumulation, and pass output;
- direct differential validation against Blender Cycles linear render passes.

See [docs/architecture.md](docs/architecture.md) for the design and current
scope and [docs/cycles-compatibility.md](docs/cycles-compatibility.md) for the
shader-graph path, integrator contract, and explicit compatibility gaps.

## Build

Clone with submodules and build the enabled Luisa backends:

```bash
git clone --recurse-submodules https://github.com/LuisaGroup/Psycles.git
cd Psycles
cmake -S . -B build -G Ninja
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure -j32
```

Psycles pins `third_party/LuisaCompute` (`next@db53c6b76`) as a CMake
subdirectory so normal builds and validation share the exact tested Luisa
revision. To also build the vectorized CPU SIMD backend, add
`-DPSYCLES_ENABLE_LUISA_SIMD=ON`. The commands above use all 32 hardware
threads on the validated 16-core/32-thread workstation; adjust `--parallel` /
`-j` to match your host.

For a dependency-free contract-core build:

```bash
cmake -S . -B build-core -G Ninja -DPSYCLES_ENABLE_LUISA=OFF
```

See [BUILD.md](BUILD.md) for prerequisites (Windows/Linux/macOS, ROCm/HIP,
DirectX 12, Vulkan, LLVM/Embree), backend options, cache behavior, and
troubleshooting. Current implementation status is in
[docs/cycles-compatibility.md](docs/cycles-compatibility.md); development rules
and completion gates are in [DEVELOP.md](DEVELOP.md).

## Render through Luisa

```bash
./build/bin/psycles_luisa_render_demo psycles-luisa.ppm fallback 640 400 256
```

To render an exported Blender scene package with any enabled Luisa backend
(`hip`, `dx`, `vk`, `metal`, `fallback`, or `simd`):

```bash
./build/bin/psycles_render_blender_scene /path/to/export-package output.ppm vk 1920 1080 256
```

All backends execute the exact same Luisa DSL / DirectJit Cycles SVM program;
neither `fallback` nor `simd` is a separate Psycles CPU reference renderer.

Correctness is not inferred from another Psycles implementation. The
regression harness renders the same `.blend` with official Blender Cycles to
linear multilayer EXR and compares those passes directly with Psycles-Luisa
output.

The [scene benchmark](docs/scene-benchmark.md) covers Blender Cycles (CPU and
HIP/Metal) alongside Psycles across GPU (`hip`, `dx`, `vk` native XIR-to-SPIR-V
and `vk-dxc`, `metal`) and CPU (`fallback`, `simd`) backends. It compares
Cycles' main-loop wall time with Psycles' render-only wall time and records
scene compilation, session JIT/PSO initialization, EXR hashes, numeric pass
comparisons, and device-labeled triptychs separately.

When a pass comparison diverges, the
[per-path Cycles oracle](docs/cycles-path-trace.md) compares RNG dimensions,
raw closures, light/BSDF sampling, PDFs, and path-state transitions at the
first mismatching event. Discrete state and random samples are exact gates;
continuous fields have explicit float32 bounds.

## Current execution path & verified performance

Native Cycles SVM (DirectJit + staged wavefront on GPU, megakernel on CPU) is
the default material path across surface, background, sampled/forward light
emission, and volume consumers. The staged GPU path tracer uses Cycles-aligned
coroutine boundaries, a staged direct-light shadow queue, and generic Luisa
scheduler extensions for surface sorting. Local SVM arrays are bounded by
static compiler analysis, not scene profiles.

On **LuisaCompute `next@db53c6b76`** (AMD Ryzen 9 9950X3D + AMD Radeon RX 9070
XT, vs **Blender Cycles 5.3.0**), all five official scenes compile and render
on all six validated backends with **0 invalid pixels** and beat Cycles across
every scene:

### GPU — 256 spp (`wavefront-staged`) vs Blender Cycles 5.3.0 HIPRT

| Scene | Cycles 5.3.0 HIPRT | Psycles `hip` | Psycles `dx` | Psycles `vk-dxc` | Psycles `vk-xir` | Best Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 3.238 s | 1.859 s *(1.74×)* | **1.019 s** *(3.18×)* | 1.229 s *(2.64×)* | 1.037 s *(3.12×)* | **3.18× (`dx`)** |
| `monster` (1024×1024) | 13.348 s | 7.905 s *(1.69×)* | 5.827 s *(2.29×)* | 6.306 s *(2.12×)* | **5.356 s** *(2.49×)* | **2.49× (`vk-xir`)** |
| `lone-monk` (1440×1080) | 13.315 s | 11.586 s *(1.15×)* | 4.826 s *(2.76×)* | 5.384 s *(2.47×)* | **4.665 s** *(2.85×)* | **2.85× (`vk-xir`)** |
| `classroom` (1920×1080) | 17.046 s | 13.273 s *(1.28×)* | 7.538 s *(2.26×)* | 8.004 s *(2.13×)* | **7.137 s** *(2.39×)* | **2.39× (`vk-xir`)** |
| `barbershop` (2048×858) | 24.368 s | 22.366 s *(1.09×)* | **16.912 s** *(1.44×)* | 17.091 s *(1.43×)* | 18.433 s *(1.32×)* | **1.44× (`dx`)** |

### CPU — 64 spp (`megakernel`, idle-host medians) vs Blender Cycles 5.3.0 CPU

| Scene | Cycles 5.3.0 CPU | Psycles `fallback` | Psycles `simd` (W8) | Best Speedup |
| --- | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 2.894 s | 2.277 s *(1.27×)* | **2.127 s** *(1.36×)* | **1.36× (`simd`)** |
| `monster` (1024×1024) | 12.686 s | **10.307 s** *(1.23×)* | 10.956 s *(1.16×)* | **1.23× (`fallback`)** |
| `lone-monk` (1440×1080) | 22.948 s | 16.096 s *(1.43×)* | **15.445 s** *(1.49×)* | **1.49× (`simd`)** |
| `classroom` (1920×1080) | 24.498 s | 20.648 s *(1.19×)* | **19.814 s** *(1.24×)* | **1.24× (`simd`)** |
| `barbershop` (2048×858) | 37.492 s | **29.663 s** *(1.26×)* | 34.093 s *(1.10×)* | **1.26× (`fallback`)** |

See the [2026-11-22 six-backend validation report](docs/validation/2026-11-22/six-backend-benchmark/README.md)
and [validation index](VALIDATION.md) for cold/warm JIT timings, RMSE tables,
unit test gates, and methodological notes.

This is not a claim of complete Cycles SVM feature coverage: eight semantic
opcodes remain unimplemented, indirect-light/path residuals remain under
diagnosis, and the displacement prepass still retains a private legacy
evaluator slated for removal. See [the current compatibility status](docs/cycles-compatibility.md)
for exact coverage boundaries.

Unsupported execution features are explicit compilation diagnostics rather
than silent approximations. Node and render semantics are accepted only after
comparison with official Cycles output.
