# Optimization Architecture & Benchmark Reference

This document details Psycles' native Cycles SVM specialization, GPU wavefront scheduling, backend-specific code generation, and verified performance across all six active backends on `LuisaCompute next@db53c6b76`.

---

## 1. Execution Model Overview

Psycles lowers every Cycles shader domain — **surface, background, light emission, volume, and true/bump displacement** — into native Cycles 5.2/5.3 SVM word streams and executes them through a single LuisaCompute C++ DSL implementation across both GPU and CPU targets:

| Target Family | Scheduler | Shader Translation | Hardware / Runtime Stack |
| --- | --- | --- | --- |
| GPU (`hip`, `dx`, `vk`) | `wavefront-staged` (default benchmark path) | DirectJit SVM (`<= 80` shaders/bucket) + bytecode VM fallback | HIP / HIPRT 3.0, DirectX 12 SM 6.6+, Vulkan 1.4 (`vk-xir` & `vk-dxc`) |
| CPU (`fallback`, `simd`) | `megakernel` | DirectJit SVM | LLVM 24 ORC JIT, Embree 4.4.1, oneTBB 2023.1.0 |

---

## 2. DirectJit Cycles SVM & Shader Specialization

1. **Dual-mode SVM Translation (`DirectJitTranslator` vs `BytecodeVmTranslator`):**
   - `DirectJitTranslator` specializes each distinct shader group directly into structured Luisa DSL control flow rather than running a dynamic bytecode PC loop at runtime (`src/luisa/cycles_svm_translator.cpp`).
   - By default (`default_auto_direct_jit_max_shaders = 80u`, tunable via `PSYCLES_SVM_JIT_MAX_SHADERS` or overridden via `PSYCLES_SVM_BACKEND=jit|vm`), domains/buckets with `<= 80` active non-empty shaders compile through `DirectJitTranslator`, while larger unpartitioned shader tables fall back to `BytecodeVmTranslator`.
2. **Virtual SSA Stack Slots (`Stack::direct_jit`):**
   - Bytecode VM mode allocates a scene-bounded `Local<float>` array (`peak_stack_usage` from static compiler analysis, avoiding fixed `SVM_STACK_SIZE` allocations).
   - DirectJit mode replaces the stack array with per-case virtual SSA scalar slots and lazy mutable scalar variables (`src/luisa/cycles_svm_stack.cpp`) only inside conditional scopes, eliminating function-wide stack `alloca`s and cross-shader PHI merges.
3. **Compile-Time Domain, Opcode, and Feature Pruning:**
   - Separate specialization masks (`CyclesSvmShaderDomain`: `surface_only`, `shadow_surface`, `light_emission`, `world`, `volume`) and `ClosureUsage` bitmasks (`KernelShader` metadata) omit unreachable SVM opcode cases, closure evaluation/sampling branches, BSSRDF exit paths, and ray-portal handling at host DSL recording time.
   - `scene_has_shadow_terminator_geometry_offset` and object shading-offset checks elide shadow-terminator math on scenes without terminator offsets.
   - Native geometry and transform reads (`cycles_object_index`, packed triangle/curve buffers, lazy `TransformState`) defer object matrix and vertex loads to actual consumers rather than eager per-hit struct copies.
4. **Callable Outlining & Surface Multi-Bucketing:**
   - Shared heavy helpers (such as RGB ramp/curve lookup tables and procedural noise evaluators) use shared Luisa `Callable`s or `$outline_when_with_name` boundaries (`auto_outline_shared_value_callables` in LuisaCompute requiring `call_count >= 2` and `<= 256` B/lane value parameters) while keeping hot BSDF/native math inlined to avoid register spills.
   - Large multi-material scenes partition surface shading into specialized shader buckets (`src/luisa/path_kernel_surface_buckets.cpp`, configurable via `PSYCLES_SURFACE_BUCKET_MODE` and `PSYCLES_SURFACE_BUCKET_MAX_SHADERS`, default `44`) so complex shaders stay separated from simple diffuse/emissive surfaces during dispatch and sorting.
5. **Safe Native Fast Math:**
   - Production kernels compile with `enable_fast_math = true` (opt-out diagnostic knob: `PSYCLES_NATIVE_FAST_MATH=0`).
   - On LLVM CPU backends (`fallback` and `simd`), fast-math flags emit safe algebraic reassociation/contraction (`arcp contract afn reassoc nsz`) **without** `nnan`/`ninf` poison annotations and preserve exact predicates around `ISNAN`, `ISINF`, `COPYSIGN`, `FMOD`, and `FRem`.

---

## 3. Wavefront Coroutine Scheduling, Surface Sorting & Shadow Queue

When running with `wavefront-staged` on GPU (`src/luisa/path_kernel_executor.cpp`, `src/luisa/path_kernel_pipeline.cpp`):

- **Four-Stage Path Coroutine Split:** Persistent per-path state lives in pool-managed coroutine frames, while transient shader data and closure pools stay short-lived within individual continuations:
  1. Primary/continuation ray intersection (`intersect_closest`)
  2. Volume integration when statically required by the scene
  3. Surface-sorted SVM evaluation and shading buckets (`shade_surface`, default block size `64`, tunable via `PSYCLES_SHADE_SURFACE_BLOCK_SIZE` / `PSYCLES_SHADE_SURFACE_BUCKET0_BLOCK_SIZE`)
  4. Scatter continuation and direct-light sampling
- **Wavefront Queue Policies (`WavefrontCoroSchedulerConfig`):**
  - `largest_continuation_first = true` prioritizes the densest work queue.
  - `incremental_continuation_counts = true` and `fused_continuation_counts = true` minimize launch/compaction overhead.
  - `frame_buffer_compaction = true` (`PSYCLES_WAVEFRONT_FRAME_COMPACTION=0` to disable) and `refill_threshold = 0.5 * capacity` (`PSYCLES_WAVEFRONT_REFILL_RATIO` / `PSYCLES_WAVEFRONT_REFILL_THRESHOLD`) sustain wave occupancy as paths terminate.
  - `shader_compile_threads = clamp(hardware_concurrency, 1, 8)` (`PSYCLES_SHADER_COMPILE_THREADS`) parallelizes multi-stage PSO/JIT compilation.
- **Shader-Key Surface Sorting:** Surface shading partitions hits using shader-derived sort keys so wavefront lanes execute coherent SVM branches.
- **Fused Staged Direct-Light Shadow Queue (`path_kernel_direct_light_queue.cpp`):** Decouples shadow ray evaluation into `shade_light_nee` -> `intersect_shadow` -> `shade_shadow` with compact SoA task/traversal buffers and fused unoccluded handling.

---

## 4. Backend-Specific Compilation & Tuning

- **DirectX 12 (`dx`):**
  - Targets Shader Model 6.6+ with native `[WaveSize(32)]` on variable-wave GPUs (`gfx1201`) and preserves `[WaveSize(N)]` across DXIL binary bitcode.
  - Uses fast large-DXIL `-O3` optimization pipelines, `RayQuery` `TMin` origin handling, and `O(|active|)` bindless command reordering.
- **Vulkan (`vk-xir` default vs `vk-dxc`):**
  - Defaults to LuisaCompute's **native XIR-to-SPIR-V compiler** (`LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=ON`; examples also enforce `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` unless overridden), avoiding DXC overhead while using subgroup-size pipeline caching and streamlined SPIR-V CFGs (`~419k` words on `barbershop`).
  - `vk-dxc` remains buildable as a control target with `-DLUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=OFF`.
- **ROCm HIP (`hip`):**
  - Compiles via LuisaCompute's HIP LLVM CodeGen + ROCm `comgr` with HIPRT 3.0 ray tracing.
  - On systems with both an APU iGPU and a discrete Radeon GPU, set `HIP_VISIBLE_DEVICES=1` (or the discrete device index) to select the discrete GPU.
- **CPU (`fallback` and `simd`):**
  - Both compile through LLVM 24 ORC JIT + Embree 4.4.1 + oneTBB 2023.1.0 with worker-owned barrier-frame arenas (4 MiB fast buffer + reusable pointer-stable overflow chunks).
  - `simd` enables packet-vectorized W8 execution (`PSYCLES_ENABLE_LUISA_SIMD=ON` at configure time) and fast `NORMALIZE`, while normal `luisa.cpu.native_math` helpers stay inlined in `LLVMJIT` to avoid x86-64 `FastCC` vector-register spills.

---

## 5. Verified Six-Backend Benchmark Reference (`db53c6b76`)

Measured on `LuisaCompute next@db53c6b76` (`db53c6b76158918007384b17735c5e2e24634211`) against official **Blender Cycles 5.3.0** (`02f0899d78e3`) using `tools/run_scene_benchmark.py` (equal 15-pass linear EXR comparison, render-only wall interval vs Cycles main-loop wall interval).

### Testbed Configuration

| Component | Baseline Spec |
| --- | --- |
| Host CPU | AMD Ryzen 9 9950X3D (16 cores / 32 threads) |
| Host GPU | AMD Radeon RX 9070 XT (`gfx1201`, RDNA 4) |
| GPU Toolchains | ROCm 10.1.0 / HIP `7.16.26385` + HIPRT 3.0, Vulkan API `1.4.359`, DirectX 12 SM 6.6+ (`[WaveSize(32)]`) |
| CPU Toolchains | LLVM 24 (`clang-cl` / ORC JIT), Embree 4.4.1, oneTBB 2023.1.0 |
| Schedulers & Math | GPU: `wavefront-staged` (DirectJit SVM, native surface sort, direct-light shadow queue); CPU: `megakernel`; `enable_fast_math = true` (safe FMF) |

### GPU Results — 256 spp (`wavefront-staged`, Warm `r1` Render Time) vs Cycles 5.3.0 HIPRT

All 20 `(scene, backend)` GPU runs produced **0 invalid pixels**.

| Scene (Resolution) | Cycles 5.3.0 HIPRT | Psycles `hip` | Psycles `dx` | Psycles `vk-dxc` | Psycles `vk-xir` | Best GPU Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 3.238 s | 1.859 s *(1.74×)* | **1.019 s** *(3.18×)* | 1.229 s *(2.64×)* | 1.037 s *(3.12×)* | **3.18× (`dx`)** |
| `monster` (1024×1024) | 13.348 s | 7.905 s *(1.69×)* | 5.827 s *(2.29×)* | 6.306 s *(2.12×)* | **5.356 s** *(2.49×)* | **2.49× (`vk-xir`)** |
| `lone-monk` (1440×1080) | 13.315 s | 11.586 s *(1.15×)* | 4.826 s *(2.76×)* | 5.384 s *(2.47×)* | **4.665 s** *(2.85×)* | **2.85× (`vk-xir`)** |
| `classroom` (1920×1080) | 17.046 s | 13.273 s *(1.28×)* | 7.538 s *(2.26×)* | 8.004 s *(2.13×)* | **7.137 s** *(2.39×)* | **2.39× (`vk-xir`)** |
| `barbershop` (2048×858) | 24.368 s | 22.366 s *(1.09×)* | **16.912 s** *(1.44×)* | 17.091 s *(1.43×)* | 18.433 s *(1.32×)* | **1.44× (`dx`)** |

#### GPU Cold `r0` / Warm `r1` Session JIT & Combined RMSE

| Scene | `hip` (Cold / Warm JIT, RMSE) | `dx` (Cold / Warm JIT, RMSE) | `vk-dxc` (Cold / Warm JIT, RMSE) | `vk-xir` (Cold / Warm JIT, RMSE) |
| --- | ---: | ---: | ---: | ---: |
| `bmw27` | 6.59 s / 0.85 s *(0.08608)* | 3.80 s / 0.97 s *(0.08605)* | 4.66 s / 0.94 s *(0.08604)* | 3.49 s / 0.95 s *(0.08604)* |
| `monster` | 10.62 s / 1.49 s *(0.06637)* | 12.88 s / 1.98 s *(0.06635)* | 11.87 s / 1.88 s *(0.06635)* | 9.80 s / 1.93 s *(0.06635)* |
| `lone-monk` | 10.79 s / 1.01 s *(0.06889)* | 14.09 s / 1.37 s *(0.06863)* | 12.52 s / 1.42 s *(0.06893)* | 10.21 s / 1.46 s *(0.06891)* |
| `classroom` | 11.36 s / 2.12 s *(0.05900)* | 9.94 s / 2.79 s *(0.05549)* | 7.50 s / 2.86 s *(0.05553)* | 11.79 s / 2.83 s *(0.05554)* |
| `barbershop`\* | 113.06 s / 8.69 s *(0.00244)* | 107.10 s / 10.99 s *(0.00257)* | 100.84 s / 9.92 s *(0.00253)* | 103.58 s / 10.16 s *(0.00252)* |

\* On `barbershop` `r0`, updating LuisaCompute to `db53c6b76` invalidated the AMD driver disk cache (`~68–78 s` one-time driver compile), while LuisaCompute's internal compiler stages improved substantially over `9672b97bb`: coroutine split `-26%` to `-35%`, HIP LLVM codegen `-31%`, `vk-xir` SPIR-V compile `-12%`, and warm `r1` JIT `-19%` to `-32%`.

### CPU Results — 64 spp (`megakernel`) vs Cycles 5.3.0 CPU

All 10 `(scene, backend)` CPU runs produced **0 invalid pixels**, with `fallback` and `simd` matching each other within `0.00003` Combined RMSE on every scene.

| Scene (Resolution) | Cycles 5.3.0 CPU | `fallback` (Idle / Daytime) | `simd` W8 (Idle / Daytime) | Best Speedup (Idle) | Cold / Warm JIT (`fallback` \| `simd`) | RMSE (`fallback` / `simd`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 2.894 s | 2.277 s / 2.350 s | **2.127 s** / 2.502 s | **1.36× (`simd`)** | 1.59 s / 0.09 s \| 7.10 s / 0.07 s | 0.09270 / 0.09269 |
| `monster` (1024×1024) | 12.686 s | **10.307 s** / 10.277 s | 10.956 s / 11.916 s | **1.23× (`fallback`)** | 4.48 s / 0.20 s \| 16.94 s / 0.15 s | 0.07099 / 0.07099 |
| `lone-monk` (1440×1080) | 22.948 s | 16.096 s / 16.268 s | **15.445 s** / 17.246 s | **1.49× (`simd`)** | 4.10 s / 0.24 s \| 19.16 s / 0.17 s | 0.17762 / 0.17761 |
| `classroom` (1920×1080) | 24.498 s | 20.648 s / 20.891 s | **19.814 s** / 22.101 s | **1.24× (`simd`)** | 5.30 s / 0.30 s \| 22.71 s / 0.24 s | 0.06954 / 0.06957 |
| `barbershop` (2048×858) | 37.492 s | **29.663 s** / 30.438 s | 34.093 s / 34.868 s | **1.26× (`fallback`)** | 22.38 s / 0.69 s \| 114.30 s / 0.62 s | 0.04426 / 0.04424 |
