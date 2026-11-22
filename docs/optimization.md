# Optimization Architecture & Benchmark Reference

This document details Psycles' native Cycles SVM specialization, GPU wavefront scheduling, backend-specific code generation, and verified performance across all six active backends on `LuisaCompute next@2beb5a275`.

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

## 5. Verified Six-Backend Benchmark Reference (`2beb5a275`)

Measured on `LuisaCompute next@2beb5a275` (`2beb5a2757fa46868da99b22be10ef571f77e443`) against official **Blender Cycles 5.3.0** (`02f0899d78e3`) using equal 15-pass linear EXR comparison (render-only wall interval vs Cycles main-loop wall interval). Static-mesh pre-transformed emissive-triangle coordinates and instance normal matrices were also repaired in Psycles, slashing `bmw27` / `monster` / `classroom` GPU Combined RMSE from `0.055–0.086` down to `0.0014–0.0079` and matching Cycles `bmw27` mean scene luminance (`0.9998×`).

### Testbed Configuration

| Component | Baseline Spec |
| --- | --- |
| Host CPU | AMD Ryzen 9 9950X3D (16 cores / 32 threads) |
| Host GPU | AMD Radeon RX 9070 XT (`gfx1201`, RDNA 4) |
| GPU Toolchains | ROCm 10.1.0 / HIP `7.16.26385` + HIPRT 3.0, Vulkan API `1.4.359`, DirectX 12 SM 6.6+ (`[WaveSize(32)]`) |
| CPU Toolchains | LLVM 24 (`clang-cl` / ORC JIT), Embree 4.4.1, oneTBB 2023.1.0 |
| Schedulers & Math | GPU: `wavefront-staged` (DirectJit SVM, native surface sort, direct-light shadow queue); CPU: `megakernel`; `enable_fast_math = true` (safe FMF) |

### GPU Results — 256 spp (`wavefront-staged`, Warm `r1` Render Time) vs Cycles 5.3.0 HIPRT

All 20 `(scene, backend)` GPU configurations produced **0 invalid pixels**.

| Scene (Resolution) | Cycles 5.3.0 HIPRT | Psycles `hip` | Psycles `dx` | Psycles `vk-dxc` | Psycles `vk-xir` | Best GPU Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 3.238 s | 2.418 s *(1.34×)* | 1.217 s *(2.66×)* | 1.489 s *(2.17×)* | **1.200 s** *(2.70×)* | **2.70× (`vk-xir`)** |
| `monster` (1024×1024) | 13.348 s | 8.260 s *(1.62×)* | 6.078 s *(2.20×)* | 6.802 s *(1.96×)* | **5.665 s** *(2.36×)* | **2.36× (`vk-xir`)** |
| `lone-monk` (1440×1080) | 13.315 s | 10.984 s *(1.21×)* | 4.847 s *(2.75×)* | 5.463 s *(2.44×)* | **4.685 s** *(2.84×)* | **2.84× (`vk-xir`)** |
| `classroom` (1920×1080) | 17.046 s | 14.414 s *(1.18×)* | 7.583 s *(2.25×)* | 8.047 s *(2.12×)* | **7.454 s** *(2.29×)* | **2.29× (`vk-xir`)** |
| `barbershop` (2048×858) | 24.368 s | 22.495 s *(1.08×)* | **17.031 s** *(1.43×)* | 17.042 s *(1.43×)* | 18.158 s *(1.34×)* | **1.43× (`dx`)** |

#### GPU Cold `r0` / Warm `r1` Session JIT & Combined RMSE

| Scene | `hip` (Cold / Warm JIT, RMSE) | `dx` (Cold / Warm JIT, RMSE) | `vk-dxc` (Cold / Warm JIT, RMSE) | `vk-xir` (Cold / Warm JIT, RMSE) |
| --- | ---: | ---: | ---: | ---: |
| `bmw27` | 7.08 s / 0.88 s *(0.00787)* | 3.54 s / 0.94 s *(0.00791)* | 4.95 s / 0.96 s *(0.00794)* | 3.43 s / 0.96 s *(0.00790)* |
| `monster` | 10.29 s / 1.49 s *(0.00145)* | 13.33 s / 1.87 s *(0.00146)* | 11.73 s / 1.91 s *(0.00144)* | 9.92 s / 1.85 s *(0.00144)* |
| `lone-monk` | 10.22 s / 1.05 s *(0.06870)* | 13.90 s / 1.26 s *(0.06843)* | 12.18 s / 1.47 s *(0.06872)* | 10.56 s / 1.50 s *(0.06870)* |
| `classroom` | 11.23 s / 1.89 s *(0.00167)* | 17.75 s / 2.79 s *(0.00173)* | 14.66 s / 2.82 s *(0.00221)* | 12.03 s / 2.83 s *(0.00221)* |
| `barbershop`\* | 118.22 s / 8.49 s *(0.00217)* | 113.16 s / 9.69 s *(0.00233)* | 99.36 s / 9.51 s *(0.00228)* | 98.00 s / 9.54 s *(0.00227)* |

\* On `barbershop` `r0`, cold runs compiled from an empty local shader cache (`bin/.cache` / `bin_vk_dxc/.cache`); `barbershop hip` cold JIT `118.22 s` is from the initial cold `r0` pass, while warm JIT (`8.49 s`), render time (`22.495 s`), and parity (`rmse=0.00217`, `invalid_pixels=0`) are from the clean verified two-repeat HIP run (replacing an earlier single `r1` transient `render_failed` exit). Note also that updating `LuisaCompute` on `2beb5a275` restored SLEEF fast trig/inverse-trig minimax accuracy and duplicate switch-PHI handling while resolving all 6 prior `fallback`-only CTest exceptions (`421/421` passing).

### CPU Results — 64 spp (`megakernel`) vs Cycles 5.3.0 CPU

All 10 `(scene, backend)` CPU configurations produced **0 invalid pixels**, with `fallback` and `simd` matching each other within `0.00003` Combined RMSE on every scene.

| Scene (Resolution) | Cycles 5.3.0 CPU | `fallback` (Warm Mean) | `simd` W8 (Warm Mean) | Best Speedup | Cold / Warm JIT (`fallback` \| `simd`) | RMSE (`fallback` / `simd`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bmw27` (960×540) | 2.894 s | **2.486 s** | 2.744 s\* | **1.16× (`fallback`)** | 1.43 s / 0.08 s \| 7.66 s / 0.07 s | 0.03580 / 0.03583 |
| `monster` (1024×1024) | 12.686 s | **10.289 s** | 11.174 s | **1.23× (`fallback`)** | 4.54 s / 0.18 s \| 18.23 s / 0.15 s | 0.03041 / 0.03041 |
| `lone-monk` (1440×1080) | 22.948 s | 15.710 s | **15.578 s** | **1.47× (`simd`)** | 4.21 s / 0.23 s \| 18.92 s / 0.20 s | 0.17754 / 0.17753 |
| `classroom` (1920×1080) | 24.498 s | 21.282 s | **20.405 s** | **1.20× (`simd`)** | 5.36 s / 0.29 s \| 24.16 s / 0.24 s | 0.04217 / 0.04216 |
| `barbershop` (2048×858) | 37.492 s | **29.955 s** | 34.533 s | **1.25× (`fallback`)** | 23.50 s / 0.66 s \| 119.64 s / 0.70 s | 0.04425 / 0.04422 |

\* `bmw27` `simd` warm runs recorded `r2 = 2.266 s` and a single host scheduling outlier `r3 = 3.222 s` (mean `2.744 s`).
