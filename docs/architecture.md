# System Architecture

Psycles translates Blender Cycles scenes into an immutable typed scene/graph contract, compiles shader networks directly into Cycles 5.2/5.3 SVM bytecode, and executes the complete renderer on LuisaCompute across GPU (`hip`, `dx`, `vk`, `metal`, `metal4`) and CPU (`fallback`, `simd`) backends.

For performance strategies and verified multi-backend measurements, see [optimization.md](optimization.md). For compatibility boundaries, see [cycles-compatibility.md](cycles-compatibility.md).

---

## 1. End-to-End Pipeline

```text
Blender evaluated scene + node graph
  -> Immutable scene & typed ShaderGraph contract   (src/contract/, src/adapter/)
  -> Native Cycles SVM IR, bytecode & scene image   (src/compiler/)
  -> Scene-specialized DirectJit / Luisa DSL kernel (src/luisa/)
  -> Coroutine / megakernel session & EXR pass I/O  (src/luisa/, src/io/)
```

All material evaluation consumers — **surface, background, light emission, volume, and true/bump displacement** — share this single native SVM execution pipeline. No legacy pre-SVM surface evaluator or secondary displacement bridge remains.

---

## 2. Scene Contract & Blender Adapter

- **Graph Normalization (`src/adapter/`):** Preserves Blender node sockets, defaults, properties, node-group recursion/memoization, mapping transforms, and light/world trees. Unsupported nodes or sockets raise explicit diagnostics rather than host approximations.
- **Typed `ShaderGraph` & `SceneDatabase` (`include/psycles/contract/`):**
  - Explicit `surface`, `volume`, and `displacement` shader roots plus surface-normal bump projection.
  - Atomic `SceneDelta` transitions into an immutable `SceneSnapshot` with stable resource IDs (never raw Luisa handles or GPU pointers).
  - Domain-aware mesh attribute storage (`point`, `corner`, `face`) and static ribbon curve descriptors (containing-curve identity + packed segment index).
  - Displacement and normal/tangent attribute requests drive minimal GPU buffer residency.

---

## 3. Native Cycles SVM Compiler & Material Runtime

- **Compiler (`src/compiler/`):**
  - Performs Cycles graph transformations, constant folding, stack liveness/allocation, and emits typed SVM words + `ShaderJump` tables (`surface`, `volume`, `displacement` offsets).
  - `BOTH` displacement shaders emit bump prefix evaluation (`P` + derivative save/restore with updated shading normal `N`) falling through cleanly into the surface SVM stream.
  - Emits both linear word streams (for `BytecodeVmTranslator`) and structured `SvmBlockIr` (for `DirectJitTranslator`), alongside `ShaderEntryUsage` per-domain opcode/stack/noise bounds.
- **Scene Upload & Binding (`src/luisa/path_tracer_scene*.cpp`, `src/luisa/path_tracer_cycles_svm_*.cpp`):**
  - Translates static meshes (with true native SVM displacement prepass in `path_tracer_displacement_scene.cpp`), ribbon curves, `KernelObject` transforms/flags, `KernelShader` closure/emission metadata, and texture/attribute heap bindings.
  - `PathCyclesSvmKernelGlobals` supplies lazy transform and triangle/curve attribute lookups directly from device buffers without eager per-hit vertex expansion.
- **Volume Execution (`src/luisa/path_tracer_volume_*.cpp`, `src/luisa/volume_*.cpp`):**
  - Ordered volume stack tracking up to 8 active phases sharing a unified closure allocator.
  - Camera-dependent majorant hierarchy (`volume_majorant_hierarchy.cpp`) and native SVM density/absorption/scattering/emission evaluation for both main and shadow rays.
- **Subsurface Scattering (`src/luisa/subsurface_random_walk_component.cpp`, `src/luisa/path_kernel_subsurface*.cpp`):**
  - Random-walk and Burley BSSRDF continuation integrated with the staged coroutine path without repeating primary ray traversal after a subsurface exit hit.

---

## 4. Transport & Coroutine Scheduling

- **Entrypoints (`include/psycles/luisa/path_tracer.h`, `src/luisa/path_tracer_kernel.cpp`, `src/luisa/path_tracer_session.cpp`):**
  - `LuisaPathTracerBackend::compile_scene` specializes scene geometry, light distribution / LightTree (`src/sampling/`), and `cycles_svm` tables into an immutable compiled scene.
  - `create_session` initializes camera-dependent volume/background resources and compiles device kernels (`Luisa shader JIT`).
  - `SampleRange.total` anchors Tabulated Sobol sequence hashes consistently across progressive renders and single-pixel diagnostic captures.
- **Schedulers (`src/luisa/path_kernel_pipeline.cpp`, `src/luisa/path_kernel_executor.cpp`):**
  - **`megakernel`:** Single monolithic path kernel loop per thread/packet (default on CPU `fallback` / `simd`).
  - **`wavefront-staged`:** Splits path tracing into persistent pool-allocated coroutine frames (`intersect_closest`, optional volume stage, surface-sorted multi-bucket shading, and scatter + fused direct-light shadow queue `shade_light_nee -> intersect_shadow -> shade_shadow`). See [optimization.md](optimization.md) for bucket planning, refill thresholds, and shader-sorting strategy.
- **Pass Routing & Output (`src/luisa/path_kernel_film*.cpp`, `src/io/`):**
  - Supports Combined, Normal, Albedo, Diffuse/Glossy/Transmission direct/indirect, Emission, Background, Volume Direct/Indirect, and diagnostic 344-AOV `PsyTraceNNN` capture, writing 32-bit float multilayer OpenEXR (`ViewLayer.<pass>.<component>`) alongside diagnostic PFM outputs.
