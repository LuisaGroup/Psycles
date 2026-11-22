# Developer Guide

This guide documents Psycles' repository structure, semantic and performance invariants, and required verification gates.

---

## Repository Structure

```text
Psycles/
  include/psycles/
    adapter/       Blender scene importer public API
    blender_bridge/ Native C ABI declarations for host application plugins
    compiler/      Core node registry, Cycles SVM IR & bytecode compiler, sky/camera helpers
    contract/      Immutable SceneDatabase, ShaderGraph, pass & transport contracts
    io/            Multilayer OpenEXR & PFM image I/O, progressive pixel probe
    luisa/         Luisa backend API, surface/volume/sampling headers, DirectJit translator
    sampling/      Tabulated Sobol, LightTree, background & pixel-filter builders
  src/             Implementations matching each include module above
  blender_addon/   Official Blender 4.2+ / 5.x Render Engine extension (`psycles/`)
  examples/        CLI executables (render_demo, render_blender_scene, probe_nishita)
  tools/           Blender packaging, Cycles benchmark runner, path-trace oracle, policy scripts
  tests/           C++ core/device tests, Cycles oracle fixtures, Python contract suites
  docs/            Design manuals, optimization reference, compatibility spec, Cycles probe JSONs
  third_party/     Pinned LuisaCompute submodule and sky lookup tables
```

---

## Engineering & Compatibility Invariants

### 1. Native Cycles SVM Fidelity

- Blender Cycles (5.2.1 / 5.3.0) source and rendered linear passes are the sole correctness oracle. Do **not** build a separate host CPU reference renderer or duplicate device BSDF/MIS/sampling logic on the host.
- Preserve Cycles' SVM node stream, typed word payloads, stack addressing, `ShaderJump` entrypoints (`surface`, `volume`, `displacement`, plus bump fallthrough prefix), closure allocations, feature masks, and RNG sampling dimensions.
- Unsupported reachable nodes/features must emit explicit compiler diagnostics (`ShaderCompileDiagnostic`) rather than silent fallbacks or placeholder defaults.
- Specialization happens only at host recording time using immutable finalized scene/session facts (active shaders, static stack high-water mark, closure bounds, attribute residency, feature masks). Never key specialization off scene names, asset hashes, or pre-render heuristics.

### 2. Structural-Parity Performance Policy

- Keep native fast math enabled (`enable_fast_math = true`); never trade runtime performance for non-structural 1-ULP bit matching.
- Never replace native hardware math or texture filtering (`nearest`, linear bilinear `1x`, Cycles bicubic `4x`) with software loops or scalar `sqrt`/division normalization where `native_vector_math.h` `rsqrt` applies.
- Express per-shader/per-bucket specialization and surface sorting cleanly in Psycles host builders or generic Luisa `Coro` scheduler facilities; never add renderer-specific branches inside `third_party/LuisaCompute` compiler passes or runtime allocators.
- Keep `auto`-inlining heuristics natural unless a controlled cross-scene measurement demonstrates an instruction/register win (never sprinkle `noinline` pragmas ad hoc).

### 3. Source Hygiene & Submodule Discipline

- Keep every first-party source/header file at or below 2,000 lines (`python tools/check_source_size.py`).
- Keep CMake source lists (`CMakeLists.txt` and `cmake/*.cmake`) exact — no stale paths and no orphaned `.cpp` files.
- Publish or pin `third_party/LuisaCompute` to a clean upstream/fork commit before advancing the submodule gitlink in `Psycles`.
- Keep path-handling tests cross-platform (handle Windows `Path` separators and `PurePosixPath` fixture strings explicitly).

---

## Mandatory Verification Gates

Run the appropriate gates after every change:

1. **Repository Policy & Static Gates:**

   ```bash
   python tools/check_source_size.py
   python tests/test_luisa_shader_performance_policy.py
   python -m unittest discover -s tests -p "test_cycles_binding_oracle.py"
   ```

2. **Full Build & CTest Suite:**
   - Build all targets cleanly and run CPU/GPU CTest (`ctest --test-dir build --output-on-failure`).
   - Run GPU backend tests sequentially on single-GPU hosts to avoid queue contention.
3. **End-to-End Scene Parity & Throughput Benchmark:**
   - When modifying SVM handlers, path-kernel scheduling, or updating `third_party/LuisaCompute`, validate full canonical scenes via [docs/scene-benchmark.md](docs/scene-benchmark.md) against official Cycles (`hip`, `dx`, `vk-xir`, `vk-dxc`, `fallback`, `simd`) confirming `0` invalid pixels, expected Combined RMSE, and non-regressed render-only wall time.
4. **First-Divergence Diagnosis:**
   - When an EXR pass diverges from Cycles, capture a single-pixel/single-sample trace via [docs/cycles-path-trace.md](docs/cycles-path-trace.md) (`tools/compare_cycles_path_traces.py`) to pinpoint the exact diverging RNG word, closure, PDF, or path state transition before changing kernel logic.
