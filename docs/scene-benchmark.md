# Canonical Scene Benchmark Runner

`tools/run_scene_benchmark.py` compares official Blender Cycles (5.2.x / 5.3.0) against Psycles across CPU and GPU Luisa backends on unmodified `.blend` scenes using identical resolution, scene-owned seeds, fixed sample counts, disabled adaptive sampling, and disabled denoising.

For the verified five-scene, six-backend performance results (`LuisaCompute next@2beb5a275` vs Cycles 5.3.0), see [optimization.md](optimization.md).

---

## Backends Covered

| Engine | Backend Identifier | Execution Stack |
| --- | --- | --- |
| Blender Cycles | `CPU` / `HIP` / `METAL` | Official Blender Cycles main loop (explicit device-name filter) |
| Psycles/Luisa | `fallback` | LLVM/Embree scalar-per-thread CPU (`megakernel`) |
| Psycles/Luisa | `simd` | LLVM/Embree packet-vectorized W8 CPU (`PSYCLES_ENABLE_LUISA_SIMD=ON`, `megakernel`) |
| Psycles/Luisa | `hip` | AMD ROCm HIP + HIPRT 3.0 GPU |
| Psycles/Luisa | `dx` | DirectX 12 SM 6.6+ GPU (Windows) |
| Psycles/Luisa | `vk` | Vulkan 1.4 (`vk-xir` native XIR-to-SPIR-V default, or `vk-dxc` build with `-DLUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=OFF`) |
| Psycles/Luisa | `metal` / `metal4` | Apple Metal / Metal 4 GPU (macOS) |

Both `fallback` and `simd` compile and execute the exact same Psycles Luisa DSL/DirectJit SVM kernel as the GPU backends. Cycles remains the sole rendering oracle.

---

## Usage

### Worktree / Linux HIP & CPU Example

```bash
TMPDIR=/tmp/psycles-compiler-tmp \
python3 tools/run_scene_benchmark.py \
  --blender /path/to/blender \
  --psycles-render build/bin/psycles_render_blender_scene \
  --blend /path/to/scene.blend \
  --output-dir /tmp/psycles-benchmark-out \
  --cycles-hip-device-name "Radeon RX 9070 XT" \
  --compiler-tmp /tmp/psycles-compiler-tmp \
  --width 640 --height 480 --samples 64
```

### Apple Silicon Metal Example

```bash
python3 tools/run_scene_benchmark.py \
  --blender /opt/homebrew/bin/blender \
  --psycles-render build-macos/bin/psycles_render_blender_scene \
  --blend /path/to/classroom.blend \
  --output-dir build-macos/benchmarks/classroom-apple \
  --bundle build-macos/scene-exports/classroom \
  --reuse-export --skip-cycles-cpu \
  --cycles-gpu-device METAL \
  --cycles-gpu-device-name "Apple M1 Max" \
  --psycles-backends fallback,metal \
  --width 640 --height 360 --samples 64
```

### Resuming & Reusing Exports

- `--reuse-export`: reuses an existing exported bundle (`scene.json` + `geometry.bin`) for the same source `.blend`. Material node graphs, socket values, and scene geometry remain verbatim from Cycles.
- `--resume`: continues an interrupted or partially failed benchmark directory after re-validating schema, hashes, command lines, logs, and complete pass inventories (`resume.reused_stages`). Any changed bundle invalidates resume reuse.
- `--cycles-gpu-device` / `--cycles-gpu-device-name`: selects non-HIP Cycles GPU devices (`--cycles-hip-device-name` remains supported for HIP).
- `--psycles-backends`: ordered comma-separated list of Luisa backend names.

---

## Pass Contract & Timing Methodology

1. **Equal-Work 15-Pass Contract (46 Channels):**
   - Cycles headless setup strips scene-inherited auxiliary passes (AO, Depth, Mist, indices, denoising, custom AOVs, light groups) and enables the exact 15 common linear passes (46 channels including Combined alpha) without saving back to the `.blend`.
   - Pass name aliases (e.g. `VolumeDir`) are normalized during EXR validation; unexpected or missing channels fail the run.
2. **Clean Timing Boundaries (`benchmark.json`, `psycles.scene-benchmark.v3`):**
   - **Cycles render time (`render_seconds`):** extracted from `--debug-cycles` `Rendering in main loop is done in ... seconds.` (excludes scene sync/Python wrapper overhead).
   - **Psycles render time (`render_seconds`):** render-only integration wall time (`Rendered ... in ... s`).
   - **Separate overhead metrics:** scene compilation time, session initialization (`shader_jit_seconds`, covering JIT compile + PSO creation + setup/baking), total render call time, and full process wall time are recorded independently. Always report cold (`r0`) vs warm (`r1+`) initialization explicitly.
3. **Differential Reports & Visual Inspection:**
   - Every `(Cycles device, Psycles backend)` pair writes JSON error metrics (RMSE, MAE, relative error, invalid/non-finite counts) and labeled reference/actual/absolute-difference EXR/PNG triptychs across all 15 passes.
   - Always inspect invalid pixel counts and visual triptychs alongside timing; never overlap timed benchmarks with background builds or concurrent renders.
