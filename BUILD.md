# Building & Running Psycles

Psycles embeds `third_party/LuisaCompute` (`next@db53c6b76`) as a CMake subdirectory so standard builds configure, compile, and test in one pass.

---

## Prerequisites

- **Git** with submodule support
- **CMake 3.26+** and **Ninja** (or another CMake generator)
- **C++20 compiler** (`clang-cl` + `lld-link` / MSVC on Windows; GCC 13+ or Clang on Linux; Xcode Clang on macOS)
- **OpenImageIO & OpenEXR** development packages (for full-float Cycles-compatible multilayer EXR pass output)
- **Python 3** (required for binding/policy/source-size checks and differential tools)

For Blender Cycles comparison/benchmark tools (`tools/run_scene_benchmark.py`, `tools/compare_cycles.py`), install the host Python dependencies:

```bash
python -m pip install -r requirements-validation.txt
```

### Platform Toolchains

- **Windows:**
  - Validated with ROCm 10.1.0 (`clang-cl` / LLVM 24 + HIPRT 3.0), Windows 11 D3D12 SDK (`dxil.dll` / `dxcompiler.dll`), Vulkan API 1.4, Embree 4.4.1, and oneTBB 2023.1.0.
- **Linux (Arch / Ubuntu 24.04+):**
  - CPU backends (`fallback`, `simd`) require LLVM development headers, Embree 4.x, oneTBB, plus `libx11-dev` and `uuid-dev`:

    ```bash
    # Arch Linux
    sudo pacman -S --needed blender python-cattrs python-numpy python-pillow openimageio openexr

    # Ubuntu 24.04 LTS example (LLVM 22 + Embree + X11/UUID)
    sudo apt-get update && sudo apt-get install -y \
      cmake ninja-build wget gnupg lsb-release software-properties-common \
      libembree-dev libx11-dev uuid-dev
    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
    sudo ./llvm.sh 22 && sudo apt-get install -y llvm-22-dev
    ```

- **macOS (Apple Silicon):**
  - Uses Apple Clang + Metal (`PSYCLES_ENABLE_LUISA_METAL=ON`, optional `PSYCLES_ENABLE_LUISA_METAL4=ON`), OpenImageIO, and Embree/LLVM when CPU backends are enabled.

---

## Clone & Configure

```bash
git clone --recurse-submodules https://github.com/LuisaGroup/Psycles.git
cd Psycles
cmake -S . -B build -G Ninja
cmake --build build --parallel 32
```

If cloned without submodules:

```bash
git submodule update --init --recursive
```

### CMake Options

| Option | Default | Purpose |
| --- | --- | --- |
| `PSYCLES_ENABLE_OPENIMAGEIO` | `ON` | Multilayer OpenEXR output support (`ViewLayer.<pass>.<component>`) |
| `PSYCLES_ENABLE_LUISA` | `ON` | Build the LuisaCompute DSL compiler and runtime renderer (`psycles_luisa`, `psycles_luisa_runtime`) |
| `PSYCLES_ENABLE_LUISA_FALLBACK` | `ON` | Build LLVM/Embree scalar CPU backend (`fallback`) |
| `PSYCLES_ENABLE_LUISA_SIMD` | `OFF` | Build LLVM/Embree W8 packet-vectorized CPU backend (`simd`) |
| `PSYCLES_ENABLE_LUISA_HIP` | `ON` *(non-Apple)* | Build AMD ROCm HIP/HIPRT backend (`hip`) |
| `PSYCLES_ENABLE_LUISA_VULKAN` | `ON` *(non-Apple)* | Build Vulkan backend (`vk`, defaults to native XIR-to-SPIR-V) |
| `LUISA_COMPUTE_ENABLE_DX` | `ON` on Windows | Build DirectX 12 SM 6.6+ backend (`dx`) |
| `LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV` | `ON` | Use LuisaCompute native XIR-to-SPIR-V instead of DXC for `vk` |
| `PSYCLES_ENABLE_LUISA_METAL` | `ON` on Apple | Build Apple Metal backend |
| `PSYCLES_ENABLE_LUISA_METAL4` | `OFF` | Build Apple Metal 4 AIR backend |
| `PSYCLES_BUILD_TESTS` | `ON` | Build C++ test targets and register CTest gates |
| `PSYCLES_BUILD_EXAMPLES` | `ON` | Build `psycles_luisa_render_demo`, `psycles_render_blender_scene`, `psycles_probe_nishita` |
| `PSYCLES_FETCH_LUISA_NEXT` | `OFF` | Fetch Luisa `next` only when `third_party/LuisaCompute` is absent |

CMake enforces strict backend postconditions: if an enabled `PSYCLES_ENABLE_LUISA_*` backend target cannot be created by LuisaCompute, configuration fails explicitly rather than silently dropping the backend.

### Common Configure Presets

```bash
# Dependency-free contract & SVM compiler core build only
cmake -S . -B build-core -G Ninja -DPSYCLES_ENABLE_LUISA=OFF

# Workstation GPU + CPU SIMD build
cmake -S . -B build -G Ninja -DPSYCLES_ENABLE_LUISA_SIMD=ON

# Vulkan DXC control build (vk-dxc instead of default vk-xir)
cmake -S . -B build-vk-dxc -G Ninja -DLUISA_COMPUTE_ENABLE_VK_XIR_SPIRV=OFF

# Explicit LLVM/Embree prefixes
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -Dembree_DIR=/path/to/embree/lib/cmake/embree
```

---

## Build Targets & Module Layout

- `psycles_core` (`Psycles::core`): scene contract (`src/contract/`), Blender scene adapter (`src/adapter/`), core node schema & native Cycles SVM IR/bytecode compiler (`src/compiler/`), pass I/O & OpenEXR writer (`src/io/`), LightTree/Sobol/pixel filter sampling (`src/sampling/`), Nishita/Hosek sky tables (`third_party/`).
- `psycles_luisa` & `psycles_luisa_runtime`: native Cycles SVM device handlers (`src/luisa/cycles_svm*.cpp`), `DirectJitTranslator` (`src/luisa/cycles_svm_translator.cpp`), path-kernel stages (`src/luisa/path_kernel_*.cpp`), and backend session/scene runtime (`src/luisa/path_tracer_*.cpp`).
- Examples (`examples/`):
  - `psycles_luisa_render_demo`: self-contained demo scene renderer
  - `psycles_render_blender_scene`: exported Blender bundle renderer & diagnostic trace CLI
  - `psycles_probe_nishita`: Nishita sky probe utility

---

## Render CLIs

### Smoke Demo

```bash
./build/bin/psycles_luisa_render_demo output.ppm hip 640 400 256
```

### Blender Export Package Renderer

```bash
./build/bin/psycles_render_blender_scene /path/to/export-package output.ppm vk 1920 1080 256
```

Key positional arguments (`examples/render_blender_scene.cpp`):

1. `export_dir` — Blender export directory (`scene.json` + `geometry.bin`)
2. `output.ppm` — base output path (writes `-combined.pfm` / per-pass PFMs and `<stem>.exr` when OpenImageIO is enabled)
3. `backend` — `hip`, `dx`, `vk`, `metal`, `metal4`, `fallback`, or `simd`
4. `width` *(optional)* — render width (default: scene width)
5. `height` *(optional)* — render height (default: scene height)
6. `samples` *(optional)* — total AA sample count (`SampleRange.total`, which keys the full Tabulated Sobol sequence)
7. `max_samples_per_dispatch` *(optional)*
8. `path_trace_output` *(optional, `-` to skip)*, followed by `pixel_x`, `pixel_y`, `sample_index`, `sample_first`, `sample_count`, `chunk_probe_json` (`-` to skip), `chunk_size`, `probe_full_frame`
9. `scheduler` *(optional, position 17)* — `megakernel` *(default)*, `megakernel-per-sample`, `wavefront`, `wavefront-graph`, `wavefront-staged`, or `persistent`
10. `wavefront_block_size` *(optional, position 18, multiple of 32 in `[32, 1024]`, default `32`)*

When running `vk`, `psycles_render_blender_scene` enforces `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` unless already set in the environment. On multi-GPU AMD systems (APU iGPU + discrete Radeon GPU), set `HIP_VISIBLE_DEVICES=1` before running `hip`.

### Shader Cache Behavior

LuisaCompute stores compiled kernel objects in the runtime `.cache` directory (typically `build/bin/.cache`). Kernel keys capture structure, argument ABI, backend target identity, native includes, and compiler options; runtime inputs (seed, sample range, camera transforms, dynamic resource bindings) do not invalidate cached shaders.

- Delete `build/bin/.cache` (or run with `PSYCLES_DISABLE_SHADER_CACHE=1`) to measure cold JIT compilation.
- Keep separate build/runtime `.cache` directories when comparing `vk-xir` vs `vk-dxc`.

---

## Testing & Verification Gates

```bash
# Run full CTest suite
ctest --test-dir build --output-on-failure -j16

# CPU-only CTest (exclude GPU targets on headless hosts)
ctest --test-dir build --output-on-failure -j16 -E "_(hip|dx|vk|metal|metal4)$"

# Repository policy & structure checks
python tools/check_source_size.py
python tests/test_luisa_shader_performance_policy.py
python -m unittest discover -s tests -p "test_cycles_binding_oracle.py"
```

When validating Vulkan's native XIR-to-SPIR-V path explicitly:

```bash
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure --parallel 1 -R '<vulkan-test-regex>'
```

Keep GPU CTest runs sequential or bounded on a single GPU to avoid device-queue contention. Note that on Windows systems with `fallback`, six pre-existing `LuisaCompute` `fallback`-only tests (`luisa_homogeneous_volume_fallback`, `luisa_volume_triangle_fallback`, `luisa_volume_environment_fallback`, `luisa_background_sampling_fallback`, `luisa_cycles_ray_portal_render_fallback`, and `luisa_cycles_svm_hair_scattering_fallback`) are tracked upstream (`simd` and GPU variants pass).

---

## Troubleshooting

- **`LuisaCompute submodule is not initialized`:** Run `git submodule update --init --recursive`.
- **HIP selects integrated APU graphics instead of discrete Radeon GPU:** Export `HIP_VISIBLE_DEVICES=1`.
- **LLVM / Embree / OpenImageIO not found:** Pass `-DLLVM_DIR=...`, `-Dembree_DIR=...`, or `-DOpenImageIO_DIR=...` explicitly.
- **Linux fallback fails with missing `X11/Xlib.h`:** Install `libx11-dev` (required by Luisa's fallback/swapchain support module even when GUI is disabled).
- **Deleted or moved CMake source files leave stale `.obj` link errors on Windows Ninja:** Clean `CMakeFiles/psycles*.dir` and `bin/psycles_*` artifacts, then reconfigure and rebuild.
