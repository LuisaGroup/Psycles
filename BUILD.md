# Building Psycles

Psycles builds LuisaCompute as a CMake subdirectory. The repository pins the
tested Luisa revision in `third_party/LuisaCompute`; a normal user build does
not need a separate Luisa installation or a second configure step.

## Prerequisites

- Git with submodule support
- CMake 3.26 or newer
- A C++20 compiler
- Ninja or another CMake generator
- Python 3 for the versioned compatibility checks

The host-only `fallback` backend additionally needs LLVM, Embree, and X11
development packages discoverable by CMake. The active fallback validation
baseline is:

| Component | Version |
|---|---:|
| Ubuntu | 24.04 LTS (Noble) |
| GCC | 13.3 |
| CMake | 3.27.7 |
| Ninja | 1.11.1 |
| LLVM | 22.1.8 |
| Embree | 4.3.0 |

Install LLVM 22 from the
[official LLVM Debian/Ubuntu repository](https://apt.llvm.org/), then install
the development packages required by Luisa's fallback backend:

```bash
sudo apt-get update
sudo apt-get install -y \
  cmake ninja-build wget gnupg lsb-release software-properties-common \
  libembree-dev libx11-dev
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 22
sudo apt-get install -y llvm-22-dev
```

The repository disables Luisa's unused GUI component, so a headless fallback
build does not require GLFW. Luisa's fallback module still compiles its common
Vulkan swapchain support on Linux and therefore requires the X11 development
headers. GPU backend prerequisites are inherited from LuisaCompute.

## Clone and build

```bash
git clone --recurse-submodules https://github.com/LuisaGroup/Psycles.git
cd Psycles
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

CMake stops with this exact recovery command when Luisa execution is enabled
but the bundled submodule is absent. It does not silently produce a core-only
build.

## Configuration

| Option | Default | Purpose |
|---|---:|---|
| `PSYCLES_ENABLE_LUISA` | `ON` | Build the Luisa DSL/runtime renderer |
| `PSYCLES_ENABLE_LUISA_FALLBACK` | `ON` | Build the LLVM/Embree host backend |
| `PSYCLES_BUILD_TESTS` | `ON` | Build C++ tests and register compatibility gates |
| `PSYCLES_BUILD_EXAMPLES` | `ON` | Build render and inspection CLIs |
| `PSYCLES_FETCH_LUISA_NEXT` | `OFF` | Fetch `next` only when the initialized submodule is unavailable |

The source selection order is:

1. initialized `third_party/LuisaCompute`;
2. CMake `FetchContent` when `PSYCLES_FETCH_LUISA_NEXT=ON`;
3. a configuration error with an actionable submodule command.

To build only the dependency-free scene, graph, and compiler contracts:

```bash
cmake -S . -B build-core -G Ninja -DPSYCLES_ENABLE_LUISA=OFF
cmake --build build-core
ctest --test-dir build-core --output-on-failure
```

For a fallback build whose packages are outside the system prefix, provide
their package directories explicitly:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -Dembree_DIR=/path/to/embree/lib/cmake/embree-4.3.0
```

With the Ubuntu packages above, the explicit equivalent is:

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
  -Dembree_DIR=/usr/lib/x86_64-linux-gnu/cmake/embree-4.3.0
```

Configuration must print:

```text
Build with fallback backend (LLVM 22.1.8, Embree 4.3.0)
```

`PSYCLES_ENABLE_LUISA_FALLBACK=ON` is a strict postcondition: configuration
fails if Luisa does not create `luisa-compute-backend-fallback`. Psycles never
silently accepts a core-only build when fallback execution was requested.

## Render

The small Luisa smoke test is:

```bash
./build/bin/psycles_luisa_render_demo output.ppm fallback 640 400 256
```

The normalized Blender-scene renderer is:

```bash
./build/bin/psycles_render_blender_scene \
  /path/to/export-package output.ppm fallback 640 480 64
```

Its positional arguments are export directory, output path, backend, width,
height, and samples per pixel.

## Fallback shader cache

The fallback backend persists a native object and exact metadata for an
unnamed cache-enabled kernel. A valid hit skips AST-to-XIR conversion, LLVM IR
generation, optimization, and object generation. The key covers kernel
structure, argument ABI, LLVM and target-machine identity, builtins, native
includes, and code-generation options.

Runtime values intentionally remain kernel arguments or resources and are not
part of the cache identity. Changing a seed, sample range, camera value, or
material parameter therefore reuses the compiled shader when its type and
binding layout are unchanged.

With Luisa's default file-backed `BinaryIO`, CLI builds place cache entries in
the runtime `.cache` directory (normally `build/bin/.cache`). Removing that
directory forces a cold compile. Applications that supply a custom `BinaryIO`
control cache storage themselves.

The current Lone Monk fallback baseline uses the same cached main kernel for
both 64×48/1 spp and 640×480/64 spp:

| Frozen-runtime measurement | Time |
|---|---:|
| Cold shader JIT | 327.574 s |
| Hot shader JIT | 0.682609 s |
| Hot main-object load | 1.8655 ms |
| 640×480/64 spp hot shader setup | 0.789711 s |

The cold and hot 64×48 runs are byte-identical across all 13 emitted linear
passes. These figures measure cache behavior, not acceptable first-build
latency: replacing per-material expanded AST with the shared device
instruction executor remains an active development goal.

## Validation

The normal release gate is:

```bash
cmake --build build
ctest --test-dir build --output-on-failure
```

This includes Luisa AST construction, Blender import, core contracts, the
real fallback-device tabulated-Sobol bit fixture, versioned Cycles node
inventory, integrator baselines, and analytic-light baselines. The device
fixture is registered only when the fallback target exists; it disables
shader caching and fast math, dispatches runtime-uniform hash/sample/path-step
inputs, and compares device-side bitcasts rather than host-converted floats.

The focused sampling gate is:

```bash
cmake --build build --target \
  psycles_tabulated_sobol_tests \
  psycles_luisa_compile_tests \
  psycles_luisa_sobol_fallback_tests
ctest --test-dir build \
  -R 'psycles\.(tabulated_sobol|luisa_ast|luisa_sobol_fallback)$' \
  --output-on-failure
```

A full compatibility claim additionally requires the focused official-Cycles
linear-pass probes described in
`docs/cycles-compatibility.md`.

## Troubleshooting

- `LuisaCompute submodule is not initialized`: run
  `git submodule update --init --recursive`.
- `LLVM or Embree not found`: set `LLVM_DIR` and `embree_DIR`, or disable
  fallback with `-DPSYCLES_ENABLE_LUISA_FALLBACK=OFF` when another Luisa
  backend is available. Do not treat a configure that omits the exact fallback
  version line above as a passing fallback build.
- `X11/Xlib.h` not found: install `libx11-dev`; disabling Luisa GUI does not
  remove this fallback-module dependency.
- A dependency changed but CMake retained old feature tests: re-run the CMake
  configure command before rebuilding.
- A generated shared library is empty or is not reported as ELF/PE/Mach-O:
  treat it as an interrupted/corrupt build artifact and rebuild that target;
  do not accept passing stale test executables as validation.
