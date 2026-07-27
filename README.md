# Psycles

Psycles is a new renderer implementation that treats Blender Cycles as an
interface and semantic contract, while expressing execution through
LuisaCompute's multistage C++ DSL and runtime JIT.

The project deliberately does **not** translate Cycles SVM bytecode, reuse the
Cycles GPU kernel ABI, or commit to a wavefront/megakernel schedule. Its current
milestone establishes the semantic and compilation boundaries that later
geometry, transport, and scheduling implementations must obey:

- a typed, extensible ShaderGraph contract with explicit surface, volume, and
  displacement roots;
- a pre-SVM Cycles graph adapter with explicit node/socket coverage failures;
- a parameterized `SurfaceProgram` that preserves add/mix closure trees;
- structural-versus-parameter change analysis for correct JIT invalidation;
- an atomic incremental material library that reuses programs on parameter-only
  edits;
- transactional scene snapshots with stable resource identifiers;
- a Luisa DSL `Polymorphic<Surface>` protocol retained as the canonical
  device-side material dispatch boundary;
- a Luisa device execution path with camera rays, hardware/fallback RayQuery,
  transformed instances, multi-bounce transport, direct-light sampling, film
  accumulation, and pass output;
- direct differential validation against Blender Cycles linear render passes.

See [docs/architecture.md](docs/architecture.md) for the design and current
scope and [docs/cycles-compatibility.md](docs/cycles-compatibility.md) for the
shader-graph path, integrator contract, and explicit compatibility gaps.

## Build

Clone with submodules and build the default Luisa/fallback configuration:

```bash
git clone --recurse-submodules https://github.com/LuisaGroup/Psycles.git
cd Psycles
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build --output-on-failure
```

Psycles uses `third_party/LuisaCompute` as a CMake subdirectory. An external
LuisaCompute checkout can still override the submodule:

```bash
cmake -S . -B build -G Ninja \
  -DPSYCLES_LUISA_SOURCE_DIR=/path/to/LuisaCompute
```

For a dependency-free contract-core build:

```bash
cmake -S . -B build-core -G Ninja -DPSYCLES_ENABLE_LUISA=OFF
```

See [BUILD.md](BUILD.md) for prerequisites, backend options, cache behavior,
and troubleshooting. Current implementation status and release gates are in
[DEVELOP.md](DEVELOP.md).

## Render through Luisa

```bash
./build/bin/psycles_luisa_render_demo psycles-luisa.ppm fallback 640 400 256
```

The arguments are output path, Luisa backend, width, height, and samples per
pixel. `fallback` is LuisaCompute's current host fallback backend; it executes
the same Luisa DSL program and is not a separate Psycles renderer.

Correctness is not inferred from another Psycles implementation. The
regression harness renders the same `.blend` with Blender Cycles to linear
multilayer EXR and compares those passes directly with Psycles-Luisa output.

## Current vertical slice

The implemented slice can:

1. accept a normalized, pre-SVM Cycles graph representation;
2. map constant, RGB, geometry, math add/multiply, color mix, diffuse,
   emission, transparent, add-closure, and mix-closure nodes;
3. validate node coverage, sockets, roots, graph types, and acyclicity;
4. compile the surface root to typed float, float3, and closure instructions;
5. keep editable values in a generated parameter block while using graph
   structure and static properties as the program signature;
6. rebind material parameters without regenerating a `SurfaceProgram`, or
   recompile atomically when structure changes;
7. trace the program through Luisa DSL as a polymorphic `GraphSurface`;
8. evaluate Lambertian and current Cycles rough diffuse semantics, emission,
   RGB transparent extinction, and add/mix closure sample weights;
9. apply complete scene deltas atomically and reject dangling references;
10. upload cameras, transformed triangle instances, smooth normals, material
    overrides, point/spot/area/distant/background lights, and world emission
    to Luisa resources and acceleration structures;
11. render multi-bounce transport with next-event estimation, Russian
    roulette, deterministic sampling, and Film accumulation in Luisa kernels;
12. output Combined, Normal, Albedo, Denoising Normal/Albedo, and Sample Count
    passes.

Unsupported execution features are explicit compilation diagnostics rather
than silent approximations. Node and render semantics are accepted only after
comparison with official Cycles output.
