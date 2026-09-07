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

Psycles uses the pinned `third_party/LuisaCompute` submodule as a CMake
subdirectory so that normal builds and validation use the same tested Luisa
revision. The commands above use every hardware thread on the validated
16-core/32-thread workstation; use the machine's hardware-thread count on
another host.

For a dependency-free contract-core build:

```bash
cmake -S . -B build-core -G Ninja -DPSYCLES_ENABLE_LUISA=OFF
```

See [BUILD.md](BUILD.md) for prerequisites, backend options, cache behavior,
and troubleshooting. Current implementation status is in
[docs/cycles-compatibility.md](docs/cycles-compatibility.md); development rules
and completion gates are in [DEVELOP.md](DEVELOP.md).

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

The [scene benchmark](docs/scene-benchmark.md) supports the full Cycles CPU/HIP
and Psycles fallback/HIP/Vulkan matrix, or an explicit focused HIP campaign.
It compares Cycles' original main-loop wall time with Psycles' render-only
wall time and records compilation, enclosing process times, EXR hashes,
numeric pass comparisons and device-labeled triptychs separately.

When a pass comparison diverges, the
[per-path Cycles oracle](docs/cycles-path-trace.md) compares RNG dimensions,
raw closures, light/BSDF sampling, PDFs, and path-state transitions at the
first mismatching event. Discrete state and random samples are exact gates;
continuous fields have explicit float32 bounds.

## Current execution path

Native Cycles 5.2.1 SVM is the default material path, including surface,
background, sampled/forward light emission and volume consumers. The staged
path tracer uses Cycles-aligned coroutine boundaries and generic Luisa
scheduler extensions for application-specific surface sorting. Local SVM
arrays are bounded by static compiler analysis, not rendering profiles.

Lone Monk, Monster, Classroom and Barbershop run on HIP at 256 spp, with
original Cycles linear-pass comparisons. This is not a claim of complete
SVM coverage or performance parity. Nine semantic opcodes remain unsupported,
indirect-light/path residuals remain under diagnosis, and the displacement
prepass still contains a private legacy evaluator that must be removed.
See [the current compatibility status](docs/cycles-compatibility.md) and
[validation index](VALIDATION.md) for exact gates and evidence.

Unsupported execution features are explicit compilation diagnostics rather
than silent approximations. Node and render semantics are accepted only after
comparison with official Cycles output.
