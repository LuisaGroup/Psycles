# Psycles handoff — 2026-07-28

## Branch and scope

Continue on `refactor/path-tracer-modules`, based on
`main@ad360032f064c98837dcfa06300a3839c54d3817`. The branch is intentionally
ahead of `main` in small recovery checkpoints. Do not restart from `main` or
from an older scratch clone.

The required order remains:

1. integrate the verified Blender 4.5.10 tabulated-Sobol stream into the
   production path kernel with its fixed random dimensions;
2. establish the intentionally changed production-kernel cache identity, then
   split `src/luisa/path_tracer.cpp` by stable responsibility without further
   changing rendering semantics or its nine-argument kernel ABI;
3. add the Cycles single-light distribution and Nishita importance CDF;
4. publish each fully tested rendering checkpoint with official Cycles
   linear-pass comparisons and viewable images;
5. validate additional Blender demo scenes;
6. replace expanded per-material DSL ASTs with a buffer-driven shared device
   instruction executor and remeasure cold/hot JIT.

## Published checkpoints

- `80446d6`: remove `PSYCLES_LUISA_SOURCE_DIR` and use the pinned Luisa
  submodule as the normal source.
- `059a5a6`: record the clean GNU 13.3/CMake 3.27.7 core build and 4/4 tests.
- `7ab57cf`: restore the official Blender 4.5.10 host tabulated-Sobol
  generator, fixed dimensions, whole-table IEEE-754 fingerprint, and 5/5
  clean core gate.
- `74c2f45`: require a real fallback target and pin the Luisa LLVM 22
  shared-library fix after building the LLVM 22.1.8/Embree 4.3.0 module.

Checkpoint 1b is now green. The real fallback fixture loads the backend
module, initializes Embree 4.3.0, compiles the Luisa sampler with cache and
fast math disabled, dispatches it, and compares device-side `uint4` bitcasts.
The complete fallback CTest gate passes 8/8.

## Exact current boundary

- `include/psycles/luisa/cycles_sampler.h` contains the Luisa DSL lowering for
  pixel hashing, Owen scrambling, pattern shuffling, fixed dimensions, and
  table lookup. Buffer variables are passed by const reference because Luisa
  DSL variables are non-copyable.
- `tests/test_luisa_compile.cpp` instantiates that lowering in a Luisa kernel,
  while `tests/test_luisa_sobol_fallback.cpp` executes it on the fallback
  device. Runtime uniforms prevent the hash, bounce, and scramble path from
  collapsing into a compile-time-only fixture; sentinel output uploads catch
  skipped dispatches.
- The device fixture locks camera sample 0/dimension 0, first-path-step
  light/BSDF dimensions 17/19, next-path-step light dimension 33, a
  16-dimension stride, pixel hash `0x4bf378cb`, shuffled indices
  38201/54017/10898/30843, and all 16 returned IEEE-754 lanes.
- `src/luisa/path_tracer.cpp` is untouched by this recovery. It still uses the
  old PCG-style state and still samples light classes separately.
- No single-light distribution, world-area emissive CDF, Nishita importance
  CDF, or production kernel Sobol integration has landed yet.
- The pinned Luisa submodule is `9f0c3287`; it respects LLVM's exported shared
  library policy and avoids the LLVM 22 component-archive `try_compile`
  failure.
- Ubuntu 24.04.3, LLVM 22.1.8, Embree 4.3.0, GCC 13.3, CMake 3.27.7, and
  Ninja 1.11.1 configure and link a non-empty ELF
  `libluisa-backend-fallback.so`. Installation and CMake paths are in
  `BUILD.md`.
- Psycles now fails configuration when fallback is requested but Luisa does
  not create the fallback target. A silent core-only downgrade is not
  accepted.

## First actions in the next thread

1. Add an explicit total-AA-sample-count contract. Derive the table sequence
   size from that total, never from a progressive render chunk.
2. Bind the table resource and sequence size in the production path kernel,
   replacing PCG calls with the fixed camera and per-`path_step` dimensions.
   Transparent events advance `path_step` even when regular bounce depth does
   not.
3. Record the resulting intentional kernel hash/cache identity, then move the
   monolith into private sampling, geometry, environment, lights, kernel,
   render-session, and scene-compiler modules while preserving the exact
   nine-argument ABI and rendered bits.
4. Implement Cycles' unified single-light selection density, then the
   device-built Nishita conditional and marginal importance CDFs.
5. Compare every transport/sampling boundary against official Blender 4.5.10
   Cycles linear passes and upload new meaningful comparison images.
6. Update all three handoff/build documents and push immediately after each
   passing boundary.

Do not spend time building a separate CPU renderer or expanding CPU-only
gates. The host Sobol code remains solely LUT-generation infrastructure for
Luisa. Rendering acceptance is Luisa fallback execution plus official Cycles
linear-pass comparison.
