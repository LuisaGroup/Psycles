# Psycles handoff — 2026-07-28

## Branch and scope

Continue on `refactor/path-tracer-modules`, based on
`main@ad360032f064c98837dcfa06300a3839c54d3817`. The branch is intentionally
ahead of `main` in small recovery checkpoints. Do not restart from `main` or
from an older scratch clone.

The required order remains:

1. finish Blender 4.5.10 tabulated-Sobol device parity, fixed random
   dimensions, single-light distribution, and Nishita importance sampling;
2. publish a fully tested rendering checkpoint;
3. split `src/luisa/path_tracer.cpp` by stable responsibility without changing
   rendering semantics, kernel argument ABI, or fallback-cache identity;
4. validate additional Blender demo scenes;
5. replace expanded per-material DSL ASTs with a buffer-driven shared device
   instruction executor and remeasure cold/hot JIT.

## Published checkpoints

- `80446d6`: remove `PSYCLES_LUISA_SOURCE_DIR` and use the pinned Luisa
  submodule as the normal source.
- `059a5a6`: record the clean GNU 13.3/CMake 3.27.7 core build and 4/4 tests.
- `7ab57cf`: restore the official Blender 4.5.10 host tabulated-Sobol
  generator, fixed dimensions, whole-table IEEE-754 fingerprint, and 5/5
  clean core gate.

The handoff commit after those checkpoints stages the Luisa sampler lowering,
the corresponding AST instantiation, headless Luisa configuration, and this
documentation. It is deliberately marked unverified.

## Exact current boundary

- `include/psycles/luisa/cycles_sampler.h` contains the Luisa DSL lowering for
  pixel hashing, Owen scrambling, pattern shuffling, fixed dimensions, and
  table lookup.
- `tests/test_luisa_compile.cpp` instantiates that lowering in a Luisa kernel.
- `src/luisa/path_tracer.cpp` is untouched by this recovery. It still uses the
  old PCG-style state and still samples light classes separately.
- No single-light distribution, world-area emissive CDF, Nishita importance
  CDF, or production kernel Sobol integration has landed yet.
- The first Luisa dependency build was stopped at the user's request while
  XIR compilation reported 95%. That is not a passing AST or fallback gate.
- The next fallback environment is Ubuntu 24.04, LLVM 22.1.8, Embree 4.3.0,
  GCC 13.3, and CMake 3.27.7. Installation and CMake paths are in `BUILD.md`.

## First actions in the next thread

1. Install/verify LLVM 22 and Embree 4.3 as documented in `BUILD.md`.
2. Configure with `PSYCLES_ENABLE_LUISA_FALLBACK=ON` and confirm CMake prints
   the exact LLVM and Embree versions rather than silently disabling fallback.
3. Finish and run `psycles_luisa_compile_tests`.
4. Add a real fallback device fixture that uploads the 256-pattern table,
   executes camera, first-bounce light/BSDF, and next-bounce light lookups, and
   compares every output as IEEE-754 bits against
   `tests/test_tabulated_sobol.cpp`.
5. Only after that fixture passes, bind the table resource and sequence size
   in the production Luisa path kernel and replace the PCG calls with fixed
   dimensions.
6. Update `DEVELOP.md` and push immediately after each passing boundary.

Do not spend time building a separate CPU renderer or expanding CPU-only
gates. The host Sobol code remains solely LUT-generation infrastructure for
Luisa. Rendering acceptance is Luisa fallback execution plus official Cycles
linear-pass comparison.
