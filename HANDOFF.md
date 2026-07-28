# Psycles handoff — 2026-07-28

## Branch and published boundary

Continue on `refactor/path-tracer-modules`; do not restart from `main` or an
older scratch checkout.

The important published checkpoints are:

- `74c2f45`: require the real Luisa fallback target and validate Ubuntu 24.04,
  LLVM 22.1.8, Embree 4.3.0, GCC 13.3, CMake 3.27.7, and Ninja 1.11.1.
- `a4ba31c`: execute the tabulated-Sobol fixture on the fallback device and
  compare all returned `float4` values through device-side `uint4` bitcasts.
- LuisaCompute `next@f42f3c6e`: integrate the LLVM shared-library discovery
  fix and repair an XIR `local_load_elimination` heap use-after-free.
- `35f04ae`: integrate the verified Sobol stream into the production path
  kernel while retaining its nine top-level arguments.
- `cba0428`: split the production Luisa path tracer into private host/device
  modules without changing its public backend API or rendered results.

The Psycles submodule now pins LuisaCompute `f42f3c6e`. That commit is directly
on `LuisaGroup/LuisaCompute:next`; no PR or force push was used.
The final pin was reconfigured against LLVM 22.1.8/Embree 4.3.0, completed
245 incremental fallback build steps, and passed the complete Psycles CTest
gate 8/8.

## Exact verified state

### Production sampling

- `SampleRange.total` is the total AA sample count for the whole render, not
  the size of a progressive chunk. A render session rejects zero totals,
  out-of-range chunks, and attempts to change the total mid-session.
- Both render CLIs set `count` and `total` from the requested spp.
- The production kernel uploads a `Buffer<float4>` Sobol table, keeps the
  existing nine-argument ABI by replacing the old scalar seed slot with that
  buffer, and stores seed/sequence size in `RenderKernelParameters`.
- Camera filter uses Cycles dimension 0 with sample 0 forced to the pixel
  center. Lens/time uses dimension 1. Each `path_step` consumes the fixed
  terminate, light, light-terminate, and surface-BSDF lanes with the
  16-dimension stride. Light/BSDF component selection uses `z`; shape
  sampling uses `x/y`.
- The old PCG generator has no remaining production call sites.

### Path-tracer architecture

`src/luisa/path_tracer.cpp` is now a 51-line public-backend façade. The
runtime target compiles that façade with nine private translation units:

| Module | Responsibility |
|---|---|
| `path_tracer_common.cpp` | Host conversions, packed callable values, pass helpers, and diagnostics |
| `path_tracer_sampling.cpp` | Cycles-compatible pixel-filter callable |
| `path_tracer_lighting.cpp` | MIS, contribution splitting, roulette, and emissive-triangle PDF callables |
| `path_tracer_surfaces.cpp` | Strongly typed surface evaluate/emission/sample/AOV dispatch callables |
| `path_tracer_environment.cpp` | World graph, environment-sun, and Nishita-sun callables |
| `path_tracer_geometry.cpp` | Transparent shadow traversal and material lookup |
| `path_tracer_kernel.cpp` | The single production path-state machine and shader compilation |
| `path_tracer_session.cpp` | Sobol upload, progressive-range validation, dispatch, and pass readback |
| `path_tracer_scene.cpp` | Scene/material/geometry/light compilation and GPU resource upload |
| `path_tracer.cpp` | Backend construction and render-session façade |

Private resource/session declarations live in `path_tracer_internal.h`;
device ABI structs in `path_tracer_types.h`; buffer-backed graph services and
callable signatures have their own private headers. The main path-state
machine remains cohesive in `path_tracer_kernel.cpp`; this refactor did not
invent a new public path-state ABI.

The shader still has nine explicit top-level arguments. Fallback metadata
retains `ARGUMENT_HASH cf9ee8fec3c444f6` and `ARGUMENT_COUNT 19`. Introducing
explicit Luisa callable boundaries legitimately changed the structural cache
key from the historical `kernel_70ce93bbfda41afc` to
`kernel_4c0f6e0d82a53e90`; do not force the old name. The new key hot-loads in
about 8–11 ms.

As an architecture regression gate, all 13 PFM outputs from each of
`emission_surface`, `diffuse_bsdf_matrix`, and `diffuse_surface` match the
pre-refactor Psycles outputs byte for byte (39/39). A final repeated
`emission_surface` run also matched all 13 passes byte for byte across
processes. This is a refactor-equivalence result, separate from the
Psycles-versus-Cycles accuracy measurements below.

### Cycles 4.5.10 comparisons

All comparisons use official Blender 4.5.10 linear passes:

| Probe | Settings | Result |
|---|---:|---|
| `emission_surface` | 64×64, 4 spp | Combined, Emit, and Normal RMSE/max error are all 0 |
| `diffuse_bsdf_matrix` | 64×64, 4 spp | Combined, DiffCol, DiffDir, and Normal RMSE/max error are all 0 |
| `diffuse_surface` | 64×64, 16 spp | Combined RMSE `0.006317606`, mean-energy ratio `0.999740158`, invalid pixels 0 |

Do not describe `diffuse_surface` as bitwise or pixel-exact. Its remaining
error is concentrated around geometry silhouettes and direct-light sampling,
which still lacks the Cycles unified light distribution.

### Luisa XIR stability

The first production cold compile exposed a nondeterministic XIR crash:

- before the fix, 6/20 cold compiles crashed;
- ASan located a heap use-after-free in
  `src/xir/passes/local_load_elimination.cpp`: the pass retained a reference
  to a dense-map value and then inserted missing predecessor entries, allowing
  reallocation to invalidate that reference;
- the fix pre-creates every block/predecessor data-flow entry and permits only
  non-inserting `find` access during iteration;
- the new
  `local_load_elim_loop_fanout_keeps_analysis_storage_stable` fixture aborts
  with the old implementation and passes with the fix under ASan/UBSan;
- the release production-kernel stress gate improved from 6/20 failures to
  20/20 successful cold compiles;
- a subsequent hot load took about 9.498 ms, and all 13 cold/hot PFM passes
  were byte-identical.

The targeted UAF is clean. Do not claim that the entire Luisa ASan test suite
is globally clean: unrelated existing tests still contain sanitizer findings.

## Work that is deliberately still open

- Analytic lights, emissive triangles, and the environment still have separate
  class-specific selection paths. Cycles' unified single-light distribution
  and its exact selection PDF are not implemented.
- World/emissive importance distributions and the Nishita conditional and
  marginal importance CDFs are not implemented.
- The full per-event random-dimension trace is not complete even though the
  production camera/light/BSDF/RR lanes now use the fixed Cycles dimensions.
- No new persistent comparison image was uploaded during this closeout; the
  next meaningful renderer change must include viewable Cycles/Psycles
  comparison and difference images, not only scalar metrics.

## First actions in the next thread

1. Read this file, `DEVELOP.md`, and `BUILD.md`, then confirm the branch and
   submodule hashes before editing.
2. Implement Cycles' unified single-light selection distribution and exact
   MIS selection density.
3. Re-run the focused official Cycles linear-pass probes and upload viewable
   comparison/difference images for that renderer change.
4. Add world/emissive distributions, then the device-built Nishita
   conditional and marginal importance CDFs.
5. Complete a per-event random-dimension trace for camera, light, BSDF,
   transparency, and Russian roulette.
6. Update these documents and push immediately after each passing boundary.

Do not build a separate CPU renderer or CPU oracle. Cycles 4.5.10 is the sole
rendering reference; acceptance is real Luisa execution plus official Cycles
linear-pass comparison.
