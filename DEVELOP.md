# Psycles development status

This is a living implementation plan, last updated 2026-07-28. “Implemented”
does not mean “Cycles compatible”: compatibility requires an official Blender
4.5.10 Cycles probe and a versioned linear-pass baseline.

## Active handoff checkpoint

The modular path-tracer work was merged by PR #1 at `main@2f49868`; this
handoff continues directly on `main` with the Blender evaluated-geometry cache
repair. Every independently validated stage is recorded here and pushed before
the next long-running compile or render:

1. [x] split `src/luisa/path_tracer.cpp` by stable responsibility without
   changing rendering semantics, the kernel argument ABI, or automatic cache
   behavior;
2. implement Cycles' unified single-light distribution, world/emissive
   distributions, and Nishita conditional/marginal importance CDFs;
3. expand official Cycles linear-pass validation to additional Blender demo
   scenes;
4. replace per-material expanded DSL ASTs with a buffer-driven shared device
   instruction executor and remeasure cold/hot compilation.

The Git recovery boundary is important: the interrupted session's experimental
unified-light, triangle/area-light sampling, and Nishita changes were not
published and are not present in this tree. The repository contains the
validated modular split, production Sobol work, and the exporter cache repair
described below. Resume unified-light work from the unchecked P1 items and
Cycles 4.5.10 source rather than assuming the chat-only experiment survived.

The tabulated-Sobol host and fallback-device fixtures were independently
recovered and verified on this branch, and the same stream is now integrated
into the production path kernel. The production result is accepted only from
official Blender 4.5.10 linear-pass probes; historical chat statistics are not
treated as evidence.

Checkpoint 0 removes the `PSYCLES_LUISA_SOURCE_DIR` override so the pinned
submodule is the single normal source of LuisaCompute. Repository-wide option
references and whitespace checks pass. A clean GNU 13.3/CMake 3.27.7
`PSYCLES_ENABLE_LUISA=OFF` configure and build also pass with the Unix
Makefiles generator, followed by 4/4 core CTest groups. The existing
Luisa/fallback 6/6 gate remains the required full-build check before any
rendering change is published; checkpoint 1b expands that gate to 8/8.

Checkpoint 1a restores the Blender 4.5.10 tabulated-Sobol host contract as an
independent `psycles::sampling` module. Its test locks the complete
256-pattern × 256-sample × float4 table with an IEEE-754 FNV-1a fingerprint,
plus pixel hash, camera, first-bounce light/BSDF, next-bounce light, and
16-dimension bounce-stride fixtures. The fixture was independently regenerated
from the authoritative Cycles algorithm; the clean core gate is now 5/5.
Luisa device lowering and path-kernel integration are intentionally tracked as
the next checkpoint rather than implied by this host-only result.

Checkpoint 1b is a passing device-sampling gate. It adds a Luisa-native
tabulated-Sobol lowering in
`include/psycles/luisa/cycles_sampler.h`, extends the Luisa AST test to
instantiate it, and executes the lowering in
`tests/test_luisa_sobol_fallback.cpp`. The fixture uploads the complete table,
passes pixel/sample/path-step values as runtime uniforms, initializes outputs
to sentinels, performs device-side float-to-uint bitcasts, and locks four
camera/light/BSDF probes plus dimension/index metadata. Shader cache and fast
math are disabled for this fixture.

The recovery environment configures and builds a non-empty ELF
fallback module with Ubuntu 24.04.3, GCC 13.3, CMake 3.27.7, Ninja 1.11.1,
LLVM 22.1.8, and Embree 4.3.0. CMake also rejects
`PSYCLES_ENABLE_LUISA_FALLBACK=ON` if Luisa silently disables the backend.
The focused AST/host/device gate passes 3/3 and the complete fallback gate
passes 8/8.

Checkpoint 1c integrates that stream into the production kernel. It adds a
whole-render `SampleRange.total` contract, derives the Sobol sequence size from
that total rather than a progressive chunk, uploads the production
`Buffer<float4>` table, and replaces the old scalar seed argument while
retaining the nine-argument kernel ABI. Seed and sequence size are runtime
parameters. Camera filter/lens, per-`path_step` light, light roulette, BSDF,
and Russian-roulette sampling use the fixed Cycles dimensions; no PCG call
sites remain. Do not extend this work into a CPU renderer or additional
CPU-only validation.

Checkpoint 1d is the architecture-only production split published as
`cba0428`. The public backend façade is 51 lines; common host utilities,
sampling, lighting, surfaces, environment, geometry, the kernel, the render
session, and the scene compiler now have separate private implementation
boundaries. The production path-state machine remains one cohesive kernel.
The nine explicit shader arguments, `SampleRange.total`, fixed Sobol lanes,
and lowered argument metadata are unchanged.

The split completed a full fallback build and the complete CTest gate 8/8.
Across `emission_surface`, `diffuse_bsdf_matrix`, and `diffuse_surface`, all
39 emitted Psycles PFM passes match their pre-refactor baselines byte for
byte. The explicit callable boundaries produce a legitimate one-time
structural-cache migration; the modular focused key is
`kernel_4c0f6e0d82a53e90`, with the same
`ARGUMENT_HASH cf9ee8fec3c444f6` and `ARGUMENT_COUNT 19` as the historical
production-Sobol key.

The current Lone Monk 640x480 end-to-end diagnostic exposed an exporter
geometry-cache regression. `arch.005` through `arch.008` share one source Mesh
datablock but use object-specific Mirror modifier inputs; their evaluated
render widths are approximately 8.88, 15.07, 15.09, and 8.88 scene units.
Caching every evaluated geometry by the source datablock incorrectly mapped
all four instances to the small `arch.005` mesh. The repair only shares
unmodified render Mesh datablocks, including the effective material-slot
signature, while modified and non-Mesh objects retain object-specific
evaluated geometry. The Blender regression exports four objects as exactly
three geometries: two unmodified instances share, while Array=2 and Array=4
objects sharing one source Mesh retain distinct 4/8-triangle results.

The repaired full Lone Monk export contains 7,543 instances and 350
geometries. `arch.005` through `arch.008` map to distinct geometry IDs
262/263/264/265 with independently hashed vertex streams and evaluated widths
8.8776/15.0709/15.0898/8.8762; 841 `grass_blade.002` instances still share
geometry 340. A clean Release fallback build and all eight existing tests plus
the Blender regression pass (9/9 with Blender 4.5.10 available). The fresh
640x480, 64 spp, seed-0 Psycles render had not completed its first fallback JIT
when this handoff was requested and produced no image, so visual acceptance of
the repaired arches remains the immediate next gate.

The production integration exposed a Luisa XIR
`local_load_elimination` heap use-after-free. LuisaCompute `next@f42f3c6e`
pre-creates all block/predecessor data-flow entries and forbids map insertion
while a block result reference is live. The new loop/fanout fixture fails the
old implementation under ASan and passes the fix. Production cold JIT changed
from 6/20 crashes to 20/20 successes; a subsequent hot load took about
9.498 ms and all 13 PFM outputs were byte-identical.

Every transport, light-distribution, or environment-sampling boundary must be
compared against the official Blender 4.5.10 Cycles linear passes. New
meaningful render comparisons are uploaded as viewable images in addition to
recording RMSE, energy ratios, invalid-pixel counts, and deterministic-cache
status.

## Current checkpoint

| Area | Verified state |
|---|---|
| Shader inventory | 96 Cycles-applicable nodes tracked from 105 Blender shader node types |
| Complete coverage | 43/96 complete: 41 `cycles_verified` device nodes and 2 structural output adapters |
| Remaining nodes | 12 partial and 41 pending; no implemented node is waiting for a probe, and 1 Cycles OSL-only node is tracked separately |
| Automated gate | 5/5 clean core groups pass; final Luisa `f42f3c6e` completed a 245-step fallback rebuild with LLVM 22.1.8/Embree 4.3.0, the device fixture executes exact bit comparisons, the complete fallback gate passes 8/8, and the optional Blender 4.5.10 exporter regression raises the full gate to 9/9 |
| Path-tracer architecture | Public façade reduced to 51 lines; nine private implementation translation units; full fallback build and 39/39 pre/post-refactor PFM byte matches |
| Production Sobol probes | `emission_surface` and `diffuse_bsdf_matrix` selected linear passes are pixel-exact at 64×64/4 spp; `diffuse_surface` 64×64/16 spp Combined RMSE is `0.006317606`, mean-energy ratio is `0.999740158`, and invalid pixels are 0 |
| Analytic lights | 11 Point/Spot/Area/Sun baselines, including shapes, spread, finite Sun disk, and light node trees |
| Full-scene geometry/AOV | Negative-scale normal transforms and closure-weighted glossy normals are fixed |
| Full-scene transport | At 640×480/64 spp, Lone Monk Combined RMSE is `0.26116`, Normal is `0.03696`, and DiffCol is `0.01175`; Combined mean energy is 95.75% of Cycles |
| Cold/hot fallback JIT | The pre-Sobol Lone Monk baseline is `327.574 s` cold and `0.682609 s` hot; the historical focused production-Sobol kernel `kernel_70ce93bbfda41afc` passed 20/20 cold compiles after the XIR fix; the modular focused key `kernel_4c0f6e0d82a53e90` cold-compiled in about 0.407 s and subsequently hot-loaded in about 8–11 ms |
| Persistent fallback cache | Native object plus exact metadata implemented, 8/8 isolated assertions pass, and the full-scene cross-process run is bitwise equal across 13 passes |
| Upstream integration | LuisaCompute PR [#253](https://github.com/LuisaGroup/LuisaCompute/pull/253) is merged as `98f0150e`; Psycles pins `next@f42f3c6e`, which includes the LLVM 22 shared-library export fix and the XIR local-load analysis storage fix |

The latest glossy-normal probe reduced Normal RMSE from `0.399218` to
`0.00192210` (about 99.5%) and measures Combined relative RMSE `0.5273%`.
The earlier 64×48/256 spp glossy-normal checkpoint measured Normal RMSE
`0.02515`. The current 640×480/64 spp component run measures 95.75% Combined,
97.91% Diffuse Direct, 86.21% Diffuse Indirect, 95.15% Glossy Direct, and
84.33% Glossy Indirect mean energy relative to Cycles. This separates the
remaining indirect-transport deficit from exposure and geometric-normal
errors.

## Binding and JIT policy

Values that can change without changing generated control flow belong in a
kernel argument, resource buffer, or bindless resource:

- render seed, sample start/count, resolution, and exposure;
- camera and transform values;
- integrator thresholds, bounce limits, and clamps;
- material socket values and texture/resource handles;
- light parameters and scene data.

Topology, static node modes, callable signatures, and resource binding types
remain structural. The cache key represents that structural program and its
code-generation environment; it must not absorb frequently changing values.
Each migration to a runtime argument needs two tests: changing the value must
change the result, and it must not create a new shader cache entry.

Changing private callable boundaries is itself a structural program change, so
the architecture split correctly created a new automatic cache key even
though rendered bits and the argument ABI stayed fixed. Never pin
`ShaderOption.name` to an old key to hide such a migration.

The existing per-material `GraphSurface` specialization still makes AST size
grow with material topology. Shared callables are an intermediate reduction.
The intended endpoint is a compact, buffer-driven typed-value instruction
stream whose kernel structure stays stable as ordinary material parameters
and most graph data change.

## Sampling parity

Psycles is deterministic for a fixed seed, and the production path kernel now
uses the official Cycles 4.5 tabulated-Sobol LUT, pixel hashing, Owen
scrambling, pattern shuffling, camera dimensions, per-`path_step`
light/BSDF/Russian-roulette dimensions, and 16-dimension stride. Checkpoint 1b
locks the Luisa lowering bit for bit on the real fallback device; checkpoint
1c connects the same lowering to production. The integration removes the old
PCG generator but does not by itself prove that every conditional Cycles event
consumes an identical dimension.

No “exact RNG” claim will be made from converged image statistics. The release
gate is a trace probe that records, for fixed pixel/seed/sample indices:

1. pixel-filter and lens samples;
2. light selection and light-shape dimensions;
3. BSDF component and direction dimensions at every bounce;
4. transparency and Russian-roulette dimensions;
5. the final consumed dimension index.

Every value and advancement decision must match the corresponding Cycles trace
before the full trace gate can turn green. Current official probes establish
pixel-exact camera/emission and focused diffuse-matrix paths, while the
16-spp diffuse sphere retains Combined RMSE `0.006317606` pending unified
direct-light sampling.

## Roadmap

### P0 — iteration latency and reproducibility

- [x] Validate fallback persistent cache on a full Lone Monk cold/hot
  cross-process run, including 13 bitwise-identical linear passes.
- [x] Submit the cache as LuisaCompute draft PR
  [#253](https://github.com/LuisaGroup/LuisaCompute/pull/253) and pin the
  tested commit through the Psycles submodule.
- [x] Move render window/resolution, sample range/seed, continuous camera
  values, bounce limits, clamps, background, filter width, and pass alpha
  threshold out of the AST and into kernel arguments.
- [x] Record cache hit/miss and compile-stage timings: `183.578 s` main
  AST-to-XIR cold, `327.574 s` total cold JIT, `0.682609 s` hot JIT, and
  `1.8655 ms` main-object load.

Before production Sobol integration, changing the render from 64×48/1 spp to
640×480/64 spp retained the pre-Sobol Lone Monk key
`kernel_bb7a6886f6f75b90` and completed shader setup in `0.789711 s`. The
historical monolithic production-Sobol key was
`kernel_70ce93bbfda41afc`; the modular focused key is
`kernel_4c0f6e0d82a53e90`. These differently scoped or structurally distinct
probes must not be compared as a performance regression. Structural modes
such as projection type, static node modes, and callable topology remain
deliberately outside the runtime argument block.

### P1 — transport parity

- [x] Execute the tabulated-Sobol lowering on LLVM 22.1.8/Embree 4.3.0
  fallback and lock the camera/light/BSDF results bit for bit.
- [x] Integrate that stream into the production kernel using total AA samples
  and `path_step`, then validate focused official Cycles linear passes.
- [x] Split the 6,848-line production monolith by stable responsibility while
  preserving its nine-argument ABI, Sobol dimensions, and rendered bits;
  validate the full fallback build, CTest 8/8, and 39/39 PFM regression
  outputs.
- Implement Cycles' unified single-light selection distribution and exact
  selection PDF.
- Implement Cycles-compatible environment and emitter importance
  distributions, including PDFs used by MIS and the Nishita
  conditional/marginal CDFs.
- Close indirect-transmission and glossy-indirect energy gaps.
- Complete the remaining Principled lobes and event labels.
- Finish Bump/Normal modes and derivative behavior, then re-run full-scene
  AOV and Combined gates.
- Add the bitwise Cycles random-dimension trace and replace the current RNG
  path only when each event is covered.

### P2 — shader graph coverage

- Convert each of the 16 partial nodes into focused mode/socket probes.
- Implement the 41 pending Cycles nodes in dependency order.
- Keep `tools/check_cycles_shader_node_coverage.py --require-complete` red
  until every applicable node is verified or explicitly classified outside
  device scope.
- Add volume and displacement roots after their isolated contracts exist;
  do not silently route them through Surface.

### P3 — performance architecture

- Replace per-material expanded DSL graphs with a shared device instruction
  executor and immutable program buffers.
- Deduplicate texture/attribute services and material parameter layouts.
- Add GPU-backend differential/performance runs after fallback semantics are
  stable.
- Evaluate scheduling or material clustering only after semantic gates remain
  green under the shared device IR.

## Completion criteria

Psycles is not complete merely because a showcase scene renders. A release
candidate requires:

- all 96 applicable shader nodes resolved by versioned policy and all supported
  modes officially probed;
- Surface, Volume, and Displacement behavior explicitly supported or rejected
  at import with no silent approximation;
- integrator, analytic-light, world, BSDF, transparency, pass, and color
  management gates green;
- exact sampling-trace status documented (green if exact, otherwise a named
  compatibility limitation);
- no invalid pixels, deterministic cache behavior, reproducible builds, and
  accepted full-scene linear-pass thresholds;
- cold and warm JIT, render time, and memory baselines recorded for both the
  fallback backend and at least one supported GPU backend.
