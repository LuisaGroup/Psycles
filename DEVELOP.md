# Psycles development status

This is a living implementation plan, last updated 2026-07-29. “Implemented”
does not mean “Cycles compatible”: compatibility requires an exact-revision
Cycles render, linear-pass metrics, and visual comparison. The current
commands, machine, reports, and triptychs are in
[VALIDATION.md](VALIDATION.md).

## Mandatory Cycles SVM implementation lock

Until the project owner gives an explicit superseding instruction, the sole
implementation task is an isomorphic Luisa DSL implementation of Blender
Cycles 5.2.1 SVM. This is a hard development constraint, not a design
preference:

- reproduce Cycles' SVM node stream, typed node payloads, stack addressing,
  program-counter loop, single node-type dispatch, closure state, feature
  masks, and surface/volume/displacement control flow before attempting any
  alternative architecture;
- do not retain or extend Psycles-specific substitutes such as execution-family
  plus semantic-subtype dispatch, independently scheduled value/closure
  programs, or a second closure-leaf decode layer;
- do not redesign, generalize, fuse, split, or otherwise "improve" the Cycles
  execution model. Luisa multistage/JIT facilities may erase node cases and
  closures that Cycles feature masks prove unreachable, but may not change the
  observable state machine or bytecode semantics;
- do not advance unrelated renderer features or optimizations while this SVM
  replacement remains incomplete;
- every migrated node family requires a field/state mapping to the exact
  Cycles 5.2.1 source, a Cycles-oracle regression, and whole-program validation.

No deviation is implied by temporary regressions or by the convenience of the
existing implementation. Only an explicit project-owner instruction can relax
this lock.

## Mandatory structural-parity performance lock

The Cycles lock applies to the observable renderer state machine, not to an
incidental floating-point bit pattern selected by one compiler or device. Until
the project owner explicitly says otherwise, a non-structural one-ULP or
last-bit difference must never justify a runtime performance cost:

- do not disable fast math, emulate native math or texture filtering in
  software, add branches or memory traffic, or replace a hardware instruction
  solely to reproduce the final bit of one Cycles build;
- keep structural facts exact: SVM words and cursor movement, control flow,
  visibility/support, RNG dimensions, sampling probabilities, closure state,
  energy, texture addressing/filter type, and finite/invalid behavior;
- external-oracle regressions must compare structural words exactly and use a
  documented numerical tolerance only for host/backend arithmetic whose small
  representation difference does not alter those structural facts;
- stricter arithmetic is permitted only with a written proof obligation, such
  as maintaining an outward probability bound. Each exception must be narrowly
  allowlisted and regression-tested rather than inferred from an exact-hash
  fixture.

Native texture operations are part of this lock: nearest and linear filtering
must remain one native sample, and Cycles cubic filtering must remain its four
native bilinear samples. A backend-specific software texel loop is not an
acceptable way to match interpolation rounding.

Vector normalization follows the same rule. Cycles' zero, fallback, and
near-zero domain predicates remain explicit because they can change control
flow or finite/invalid behavior; arithmetic inside the accepted domain uses
the shared native reciprocal-square-root implementation in
`native_vector_math.h`. Do not reintroduce scalar `sqrt` plus division merely
to reproduce a CPU/GPU rounding sequence.

## Authoritative-reference policy

The checked-out latest Blender/Cycles source and renders from the same
`.blend`, frame, integrator settings, seed, samples, and linear passes are the
only correctness oracle for Psycles rendering and sampling.

- Do not build a CPU renderer, CPU sampling implementation, or host-side
  “reference” evaluator for any Luisa device algorithm.
- Host code may compile immutable scene data for device upload, but must not
  duplicate a Luisa selection, BSDF, light-shape, transport, or MIS algorithm
  for validation.
- Regression tests for renderer behavior must execute the Luisa DSL/JIT path
  on the supported backends and compare against current Cycles outputs or
  explicitly versioned Cycles fixtures.
- Reading Cycles source establishes both SVM semantics and, under the active
  implementation lock above, the required SVM execution model. Luisa DSL is
  the implementation language; it is not permission to substitute a different
  interpreter architecture.

## Active handoff checkpoint

Continue on `main`. The published renderer boundary is `dcb96e3`; it pins the
published LuisaCompute `next@d57720955`. The clean Blender/Cycles source
checkout is `main@4fe17ef6`, and its Blender 5.3.0 Alpha / `gfx1201` HIP build
produced the primary 1440×1080/256 spp Lone Monk reference. The current
Cycles/Psycles report and all 13 real triptychs are committed. The older
640×480 Blender 5.2.0 LTS result remains an explicitly versioned historical
checkpoint.

The active sequence is:

1. [x] implement and upload the current Cycles flat-light distribution for
   emissive triangles, analytic lights, and the sampled background;
2. [x] preserve material emission/world sampling metadata and compile the
   original closure graph through Luisa without Blender/Cycles pre-baking;
3. [x] repair Luisa XIR restructuring from CFG, dominance, scope, affine-state,
   and executable-semantics invariants; publish the regressions to `next`;
4. [x] enable strict fallback/HIP/Vulkan targets, validate HIP and Vulkan on
   the RX 9070 XT, and eliminate opaque transparent-dispatch code through
   semantic JIT capability specialization;
5. [x] add Cycles-compatible full-float multilayer EXR output, EXR-to-EXR
   comparison, and committed Cycles/Psycles/difference triptychs;
6. [x] run the unmodified 35-material Lone Monk scene at 640×480/64 spp on
   Cycles HIP and Psycles Vulkan on the same RX 9070 XT, record every pass,
   inspect real triptychs, and fix the missing Blender 5.2
   `SINGLE_SCATTERING` sun-guiding contract with regressions;
7. [x] use the completed exact-current-Cycles HIP build to repeat Lone Monk
   at 1440×1080/256 spp with peak-memory collection; repair Release Vulkan
   error propagation and formally partition samples into watchdog-safe exact
   batches; use the pass evidence to rank remaining differences;
8. [ ] implement environment-map importance CDFs, remaining MIS paths, and
   formal automatic-emission classification exposed by the full-scene gate;
9. [ ] replace per-material expanded DSL ASTs with a buffer-driven shared
   device instruction executor and remeasure cold/hot compilation.

Every independently validated stage is committed and pushed before the next
long-running build or render. Historical checkpoints below explain how the
current architecture was reached; their old Blender 4.5 measurements remain
historical evidence and do not supersede the 2026-07-29 record.

## Historical checkpoints

Checkpoint 0 removes the `PSYCLES_LUISA_SOURCE_DIR` override so the pinned
submodule is the single normal source of LuisaCompute. Repository-wide option
references and whitespace checks pass. A clean GNU 13.3/CMake 3.27.7
`PSYCLES_ENABLE_LUISA=OFF` configure and build also pass with the Unix
Makefiles generator, followed by 4/4 core CTest groups. The existing
Luisa/fallback 6/6 gate remains the required full-build check before any
rendering change is published; checkpoint 1b expands that gate to 8/8.

Checkpoint 1a historically introduced the tabulated-Sobol resource contract.
The current host module retains only sequence sizing and construction of the
256-pattern × float4 lookup table required for device upload; its test locks
that immutable table with an IEEE-754 FNV-1a fingerprint. The old host
pixel-hash, dimension, shuffle, and sample-lookup mirror has been deleted, so
there is no CPU sampling oracle. Camera, bounce, light, BSDF, and shuffle
semantics are exercised only through the Luisa device regression in checkpoint
1b and compared with exact-revision Cycles renders.

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

An earlier Lone Monk 640×480 end-to-end diagnostic exposed an exporter
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
the Blender regression pass (9/9 with Blender 4.5.10 available). A fresh
production fallback render compiled 350 geometries, 7,543 instances, and 37
materials. Cold shader JIT took 3,947.13 seconds and 640x480/64 spp rendering
took 61.20 seconds. The Combined preview using the scene's Filmic, Medium
Contrast, -2 EV display settings and the separate linear Normal comparison
both confirm one continuous foreground arch matching the Cycles structure.
This accepts the geometry repair only: existing lighting/environment and pass
semantics differences remain visible and are not hidden by the display
transform. The unusually slow cold JIT is a separate full-scene performance
issue.

The production integration exposed a Luisa XIR
`local_load_elimination` heap use-after-free. The historical
LuisaCompute `next@f42f3c6e` checkpoint pre-created all
block/predecessor data-flow entries and forbade map insertion while a block
result reference was live. The new loop/fanout fixture failed the old
implementation under ASan and passed the fix. Production cold JIT changed
from 6/20 crashes to 20/20 successes; a subsequent hot load took about
9.498 ms and all 13 PFM outputs were byte-identical. The current pin
`f83725d27` includes this repair and the later formal CFG restructuring work.

Every transport, light-distribution, or environment-sampling boundary must be
compared against an exact-revision current Cycles render. New meaningful
render comparisons are committed as viewable triptychs in addition to
recording RMSE, energy ratios, invalid-pixel counts, and
deterministic-cache status.

AMD full-scene runs use the same configurable sysfs sampler for both
renderers (50 ms for current Cycles, 20 ms for current Psycles):

```bash
python3 tools/measure_amd_vram.py \
  --output /tmp/render-vram.json --interval 0.05 -- \
  <renderer> <arguments...>
```

The report records the command, DRM device, machine-wide pre-launch baseline,
absolute peak, increase over baseline, final usage, duration, sample count,
and child exit status. Performance reports must retain both the absolute peak
and baseline-relative increase; desktop VRAM already in use is not renderer
memory.

Long Luisa renders must also preserve bounded device progress. The current
sample scheduler partitions `[first, first + count)` so adjacent nonempty
batches meet exactly, every batch has at most
`max_samples_per_dispatch` samples, and the final endpoint equals the
requested endpoint. The default is 8. Each dispatch is independently
synchronized, and the global sample index plus whole-sequence sample total
are passed unchanged to the device sampler. Do not replace this invariant
with scene-specific sample counts. Larger-image tiling must define the same
kind of exact ordered cover over pixels and prove output equivalence.

## Current checkpoint

| Area | Verified state |
|---|---|
| Shader inventory | 96 Cycles-applicable nodes tracked from 105 Blender shader node types |
| Complete coverage | 48/96 complete: 46 `cycles_verified` device nodes and 2 structural output adapters |
| Remaining nodes | 13 partial and 35 pending; no implemented node is waiting for a probe, and 1 Cycles OSL-only node is tracked separately |
| Automated gate | Release configuration builds fallback, HIP, and Vulkan with 32 jobs; Psycles passes 171/171 in 6.69 s. Luisa `test_device_math vk` passes 388 assertions across native SPIR-V and HLSL-to-SPIR-V, and the CUDA backend target compiles with 32 jobs. |
| Path-tracer architecture | Public façade plus private modules; unified flat-light CDF is uploaded once and selected through one Luisa upper-bound callable; opaque transparent-extinction dispatch is removed by semantic capability specialization; sample intervals are exact ordered partitions with at most 8 spp per synchronized dispatch |
| Production Sobol probes | Historical 4.5 emission/diffuse probes remain recorded; current 5.2 flat-light 64×64/256 spp Combined RMSE is `0.000256543`, luminance ratio `1.000089120`, invalid pixels 0, with DiffCol and Normal exact |
| Analytic lights | 11 Point/Spot/Area/Sun baselines, including shapes, spread, finite Sun disk, and light node trees |
| Transparent closure probes | `transparent_mix` Combined RMSE is `7.93035e-7`; `transparent_data_pass` Combined RMSE is `0.000179912`; selected data passes are exact and all invalid-pixel counts are 0 |
| Image pipeline | Psycles writes full-float Cycles-compatible multilayer EXR; EXR readback locks channel names, exact float values, and `lin_rec709_scene` identity; the differential runner commits Cycles/Psycles/absolute-difference triptychs |
| AMD backends | RX 9070 XT / gfx1201 passes focused HIP and Vulkan renders; Release Vulkan results are never discarded; the 1440×1080/256 spp render completes in 32 bounded dispatches without a timeout/reset |
| Full-scene geometry/AOV | Negative-scale normal transforms and closure-weighted glossy normals are fixed |
| Full-scene transport | Current Blender-main same-RX-9070-XT 1440×1080/256 spp Lone Monk Combined RMSE is `0.216918692`, relative RMSE `0.135484421`, luminance ratio `1.0224346`, and invalid pixels 0; diffuse/glossy indirect remain 8.63%/6.07% low |
| Cold/hot fallback JIT | The pre-Sobol Lone Monk baseline is `327.574 s` cold and `0.682609 s` hot; the historical focused production-Sobol kernel `kernel_70ce93bbfda41afc` passed 20/20 cold compiles after the XIR fix; the modular focused key `kernel_4c0f6e0d82a53e90` cold-compiled in about 0.407 s and subsequently hot-loaded in about 8–11 ms |
| Persistent fallback cache | Native object plus exact metadata implemented, 8/8 isolated assertions pass, and the full-scene cross-process run is bitwise equal across 13 passes |
| Upstream integration | LuisaCompute PR [#253](https://github.com/LuisaGroup/LuisaCompute/pull/253) is merged as `98f0150e`; Psycles pins published `next@d57720955`, including GPU subdirectory paths, formal XIR repairs, read-only callable preservation, the frozen module argument-layout ABI, and build-independent Vulkan result checking |

An earlier glossy-normal probe reduced Normal RMSE from `0.399218` to
`0.00192210` (about 99.5%) and measures Combined relative RMSE `0.5273%`.
The earlier 64×48/256 spp glossy-normal checkpoint measured Normal RMSE
`0.02515`. That historical 640×480/64 spp component run measured 95.75% Combined,
97.91% Diffuse Direct, 86.21% Diffuse Indirect, 95.15% Glossy Direct, and
84.33% Glossy Indirect mean energy relative to Cycles. The historical
`e13a1c0` 640×480 result measures 102.29% Combined, 102.40% Diffuse Direct,
91.16% Diffuse Indirect, 101.54% Glossy Direct, and 94.42% Glossy Indirect.
The current 1440×1080/256 spp result measures 102.24% Combined, 102.19%
Diffuse Direct, 91.37% Diffuse Indirect, 101.53% Glossy Direct, and 93.93%
Glossy Indirect.
This keeps the remaining indirect-transport deficit separate from the
repaired exposure/sun-sampling path and geometric-normal errors.

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
- [x] Implement Cycles' flat unified single-light selection distribution and
  exact selection PDF.
  - [x] Checkpoint the production host-side Cycles 4.5 flat-distribution
    builder and its exact upper-bound lookup. Emissive triangles are ordered
    by scene/primitive order and weighted by world-space area; lamps are
    uniform, with triangles and lamps each receiving 50% probability when
    both classes exist. The focused test locks the CDF, per-emitter selection
    PDFs, zero-area intervals, and boundary behavior.
  - [x] Populate the distribution from the compiled scene, upload one CDF, and
    replace the independent environment/triangle/all-lamp NEE selection with
    one device-side upper-bound lookup. Import raw material emission-sampling
    and world-sampling metadata, and validate the result through current
    Cycles EXR metrics and triptychs.
- Implement the Cycles light tree without changing flat-distribution
  semantics when the tree is disabled.
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
- [x] Add focused HIP/Vulkan differential and timing runs on the RX 9070 XT.
- [x] Add a packaged-Cycles same-device 640×480 full-scene performance run.
- [x] Build current Cycles and add the 1440×1080/256 spp full-scene run plus
  peak-memory reporting. Current render-only throughput is `0.729514×`
  Cycles; baseline-relative peak VRAM is 1,711,570,944 bytes for Psycles
  versus 2,659,450,880 bytes for Cycles.
- Add an exact pixel/tile partition before larger images if the fixed 8-spp
  dispatch bound is not sufficient for a backend watchdog.
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
