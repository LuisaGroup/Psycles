# Psycles handoff — 2026-07-29

## Current continuation — 2026-08-07

The current renderer implementation boundary is Psycles `main@8c6fa43` with
LuisaCompute `next@3e63df0c6` and Blender/Cycles 5.3 Alpha
`82186b01ad2e`. The older published-boundary section below remains a
historical record; do not reset to its July revisions.

Two official complex-scene current-head checkpoints now exist:

- [Monster Under the Bed](docs/validation/2026-08-07/monster-current-head/README.md)
  completes the canonical five-way matrix at 960x960x128 after formal Luisa
  fallback hit-kind and Vulkan dispatch-bound repairs plus an independent
  Principled Coat Normal correction. Its remaining Combined relative RMSE is
  `0.156101`, so higher-spp transport alignment is still open.
- [Lone Monk](docs/validation/2026-08-07/lone-monk-current-head/README.md)
  completes the five-way matrix at 960x720x128. The old grass mismatch is no
  longer structurally visible; Psycles HIP Combined relative RMSE is
  `0.026193` with a `1.001109` mean-luminance ratio. Performance is now the
  urgent failure: HIP is `4.09x` slower than Cycles HIP, fallback is `9.86x`
  slower than Cycles CPU, and Vulkan is `103.54x` slower than Cycles HIP with
  a 19.56-minute cache-cold JIT.

The next correctness gate is a higher-spp Lone Monk/Monster convergence run
against Cycles, followed by fresh current-head Classroom, Barbershop, and
Blender 4.1 Splash matrices. The next performance gate must separate exact
traversal cost from the monolithic path shader and reduce generated-kernel
size without pre-baking or weakening raw closure semantics.

## Published boundary

Continue on `main`; do not restart from a historical refactor branch.

- Psycles renderer implementation: `dcb96e3`, published to
  `LuisaGroup/Psycles:main`.
- LuisaCompute pin: `d57720955`, including the formal XIR repairs,
  read-only callable preservation, frozen module argument-layout ABI, and
  build-independent Vulkan result checking, published directly to
  `LuisaGroup/LuisaCompute:next`.
- Current Blender/Cycles source checkout:
  `/home/mike/Projects/blender-cycles`,
  clean `main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`.
- Current-source renderer:
  `/home/mike/Projects/blender-install-4fe17ef6/blender`, Blender 5.3.0
  Alpha Release with a `gfx1201`-only Cycles HIP kernel. Its matched Lone
  Monk 1440×1080/256 spp reference is the primary current quality gate.
- Historical 640×480 and focused reference-render executable: Blender 5.2.0
  LTS hash `fbe6228777e7`, built 2026-07-15. Those older pixels remain
  explicitly labeled and are not described as current `main` pixels.
- Exact commands, environment, metrics, timings, reports, limitations, and
  visual inspection:
  [VALIDATION.md](VALIDATION.md).

Both remotes were refreshed. At this boundary, Psycles `main` contains the
`dcb96e3` renderer plus the following validation-only handoff commit;
LuisaCompute `next` and the local Blender/Cycles `main` checkout match the
exact revisions above. All three worktrees match their tracking refs.

## Non-negotiable correctness policy

Latest exact-revision Cycles is the sole rendering and sampling oracle.

- Never add a CPU renderer, CPU sampler, or host mirror of a Luisa device
  algorithm.
- Blender/Cycles must not pre-bake a material. Export the original nodes,
  sockets, links, closure topology, and scene metadata; evaluate them through
  Luisa DSL/JIT.
- Host code may normalize immutable scene data and build device resources,
  but may not replace BSDF, closure, light, transport, or MIS evaluation with
  a host approximation.
- A renderer change is accepted only with a real Luisa backend run against
  the same Cycles scene/settings/seed/samples and linear passes, including
  numeric metrics and viewable Cycles/Psycles/difference triptychs.
- XIR changes must follow explicit CFG, dominance, SSA, scope, ownership, and
  executable-semantics invariants. Do not accumulate scene-shaped special
  cases.
- Every discovered defect needs a regression before its fix is published.
- Use all 32 hardware threads for builds and test scheduling on this machine.
- Commit and push each independently passing boundary before the next long
  compile or render.

## What is now implemented

### Three Luisa backends on the AMD workstation

Psycles has top-level, default-`ON`, strict CMake options for fallback, HIP,
and Vulkan. Configuration fails if a requested Luisa backend target is not
created. The validated release build enables:

```text
PSYCLES_ENABLE_LUISA_FALLBACK=ON
PSYCLES_ENABLE_LUISA_HIP=ON
PSYCLES_ENABLE_LUISA_VULKAN=ON
PSYCLES_ENABLE_OPENIMAGEIO=ON
```

The complete Psycles build uses `cmake --build build --parallel 32`; CTest
uses `-j32`. The final gate passes 13/13.

### Cycles flat-light selection

The renderer now builds one Cycles-style flat distribution over:

- emissive triangles weighted by world-space area;
- analytic lights with uniform lamp probability;
- the sampled background as a lamp entry;
- a 50/50 triangle/lamp class split when both classes are present.

The CDF and exact selection PDFs are uploaded once and selected through one
Luisa upper-bound callable. Raw material emission-sampling metadata and world
sampling metadata are imported without replacing the material graph.

The `flat_light_distribution` Vulkan probe at 64×64/256 spp measures:

```text
Combined RMSE        0.000256542553
Combined luminance   1.000089120 × Cycles
Diffuse Direct RMSE  0.000557016116
Diffuse Direct lum.  1.000055316 × Cycles
Diffuse Color        exact
Normal               exact
invalid pixels       0
```

### Vulkan shader-size repair

RADV originally reset on the opaque focused scene because
`transparent_extinction` recorded every material closure program. The
semantics-preserving repair specializes the dispatch table from the static
`may_be_transparent` capability:

- provably opaque programs are absent from this one callable;
- potentially transparent programs remain unchanged;
- all original closure programs remain available for other operations;
- no material values are baked.

The opaque shader fell from about 130,021 to 61,528 SPIR-V words and now runs
on the RX 9070 XT. A recording-counter regression proves opaque extinction is
recorded zero times and transparent extinction exactly once.

### Luisa XIR repair

The published Luisa sequence is:

- `6ead8e714`: make HIP/Vulkan/fallback backend resources work from a CMake
  subdirectory;
- `83a04feb8`: preserve CFG and SSA invariants;
- `f83725d27`: preserve executable semantics;
- `0e6f4376e`: process a full stable if-candidate batch, reject post-merge
  selection re-entry, and expose opt-in pass tracing;
- `23691a5a6`: lock linear restructuring for loop exits;
- `5cf0c548d`: preserve declared loop-merge boundaries;
- `30602e640`: preserve uniquely rooted read-only resource callables instead
  of forcing their code into the monolithic kernel;
- `eb167454a`: freeze the validated module argument-layout ABI before
  post-order callable/kernel emission;
- `d57720955`: check every Vulkan result in Release and Debug builds, with a
  forced-`NDEBUG` device-loss regression.

The final transformation uses dominance-constrained loop membership, treats
nested break scopes atomically, preserves non-trivial update-region execution,
node-splits selection re-entry, clones affine opaque ray-query storage, and
ignores disconnected edges when judging executable constructs. Its batch
progress measure is the strictly decreasing raw conditional count. Its
selection postcondition forbids an edge from an `M`-dominated region back into
the pre-`M`, `H`-dominated interior for selection `(H, M)`.

`test_xir_pass_restructure_cfg` passes 51 tests / 1013 assertions. All 21
structural SPIR-V tests pass, and the RX 9070 XT Vulkan SPIR-V runtime gate
passes 86/86 tests / 2029 assertions. The complete Luisa CTest gate passes
115/116. The sole failure is the independently reproducible pre-existing
`test_eastl_allocation` set of eight `fixed_vector` assertions; do not hide it
or attribute it to the XIR patch.

### Unmodified Lone Monk compiler boundary

The `column marble` graph used to expand whole-scene attribute metadata into
shader control flow. Psycles now uploads one binding table and one range per
geometry, and a Luisa loop searches only the current geometry's range. For a
fixed shader program and number of attribute operations, recorded AST/XIR
control-flow size is now independent of scene attribute-table cardinality.
The 512-binding regression measures the real shader service and caps its
structured selection count at eight. No node, closure, lookup table, or
material value was removed or baked.

Preserving read-only callables then exposed a module ABI ordering defect:
post-order SPIR-V emission visits callees before the kernel, but direct-buffer
metadata offsets had been initialized only inside kernel emission. The repair
freezes one validated `SpirvKernelArgumentLayoutPlan` before any function and
asserts that the sole kernel observes that immutable layout. Its runtime
regression uses a nonzero buffer subview, a scalar that moves the metadata
trailer to word two, and a real `OpFunctionCall`; it returns
`{18, 29, 40, 51}`.

At `next@d57720955`, the original 35-material export renders unchanged through
strict-native Vulkan. A true cold run produced a 2,611,188-word shader,
completed scene compilation in 0.737443 seconds, main-kernel JIT in 226.27
seconds, and 64×48/1 spp rendering in 0.0114453 seconds. Its 40-channel EXR is
finite, but this one-sample result is only a compiler/first-pixel record.

### Lone Monk 480p Cycles differential

The committed real baseline is scene `daylight`, frame 4, camera `cam.001`,
640×480, 64 fixed spp, seed zero, no adaptive sampling, and no denoising.
Cycles enabled only `HIP_AMD Radeon RX 9070 XT_0000:03:00`; Psycles selected
`AMD Radeon RX 9070 XT (RADV GFX1201)`. All 35 raw material graphs remain in
the export.

The initial Combined comparison was globally dark and noisy despite close
Diffuse Color and Normal passes. Lone Monk has no analytic lights and uses a
procedural sky. Blender 5.2 exports `SINGLE_SCATTERING` plus
`aerosol_density`; the importer recognized only legacy `NISHITA` plus
`dust_density`. Raw background-ray evaluation survived, but the environment
distribution lost explicit sun sampling and used uniform-sphere sampling.

`e13a1c0` makes the compatibility contract explicit: current
`SINGLE_SCATTERING` and legacy `NISHITA` enter the implemented
single-scattering path; current `aerosol_density` is preferred with
`dust_density` as the legacy fallback. The regression exercises both
versioned pairs. `MULTIPLE_SCATTERING` is distinct and is not claimed by the
simple-world sampler.

This reduced Combined RMSE from `22.190855` to `0.262420535`, relative RMSE
from `14.23354` to `0.168320400`, restored the luminance ratio from `0.819773`
to `1.022870`, and reduced the maximum error from `4505.75` to `10.2459`.
Diffuse/Glossy Direct mean luminance is within 2.4%/1.6%, but Diffuse/Glossy
Indirect remains about 8.8%/5.6% low. This is not a final 1:1 pass.

The [machine report](docs/validation/2026-07-29/lone-monk/report-640x480-64.json)
and all 13 [real triptychs](docs/validation/2026-07-29/lone-monk/triptychs-640x480-64/)
are published. Combined, Diffuse Color, Normal, Diffuse Direct, Diffuse
Indirect, and Glossy Indirect were opened at original resolution and their
visual findings are written in the
[full process record](docs/validation/2026-07-29/lone-monk/README.md).

Cycles internal render-only time is 0.96 seconds. Psycles is 1.48413 seconds,
so current same-device throughput is `0.6468×` Cycles (about `1.546×` slower);
there is no speedup claim. Psycles cold scene-plus-JIT setup is approximately
244.841 seconds. Peak VRAM was not captured for the short matched run.

### Current Lone Monk 1080p Cycles differential

The primary baseline now uses the locally built Blender/Cycles
`main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5` for both Cycles pixels and
the Psycles scene export. It is scene `daylight`, frame 4, camera `cam.001`,
1440×1080, 256 fixed spp, seed zero, no adaptive sampling, and no denoising.
Both renderers selected the RX 9070 XT; all 35 raw material graphs remain
unbaked.

The first Psycles attempt submitted all samples in one compute dispatch and
triggered an AMDGPU watchdog reset. It also exposed Luisa's Release macro
discarding `VK_ERROR_DEVICE_LOST`, which falsely returned success with an
almost-all-zero EXR. That corrupt output is recorded but excluded from all
accepted timing and quality claims.

The two formal repairs are:

- Luisa `d57720955`: every Vulkan expression is evaluated exactly once and
  every non-success result terminates in every build configuration. A
  forced-`NDEBUG` child-process regression requires device loss to abort.
- Psycles `dcb96e3`: `[first, first + count)` is partitioned into an ordered,
  contiguous, non-overlapping exact cover with at most 8 samples per
  synchronized dispatch. The exhaustive scheduler regression and a real
  single-batch/two-batch Vulkan pixel-equivalence check pass.

The accepted 32-dispatch run has no timeout/reset record. Cycles render-only
is `18.961390479 s`; Psycles is `25.9918 s`, so Psycles throughput is
`0.729514×` Cycles and is `1.370775×` slower. Baseline-relative peak VRAM is
2,659,450,880 bytes for Cycles and 1,711,570,944 bytes for Psycles.

Combined RMSE is `0.216918692`, relative RMSE `0.135484421`, luminance ratio
`1.022434553`, and invalid pixels zero. Diffuse/Glossy Direct means are
2.19%/1.53% high; Diffuse/Glossy Indirect remain 8.63%/6.07% low. All 13
[current triptychs](docs/validation/2026-07-29/lone-monk/triptychs-1440x1080-256-main-4fe17ef6/)
were generated and inspected at original resolution. Geometry, framing,
materials, and large-scale normals align; the persistent visual defect is
darker indirect transport, not a scene/export permutation. The
[machine report](docs/validation/2026-07-29/lone-monk/report-1440x1080-256-main-4fe17ef6.json)
and
[complete process record](docs/validation/2026-07-29/lone-monk/README.md)
contain every command, hash, channel audit, failed attempt, regression,
timing, VRAM value, and visual note.

### Multilayer OpenEXR and triptychs

Psycles uses OpenImageIO/OpenEXR to write one full-float EXR with
`ViewLayer.<pass>.<component>` channels. It keeps PFM only for legacy
diagnostics. The shader-probe runner compares Cycles EXR directly with Psycles
EXR and always emits triptychs.

The EXR regression reopens the generated file, checks Cycles-compatible
channel names, compares all float values exactly, and asserts
`oiio:ColorSpace` and `colorInteropID` are `lin_rec709_scene`. This fixes an
interchange bug where the unstable `scene_linear` OCIO role labeled unchanged
Rec.709 pixels as `lin_ap1_scene`. The three committed focused validation sets
are:

- [flat light](docs/validation/2026-07-29/flat-light-vk/report.json);
- [transparent mix](docs/validation/2026-07-29/transparent-mix-vk/report.json);
- [transparent data passes](docs/validation/2026-07-29/transparent-data-pass-vk/report.json).

The triptych panels are Cycles, Psycles, and independently amplified absolute
difference. They were inspected at original resolution. No geometry-edge,
silhouette, missing-light, or systematic shading discrepancy is visible in
these focused probes.

## Exact next work

1. Diagnose the remaining diffuse/glossy indirect energy gap and stochastic
   distribution difference from pass evidence and current Cycles semantics.
   Do not infer the cause from the likely-open list and do not add a CPU
   renderer/sampler. Any discovered defect gets a minimal Luisa-path
   regression before its fix.
2. Add a formally exact pixel/tile partition before increasing beyond this
   1080p gate if an 8-spp dispatch approaches a backend watchdog on a larger
   image. Preserve global pixel/sample indices and add partition and
   pixel-equivalence regressions.
3. Correct the raw Sky Texture compiler contract for Blender 5.2
   `MULTIPLE_SCATTERING` rather than silently treating it as the implemented
   single-scattering model. Add a current-Cycles fixture when implementing the
   missing equations and importance sampling.
4. Continue the evidence-ranked gaps: bring the reciprocal flattened Light
   Tree checkpoint to Cycles-exact mesh/instance topology and specialized
   emitter importance, then environment-map importance CDFs, automatic
   emissive sampling classification, visible-light forward MIS, and
   additional complex Blender demo scenes. Preserve raw closure graphs and
   commit/push every passing boundary.

## Known limitations

- The current 1440×1080/256 spp Lone Monk result is a real current-source
  full-scene baseline, but Combined relative RMSE is 13.55% and indirect
  energy remains low; it has not passed final quality acceptance.
- Psycles render-only is currently about 1.371× slower than current Cycles
  HIP on the same RX 9070 XT. Peak VRAM is measured and lower for Psycles,
  but this does not offset the missing throughput or quality parity.
- The current sample partition bounds samples per dispatch, not total
  pixel work. Much larger images still need an exact tile partition for a
  backend-independent watchdog guarantee.
- The simple-world sampler supports Blender 5.2 `SINGLE_SCATTERING`, not the
  distinct `MULTIPLE_SCATTERING` model.
- Environment-map importance CDFs and
  `world_sample_map_resolution` are not connected.
- Light-tree selection and reverse MIS are implemented on fallback/HIP/Vulkan,
  but mesh/instance subtrees, specialized emitter importance, light linking,
  and finite-sample proposal identity with Cycles remain open. See
  `docs/validation/2026-08-07/light-tree/README.md`.
- Automatic emissive sampling classification still needs a formal
  Cycles-aligned static analysis; a host pre-evaluation shortcut is forbidden.
- Imported light MIS metadata is preserved, but all corresponding forward-MIS
  behavior is not complete.
- Volume, displacement, subdivision, motion, denoising, pass, and remaining
  node coverage are not yet sufficient for a 1:1 feature claim.

Psycles is being developed as a production renderer, not a demo. Do not
replace missing semantics with showcase-specific tricks; preserve the
data-oriented Luisa architecture and make each compatibility boundary
reproducible.
