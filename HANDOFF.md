# Psycles handoff — 2026-07-29

## Published boundary

Continue on `main`; do not restart from a historical refactor branch.

- Psycles renderer implementation: `a10d686`, published to
  `LuisaGroup/Psycles:main`.
- LuisaCompute pin: `0e6f4376e`, including the linear/verifiable
  conditional-batch repair,
  published directly to `LuisaGroup/LuisaCompute:next`.
- Current Blender/Cycles source checkout:
  `/home/mike/Projects/blender-cycles`,
  `main@9353fed6d7cdc25b2aa03c30a155b044b313c8ec`.
- Focused reference-render executable: Blender 5.2.0 LTS hash
  `fbe6228777e7`.
- Exact commands, environment, metrics, timings, reports, limitations, and
  visual inspection:
  [VALIDATION.md](VALIDATION.md).

Refresh both remotes before continuing. At this boundary, `main` and
LuisaCompute `next` were fetched and their tracking refs contained the
published commits above.

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
uses `-j32`. The final gate passes 12/12.

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
  selection re-entry, and expose opt-in pass tracing.

The final transformation uses dominance-constrained loop membership, treats
nested break scopes atomically, preserves non-trivial update-region execution,
node-splits selection re-entry, clones affine opaque ray-query storage, and
ignores disconnected edges when judging executable constructs. Its batch
progress measure is the strictly decreasing raw conditional count. Its
selection postcondition forbids an edge from an `M`-dominated region back into
the pre-`M`, `H`-dominated interior for selection `(H, M)`.

`test_xir_pass_restructure_cfg` passes 51 tests / 1013 assertions. The complete
Luisa gate passes 87/88. The sole failure is the independently reproducible
pre-existing `test_eastl_allocation` set of eight `fixed_vector` assertions;
do not hide it or attribute it to the XIR patch.

### Multilayer OpenEXR and triptychs

Psycles uses OpenImageIO/OpenEXR to write one full-float EXR with
`ViewLayer.<pass>.<component>` channels. It keeps PFM only for legacy
diagnostics. The shader-probe runner compares Cycles EXR directly with Psycles
EXR and always emits triptychs.

The EXR regression reopens the generated file, checks Cycles-compatible
channel names, and compares all float values exactly. The three committed
focused validation sets are:

- [flat light](docs/validation/2026-07-29/flat-light-vk/report.json);
- [transparent mix](docs/validation/2026-07-29/transparent-mix-vk/report.json);
- [transparent data passes](docs/validation/2026-07-29/transparent-data-pass-vk/report.json).

The triptych panels are Cycles, Psycles, and independently amplified absolute
difference. They were inspected at original resolution. No geometry-edge,
silhouette, missing-light, or systematic shading discrepancy is visible in
these focused probes.

## Exact next work

1. Build the checked-out current Blender/Cycles revision, or choose another
   exact common source/binary revision. The packaged Blender 5.2 executable is
   older than the inspected 2026-07-28 source, so do not call it exact
   current-`main`.
2. Fix the formally isolated `column marble` material-lowering explosion.
   Its single raw graph currently creates 45,900 XIR blocks and 1,543,437
   instructions before restructuring. Prove whether sampled-table or
   dependency lowering violates a bounded-code-size invariant; add a red/green
   regression and keep the raw closure graph intact.
3. Render Lone Monk through both Cycles and Psycles on the same RX 9070 XT.
   Start at 480p; attempt 1080p if memory permits. Use the same frame, seed,
   samples, integrator, raw materials, and linear pass set. Record device
   selection, scene/export counts, cold and warm compilation,
   render-only and wall time, peak memory, per-pass RMSE/energy/invalid pixels,
   and same-device speedup. Commit all reports and triptychs.
4. Fix the first formal renderer mismatch exposed by that gate. Likely open
   areas are environment importance CDFs, automatic emission classification,
   visible-light forward MIS, the light tree, and complex material/geometry
   coverage. Do not assume the likely list is the diagnosis.
5. Continue through other complex Blender demo scenes after Lone Monk, keeping
   exact-revision Cycles as the reference and adding a regression for every
   defect.

The scene is now present and its first backend bring-up is recorded in
[the Lone Monk report](docs/validation/2026-07-29/lone-monk/bringup.json).
Vulkan initially exposed a greater-than-20-minute monolithic render-kernel
cold JIT and HIP exposed a greater-than-10-minute HIPRT
acceleration-structure build. The XIR batch/merge repair now renders a
full-geometry, six-material controlled Vulkan input in a 22.5549-second JIT.
An eleven-material trace and five single-material trials isolate the remaining
Vulkan code-size blocker to the raw `column marble` graph. The reduced input
is diagnostic only; do not call it a Lone Monk quality gate or manufacture a
triptych from it.

## Known limitations

- The 2026-07-29 committed probes are focused 64×64 tests, not a full-scene
  quality or speed acceptance.
- The focused Cycles process selected CPU, so no same-device speedup is
  claimed yet.
- Environment-map importance CDFs and
  `world_sample_map_resolution` are not connected.
- Cycles light-tree selection is not implemented.
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
