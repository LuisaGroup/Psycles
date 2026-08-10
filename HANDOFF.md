# Psycles handoff — 2026-07-29

## Current continuation — 2026-08-10

The current renderer implementation boundary advances from Psycles main with
LuisaCompute `next@8c6951520` and Blender/Cycles 5.3 Alpha
`82186b01ad2e`. The older published-boundary section below remains a
historical record; do not reset to its July revisions.

The current official complex-scene checkpoints are:

- [Post-population surface-closure ABI](docs/validation/2026-08-10/surface-closure-point-abi/README.md)
  makes the physical closure dependency cut a strong DSL type and transports
  it through a 48-byte packed callable ABI. The full Lone Monk HIP kernel drops
  from 3,676 to 2,704 scratch bytes per thread (-26.4%); warm 960x540/64-spp
  throughput improves by a measured 4.2--4.6%. Fallback, HIP, and native Vulkan
  ABI/closure regressions pass, twelve of fifteen linear passes are exact, and
  the original-resolution triptych has no structured difference. Do not infer
  a Cycles speed lead until the pending matched 1080p matrix is complete.

- [Sparse XIR restructure analyses](docs/validation/2026-08-10/xir-restructure-sparse-analyses/README.md)
  reuses loop-boundary facts per CFG version, derives construct parents by an
  event walk over the sparse immediate-dominator tree, and audits post-merge
  re-entry through exact dominance frontiers. The follow-up carries enclosing
  loops as persistent contexts and replaces per-arm graph searches with block
  value numbering plus one sparse reverse-CFG dataflow per loop. Luisa
  `next@8c6951520` is published. Its follow-ups replace the quadratic
  repeated DCE scan with an equivalent reverse-use least-fixed-point
  worklist and replace per-arm loop-boundary merge graph searches with one
  versioned sparse dataflow per loop plus explicit batch invalidation. The
  latest follow-up replaces 1,723 per-candidate dominator rebuilds with an
  exact immutable-tree overlay whose transparent merges carry
  nearest-common-dominator anchors. Dynamic merge inference continues to see
  the mutated graph; the immutable lexical merge is only the contracted-graph
  fallback. The newest stage moves selection-merge inference into a standalone
  value-numbered batch component with persistent loop contexts and reusable
  dense epoch arrays. The complete XIR suite passes `48/48`, and
  the RX 9070 XT native Vulkan path passes `92/92` tests with `2,096`
  assertions. On the unchanged Lone Monk module, `drain_selection_exits`
  falls 18.30x to `0.723 s`; DCE subsequently falls 4.72x from `14.182 s` to
  `3.004 s`. `restructure_cfg` falls from `35.226 s` to
  `17.945 s`; AST-to-SPIR-V falls from `75.920 s` to `57.766 s`, then to
  `47.533 s` after DCE, then to `41.046 s` after merge batching, then to
  `35.714 s` after dominance-overlay batching. The if batch itself falls
  3.79x from `7.930 s` to `2.091 s`, `restructure_cfg` reaches `6.826 s`,
  then dense merge inference takes the if batch to `0.299 s` (another 6.99x),
  `restructure_cfg` to `5.191 s`, and AST-to-SPIR-V to `33.752 s`. Versioned
  loop-continue analysis then takes its phase from `1.531 s` to `0.552 s`,
  `restructure_cfg` to `4.053 s`, XIR legalization to `18.883 s`, and
  AST-to-SPIR-V to `32.691 s`. The newest selection-exit stage defers
  post-dominator refreshes until the drain boundary, restricts loop dataflow
  to successor-closed active regions, and locally invalidates one-target
  funnel dependencies. Its drain falls from `0.726 s` to about `0.408 s`,
  relation construction to `0.069 s`, and `restructure_cfg` to `3.555 s`.
  The newest stage value-numbers each immutable CFG once, stores sparse CSR
  edges in both directions, and solves post-dominance entirely with dense RPO
  IDs on the historical sink-reachable domain. Aggregate post-dominator time
  falls 3.73x from `0.649 s` to `0.174 s` across 229 calls;
  `restructure_cfg` reaches `3.036 s`, XIR legalization `17.722 s`, and
  AST-to-SPIR-V `31.228 s`.
  A decomposed profile then shows that 90.4% of loop-continue time is exact
  dominator rebuilding rather than region discovery. Intermediate mutation
  versions now rebuild idom ancestry after every mutation but defer the
  unobserved frontier relation to the final retained tree. Loop-continue falls
  from `0.520 s` to `0.356 s`, `restructure_cfg` to `2.877 s`, and
  AST-to-SPIR-V to `30.921 s`; 129 invalidations still cause 129 ancestry
  rebuilds but only eight frontier materializations.
  The current dense-dominator stage then value-numbers the historical
  reachable CFG once, stores sparse predecessor CSR, and solves CHK entirely
  on RPO IDs. Dominance ancestry falls from `319.951 ms` to `230.078 ms`
  (-28.1%), and `restructure_cfg` from `3.056 s` to `2.745 s` (-10.2%). The
  129 rebuilds converge in exactly 258 passes over 645,720 blocks and 807,853
  edges; output and SPIR-V remain identical after full XIR, system-STL, and
  native Vulkan gates.
  Selection-exit SSA transport is now delayed to the drain's final CFG fixed
  point. Intervening queries observe graph structure but not operands, and
  state dispatch preserves the original dynamic successor, so the final exact
  repair is trace-equivalent to repairing each rewrite eagerly. Nine logical
  requests become one physical repair; site scanning falls from `307.857 ms`
  to `58.156 ms`, the drain from `392.109 ms` to `163.168 ms`,
  `restructure_cfg` to `2.428 s`, XIR legalization to `17.201 s`, and
  AST-to-SPIR-V to `30.399 s`. Full gates, SPIR-V, and output remain exact.
  Selection-merge scoring now walks only the query's aggregate support, while
  enclosing-selection fallback walks exactly the header's dominator ancestors
  and dense IDs retain historical tie order. The if batch falls from
  `271.223 ms` to `101.746 ms` (-62.5%) and `restructure_cfg` to `2.271 s`;
  output and both SPIR-V modules remain exact after all gates.
  The merge
  canonicalizer itself remains at `0.084 s`. Peak RSS
  falls from `9,415,608 KiB` to `1,654,768 KiB` in the matched driver-cache
  state, with identical SPIR-V sizes and byte-identical output. The DCE run
  retriggered RADV compilation, so its process peak is not compared across
  cache states. Dense post-dominance is no longer a primary perf hotspot; the
  next measured target is `0.269 s` loop-continue normalization, including
  `0.219 s` of exact dominance rebuilding. If-batch is now `0.102 s` and
  selection-exit drain `0.162 s`.

- [Sparse XIR verifier dominance](docs/validation/2026-08-10/xir-verifier-sparse-dominance/README.md)
  gives every locally reachable block a numeric RPO ID, stores predecessors
  as sparse CSR and only one idom parent per block, then answers dominance by
  ancestry intervals. Luisa `next@f6a9b2728` is published. The complete XIR
  suite passes `48/48`; the RX 9070 XT native Vulkan path passes `92/92` tests
  and `2,096` assertions. On the unchanged Lone Monk production module,
  cache-cold Vulkan JIT falls from `569.378 s` to `180.533 s` with identical
  raw and optimized SPIR-V sizes. Handoff verification is `1.009 s`.
  Restructure still costs `53.138 s` and RADV pipeline creation `86.780 s`, so
  the immediate compiler follow-up is a formally invalidated worklist for the
  repeated post-restructure scans, not weaker verification.

- [Monster BSSRDF exit normal](docs/validation/2026-08-07/monster-bssrdf-exit-normal/README.md)
  formally aligns Cycles' retained-BSSRDF closure-normal reduction and the
  synthetic unit Lambert used at a subsurface exit. The exact Monster path
  now agrees through the exit normal, cosine sample, and following
  object/primitive. At 960x960x512, Combined relative RMSE falls 1.69x from
  `0.082721` to `0.048905`, and the mean-luminance ratio becomes `1.003553`.
  Original-resolution Combined, direct/indirect diffuse, and indirect glossy
  triptychs were inspected; 218/218 tests pass. The correctness repair is
  published as `fb69b15`. Follow-up `dc98dd0` maps the parameter-aware Cycles
  `has_surface_bssrdf` material set onto deduplicated Luisa surface tags, so
  the exit callable omits provably unreachable graphs without baking closure
  values. Current `8b688ec` tightens that superset to Cycles'
  `SD_HAS_BSSRDF_BUMP` predicate, including immediate Normal-parent topology,
  BUMP/BOTH displacement policy, direct Thin Wall semantics, and per-real-
  BSSRDF closure attribution. Monster keeps only its linked-normal material;
  the unbumped child-skin material skips exit re-evaluation exactly as Cycles
  does. Cold JIT falls another 2.12% to `126.944 s` (5.38% below the
  unfiltered callable), with HIP linking still dominant at `96.676 s`.
  Current 960x960x512 Combined relative RMSE remains `0.04890434`; the warm
  run is `137.286 s`, or `5.172x` Cycles HIP, so no runtime speedup is claimed.
  Two new original-resolution triptychs were inspected and 218/218 tests pass.

- [Lone Monk background-Sun sampling](docs/validation/2026-08-07/lone-monk-background-sun-sampling/README.md)
  aligns the complete Sobol-to-guided-Nishita-Sun relation. The old polar-cap
  sampler had the right density but chose a different point on the solar disc
  than Cycles. The real path's direction error falls from `0.776416` degrees
  to one float32 ULP, sampled sky-radiance error falls from 23.21% to 0.0283%,
  and 960x720x512 Combined relative RMSE falls from `0.015995` to `0.012203`.
  Diffuse Direct relative RMSE falls 1.83x with no measurable render-time
  cost. The fallback/HIP/Vulkan regression and the full 215/215 gate pass;
  the repair is published as `9ac8bab`.

- [Lone Monk muted-node bypass](docs/validation/2026-08-07/lone-monk-muted-node-bypass/README.md)
  formally aligns Blender/Cycles muted-node graph semantics. The defect was
  not UV or transform drift: `paper - page / Mix.001` is muted and Cycles
  follows its runtime `A_Color -> Result_Color` internal link, while the old
  exporter lost both facts and evaluated the Mix. The generic exporter and
  topology repair reduces 960x720x512 Diffuse Color RMSE 20.52x to
  `0.00020008`; the exact first closure weight moves from a 20--24% error to
  about 0.1%. A fresh Cycles CPU/HIP plus Psycles fallback/HIP/Vulkan matrix
  completes, all three Psycles Diffuse Color relative RMSEs are
  `0.001106--0.001160`, and the original-size triptychs have been inspected.
  The repair is published as `d5c7730`, with 215/215 tests passing after a
  32-job build.

- [Monster Under the Bed](docs/validation/2026-08-07/monster-current-head/README.md)
  completes the canonical five-way matrix at 960x960x128 after formal Luisa
  fallback hit-kind and Vulkan dispatch-bound repairs plus an independent
  Principled Coat Normal correction. Its remaining Combined relative RMSE is
  `0.156101`, so higher-spp transport alignment is still open.
- [Lone Monk](docs/validation/2026-08-07/lone-monk-current-head/README.md)
  remains the preceding grass/current-head checkpoint. The refreshed
  muted-node matrix measures Psycles HIP Combined relative RMSE `0.026161`
  with a `1.001226` mean-luminance ratio. Performance remains the urgent
  failure: HIP is `4.10x` slower than Cycles HIP, fallback is `10.06x` slower
  than Cycles CPU, and Vulkan is `103.61x` slower than Cycles HIP with a
  19.75-minute cache-cold JIT.

The 512-spp Lone Monk Combined relative RMSE is `0.012203`; the 512-spp
Monster result is now `0.048905` after exact BSSRDF exit-frame alignment. The
next correctness gate is to isolate the remaining Monster indirect residuals,
implement the currently explicit Principled Thin Wall runtime gap, and promote
fresh current-head Classroom, Barbershop, Blender 4.1 Splash, and Monster
matrices across all five backends. The immediate performance gate is the
still-dominant HIP bitcode link and broader monolithic path-shader code
generation, followed by exact-vs-hardware traversal measurement, without
pre-baking or weakening raw closure semantics.

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
