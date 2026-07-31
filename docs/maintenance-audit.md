# Maintainability audit

This audit tracks source structure independently from Cycles feature coverage.
Line count is only a screening signal: a large table or a collection of small
probe builders is less risky than one function that records an entire path
integrator.

The baseline below was measured on `main` at `4e022f4`, excluding
`third_party`, build outputs, assets, and generated validation images.

The working target is at most 2,000 lines per hand-written source file.
Generated sources, imported third-party code, and declarative data tables are
reported separately rather than split mechanically. Exceeding the target
requires a documented reason and a semantic decomposition plan.

## Baseline

The screened C++, headers, and Python sources contain 57,613 lines. The main
concentration points are:

| File | Lines | Concentration | Risk |
|---|---:|---|---|
| `tools/create_cycles_shader_probe.py` | 5,518 | More than 100 independent probe builders; largest builders are feature matrices | Medium |
| `include/psycles/luisa/graph_surface.h` | 4,882 | One class; `trace_values()` spans about 2,350 lines | Critical |
| `src/adapter/blender_scene.cpp` | 4,440 | Graph normalization and scene decoding share one file; `lower_natural_output()` spans about 1,900 lines | Critical |
| `src/luisa/path_tracer_kernel.cpp` | 3,978 | Almost the whole file is one `LuisaRenderSession::initialize()` function | Critical |
| `src/compiler/surface_program.cpp` | 2,416 | `SurfaceProgramBuilder::lower_node()` spans about 1,540 lines | Critical |
| `src/luisa/path_tracer_scene.cpp` | 1,865 | `compile_scene()` owns most scene validation, material compilation, uploads, and acceleration setup | High |
| `include/psycles/luisa/cycles_noise.h` | 1,386 | One mathematical feature family split into many helpers | Medium |
| `tests/test_main.cpp` | 1,226 | Independent contract cases in one executable | Medium |

The first five critical entries combine size with a single dispatch function
or class. They make reviews harder, increase recompilation fan-out, and tempt
new features to cross semantic boundaries.

## Complete first-party scan

The 2026-07-31 scan covered every first-party C/C++ source and header, Python
module, CMake source, and `.inl` file: 143 files and 59,835 lines. It excluded
only `third_party`, build trees, and generated validation artifacts. The
initial scan found exactly five hand-written files over 2,000 lines:

| File | Baseline lines | Planned semantic modules |
|---|---:|---|
| `tools/create_cycles_shader_probe.py` | 5,518 | camera/light, closure, texture, color/value, and graph-composition probe families |
| `include/psycles/luisa/graph_surface.h` | 4,882 | value evaluation, closure construction, closure sampling, and trace ABI |
| `src/adapter/blender_scene.cpp` | 4,440 | scene decoding, graph normalization, and natural-output lowering by node family |
| `src/luisa/path_tracer_kernel.cpp` | 3,789 | path state/RNG, traversal, shading, direct lighting, volume, and film |
| `src/compiler/surface_program.cpp` | 2,416 | typed node lowering split by value, texture, and closure families |

The next largest hand-written source is
`src/luisa/path_tracer_scene.cpp` at 1,865 lines, so it is below the hard
screening threshold but remains on the high-risk watch list.

`docs/cycles-shader-nodes-4.5.10.json` is 7,691 lines. It is a generated,
declarative Cycles node inventory rather than hand-written executable code and
is intentionally kept as one versioned compatibility contract.

## Progress

The first path-kernel slice moved Cycles camera dimensions and primary-ray
construction into `path_tracer_camera.{h,cpp}`. The kernel decreased from 3,978
to 3,789 lines, the new implementation file is 244 lines, and old/new
differential renders are byte-identical. The extraction added fallback, HIP,
and Vulkan camera tests and is documented under
`validation/2026-07-31/camera-vulkan`.

The first `GraphSurface` size-control checkpoint partitioned the implementation
along renderer semantics: state and value access, Cycles scattering, value
evaluation by node family, raw closure traversal, and the public surface/
volume API. It reduced the 4,882-line header to a small facade, but retained
the implementation as textual `.inl` fragments in one class scope. That was
an intentionally conservative equivalence checkpoint, not the target
architecture. Every extracted source range was checked byte-for-byte against
the original header before compilation.

After this checkpoint the complete scan covered 153 source files and 60,008
lines, with four remaining files over 2,000 lines. The
`psycles.source_size` CTest contract rejects every new over-limit first-party
source and rejects growth in the explicitly budgeted debt files. A debt entry
is removed as soon as its semantic decomposition reaches the target.

The split passed the 32-worker full build and all 42 CTest contracts. A
64×64, 256 spp material fixture was then rendered through fallback, HIP, and
Vulkan. All 13 linear passes from each backend (39 PFM files total) are
byte-identical to their pre-split baselines; all three Combined outputs have
SHA-256
`4ac47cfe7d528e14da89e116f1dffd3b8dbf6536f8ff0817d87ef66a4f3409b0`.
The existing Cycles/Psycles triptych below remains the visual record for this
fixture because the current Psycles panel is byte-identical:

![Cycles, Psycles, and absolute difference for the material fixture](validation/2026-07-31/camera-vulkan/blackman-harris-vk-vs-cycles.png)

`GraphSurface` is now a real compiled host-stage surface compiler. Its public
91-line header contains only the stable `Surface` override ABI and a private
implementation pointer. State/color helpers, scattering, closure traversal,
value-graph tracing, and the surface API are ordinary functions in separate
translation units; the largest implementation file is 999 lines.

The value graph uses an explicit host-side `ValueNode` interface. At
`GraphSurfaceImplementation` construction, every immutable compiler IR
instruction is bound once to a math, context, image, or procedural node
object. Calling the virtual `evaluate()` method while a Luisa kernel is being
recorded emits DSL expressions into the current AST. The virtual dispatch is
therefore C++ metaprogramming at JIT/trace time: it does not add device-side
virtual calls or split the fused shader into Luisa `Callable` kernels. Closure
visitors and the traced/untraced sampling choice use the same host-stage
boundary.

All nine non-generated `GraphSurface` `.inl` files have been deleted. The
source-size regression rejects every new hand-written `.inl`. At this
checkpoint the eight path-kernel fragments remained an explicit shrinking
debt list; they were deleted by the typed path-pipeline checkpoint below. The
versioned Cycles BSDF table remains classified as generated declarative data.

The compiled architecture passed the 32-worker full build and all 42 tests.
The 64×64, 256 spp fixture was rerun through fallback, HIP, and Vulkan; all 13
linear passes on every backend (39 files total) remain byte-identical to the
pre-refactor baseline. The HIP output was also opened and inspected at native
resolution. The complete source gate now covers 183 first-party files and
61,298 lines, with no file over 2,000 lines.

The surface-program compiler is now a second semantic checkpoint. Its public
implementation decreased from 2,416 to 135 lines. Builder state and
diagnostics, graph-context nodes, typed value nodes, texture/procedural nodes,
and raw surface/volume closure nodes live in separate translation units; the
largest is 731 lines. The old 1,540-line `lower_node()` chain is now a family
dispatcher. Each family returns `true` when it recognizes a node even if
invalid inputs prevent emission, preserving the distinction between an input
diagnostic and an unsupported-node diagnostic.

The split passed the 32-worker full build and all 42 CTest contracts. The same
fallback/HIP/Vulkan fixture again produced 39 linear pass files that are
byte-identical to the pre-split baselines, including the Combined hash above.
At that checkpoint the scan covered 159 files and 60,210 lines, with three
remaining
over-limit debt files. `surface_program.cpp` has therefore been removed from
the size-gate allowlist.

The first stateful path-kernel checkpoint partitioned the source without
introducing new Luisa `Callable` boundaries.
`path_tracer_kernel.cpp` decreased from 3,789 to 341 lines. Eight included
implementation phases covered sample setup, closest-event and forward-light
handling, shading-point reconstruction, surface emission/data passes,
environment NEE, emissive-mesh NEE, analytic-light NEE, and BSDF
continuation/film writes. The largest phase was 595 lines. This preserved the
original lexical scopes and established an equivalence baseline, but textual
inclusion was not the target component architecture.

Every phase body compares byte-for-byte with its pre-split source range. The
32-worker full build and all 42 tests pass, and another three-backend render
made all 39 linear passes byte-identical to the preceding baseline. Moving
source locations invalidated the shader cache key once; cold JIT timings were
0.203 s for fallback, 0.388 s for HIP, and 0.799 s for Vulkan, consistent with
the earlier cold Vulkan trace. The current scan covers 167 files and 60,239
lines, leaving only two over-limit files. `path_tracer_kernel.cpp` is no
longer allowlisted by the size gate.

The path kernel is now a real host-stage component pipeline in ordinary
headers and translation units. `PathKernelPipeline` owns the Luisa path-loop
scope and invokes virtual closest-event, geometry, shading, and scattering
stages while the kernel AST is recorded. Its direct-light stage owns an
ordered registry of environment, emissive-mesh, and analytic-light
components. Typed invocation, sample, bounce, geometry, and shading contexts
make cross-stage lifetime explicit. Sample setup, path execution, per-sample
accumulation, and final film writes each own their balanced lexical scope.
There are no new device `Callable` boundaries and no device-side virtual
dispatch.

The closest-event stage has subsequently been split at its semantic
boundaries. `PathBounceSetupStage` owns one RNG-dimension set and one mesh
trace, `ClosestEventStage` selects a typed analytic-light/surface/background
event, and dedicated forward-light and background stages resolve it. A lamp
remains a transparent event inside the same bounce. This exposes the exact
segment distance needed by volume transport without duplicating sampling or
retracing the mesh, while keeping every implementation in a real `.cpp`
component.

All eight path-kernel `.inl` files have been deleted. The only remaining
first-party `.inl` is the generated Cycles 4.5.10 BSDF table. The session
initializer is 300 lines, the internal interface is 341 lines, and the largest
implementation is the 620-line setup/film module. The complete source gate
now covers 187 first-party files and 61,412 lines; every hand-written file is
below 2,000 lines and the hand-written `.inl` debt set is empty.

The checkpoint passed the 32-worker full build and all 42 CTest contracts.
The 64×64, 256 spp camera/material fixture was rendered through fallback,
HIP, and Vulkan. All 13 linear passes on all three backends (39 PFM files)
are byte-identical to the pre-component baseline, including Combined
SHA-256
`4ac47cfe7d528e14da89e116f1dffd3b8dbf6536f8ff0817d87ef66a4f3409b0`.
Observed JIT times after the source move were 0.199 s for fallback, 0.571 s
for HIP, and 1.166 s for Vulkan; render times were 0.0176 s, 0.00668 s, and
0.0178 s respectively. A trace-enabled 32×32 point-light render also produced
a raw indexed path-trace JSON file byte-identical to the pre-component
baseline, covering the diagnostic-only closure, light, state, and
post-scatter records that the ordinary image fixture does not enable.

The exact historical Lone Monk bundle was then rerendered at 640×480 and
64 spp through fallback. It compiled 348 geometries, 87,541 instances, and all
37 raw material graphs; all 13 linear passes are byte-identical to the
pre-component baseline. Scene compilation took 1.176 s, cold JIT compilation
took 17.610 s, and rendering took 5.721 s. The Combined image was opened at
native resolution and checked across the grass band, courtyard architecture,
roof highlights, dark foreground arches, and cast-shadow structure. No visual
change is present, as required by the byte comparison. Because the current
Psycles panel is identical, the existing Cycles-HIP/Psycles-fallback/absolute
difference triptych remains the visual record:

![Lone Monk Cycles HIP, Psycles fallback, and absolute difference](validation/2026-07-30/lone-monk-five-way/triptychs/psycles-fallback-vs-cycles-hip-combined.png)

The first Blender-adapter checkpoint separated the pipeline stages. Binary
geometry and public scene assembly remain in `blender_scene.cpp`, which
decreased from 4,440 to 1,184 lines, and JSON decoding is a 348-line
translation unit. The natural-output dispatcher was initially partitioned as
four textual family fragments in its original lexical context. As with the
first `GraphSurface` split, this established a low-risk equivalence baseline
but was not the final component boundary.

All four family bodies compare byte-for-byte with their pre-split source
ranges. The split passed the 32-worker full build and all 42 tests. The
64×64, 256 spp fallback/HIP/Vulkan material fixture again produced 39 linear
pass files byte-identical to the preceding baseline. The complete scan now
covers 174 files and 60,394 lines, and only the probe generator remains in the
temporary debt budget. `blender_scene.cpp` is no longer allowlisted by the
size gate.

The normalizer now delegates to real `BlenderNodeLoweringComponent` host
objects for input/context, color/value, procedural, and closure nodes. A typed
context interface owns graph mutation, socket binding, images, properties,
tables, and diagnostics. The normalizer retains only recursion, memoization,
group-context restoration, and ordered component dispatch. Its `finish`
callback still removes the active recursion marker at exactly the original
successful-lowering points. Components translate the exported raw Cycles
graph; they do not bake values, materials, or closures through Blender or
Cycles.

All four old adapter `.inl` files have been deleted. The largest family
translation unit is 702 lines and the central normalizer is 1,182 lines. The
32-worker build and all 42 tests pass. Re-rendering the 64×64, 256 spp fixture
through fallback, HIP, and Vulkan again made all 39 linear pass files
byte-identical to the immediately preceding compiled-`GraphSurface` baseline.
The complete gate now covers 185 first-party files and 61,639 lines with no
file over the limit.

The component path also normalized and compiled all 37 raw material graphs in
the current Lone Monk bundle; its only diagnostics were the already-known
adaptive-sampling and denoising configuration notices. The first render check
used a newer exported scene whose JSON differs from the historical smoke
baseline, so it was not treated as a source-regression comparison. Repeating
with the exact baseline bundle
`psycles-lone-monk-five-way-20260730/export` produced all 13 fallback passes
byte-for-byte unchanged at 640×480 and 64 spp. Scene compilation took 1.184 s,
warm JIT loading 0.274 s, and rendering 5.438 s. The Combined image was opened
at full resolution; camera, architecture, foliage/grass, material regions,
and shadow structure are unchanged, as required by the byte comparison.

The same binary also completed a 640×480, 64 spp Lone Monk smoke test on all
three Luisa backends, loading 348 geometries, 87,541 instances, and 37 raw
material graphs. The images were inspected at full resolution and the HIP/
Vulkan structure remains visually consistent; their Combined relative RMSE
is 0.0201 at this low sample count. The old five-way images are not a valid
byte baseline for this refactor because they predate the intervening RNG,
Light Path, transparent-shadow, and volume-closure fixes.

An immediate same-binary repeat was byte-identical for all 13 fallback passes
and all 13 Vulkan passes. HIP repeated exactly for five passes; eight passes
contained sparse differences (99% of Combined pixels were identical),
with Combined RMSE 0.000565 and relative RMSE 0.000363. Because this occurs
between two executions of the same binary, it is recorded as a separate
HIP/HIPRT determinism finding rather than attributed to the source split.

This cold run also isolated complex-shader compilation costs. Fallback JIT
took 18.4 s. HIP JIT took 217.0 s: AMDGPU code generation took 19.1 s, while
linking the HIP LLVM bitcode into the code object took 194.3 s. Vulkan JIT
took 135.7 s. Warm-cache JIT times were 0.278 s, 0.273 s, and 1.157 s for
fallback, HIP, and Vulkan respectively. Render-only times were 5.43 s,
2.37 s, and 2.08 s.

The Cycles probe generator is the final completed decomposition.
`create_cycles_shader_probe.py` decreased from 5,518 to 189 lines and is now
only the stable Blender CLI, canonical registry, and shared scene setup.
Builders live in modules for common construction, camera/lights, closures,
texture inputs, procedural textures, and color/value operations. The largest
module is `texture_inputs.py` at 1,868 lines.

All 111 moved function bodies compare exactly with their original source, and
all 89 probe-name/function mappings retain their original order and targets.
The Blender regression now also requires the creator registry to equal the
canonical runner inventory. Representative probes from every module family
were created through the real Blender CLI. Re-exporting the camera/filter
probe produced an identical `geometry.bin` and an identical `scene.json`
after removing only the expected absolute source-`.blend` path.

The final 32-worker build and all 42 tests pass. The complete source gate now
covers 181 first-party files and 60,499 lines with no exceptions: every
hand-written first-party source file is at most 2,000 lines, and the temporary
debt budget is empty.

## Cycles-domain geometry component

The next component checkpoint replaced the flattened scene geometry contract
with explicit point, corner, and face attributes. Blender schema v2 now
preserves shared triangle indices and the same normal/UV/Generated
cardinalities used by current Cycles. Historical v1 bundles remain readable
for isolated same-binary regression comparisons.

Repeated bindless triangle reads from closest-hit, emissive-light, and
transparent-shadow code are now generated by a real
`TriangleGeometryComponent` host object in its own header and translation
unit. Generic named attributes select point, corner, or face indices in one
shader-service implementation. The full-path regression combines all three
domains in one raw shader graph and passes on fallback, HIP, and Vulkan.

The 32-worker build and all 47 tests pass. `path_tracer_scene.cpp` remains the
largest hand-written implementation at 1,972 lines, still below the enforced
2,000-line ceiling but close enough to require the next scene-upload
component split rather than further growth.

Unmodified exports reduced `geometry.bin` by 25.66% for Lone Monk, 28.76% for
Classroom, and 20.30% for the 28,089,460-triangle Blender 4.1 Splash. A
same-current-binary Lone Monk v1/v2 comparison has Combined relative RMSE
`0.0001702`; visual structure is unchanged, and Normal RMSE is
`5.42e-8`. The v2 export follows Cycles' point-versus-corner normal decision,
so byte identity with the old unconditional loop-normal representation is not
the semantic target.

The complex v2 scene also completed 640x480/64 spp renders on all three Luisa
backends. Cold JIT was 19.10 s on fallback, 222.14 s on HIP, and 137.13 s on
Vulkan. HIP spent 199.98 s in final device-bitcode linking; Vulkan optimized
1,606,220 SPIR-V words to 1,349,527. Warm JIT was 0.266 s and 1.133 s for HIP
and Vulkan. Their render-only times were 2.244 s and 2.213 s, versus 1.926 s
for official Cycles HIP on the same RX 9070 XT.

The complete numerical reports and the inspected triptychs are in
[`validation/2026-07-31/compact-geometry`](validation/2026-07-31/compact-geometry/README.md).

## Target boundaries

The path tracer is split by renderer semantics:

- session resource allocation and immutable kernel parameters;
- camera sample and primary-ray construction;
- path state and Cycles RNG dimensions;
- closest-event traversal and self-intersection policy;
- surface shading and data-pass extraction;
- direct-light and forward-light estimators;
- volume stack, free-flight, phase sampling, and volume direct lighting;
- film/pass accumulation and diagnostic path tracing.

The material stack is split by compilation stage:

- Blender JSON and binary scene decoding;
- Blender node-tree normalization by node family;
- typed value-program lowering by node family;
- surface closure evaluation/sampling;
- volume closure coefficients and phase functions;
- textures, procedural nodes, and color operations.

Probe-generation modules are grouped by feature family. Their functions
remain independent, the canonical registry is checked against the runner, and
representative generated `.blend` files are re-exported to verify the raw
Cycles graph contract.

## Refactoring rules

Structural commits must not change renderer semantics. Each slice therefore:

1. moves one named responsibility behind an explicit internal interface;
2. keeps runtime kernel arguments, material parameter binding, and RNG
   dimension consumption unchanged;
3. uses the 32-thread full build and complete CTest suite;
4. runs focused fallback, HIP, and Vulkan device fixtures for touched DSL
   code;
5. runs the relevant Cycles/Psycles differential probe when a refactor can
   affect emitted AST control flow or numerical ordering;
6. records compile-stage or render-performance changes when a callable
   boundary changes generated code.

Feature work must use the new boundary instead of appending code back to a
known concentration point. File size is re-measured after each structural
checkpoint.
