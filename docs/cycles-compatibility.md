# Cycles compatibility status

The [microfacet anisotropy checkpoint](validation/2026-08-27/microfacet-anisotropy/README.md)
implements Blender 5.2/Cycles anisotropy for standalone Glossy and the
Principled metallic/dielectric lobes without baking the shader graph. Authored
anisotropy, rotation, and tangent remain typed SVM operands; closure setup
projects them once to `(T, alpha_x, alpha_y)`, and GGX/Beckmann evaluation and
visible-normal sampling consume that physical state. Complete, physical, and
branch-local closure transports now have field-complete round-trip regressions.
The analytic rotation-covariance and ABI suite passes on fallback, HIP, and
strict native XIR-to-SPIR-V Vulkan. Full-scene image/performance promotion is
recorded separately and is not implied by this checkpoint.

The [Blender 5.2 surface/SVM gap audit](validation/2026-08-27/cycles-5.2-surface-svm-gap-audit/README.md)
separates the already data-driven typed SVM from the remaining node/closure
coverage and post-population tagged-sum consumption gaps. It records the exact
5.2 inventory, current Barbershop program census, source-level Cycles
comparison, rejected family-decoder experiments, and the normalized 2.676x HIP
`shade_surface` gap.

The [Cycles 5.2 SVM closure-core checkpoint](validation/2026-09-02/cycles-5.2-svm-closure-core/README.md)
copies the closure allocator and typed setup transitions into the replacement
Luisa stack/PC interpreter. Its current standalone families include BSSRDF,
Sheen/Velvet, Diffuse/Glossy Toon, and Ray Portal. Ray Portal preserves the
four-word typed payload, implicit Position link, signed weight, extinction
ordering, zero-Direction fallback, direct allocation, and portal feature gate.
Exact compiler words and runtime state pass on fallback, HIP, and strict native
XIR-to-SPIR-V Vulkan. This is a staged interpreter checkpoint, not yet a claim
that production shade-surface or closure evaluation/sampling has switched over.

The [packed typed surface-operand checkpoint](validation/2026-08-27/surface-value-packed-operands/README.md)
encodes two complete typed graph edges per 32-bit word and embeds operands for
zero-, one-, and two-input nodes. It formally preserves the topological typed
IR while reducing complex-scene operand traffic by 48.3%--54.7%. Paired HIP
profiling reduces normalized Barbershop/Classroom surface cost by 6.87%/6.06%
and render-only time by 2.88%/3.85%; Monster is explicitly neutral. Expanded
versus compact evaluation passes on fallback, HIP, and native-XIR Vulkan.

The [HIP BSSRDF local-intersection domain checkpoint](validation/2026-08-13/hip-bssrdf-local-domain/README.md)
formally narrows spatial subsurface traversal from the whole scene to complete
triangle objects that can originate BSSRDF transport. On Monster it reduces
the domain from 36 instances to 3, cuts the HIP subsurface stage by 31.59%,
and reduces the warm 640x480x64 render median by 10.69%, with three-backend
regressions and 640x480 EXR triptychs.

The [surface-topology callable checkpoint](validation/2026-08-13/hip-surface-topology-callables/README.md)
keeps original typed closure graphs behind per-topology Luisa callables and
formally bounds HIP inlining over CFG-proven alternative frontiers. On Monster
it reduces the warm 640x480x64 HIP median another 8.46% to 2.24823 s, 4.09%
faster than the same-device Cycles HIP checkpoint. Full-pass, three-backend,
strict native Vulkan, profiler, and visual-triptych evidence is recorded.

The [HIP retained-callable aggregate ABI checkpoint](validation/2026-08-13/hip-callable-aggregate-abi/README.md)
formally narrows post-IPO material-callable arguments and records a 6.67%
Monster render-time reduction, full-pass numerical validation, profiler
evidence, and a visual triptych.

Psycles treats Blender Cycles as the semantic reference. A feature is complete
only after the same Blender scene and render settings have been evaluated by
official Cycles and Psycles-Luisa and their linear passes have passed a
feature-specific differential threshold.

## Current complex-scene checkpoints

The official Monster Under the Bed and Lone Monk scenes now both complete on
the canonical five-way matrix: Cycles CPU/HIP and Psycles fallback/HIP/Vulkan.
The [Monster BSSRDF exit-normal checkpoint](validation/2026-08-07/monster-bssrdf-exit-normal/README.md)
now aligns the closure-weighted normal of Cycles' synthetic exit Lambert. The
exact path agrees through the next primitive, and 960x960x512 Combined
relative RMSE falls 1.69x to `0.048905` with a `1.003553` mean-luminance
ratio. Its exact scene-level BSSRDF tag specialization preserves all 15 AOVs
pixel-for-pixel at 1 and 512 spp, reduces the HIP code object by 8.20%, and
reduces cold JIT by 3.33% to `129.689 s`; the 98.8-second link remains open.
The preceding
[Monster current-head checkpoint](validation/2026-08-07/monster-current-head/README.md)
retains the five-way 128-spp matrix and independent Principled Coat Normal
evidence. The
[current Lone Monk checkpoint](validation/2026-08-07/lone-monk-current-head/README.md)
records 0.0248--0.0262 Combined relative RMSE across the three Psycles
backends at 960x720x128, with visually aligned fine grass and data passes. The
[muted-node follow-up](validation/2026-08-07/lone-monk-muted-node-bypass/README.md)
then proves and repairs a stable book-material residual: the source graph's
muted Mix is replaced by its exported runtime internal link. At 512 spp,
Diffuse Color RMSE falls 20.52x to `0.00020008`; a fresh five-way matrix puts
all three Psycles Diffuse Color relative RMSEs between `0.001106` and
`0.001160`.

The subsequent
[scene-union closure-reachability checkpoint](validation/2026-08-10/closure-reachability/README.md)
keeps those raw graphs but records only physical Principled families reachable
by at least one parameter block sharing each topology. On Lone Monk it reduces
cold fallback/HIP/Vulkan path-kernel JIT by 3.25x/2.47x/2.06x, reduces native
Vulkan optimized SPIR-V by 62.4%, and lowers its peak compiler RSS from about
38.1 GiB to 8.98 GiB. The proof treats linked and non-finite inputs as unknown
and has both host reachability and generated-XIR regressions.

Neither result is a blanket feature-parity claim. Monster's remaining
high-spp indirect residual and dominant HIP link/render cost remain open,
while Lone Monk's indirect convergence and severe current
performance gaps also remain open. Classroom, Barbershop, and Blender 4.1
Splash retain historical evidence but still require fresh current-head
five-way promotion runs.

## Shader graph path

Psycles does not ask Blender/Cycles to pre-bake materials or execute closures
on the host. The active implementation is nevertheless required to compile the
original graph into a Cycles-5.2.1-isomorphic SVM stream and execute the same
stack/PC/node/closure state machine in Luisa DSL. The existing custom typed
topological executor is transitional code scheduled for wholesale replacement,
not an allowed competing architecture. Shader data follows this target path:

1. Blender exports the original node trees, links, socket defaults, static
   node properties, mute state, runtime internal links, image identities, and
   evaluated scene geometry.
2. The Blender adapter walks the active Surface and Volume roots recursively.
   Before node-specific lowering, it replaces muted nodes with their
   Cycles-equivalent same-type internal-link proxies and removes dead input
   branches. The replacement compiler then applies the same socket
   conversions, stack allocation, closure transforms, and typed SVM-node
   emission as Cycles 5.2.1. Displacement is part of the same shader program
   and shader-jump control flow.
3. Surface, volume, and displacement use the same Cycles-isomorphic node
   stream and stack contract. Closure nodes update the same logical closure
   state at the same point in that stream; there is no independent Psycles
   closure-tree executor.
4. Shader node groups are recursively expanded through their exported Group
   Input/Output interfaces. Group instance names have no semantic role;
   missing and recursive groups produce explicit diagnostics.
5. Image pixels, mesh attributes, parameter blocks, and geometry are uploaded
   to Luisa resources. No host-side shader evaluator is used.
6. Bump dependencies use Cycles' enter/leave-bump and derivative node
   semantics in the same stream; Psycles does not substitute a separately
   scheduled derivative program.
7. One Luisa SVM interpreter performs the Cycles PC loop and one primary
   node-type dispatch. JIT scene pruning may omit unreachable Cycles cases but
   may not add a family/subtype dispatch layer or change node state flow. The
   same evaluator runs on Luisa `fallback` and GPU backends.

Muted-node normalization is relational rather than node-specific. A muted
node's concrete implementation is absent, only the internal input reachable
from the requested output is lowered, an unconnected proxy is the zero value
of its output type, and a muted output without a proxy leaves the destination
input default intact. Invalid, explicitly muted, and unavailable-socket links
are absent at export. The Blender exporter regression, typed-topology
regression with a deliberately unsupported dead branch, exact Lone Monk path
trace, 512-spp HIP differential, and five-way full-scene rerun are recorded in
the [muted-node checkpoint](validation/2026-08-07/lone-monk-muted-node-bypass/README.md).

Image transport preserves Blender datablock ownership. Packed payloads are
copied exactly, external `//` paths are resolved relative to the owning linked
library (not blindly relative to the main file), and `GENERATED` datablocks are
encoded from their pixel buffers without mutating the source scene. Both
linked-library and generated-image paths have Blender-side regressions and
were exercised by the official Classroom asset.

## Geometry attribute contract

Blender scene schema v2 uses Cycles' native triangle attribute cardinalities
instead of flattening every corner into a synthetic vertex:

- positions and Generated coordinates are indexed by Blender point;
- evaluated UVs and MikkTSpace tangents/signs are indexed by triangle corner;
- normals use point storage unless Blender/Cycles selects corner normals;
- arbitrary named color, UV, and tangent attributes retain point, corner, or
  face domains;
- triangle material, smooth, and random-per-island values remain face-domain.

The exporter writes actual shared triangle indices. Luisa uploads every
attribute once at its natural cardinality, and a common shader-service lookup
maps `(primitive, barycentric, domain)` to the same point/corner/face elements
used by Cycles. A lower-level host-stage triangle-primitive component resolves
face material slots, instance overrides, smooth flags, exact Cycles
shader/object identities, and the graph-derived volume capability bit.
Surface, emissive-mesh, transparent-shadow, and the pending volume-only
traversal then share that primitive boundary and one triangle-geometry
component, so resource or material semantics cannot silently drift between
path estimators. Its material/override and volume-entry records pass on Luisa
fallback, HIP, and Vulkan.

The historical v1 reader remains available only to make same-binary
regression comparisons possible. Production exports use v2. The full
Lone Monk, Classroom, and Blender 4.1 Splash measurements and visual A/B are
recorded in
[the compact-geometry checkpoint](validation/2026-07-31/compact-geometry/README.md).

Unknown nodes, modes, sockets, or properties produce named coverage
diagnostics and keep the release gate red. Empty Blender material slots use
Cycles' default Principled surface (0.8 base color, 0.5 roughness, 1.5 IOR,
and Multi-GGX); they are not missing-material errors. An unconnected Cycles
Surface follows the separately probed opaque-black surface contract.
Unsupported node outputs use their exported socket default only together with
a named warning, never as an unreported compatibility claim.

`docs/cycles-shader-nodes-4.5.10.json` is the versioned Blender RNA inventory.
`tools/check_cycles_shader_node_coverage.py --require-complete` is deliberately
red until every Cycles-applicable node has a verified Luisa implementation.
An implementation used by Lone Monk is still classified as partial or
unverified until a focused Cycles linear-EXR probe covers its modes and socket
semantics.

The `node_group_color` structural probe uses two levels of arbitrarily named
group instances and currently matches Cycles exactly for Combined, Normal, and
DiffCol at 64×64 (RMSE 0 for all three passes).

The full-frame color/value probes cover RGB-to-BW scene-linear luminance,
Gamma's zero/positive/negative exponents, Brightness/Contrast clipping, both
Clamp range-order modes, RGB/HSV/HSL Separate/Combine Color dispatch, and
Hue/Saturation/Value hue wrapping, saturation clamp, value scaling, and factor
blending. All currently match Blender 4.5.10 Cycles exactly for Combined,
Normal, and DiffCol at 64×64 (RMSE and maximum absolute error 0).

Color Ramp follows Cycles' normalized pre-SVM contract rather than reimplementing
Blender's color-band evaluator in the path kernel. The exporter evaluates the
Blender ramp at the same 257 endpoints used by Cycles; the Luisa shader indexes
that immutable table with Cycles' clamp, scale, floor, and interpolation rules.
The constant matrix covers RGB/HSV/HSL, Linear/Constant/Ease/Cardinal/B-Spline,
all four hue paths, out-of-range factors, exact table boundaries, Color, and
Alpha. Against official Cycles its Combined RMSE is `1.61e-8` for Color and
exactly zero for Alpha; Normal and DiffCol are exactly zero. A spatial
Generated-coordinate probe additionally exercises dynamic indexing and measures
Combined RMSE `6.36e-4`, with the residual attributable to film sampling.
`VALTORGB` is therefore `cycles_verified`.

Volume graph preservation and coefficient evaluation now have a separate
non-rendering compatibility boundary. The Blender adapter retains raw
Absorption, Scatter, Volume Coefficients, and Principled Volume nodes,
including Add/Mix topology and phase choice. `SurfaceProgram` lowers these to
a typed volume-closure tree, and `GraphSurface` evaluates `sigma_t`, `sigma_s`,
and emission as Luisa expressions. The formulas are pinned to Cycles main
`2bad74a8`, including volume-only negative closure-weight clamping, object
density scaling, Principled `sqrt(Absorption Color)`, implicit density and
temperature attribute presence, and emission suppression for shadow or
extinction-only evaluation. The same coefficient fixture passes on Luisa
fallback, HIP, and Vulkan.

The individual volume phase families are now a second internal checkpoint.
Henyey–Greenstein, Rayleigh, Draine, Fournier–Forand, and the fitted Mie
HG/Draine split are expressed directly in Luisa DSL, including Cycles'
anisotropy/IOR/backscatter clamps, Fournier–Forand Newton inversion, and the
fast-math polynomial/bit conventions used by the Mie fit. A 28-record
evaluation and deterministic-sampling fixture was generated by compiling the
unmodified `volume_util.h` from official Cycles main
`b82c3f0da6c1813dabedc563d64e536f4d83e868`; it passes on fallback, HIP, and
Vulkan. Four additional records prove Draine's exact Rayleigh and
Henyey–Greenstein reductions.

Raw graph phase allocation and mixture selection now form the next internal
checkpoint. `GraphSurface` sends original closures to an OOP host-stage
collector without Blender/Cycles pre-baking. The fitted Mie closure expands to
its original HG and Draine pair. `VolumePhaseSet` matches Cycles' negative
weight clamp, exact-parameter merging, stable first-eight truncation,
monotonic allocation budget, scalar sample-weighted evaluation, and
single-closure reservoir selection/random-number rescaling. The combined
38-record coefficient/phase-set fixture includes the original
Draine, fitted-Mie HG, fitted-Mie Draine, and Principled HG closures plus
merge, non-refunding allocation budget, capacity, truncation, empty-set, and
combined single-trace regressions.
Its phase values and three deterministic samples are pinned to official
Cycles main `b82c3f0` and pass on fallback, HIP, and Vulkan. This is still not
a render-compatibility claim.

Volume boundary state is the third internal checkpoint. The device-local
`VolumeStack` matches Cycles' terminator-reserved capacity, exact
`(object, shader)` identity, stable enter, swap-last exit, volume/transmission
guards, shadow-stack copy, and fixed 32-slot bound. Its camera initializer
matches the bounded +Z enclosure probe, including object-only membership and
the otherwise easy-to-miss duplicate front-hit accounting. A 23-record
fixture pinned to official Cycles main `b82c3f0` passes on fallback, HIP, and
Vulkan. The entries additionally retain the raw Psycles surface dispatch
handles required to evaluate each medium without changing Cycles identity
semantics.

The camera traversal itself is now a separate OOP host-stage component rather
than a synthetic stack test. An 18-record fixture builds real triangle BLASes
and a TLAS, filters a nearer non-volume primitive in any-hit, distinguishes
open enclosing media from a closed entered/exited object, seeds the exact
world `(object, shader)` entry, advances with Cycles'
`intersection_t_offset`, and checks front/back orientation under a
negative-scale instance. The same fixture now constructs the real move-only
per-sample state twice: one sample performs the enclosure probe and a second
uses the background-only fast path. Their independent counts and terminators
prove that sample-local stacks neither alias nor leak state. It then sends
labels produced by the real Cycles event-to-label mapping through the
production surface-crossing component. Diffuse reflection leaves the stack
unchanged, transparent transmission enters exactly once, a duplicate front
crossing does not duplicate the medium, the matching back crossing exits,
and a non-volume surface is ignored. It passes on fallback, HIP, and Vulkan.
This exposed a general Luisa fallback defect:
shader arguments submitted after an asynchronous TLAS build captured the
pre-build instance-array pointer. Luisa `next` commit `c55b8d57b` replaces
that snapshot with a stable descriptor indirection, bumps the fallback
shader-cache ABI, and adds a same-stream trace/visibility/user-id/transform
regression.
The implementation and focused evidence are recorded in
[`validation/2026-07-31/camera-volume-stack`](validation/2026-07-31/camera-volume-stack/README.md).

The camera component is installed in the main per-sample path state together
with Cycles' zero-initialized volume-bounce, volume-bounds-bounce, and optical
depth fields. The main surface-scatter stage now applies the same formal
boundary transition after a valid sampled continuation and only for the
exact Cycles `LABEL_TRANSMIT` bit. It reuses the primitive's effective
material/object identity and raw graph dispatch handles, so it performs no
closure or medium pre-baking. Host specialization emits no volume locals,
ray query, or crossing call for volume-free scenes. The scene-side
specialization now follows Cycles'
preprocessing as well: effective instance material overrides determine volume
objects, transformed object AABBs determine pairwise overlap, and stack
capacity starts with the world/terminator pair before applying the same
conservative intersecting-object count and 32-slot cap. The current render
aspect and camera near-view-plane/aperture bound decide whether the +Z
enclosure query is needed. The host-only regression covers perspective,
orthographic, and panorama cameras, negative-scale bounds, world-only,
disjoint, overlapping, overridden, and saturated scenes. Details are in
[`validation/2026-07-31/volume-scene-metadata`](validation/2026-07-31/volume-scene-metadata/README.md).

Production world/object volume-point reconstruction and closest-event
homogeneous free flight are now connected. Before every mesh, analytic lamp,
or background event, `PathVolumeSegmentStage` evaluates the original stacked
closure graphs, resolves attenuation/emission or a phase collision, and
applies the total Cycles volume path-state transition. Scene compilation now
accepts raw volume programs structurally proven homogeneous and explicitly
rejects spatially varying volume dependencies at the production path gate.
Heterogeneous acceleration metadata, overlap traversal, object/World
coordinate conversion, and runtime Light Path extrema re-evaluation are
implemented, but collision/phase/direct-light integration remains open;
volume lighting outside the exact homogeneous analytic-light subset therefore
remains unverified in the public node matrix.

World volume identity is now complete in scene schema v2: the exporter retains
Cycles' background-light object index separately from the default-background
shader index, and both round-trip through the contract adapter. This is the
same `(object, shader)` pair that Cycles writes before its camera enclosure
probe; the stack does not infer it from object counts or sentinels.
Blender's `DISTANCE`, `EQUIANGULAR`, and `MULTIPLE_IMPORTANCE` material
settings now round-trip beside each raw closure graph and reach every runtime
stack entry through the effective material binding. Their exact
`1`, `2`, and `3` technique-bit representation makes the whole-stack sampling
method an order-independent union. The empty, single-technique, mixed,
explicit-MIS, and copied-stack cases pass on fallback, HIP, and Vulkan.

The analytic homogeneous segment estimator is the fourth checkpoint.
Its Luisa `.h`/`.cpp` component matches Cycles' per-channel transmittance,
second-order small-optical-depth emission integral, throughput/albedo-weighted
RGB channel selection, bounded exponential density, random-number rescaling,
and the scatter/transmit estimator weights. It also accepts Cycles' externally
guided VSPG scatter probability without changing that measure. A 59-record
fixture generated from official Cycles main `b82c3f0` `volume.h`, `mapping.h`,
and the current homogeneous control flow now also pins the formal
Distance/Equiangular/MIS selector, equiangular endpoint measure, and degenerate
PDF boundaries, finite-emitter intervals, and both spot-proposal contracts. It
passes on fallback, HIP, and Vulkan.
The Sobol contract now pins the official volume phase, reservoir, scatter
distance, expansion, shade-offset, and phase-guiding dimensions. The
production path composes this estimator before its typed closest-event stages
and enables the homogeneous subset.

Homogeneous distant-light NEE is now the next production checkpoint. It
reuses Cycles' rescaled reservoir/scatter dimensions, evaluates the original
phase closures, integrates ordered surface plus medium shadow transmittance,
and routes the result to Combined and Volume Direct. Primary rays implement
the exact empty-history VSPG defensive probability. The official Blender
5.2.0 one-sample EXR matches fallback, HIP, and Vulkan with maximum absolute
error `3.7253e-9`; reports and inspected triptychs are in
[`validation/2026-07-31/homogeneous-volume-direct`](validation/2026-07-31/homogeneous-volume-direct/README.md).
Finite point-light equiangular/MIS is now connected through a separate
proposal/re-sample protocol. The selected point light is sampled before
integration to establish the equiangular reference, the MIS branch remaps the
Volume Scatter Distance dimension before indirect free flight, and the same
emitter plus original Light coordinates are sampled again from the final
collision point. The 16-pixel official Blender 5.2.0 CPU EXR matches fallback
and HIP with maximum absolute error `1.4901e-8`, and Vulkan with
`2.3842e-7`. Reports, EXRs, and inspected triptychs are in
[`validation/2026-07-31/homogeneous-volume-point-mis`](validation/2026-07-31/homogeneous-volume-point-mis/README.md).
Finite spot-light MIS now composes two deliberately different Cycles sampling
contracts: the segment proposal always uses the visible sphere cap and keeps
zero-attenuation geometry valid, then the clipped-cone interval drives
free-flight before an ordinary spot sample from the final collision. The
16-pixel official Blender 5.2.0 CPU EXR matches fallback and HIP with maximum
absolute error `1.1921e-7`, and Vulkan with `3.2783e-7`. Reports, EXRs, and
visually inspected triptychs are in
[`validation/2026-07-31/homogeneous-volume-spot-mis`](validation/2026-07-31/homogeneous-volume-spot-mis/README.md).
Finite rectangle/ellipse area-light MIS now preserves Cycles' separate
authored-primitive segment proposal, spread-clamped collision proposal, and
shared forward-hit evaluation. Full and narrow-spread official CPU EXRs match
fallback, HIP, and Vulkan with maximum absolute error at or below
`1.1027e-6`. Reports, EXRs, and inspected triptychs are in
[`validation/2026-07-31/homogeneous-volume-area-mis`](validation/2026-07-31/homogeneous-volume-area-mis/README.md).
Mesh-emitter volume MIS now uses Cycles' distinct
`triangle_light_sample<true>` segment-area proposal and
`triangle_light_sample<false>` collision-position measure. A shared
host-stage component also owns surface NEE and forward-hit PDF evaluation,
raw emission-closure evaluation, authored front/back support, and the
inverse-transpose orientation equivalent to Cycles'
`SD_OBJECT_NEGATIVE_SCALE` correction. The latest official Cycles main
`b82c3f0d` 4×4 one-sample CPU EXR uses a reflected FRONT-only emitter and
matches fallback/HIP/Vulkan with maximum absolute errors
`6.5565e-7`, `6.4820e-7`, and `2.3451e-6`. A second production render pins
Volume Scatter visibility exclusion. Reports, EXRs, and inspected triptychs
are in
[`validation/2026-07-31/homogeneous-volume-triangle-mis`](validation/2026-07-31/homogeneous-volume-triangle-mis/README.md).
Environment volume NEE now shares the background-map sampling, forward PDF,
and raw World-closure component with surface lighting. Its latest-Cycles
4×4 oracle matches fallback, HIP, and Vulkan with maximum Combined errors
`1.4901e-8`, `1.1921e-7`, and `6.1691e-6`, including Camera and Volume
Scatter World visibility controls. Reports, EXRs, and inspected triptychs are
in
[`validation/2026-07-31/homogeneous-volume-environment-mis`](validation/2026-07-31/homogeneous-volume-environment-mis/README.md).
Emissive-triangle and environment sampling now expose radiometry-free
proposals and separate raw closure evaluators. The proposal APIs accept only
scene data and cannot evaluate shader AST; surface self/side validity and
volume visibility are established before the evaluator receives path state.
Compile-time interface regressions plus fallback/HIP/Vulkan surface and volume
renders passed `123/123`. A 640x480 Lone Monk rerun is pixel-exact against the
previous fallback and Vulkan EXRs; HIP differences remain below its measured
pre-existing cold/warm nondeterminism. Reports and inspected triptychs are in
[`validation/2026-08-03/light-proposal-emission-phase`](validation/2026-08-03/light-proposal-emission-phase/README.md).
Surface analytic, triangle, and environment NEE now also reproduce Cycles'
constant-emission scheduling boundary. A total closure-tree transfer relation
classifies non-emitting, constant, and deferred graphs from socket provenance;
linked values remain deferred even when their source is a Constant node. The
constant device callable can read only the runtime parameter block, while the
full raw closure evaluator runs after a non-zero receiving BSDF and before
shadow traversal. Constant World background hits use the same restricted
path. Full CTest passed `123/123`; the fresh five-way Lone Monk render is
pixel-exact against the previous fallback and Vulkan EXRs, with HIP inside its
measured runtime non-determinism floor. The formal relation, compile data,
reports, and inspected triptychs are in
[`validation/2026-08-03/constant-light-emission`](validation/2026-08-03/constant-light-emission/README.md).
Homogeneous and heterogeneous volume NEE now use the same compiler relation
through an explicit host-stage provider protocol: direction proposal,
constant emission, receiving phase evaluation, then guarded deferred raw
emission. Light roulette now precedes both surface and volume shadow queries
on this path. The refactor remains one fused Luisa kernel and preserves raw
Blender closures. Full CTest passed `123/123`; all six new fallback/HIP/Vulkan
EXRs are bit-for-bit identical to the retained pre-refactor Psycles images.
Latest-Cycles CPU differentials and all nine inspected triptychs are in
[`validation/2026-08-03/volume-light-emission-phase`](validation/2026-08-03/volume-light-emission-phase/README.md).
Raw Principled BSDF Emission Color and Strength now remain linked expressions
from Blender export through the typed surface program and Luisa device AST.
Principled is structurally deferred exactly because Cycles' Alpha, Sheen, and
Coat layers can affect its radiance, while a separate per-material
`output_estimate_emission`-style query handles emitter discovery without
becoming a host material evaluator. A linked full-frame probe matches current
Cycles CPU exactly in Combined and Emit on fallback, HIP, and Vulkan; reports
and all six inspected triptychs are in
[`validation/2026-08-03/principled-emission`](validation/2026-08-03/principled-emission/README.md).
The follow-up implements the exact ordered Alpha, Sheen LTC, reflective Coat
GGX/Fresnel, and Coat Tint attenuation of that raw emission as a reusable
host-stage `.h`/`.cpp` component that records Luisa device expressions. A
16-cell matrix includes linked Coat Normal and the degenerate zero-normal
boundary. Combined and Emit match current Cycles CPU within at most one or two
float ULPs on fallback, HIP, and Vulkan; all six triptychs were inspected.
Formulas, reports, timings, and images are in
[`validation/2026-08-03/principled-emission-layers`](validation/2026-08-03/principled-emission-layers/README.md).
Physical Alpha now follows Cycles' per-leaf signed cutoff, first-allocation
order, and global transparent-closure merge. A 16-cell raw-node matrix is
bit-for-bit identical to Cycles CPU in Combined and Environment on fallback,
HIP, and Vulkan; the formal relation, reports, and all six inspected triptychs
are in
[`validation/2026-08-03/principled-alpha`](validation/2026-08-03/principled-alpha/README.md).
Physical Principled Coat is now implemented as the second ordered host-stage
layer, including GGX energy compensation, Fresnel/albedo attenuation, Tint
absorption, exact reflective-caustics allocation, singular reflection, and
multi-closure delta MIS. Latest Cycles CPU differentials pass on fallback,
HIP, and Vulkan; the formal relations, direct Cycles CPU/HIP fast-math
diagnostic, reports, and inspected triptychs are in
[`validation/2026-08-03/principled-coat`](validation/2026-08-03/principled-coat/README.md).
The current-head Monster validation added the missing cross-socket topology:
when main Normal is linked but Coat Normal is not, Cycles independently
connects Coat Normal to Geometry Normal rather than inheriting the authored
main normal. The three-backend regression pins that relation, and the full
960x960 HIP differential reduces Normal relative RMSE from `2.787%` to
`0.0747%` and Glossy Direct from `54.4%` to `12.1%`. Five-way completion,
path-trace evidence, performance, and visually inspected before/after
triptychs are in
[`validation/2026-08-07/monster-current-head`](validation/2026-08-07/monster-current-head/README.md).
Physical thick Principled transmission now uses one coupled generalized-
Schlick Glass closure with Cycles' exact caustics gates, spectral tints,
front/backface eta relation, GGX energy handling, and visible-normal measure.
The shared Glass implementation now samples both GGX and Beckmann VNDFs and
uses their matching evaluation/PDF equations. Latest Cycles CPU EXR
differentials pass on fallback, HIP, and Vulkan; formulas, strict regressions,
the Luisa fallback Boolean-codegen correction, reports, EXRs, and all inspected
triptychs are in
[`validation/2026-08-03/principled-transmission`](validation/2026-08-03/principled-transmission/README.md).
Zero-thin-film Principled Thin Wall now retains the original linked Boolean,
normal, Glass, and Rough Translucent closures. A formal closure-family device
dispatch fixes a HIP non-termination caused by eagerly evaluating every
inactive BSDF family during sampled-light evaluation. The fixed 16-spp HIP
probe has Combined relative RMSE `2.37e-6` and mean-luminance ratio
`1.00000094` against Cycles HIP; profiler isolation, the three-backend
production-path regression, strict probe gates, and inspected triptychs are in
[`validation/2026-08-07/hip-closure-family-dispatch`](validation/2026-08-07/hip-closure-family-dispatch/README.md).
Thin film and the remaining thick/thin subsurface and anisotropy combinations
are still active compatibility work, so these checkpoints do not overstate
complete Principled support.

The first heterogeneous transport checkpoint now isolates the formal
null-collision transition in a Luisa `.h`/`.cpp` component. It pins current
Cycles main `b82c3f0` scalar-majorant exponential free flight, Hash Prospector
path-offset scrambling, null coefficients, throughput/albedo channel
probabilities, real/null continuation weights, deterministic absorption-only
continuation, and the defensive majorant-violation correction. The component
also exposes that violation as a predicate so an unsound preprocessing bound
cannot be mistaken for supported rendering. A 16-record fixture plus exact
32-bit hash checks passes on fallback, HIP, and Vulkan. Spatial majorant
construction and overlapping traversal were subsequently completed as
separate components. VSPG reservoir use, raw phase recovery, residual-ratio
equiangular transmittance, and distance/equiangular direct-light MIS now
compose in the production camera path. Heterogeneous scenes remain rejected
until the shadow path uses the same residual-ratio traversal rather than
homogeneous attenuation. Details of the collision checkpoint are in
[`validation/2026-07-31/heterogeneous-volume-collision`](validation/2026-07-31/heterogeneous-volume-collision/README.md).
The composed estimator and three-backend evidence are recorded in
[`validation/2026-08-01/heterogeneous-volume-vspg-direct`](validation/2026-08-01/heterogeneous-volume-vspg-direct/README.md).

Accumulated VSPG history is now connected end to end. The production path
classifies every Combined contribution into Cycles' raw scatter/transmit
passes, accumulates primary optical depth, runs the exact signed-RGBE
horizontal/vertical filter after cumulative power-of-two sample counts, and
feeds the resulting history into subsequent primary volume segments. The
majorant optical-depth mean remains a live raw statistic within a fused
multi-sample dispatch, while denoised radiance remains fixed until the next
scheduled filter. Chunked and single-call 4-sample renders both match the
official Cycles CPU EXR; maximum absolute error is `3.7253e-9` across
fallback, HIP, and Vulkan. Reports, EXRs, and visually inspected triptychs are
in [`validation/2026-07-31/volume-guiding-history`](validation/2026-07-31/volume-guiding-history/README.md).

Transparent-film termination now uses Cycles' film representation as well:
the render buffer accumulates transparency and converts it to clamped alpha
only after sample normalization. The finite absorption oracle produces zero
Combined RGB, alpha `0.361163139`, and retains the attenuated Environment
pass on all three Luisa backends.

The stacked-medium evaluator is the fifth internal checkpoint. It visits
runtime stack entries in Cycles order, evaluates each original graph with its
own parameter block and object-density scale, adds coefficients, and retains
raw phase closures in one device-local set. In particular it preserves equal
closures produced inside stack entry zero: only the second and subsequent
entries invoke exact phase merging, as in `volume_shader_eval()`. A 15-record
fixture uses two independently parameterized instances of the same raw graph
and pins single-medium, overlapping-medium, emission-suppressed, and empty
stack results to official Cycles main `b82c3f0`. It passes on fallback, HIP,
and Vulkan. Details are in
[`validation/2026-07-31/stacked-volume-evaluation`](validation/2026-07-31/stacked-volume-evaluation/README.md).

The production volume-point provider is the sixth internal checkpoint. It
reconstructs Cycles' world/object `ShaderData` state from each runtime stack
entry, including the inverse instance transform, Object/Particle Info,
normal transforms, `PRIM_NONE`, and zero primitive differentials. Scene
schema v2 now preserves Cycles' `ATTR_STD_GENERATED_TRANSFORM`; volume
Generated coordinates apply that affine transform to object-space `P` rather
than interpolating a boundary triangle. The exporter/importer consistency
check and a 26-record negative/non-uniform-scale fixture pass on fallback,
HIP, and Vulkan. The added records pin Cycles' zero-preserving
`safe_normalize()` semantics for the zero-direction bake ray, preventing an
object-space normal transform from introducing NaNs. Details are in
[`validation/2026-07-31/volume-shading-points`](validation/2026-07-31/volume-shading-points/README.md).

Mapping now implements Cycles' four device formulas directly in Luisa:
Point applies scale, Euler rotation, and translation; Texture applies inverse
translation, transposed Euler rotation, and safe inverse scale; Vector omits
translation; Normal applies safe inverse scale, rotation, and normalization.
The matrix probe covers all four modes with non-commuting rotations and
positive, negative, and zero scale components. Combined RMSE is `1.72e-8`
against official Cycles, with Normal and DiffCol exactly zero; `MAPPING` is
`cycles_verified`.

Blender's `ShaderNodeBump` is `cycles_verified`. Psycles reconstructs the same
compact Cycles ray differentials from the geometric normal, reevaluates the
height dependency subgraph at center/dx/dy, preserves the raw linked Normal
socket semantics, and follows Cycles' determinant, invert, nonnegative-strength,
zero-normal fallback, and final blend rules. The 16-cell matrix covers
Strength/Distance/Filter Width boundaries, mixed height gradients, Invert,
backfaces, linked non-unit/zero/tangent normals, and a rotated non-uniform
object transform. Against official Blender 4.5.10 Cycles, Combined and Emit
RMSE are `6.92e-7`, Normal RMSE is `5.36e-8`, and all other recorded passes
are exactly zero. A second 16-cell matrix connects a complete inner Bump to
the outer Bump's Normal input and varies both nodes independently; Combined
and Emit RMSE are `9.75e-7`, with Normal RMSE `5.36e-8`. Cycles' internal
object-space bump generated from a Displacement root remains part of the
separately release-gated Displacement domain; it is not claimed by the Blender
Bump-node result.

Normal Map is `cycles_verified` across Tangent, Object, World,
Blender Object, and Blender World spaces. Blender exports every evaluated UV
layer together with its own MikkTSpace tangent/sign data; the normalized graph
stores a named tangent attribute ID, and Luisa selects that geometry attribute
at the hit. A missing named layer follows Cycles' zero-tangent fallback to the
unperturbed shading normal. The primary 16-cell matrix covers all spaces,
strength boundaries, mirrored UVs, backfaces, and a rotated non-uniform object
transform (Combined/Emit RMSE `4.77e-8`). A second matrix selects the active
layer, two differently oriented named layers, and a missing layer
(Combined/Emit RMSE `3.61e-8`, Normal RMSE `5.50e-8`).

RGB Curves uses the same normalized pre-SVM contract as Cycles. The Blender
adapter finds the common domain of the four curves, evaluates 257 endpoints,
applies the Combined curve before the R/G/B curves, and exports the resulting
immutable table. Luisa performs Cycles' per-channel lookup, horizontal or
linear extrapolation, and unbounded Fac blend. The focused matrix covers
ordinary and expanded domains, out-of-range colors, both extension modes, and
Fac values below zero and above one. Combined RMSE is `7.63e-9`, and Normal
and DiffCol are exactly zero; `CURVE_RGB` is `cycles_verified`.

Noise Texture now lowers the Blender 4.5.10 Cycles hash, 1D–4D Perlin
gradients, coordinate precision correction, five fractal recurrences,
normalization, distortion, and color seeds directly into Luisa DSL. Five
matrix probes cover every cross-product of 1D–4D, FBM/Multifractal/Ridged/
Hybrid/Hetero Terrain, normalize off/on, and Fac/Color with nonzero
distortion and fractional detail. Against official Blender 4.5.10 Cycles,
their Combined RMSE ranges from `2.31e-8` to `1.25e-7`, with Normal and
DiffCol exactly zero; `TEX_NOISE` is therefore `cycles_verified`. Spatial
Generated-coordinate and Noise-to-Bump probes remain as integration coverage
for derivatives and film sampling. Each static dimension/fractal/output
combination is emitted once as a Luisa `Callable` and shared by every
`GraphSurface`, instead of duplicating the full Cycles noise implementation
inside every material dispatch case.

Magic Texture is `cycles_verified` without baking coordinates, colors, or
closures. The adapter retains Vector, Scale, Distortion, depth, and the
requested Color/Factor output. Depth is a finite host-stage specialization:
Luisa emits one typed callable for each used depth while all authored sockets
remain device expressions. The callable follows Cycles' exact recurrence and
Factor is the average of the resulting RGB color. A 16-cell matrix covers
every depth from 0 through 10, both outputs, implicit Generated coordinates,
signed and zero Scale/Distortion, and an extreme `1e20` coordinate. Against
Cycles CPU, Combined relative RMSE is `6.57e-7`, `8.22e-7`, and `8.20e-7` on
fallback, HIP, and Vulkan respectively, with zero invalid pixels. The extreme
case also exposed a general Luisa bug: expanding floating remainder as
`x - y * trunc(x / y)` loses the remainder for wide quotients. Luisa `next`
now preserves remainder as an IR primitive and implements exact binary32
semantics across the active code generators. Reports, regression details, and
the visually inspected triptychs are in
[`validation/2026-08-04/magic-texture`](validation/2026-08-04/magic-texture/README.md).

Wave Texture is `cycles_verified` without material baking. The adapter retains
the original typed Vector and six scalar inputs plus Wave type, Bands/Rings
direction, profile, and Color/Factor output identity. The OOP host-stage value
node expands the finite static configuration while tracing the Luisa AST, and
one shared callable evaluates Cycles' normalized signed 3D Perlin fBm rather
than duplicating it per material. Graph lowering emits only the requested
typed Color or Factor instruction, avoiding a duplicate fBm AST. The two
probes cover both types, all eight
type-specific directions, all three profiles, both outputs, implicit Generated
coordinates, signed/zero scales, linked dynamic sockets, distortion, Detail
`16`, and the formal negative-fractional Detail loop boundary. Against latest
Cycles CPU, modes relative RMSE is at most `2.50e-6` and distortion relative
RMSE at most `4.71e-5` across fallback, HIP, and Vulkan, with zero invalid
pixels. The larger high-frequency envelope is already present between official
Cycles CPU and HIP (`3.43e-5` relative RMSE), rather than evidence of a second
algorithm. The official Barbershop file contains 14 Wave nodes, including the
exact Detail-16 configuration in the probe. Reports and all visually inspected
triptychs are in
[`validation/2026-08-04/wave-texture`](validation/2026-08-04/wave-texture/README.md).

Voronoi Texture is `cycles_verified` without baking coordinates, colors, or
material results. The adapter retains Blender's raw 1D--4D node, all nine
dynamic sockets, feature, distance metric, Normalize flag, and requested
output. Static configuration is expanded while Luisa traces the shader AST;
the generated program calls a shared specialization instead of carrying a
weakly typed property switch through every shading point. The implementation
covers F1, F2, Smooth F1, Distance to Edge, N-Sphere Radius, Euclidean,
Manhattan, Chebychev, and Minkowski distance, every dimension-dependent
Distance/Color/Position/W/Radius output, fractional-detail fractal recurrence,
and Cycles normalization and socket clamps. Three probes exercise 64 cells in
total. Against Cycles CPU, Combined relative RMSE for distance, fractal, and
edge/radius is respectively `4.82e-8`, `1.18e-7`, and `3.58e-6` on fallback;
all three stay below `3.58e-6` on HIP and Vulkan, with zero invalid pixels.
The official Barbershop asset's 22 reachable Voronoi diagnostics are all
removed while preserving its original closure graphs; total scene diagnostics
fall from 52 to 30. Reports and all nine visually inspected backend triptychs
are in
[`validation/2026-08-04/voronoi-texture`](validation/2026-08-04/voronoi-texture/README.md).

Refraction BSDF is `cycles_verified` as a native Luisa microfacet closure,
not as a recolored Glass closure. The adapter retains the raw Color,
Roughness, IOR, Normal, and Beckmann/GGX distribution. Setup follows Cycles'
pure-transmission contract: Color is the ordinary closure allocation weight,
there is no Fresnel reflection branch, total internal reflection contributes
zero, backfaces invert eta, and neither GGX energy compensation nor a Glass
Fresnel tint is introduced. Linked roughness follows Cycles' square-then-
saturate rule, so negative inputs retain their magnitude rather than clamping
to a smooth lobe. Refraction keeps its exact Cycles closure IDs 20
and 21 and is both glossy and transmissive for lobe and sampled-light
filtering; unlike Glass, excluding either class removes it. A 16-cell raw-node
probe and three-backend analytic regression cover the Barbershop parameters,
both distributions, smooth/rough and unit-IOR cases, normals, backfaces, TIR,
allocation cutoff, direct evaluation/PDF, sampling, and split AOVs. Against
Cycles CPU, Combined relative RMSE is `5.12e-7` on fallback, `8.36e-8` on HIP,
and `1.43e-7` on Vulkan; `GlossCol` is exactly zero and `TransCol` is exact on
fallback. The same unchanged Barbershop export compiles both disinfectant
materials and has no Refraction diagnostic, reducing the scene total from 30
to 26. Reports and all 15 visually inspected triptychs are in
[`validation/2026-08-04/refraction-bsdf`](validation/2026-08-04/refraction-bsdf/README.md).

The legacy Attribute node is `device_partial`. Its GEOMETRY color-attribute
path now preserves raw Color, Vector, Fac, and Alpha output identity: present
RGBA attributes project to RGB, RGB, mean(RGB), and A, while a missing
attribute produces zero, zero, zero, and one exactly as Cycles does.
CORNER/BYTE_COLOR values additionally follow Cycles' sRGB decode and
OCIO-derived linear Rec.709-to-working-space conversion. A non-square
eight-cell raw-node probe matches Cycles CPU below `1.94e-7` relative RMSE on
fallback, HIP, and Vulkan, and removes all four Attribute diagnostics from the
unchanged Barbershop graph. Arbitrary float/vector attributes and the OBJECT,
INSTANCER, and VIEW_LAYER modes remain pending, so this is deliberately not
classified as `cycles_verified`. Reports and visually inspected triptychs are
in [`validation/2026-08-04/geometry-attribute`](validation/2026-08-04/geometry-attribute/README.md).
The Cycles 5.2 SVM migration additionally preserves the original
CORNER/BYTE_COLOR `uchar4` payload and Blender default-color identity in the
scene contract. The scene-linear float4 projection remains only for the
legacy expanded route; native SVM attribute fetch must consume the byte
payload and perform Cycles' device-side conversion.

Geometry Pointiness now follows Cycles' mesh-sync construction instead of
using a triangle-normal curvature approximation or a material-side bake. The
Blender adapter retains evaluated point normals and original edges only for
meshes whose raw node graphs link the Pointiness output. Psycles then applies
the same coordinate-sum duplicate quotient, normal welding, post-weld edge
deduplication, one-ring angle, and neighbor blur on the host before uploading
one point-domain standard attribute. Luisa interpolates that attribute at the
surface shading point; volume points still receive zero. A split height-field
probe covers convex and concave curvature, boundary behavior, coincident seam
vertices, and duplicate welded edges. Against latest Cycles CPU, Combined and
Emit relative RMSE are `5.65e-8` on fallback and `5.84e-8` on both HIP and
Vulkan, with maximum absolute error `1.79e-7` and zero invalid pixels. The
Geometry node remains `device_partial` because its separate Tangent and
Parametric outputs are not yet complete; this Pointiness output is nevertheless
strictly gated and documented in
[`validation/2026-08-04/geometry-pointiness`](validation/2026-08-04/geometry-pointiness/README.md).

Particle Info's non-particle sentinel contract is also explicit: its Random
output now follows Cycles rather than reusing Object Info random. The
`particle_random_nonparticle` probe matches Combined, Normal, and DiffCol
exactly at 64×64. Blender's evaluated persistent particle index is exported
per instance and carried through the Luisa hit state; Index and Random are
therefore distinct from Object Info and vary across an instanced particle
system. The current 27-instance particle probe measures Combined RMSE
`0.00402` and Normal RMSE `0.00898`, with the residual concentrated at
finite-sample sphere silhouettes. Age, Lifetime, Location, Size, Velocity,
and Angular Velocity remain explicit partial outputs.

White Noise Texture is `cycles_verified`. Its 1D–4D Value and Color paths use
Cycles' float-bit Jenkins hashes and channel permutations. A single
full-frame probe combines nontrivial constants from every dimension and both
outputs; Combined, Normal, and DiffCol all match Blender 4.5.10 exactly at
64×64/4 spp (RMSE and maximum absolute error 0).

Checker Texture is `cycles_verified`. Its Luisa lowering preserves Cycles'
float32 rounding boundaries around the coordinate-scale product and the
`(p + 1e-6) * 0.999999` precision correction, including exact integer,
negative, zero-scale, negative-scale, and large-coordinate cases. A 16-cell
constant matrix encodes both Color and Fac; Combined, Normal, and DiffCol all
match Blender 4.5.10 exactly at 64×64/4 spp (RMSE and maximum absolute error
0).

Gradient Texture is `cycles_verified` for Linear, Quadratic, Easing, Diagonal,
Radial, Spherical, and Quadratic Sphere. Its matrix covers negative and
over-one saturation, easing endpoints, radial quadrants, and Cycles'
`0.999999` spherical precision bias. Against official Cycles, Combined and
Emit RMSE are `3.73e-9` with maximum error `1.49e-8`; Normal, DiffCol, and all
unrelated light passes are exact.

Fresnel is `cycles_verified`. A 16-cell matrix covers the IOR lower clamp,
IOR below and above one, negative inputs, explicit surface normals from normal
incidence to grazing, and actual backfacing triangles with reciprocal eta.
Combined RMSE is `2.61e-8` with maximum absolute error `1.04e-7`; Normal and
DiffCol are exactly zero.

Legacy Separate/Combine RGB, HSV, and XYZ nodes are `cycles_verified`.
Their old socket identifiers and Color/Vector conversions are mapped
explicitly rather than treated as aliases by name. A 16-cell matrix makes
every component input and output reachable, including signed and
out-of-range values. Combined RMSE is `9.13e-9` with a one-ULP maximum error;
Normal and DiffCol are exactly zero.

Map Range is `cycles_verified` for both FLOAT and FLOAT_VECTOR. Its Luisa
lowering follows Cycles' distinct scalar and component-wise vector contracts,
including the scalar graph-expansion clamp, vector safe division, all four
interpolation modes, positive/zero/negative Steps, reversed From/To ranges,
zero-length From intervals, and Clamp. A 16-cell constant matrix matches
official Blender 4.5.10 Cycles exactly for Combined, Emit, Normal, DiffCol,
and every requested direct/indirect light pass at 64×64/4 spp (RMSE and
maximum absolute error 0).

Vector Math is `cycles_verified`. One 40-cell constant matrix covers all 29
Blender 4.5.10 operations plus guarded zero divisors, zero projection and
normal vectors, total internal reflection, both Faceforward branches, invalid
negative powers, zero Wrap ranges, zero Snap increments, and parallel cross
products. The official Cycles and Psycles-Luisa/fallback outputs differ by
only float roundoff: Combined and Emit RMSE are `8.41e-9`, with a maximum
absolute error of `5.96e-8`; Normal, DiffCol, and every requested light pass
are exact.

Add Shader and Mix Shader are `cycles_verified` without flattening their
closure trees. Psycles has an explicit null-closure IR/DSL operation, so
unconnected A/B sockets do not masquerade as zero-color diffuse lobes.
Four Add cells cover both, either, and neither input; eight Mix cells cover
linked factors below zero, within range, and above one plus every empty-input
combination. Combined/Emit RMSE are `3.55e-8` and `3.04e-9` respectively;
Normal, DiffCol, and all light passes are exact.

Blackbody and Wavelength are `cycles_verified`. Their polynomial intervals,
CIE table interpolation, out-of-range clamps, and working-space conversion run
inside Luisa DSL. The Blender exporter reproduces the same OCIO-derived
`rec709_to_rgb` and `xyz_to_rgb` film matrices that Cycles uploads; Psycles
does not assume the active scene-linear space is exactly Rec.709. Matrices
cover every Blackbody interval boundary and Wavelength table/range branch.
Against official Cycles, Combined/Emit RMSE are `9.84e-8` and `1.95e-8`
respectively, while Normal, DiffCol, and all unrelated light passes are exact.

Invert Color is `cycles_verified`. Its linked factor is deliberately
unbounded; treating the UI Factor subtype as an implicit `[0, 1]` device clamp
is incorrect. A 16-cell signed/HDR matrix covers factors from `-3` through `4`
and applies a verified positive bias only to make negative intermediate values
observable through Emission. Combined, Emit, Normal, DiffCol, and all unrelated
light passes match official Cycles exactly.

Layer Weight is `cycles_verified` for both Fresnel and Facing outputs, including
every Blend branch and backfacing behavior. The graph IR preserves whether the
Normal socket is linked: an unlinked socket uses the shading normal, whereas a
linked socket consumes its raw vector without normalization or a zero-vector
fallback. The matrix therefore includes non-unit, signed, and zero linked
normals as well as the default normal path. Official Cycles and
Psycles-Luisa/fallback match exactly at both 64×64/4 spp and 64×64/256 spp for
Combined, Emit, Normal, DiffCol, and all unrelated light passes.

Diffuse BSDF and Translucent BSDF are `cycles_verified`. Their Luisa closure
allocation applies Cycles' component-wise negative-weight clamp and average
weight cutoff; Diffuse preserves the exact Lambert-versus-Oren–Nayar roughness
branch, and Translucent applies Cycles' valid-reflection-normal correction
before constructing the opposite-hemisphere lobe. Two 16-cell matrices cover
zero, signed, HDR, cutoff-boundary, and linked normal inputs, including
non-unit, zero, tangent, and backfacing cases. Official Cycles and
Psycles-Luisa/fallback match pixel-for-pixel at 64×64/4 spp: Combined,
Diffuse Color, Normal, every direct/indirect light pass, emission, and
environment passes all have zero RMSE and zero maximum absolute error.

Subsurface Random Walk is now `cycles_verified` for Standard, Legacy, and Skin
transport. The typed Luisa component implements current coefficient remapping,
Skin's mixed entry, exact Tabulated Sobol domains, 256-event local transport,
similarity reduction, Henyey-Greenstein/Dwivedi proposal mixtures, and exact
object/primitive ray-query filtering. The first query uses `tmin = 0` and
rejects only the entry primitive identity; no epsilon self-intersection filter
or host prebake is used. At 64x64/4,096 spp, Combined energy is within
`1.24e-6` of Cycles and Combined relative RMSE is at most `1.27e-4` across
fallback, HIP, and Vulkan. Direct diffuse remains below `1.27e-4`; the more
sensitive indirect diffuse pass remains below `2.35e-3`. Reports, focused path
traces, performance ratios, and twelve visually inspected triptychs are in
[`validation/2026-08-04/subsurface-random-walk`](validation/2026-08-04/subsurface-random-walk/README.md).

Math and Mix are now `cycles_verified`. Two Math probes cover all 41 Blender
4.5.10 operations plus signed, zero-divisor, invalid-domain, epsilon-compare,
three-input, and output-clamp branches; all recorded passes have zero RMSE.
Modern Mix probes cover FLOAT, uniform and non-uniform VECTOR, RGBA, factor
and result clamping, every color blend mode, and guarded divide/dodge/burn/HSV
branches. Color modes and edge cases are exact; the data-type probe differs
by at most one float ULP (`5.96e-8`, RMSE `3.44e-8`). The legacy MixRGB node
also covers all 19 blend modes with exact passes, including Cycles' deliberate
choice to ignore its `use_alpha` property. The RNA-only ROTATION enum is not a
constructible Cycles mode in Blender 4.5.10; Blender itself restricts the
runtime property to FLOAT, VECTOR, and RGBA.

Environment Texture is `cycles_verified` without baking the world or image
lookup. The Blender adapter retains the original node, image binding, vector
input, interpolation, projection, and color-space metadata; an unlinked Vector
uses the world ray direction through the same `LINK_POSITION` convention as
Cycles. Luisa evaluates Cycles' equirectangular and mirror-ball projections,
repeat addressing, Closest/Linear/Cubic/Smart interpolation, and post-filter
sRGB decode directly in the generated shader. A shared spherical-geometry
primitive makes the otherwise undefined azimuth at both poles explicitly zero,
matching Cycles CPU/HIP on Vulkan as well. Two focused material probes cover zero,
axis-aligned, oblique, and pole directions plus every exposed interpolation;
a third perspective-world probe covers the implicit, unlinked Vector path.
At 64×64/4 spp against Blender 5.3 Alpha/Cycles `b82c3f0da6c1`, Combined RMSE
is `4.77e-8` on fallback, `4.67e-8` on HIP, and `9.41e-8` on Vulkan for the
projection matrix; the sampling matrix remains below `3.95e-8` on all three.
The perspective-world probe remains below `4.12e-7` on all three, including
the complete camera-to-world direction convention and background Env pass.
Reports and visually inspected triptychs are recorded in
[`validation/2026-08-04/environment-texture`](validation/2026-08-04/environment-texture/README.md).

## Sky Texture contract

Sky Texture is a versioned partial node rather than one interchangeable sky
formula. Blender 5.2 `SINGLE_SCATTERING` and legacy `NISHITA` use the existing
Cycles-compatible spectral LUT path. Legacy `HOSEK_WILKIE` now preserves its
explicit sun direction, turbidity, and ground albedo, cooks the upstream XYZ
model coefficients as immutable topology data, and evaluates the analytic
directional formula in the Luisa shader. The focused fallback and Metal probe
measures Combined energy at `1.000284x` and `1.000285x` Cycles Metal.

Guided single-scattering Nishita sampling follows the full current-Cycles
sample relation, not merely an equivalent distribution. The `1:4` map/Sun
mixture remaps the same Tabulated Sobol dimensions; the Sun branch uses the
shared concentric square-to-disk map, Cycles' canonical algebraic frame, its
cone-radius remapping, and its piecewise small-angle `1-cos` measure. Forward
PDF membership uses the stable two-vector angle with the same strict boundary.
On Lone Monk's real sample-zero path this reduces the Sun-direction error from
`0.776416` degrees to one float32 ULP and the sampled sky-radiance error from
23.21% to 0.0283%. The fallback/HIP/Vulkan oracle and 960x720x512 image
evidence are recorded in
[`validation/2026-08-07/lone-monk-background-sun-sampling`](validation/2026-08-07/lone-monk-background-sun-sampling/README.md).

`PREETHAM` and `MULTIPLE_SCATTERING` remain distinct unsupported/partial
modes. They must not be silently reported as Hosek or single-scattering
Nishita. The complete Apple scene evidence and reports are in
[`validation/2026-08-02/apple-classroom-lone-monk`](validation/2026-08-02/apple-classroom-lone-monk/README.md).

## Analytic-light contract

Point, Spot, Area, and Sun lights are represented as scene data and sampled
inside the Luisa path-tracing kernel. Their power normalization, temperature
and exposure scaling, finite-radius shape, spot attenuation, rectangular or
elliptical area, area spread, and finite Sun disk follow the Blender 4.5.10
Cycles light sampling equations. The `fallback` runs this same generated
Luisa program; there is no separate host light evaluator.

An analytic light's Blender node tree is lowered to the same parameterized
`GraphSurface` program used by geometry and world shaders. The direct-light
kernel invokes its Emission root through `Polymorphic<Surface>` at the sampled
light point. The focused `point_light_nodes` probe measures Combined relative
RMSE `0.000710` against official Cycles; its Normal and DiffCol passes match
exactly.

`docs/cycles-light-probe-baselines-4.5.10.json` records eleven official
Cycles/Psycles differential probes covering all four light families, point
disk/sphere shapes, smooth Spot, rectangle/ellipse/spread Area, finite-angle
Sun, and a Light Output node tree. High-sample ellipse and spread probes show
energy-relative errors below `3e-5`; their remaining pixel error decreases
with sample count and is treated as estimator variance, not a compatibility
claim for Cycles' random sequence.

## Integrator contract

The Blender scene package carries the sampling seed into `RenderSettings` and
these Cycles settings into `RenderSettings::integrator`:

- total, minimum, diffuse, glossy, transmission, volume, and transparent
  bounce limits;
- transparent minimum bounces;
- direct and indirect sample clamp;
- reflective and refractive caustics controls;
- direct-light sampling mode;
- film exposure, light-tree enablement, and light-sampling threshold.

The Luisa path state has separate regular and transparent depth. At a regular
bounce limit it follows Cycles' terminate-after-transparent behavior: emitter
evaluation remains active and mixed closure selection is filtered and
renormalized to transparent closures only. Transparent bounces preserve the
previous non-transparent ray flags and forward-MIS PDF. Russian roulette is
evaluated after the next surface hit and after surface emission, matching
Cycles' continuation placement. Contribution clamping uses the Cycles
sum-of-absolute-RGB rule per contribution, including the scene-sync conversion
from Blender's per-channel UI value to a device sum limit (`clamp * 3`).
Forward surface, lamp, background, and volume emission use Cycles'
`path.bounce - 1` direct/indirect classification rather than treating the
first scattered emitter as indirect.

Direct-light mode weights distinguish forward-only, NEE-only, and power-
heuristic MIS. Finite point, spot, and rectangle/ellipse area lights have
matching production NEE and forward-intersection techniques. Surface and
volume area-light NEE now use the same spread-clamped Cycles component; the
32×32 narrow-ellipse CPU oracle matches all three Luisa backends with relative
RMSE at or below `1.60e-6` against Blender/Cycles main `b82c3f0d`. The reports
and inspected triptychs are in
[`validation/2026-07-31/surface-area-light-mis`](validation/2026-07-31/surface-area-light-mis/README.md).
Surface and volume mesh-light NEE plus emissive forward-hit MIS now use the
same triangle geometry, position-dependent sampling measure, side selection,
and raw closure-evaluation component. The volume fixture above additionally
pins its segment-only area proposal and one-sided plane interval.
Surface NEE now also carries each sampled emitter's raw Cycles shader flags
into a dedicated light-evaluation query. Exclude flags project diffuse,
glossy, and transmission contributions without removing any otherwise
eligible closure from the Veach one-sample-model PDF; glass is removed only
by the combined glossy/transmission exclusion, translucent follows the
diffuse category, and missing `USE_MIS` zeroes only the competing PDF. The
formal fallback/HIP/Vulkan regressions and the five-way Lone Monk render are
recorded in
[`validation/2026-08-03/sampled-light-closure-filter`](validation/2026-08-03/sampled-light-closure-filter/README.md).
When the light tree is disabled, the Luisa NEE path applies Cycles'
`film_exposure / light_sampling_threshold` roulette to the unshadowed light
sample and compensates surviving samples by the reciprocal probability.
When it is enabled, emissive triangles, analytic lights, and the environment
share one stable emitter identity across surface/volume selection and
forward-hit MIS. Separate host-specialized Luisa callables implement Cycles'
maximum/minimum-importance mixture, leaf reservoir, random remapping, and
reverse leaf-to-root PDF. Tree construction uses final displaced support and
effective material overrides; emission metadata controls proposal importance
only, while the original closure graph remains the device radiance evaluator.
The current hierarchy is flattened and reciprocal but does not yet reproduce
Cycles' mesh/instance subtrees, light-link roots, or every specialized
per-light importance parameter. The formal boundary and fallback/HIP/Vulkan
regressions are recorded in
[`validation/2026-08-07/light-tree`](validation/2026-08-07/light-tree/README.md).
The versioned `integrator_clamp_direct` probe uses a white emission of 10 and
Blender direct clamp 2; both Cycles and Luisa/fallback produce linear RGB
`(2, 2, 2)`, with zero RMSE and maximum absolute error in all recorded passes.

The heterogeneous-volume foundation now includes Cycles' local real/null
collision measure, exact path-offset hash, raw volume-graph extrema prepass,
depth-seven 128-cubed majorant hierarchy reduction, and single-root bitwise
hierarchical DDA. Scene resource construction now applies effective instance
material overrides, creates one root per volume object/shader pair plus the
final World range, and uploads formally validated flattened root/node/range
buffers. Matching `Octree::evaluate_volume_density`, homogeneous roots
evaluate the original `GraphSurface` at `1^3 x 16` points and spatially
varying roots at `128^3 x 16`; neither class is omitted from later overlap
lookup. The prepass uses Cycles' camera/zero-direction/time-`0.5` bake state.
Its 3D sampler is bit-pinned to official Cycles, while raw-graph extrema for
three distant cells and the complete scene-resource composition pass on
fallback, HIP, and Vulkan. The hierarchy matches Cycles main `b82c3f0d`:
eight siblings are contiguous, subdivision uses
`range * diagonal * volume_scale > 1.442`, object bounds map to `[1, 2)`, and
traversal retains the root-extrema tail outside an implicit volume bound.
Partially collapsed bounds and zero instance scale follow Cycles' root-only
semantics; only a fully collapsed bound is discarded. Flattening rejects
gapped/overlapping ranges, wrong root identities, incomplete child blocks,
mismatched parents, shared/cyclic children, unreachable nodes, and invalid
extrema before any device buffer becomes visible.
Focused device regressions cover forward/reverse adjacency, parent ascent,
root exit, and the outside-root path. Cycles' hierarchy extrema come from
finite padded Sobol samples; they are not claimed as a mathematically
guaranteed bound. The multi-root reducer now preserves one active traversal,
reconstructs all other roots at each common minimum, sums extrema in stack
order, uses the exact last-equal-endpoint tie rule, and takes Cycles'
one-medium shortcut. Reverse range lookup masks high shader flags and fails
closed on missing or malformed coverage without evaluating the entry
provider. Runtime majorant violations remain explicit. Prepass details are in
[`validation/2026-07-31/volume-majorant-prepass`](validation/2026-07-31/volume-majorant-prepass/README.md);
scene-resource details are in
[`validation/2026-07-31/volume-majorant-scene-resources`](validation/2026-07-31/volume-majorant-scene-resources/README.md).
The complete stack-root-domain correction is recorded in
[`validation/2026-07-31/volume-majorant-root-domain`](validation/2026-07-31/volume-majorant-root-domain/README.md).
The ordered overlap reduction is recorded in
[`validation/2026-07-31/volume-majorant-overlap`](validation/2026-07-31/volume-majorant-overlap/README.md).
The production provider now uploads structural heterogeneous and Light Path
capability flags, applies exact object/World coordinate semantics, and follows
Cycles' one/four-sample runtime extrema policy. The Light Path bit deliberately
matches Cycles' finalized whole-shader scan, including surface-only uses;
Volume homogeneity remains a separate complete dependency reduction.
The exact device regression is recorded in
[`validation/2026-07-31/volume-majorant-runtime-provider`](validation/2026-07-31/volume-majorant-runtime-provider/README.md).
The runtime shade offset is an explicit traversal input rather than provider
construction state. The initial octree setup observes the enclosing path RNG
before tracking scramble. Every ordered root reconstruction within a later
boundary-advance operation observes the current copied tracking RNG state;
after a candidate, that local state advances by the 16-dimension bounce block.
The state contract and three-backend regression are recorded in
[`validation/2026-08-01/volume-majorant-tracking-rng`](validation/2026-08-01/volume-majorant-tracking-rng/README.md).
The candidate walker now composes that traversal with Cycles' exponential
free flight, zero-majorant skip, residual-uniform reuse, strict endpoint rule,
unscaled optical-depth statistic, and exact 1025-attempt step domain. Details
are in
[`validation/2026-08-01/heterogeneous-volume-candidate-walk`](validation/2026-08-01/heterogeneous-volume-candidate-walk/README.md).
Production `compile_scene` now derives the same root identities directly from
retained runtime material bindings, executes the raw-closure prepass after
TLAS/resource completion, and owns the flattened buffers in `LuisaSceneData`.
The three-backend resource regression, official Cycles pixel differential,
inspected triptychs, and the accompanying Luisa Vulkan optimized-XIR Accel
metadata fix are recorded in
[`validation/2026-08-01/volume-majorant-production-scene`](validation/2026-08-01/volume-majorant-production-scene/README.md).

An ordinary Blender Emission node reached from Material Output Volume now
lowers to a typed `volume_emission` operation instead of an unsupported
surface-to-volume closure conversion. Its live Color and Strength expressions,
Add/Mix weight, signed coefficient, object-volume-density scale, and
shadow/extinction query suppression follow current Cycles SVM semantics. The
exact fallback/HIP/Vulkan coefficient regressions and Cycles CPU/HIP EXR
differentials are recorded in
[`validation/2026-08-04/volume-emission`](validation/2026-08-04/volume-emission/README.md).

The official Barbershop Interior graph audit confirms that its only connected
Volume root is the `fog` material's Emission closure. It contains no reachable
volume scattering or absorption closure, and Cycles main produces identically
zero Volume Direct and Volume Indirect passes. Scene traversal models Cycles'
closed ray interval and stable primitive identity relation over candidates
enumerated by the selected acceleration backend. An earlier experiment also
synthesized whole-scene coincident/source candidates, but a Blender 5.2 audit
proved that experiment had been fitted to a mismatched Blender 5.3 Alpha
export and removed it. Endpoint, exclusion, and curve-segment regressions
remain on fallback, HIP, and Vulkan. The original differential is recorded in
[`validation/2026-08-04/barbershop-stable-traversal`](validation/2026-08-04/barbershop-stable-traversal/README.md).

Implicit conversions among the supported color/vector/point/normal/float3
family now follow Cycles' component-preserving law through a canonical typed
vector route. The exact Barbershop `Geometry.Position -> Mix.B_Color` links
remain live instead of falling back to zero. Importer regressions, all three
Luisa backends, Cycles CPU/HIP EXR differentials, and inspected triptychs are
recorded in
[`validation/2026-08-04/geometry-float3-conversion`](validation/2026-08-04/geometry-float3-conversion/README.md).

Blender's unnamed/default UV now follows the render-active map returned by
`Mesh::default_uv_map_name()`, independently of the layer selected for UI
editing; its default tangent frame follows the same map while every named UV
attribute remains addressable. Image Texture `BOX` projection now also uses
the post-shader-setup normal sign on back-facing hits, matching Cycles'
cube-face selection and orientation relation. Exporter byte-range checks, an
independently rendered back-facing projection matrix on fallback/HIP/Vulkan,
official Barbershop staged node differentials, and inspected triptychs are
recorded in
[`validation/2026-08-05/barbershop-render-uv-box`](validation/2026-08-05/barbershop-render-uv-box/README.md).

Cycles' per-object geometry representation remains explicit: each accelerator
candidate uses either static-transformed vertices or its object-space ray
relation according to geometry user count, motion, adaptive subdivision,
BSSRDF, and true-displacement gates. An FMA-ordered Pluecker component then
applies visibility, exclusion, closed-interval, and stable identity selection.
It does not fabricate another instance merely because finite support overlaps.
The representation study is recorded in
[`validation/2026-08-05/barbershop-cycles-geometry-representation`](validation/2026-08-05/barbershop-cycles-geometry-representation/README.md),
while the corrected same-build scene oracle and traversal rule are in
[`validation/2026-08-14/barbershop-blender-5.2-revalidation`](validation/2026-08-14/barbershop-blender-5.2-revalidation/README.md).

Adaptive sampling and denoising are exported and diagnosed but are not part of
the path-integrator estimator. Psycles renders fixed-count, un-denoised linear
passes; authoritative Cycles differential renders disable both. The Blender
light-tree setting is preserved and admitted by the Luisa session; unsupported
tree refinements remain explicit below rather than silently changing the
authored closure graph. A connected Volume root is preserved through Blender
import and material compilation. Its raw
dependency tree is accepted only when the homogeneous segment integrator can
represent it; a spatial dependency remains an explicit scene capability
error.

The graph-driven coroutine path now uses the main Cycles path-stage cuts,
per-continuation exact frame-field I/O, and bounded selective-queue service.
The 640x480, 64-spp HIP matrix across Lone Monk, Classroom, Monster Under the
Bed, and Barbershop separates scheduler overhead from estimator differences
with an identical per-(pixel,sample) megakernel baseline. Timings, formal
service bound, frame sizes, linear EXR metrics, and four inspected triptychs
are recorded in
[`validation/2026-08-13/cycles-wavefront-multiscene`](validation/2026-08-13/cycles-wavefront-multiscene/README.md).

The newer
[`five-scene HIP performance checkpoint`](validation/2026-08-14/multiscene-hip-performance/README.md)
repeats the comparison on Blender/Cycles 5.2 with profiler-matched GPU
boundaries. Psycles currently takes 1.74x--3.40x the Cycles path-kernel time
across Lone Monk, Classroom, Monster, Barbershop, and Flat Archiviz. Closest
intersection is at parity per dispatched item on the first three scenes;
surface shading is the common 2.4x--3.9x per-item hotspot. The same-topology
megakernel control shows coroutine staging already helps Monster and
Barbershop but regresses Lone Monk and Flat Archiviz. Five inspected Combined
triptychs, shader resource records, code-object sizes, and explicit asset/
unsupported-feature caveats are retained with the measurements.

HIP effectful shadow-all traversal now uses a formally proven native any-hit
route when query commits and post-state observation are absent. A subsequent
representation-independent private-memory proof removes unread stored
allocation identities before SROA. On the retained Barbershop profile this
reduces the shadow kernel private segment from 128 B to 8 B and improves it by
1.08%, while leaving a measured 39.8% per-launched-lane gap to Cycles 5.2
HIPRT. The formal relations, regressions, profiler records, and inspected
triptychs are in
[`validation/2026-08-16/hip-ray-query-effect-only`](validation/2026-08-16/hip-ray-query-effect-only/README.md)
and
[`validation/2026-08-16/hip-private-memory`](validation/2026-08-16/hip-private-memory/README.md).

Traversal candidate callbacks now consume a formally defined immutable scene
quotient: one 32-byte instance record and a dense material-capability table
replace full instance/geometry/material-binding loads without baking closure
values. This improves Barbershop shadow trace by another 1.11% and Classroom
triangle shadow by 1.66%. Classroom is now 34.6% faster than the retained
Cycles 5.2 HIPRT per-launched-lane trace metric, proving that the remaining
Barbershop gap is curve/callback semantic work rather than generic triangle
RayQuery traversal. Formal equivalence, rejected counterfactuals, three-backend
regressions, profiler records, all-pass differentials, and inspected
triptychs are in
[`validation/2026-08-16/scene-traversal-quotient`](validation/2026-08-16/scene-traversal-quotient/README.md).

Authored Bump height graphs are now expanded into the same topological typed
SVM stream as their consuming surface graph. Explicit differential-context SSA
values replace the recursive height interpreter while exact GVN shares cloned
invariants; no closure, texture value, or material parameter is baked. On the
retained Barbershop HIP A/B this removes 104 height programs, reduces the main
LLVM and code object by 10.40% and 11.32%, and improves the paired three-run
render median by 5.47%. The formal model, nested-Bump and invalid-graph
regressions, fallback/HIP/native-XIR Vulkan gates, full-pass repeat-floor
analysis, and inspected triptychs are in
[`validation/2026-08-26/surface-bump-expanded-svm`](validation/2026-08-26/surface-bump-expanded-svm/README.md).

Blender 5.2's standalone Sheen BSDF now remains a raw closure with static
Microfiber and Ashikhmin variants. The implementation preserves LTC setup and
sampling, analytic Velvet evaluation, the two-block physical tagged union,
and Cycles' unusual split where Ashikhmin samples with a diffuse event but is
glossy for pass routing, light exclusion, and bump-normal classification. A
16-cell 640x480, 256-spp matrix passes on HIP, fallback, and strict native-XIR
Vulkan. The probe exposed and permanently regresses a prior 0.34% aggregate
Ashikhmin energy loss caused by diffuse terminator softening. Formal relations,
reports, timings, and inspected triptychs are in
[`validation/2026-08-28/cycles-5.2-standalone-sheen`](validation/2026-08-28/cycles-5.2-standalone-sheen/README.md).

Curve/ribbon primitives now retain their primitive-kind predicate through the
post-population closure projection. Sampling uses the selected closure normal
as geometric support, curve bump shadowing is the identity, and specular setup
preserves the authored closure normal, exactly at the three boundaries where
Cycles 5.2 defines those laws. The predicate occupies an existing flags word:
the closure-point callable ABI remains 48 bytes and neither closure payload nor
coroutine frame grows. A separating Diffuse/GGX counterexample passes on
fallback, HIP, and strict native-XIR Vulkan; the broader curve and hair matrix
passes `14/14`. The formal relations, pre-fix failure, and full-suite result are
in
[`validation/2026-08-28/curve-closure-semantics`](validation/2026-08-28/curve-closure-semantics/README.md).

The following integrator work remains explicit and is not considered Cycles
compatible yet:

- Cycles' emitter importance distribution and environment importance map;
- Cycles-exact mesh/instance light-tree topology, specialized emitter
  importance, and receiver light-link roots;
- heterogeneous residual-ratio shadow transport and scene-gate removal;
- remaining distant-light forward/background behavior;
- MNEE, path guiding, shadow catcher, light linking, and light groups;
- Cycles' exact sampling sequence and random dimensions.

These gaps may change finite-sample variance even when an estimator has the
same expectation. Full-scene images are therefore diagnostic; node and
integrator probes are the acceptance tests.

## Differential policy

For every golden scene:

- official Blender Cycles writes an untonemapped 32-bit multilayer EXR;
- Psycles writes linear Combined and requested data passes;
- image origins and pass channels are canonicalized without an orientation
  search;
- the report includes RMSE, relative RMSE, P95/P99 pixel error, maximum error,
  invalid-pixel count, and energy scale;
- unsupported coverage keeps the release gate red even if the image looks
  acceptable.

Lone Monk is the first complex integration scene. Small single-node and
single-integrator-feature scenes are the source of compatibility claims.
