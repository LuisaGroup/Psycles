# Cycles compatibility status

Psycles treats Blender Cycles as the semantic reference. A feature is complete
only after the same Blender scene and render settings have been evaluated by
official Cycles and Psycles-Luisa and their linear passes have passed a
feature-specific differential threshold.

## Shader graph path

Psycles does not consume SVM bytecode and does not bake Blender materials.
Shader data follows this path:

1. Blender exports the original node trees, links, socket defaults, static
   node properties, image identities, and evaluated scene geometry.
2. The Blender adapter walks the active Surface and Volume roots recursively.
   It inserts explicit Cycles socket conversions and emits a single
   topologically ordered typed-value instruction stream. Displacement remains
   release-gated work.
3. Surface- and volume-closure-producing nodes remain typed Add/Mix trees.
   Neither domain is flattened into a fixed-size closure array.
4. Shader node groups are recursively expanded through their exported Group
   Input/Output interfaces. Group instance names have no semantic role;
   missing and recursive groups produce explicit diagnostics.
5. Image pixels, mesh attributes, parameter blocks, and geometry are uploaded
   to Luisa resources. No host-side shader evaluator is used.
6. `GraphSurface` traces the typed program while constructing Luisa DSL. Bump
   evaluates its height dependency subgraph at the center and two ray
   differential offsets.
7. `Polymorphic<Surface>` performs device-side material dispatch. The same
   generated program runs on Luisa `fallback` and GPU backends.

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
diagnostics and keep the release gate red. Missing material slots use an
explicit magenta coverage material; an unconnected Cycles Surface follows the
separately probed opaque-black surface contract. Unsupported node outputs use
their exported socket default only together with a named warning, never as an
unreported compatibility claim.

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
sample-weighted evaluation, and reservoir selection/random-number rescaling.
The combined 37-record coefficient/phase-set fixture includes the original
Draine, fitted-Mie HG, fitted-Mie Draine, and Principled HG closures plus
merge, capacity, truncation, empty-set, and combined single-trace regressions.
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

Stacked-medium evaluation, free-flight integration, heterogeneous grids, and
volume direct lighting remain open, so scene compilation still release-gates
volume materials and the four Blender volume nodes remain unverified in the
public node matrix.

World volume identity is now complete in scene schema v2: the exporter retains
Cycles' background-light object index separately from the default-background
shader index, and both round-trip through the contract adapter. This is the
same `(object, shader)` pair that Cycles writes before its camera enclosure
probe; the stack does not infer it from object counts or sentinels.

The analytic homogeneous segment estimator is the fourth internal checkpoint.
Its Luisa `.h`/`.cpp` component matches Cycles' per-channel transmittance,
second-order small-optical-depth emission integral, throughput/albedo-weighted
RGB channel selection, bounded exponential density, random-number rescaling,
and the scatter/transmit estimator weights. It also accepts Cycles' externally
guided VSPG scatter probability without changing that measure. A 31-record
fixture generated from official Cycles main `b82c3f0` `volume.h`, `mapping.h`,
and the current homogeneous control flow passes on fallback, HIP, and Vulkan.
The Sobol contract now pins the official volume phase, reservoir, scatter
distance, expansion, shade-offset, and phase-guiding dimensions. This
checkpoint still does not enable volume materials in scene compilation:
stacked graph evaluation, main-path camera/boundary hookup, VSPG history,
volume NEE, and phase-bounce state must be integrated first.

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

Direct-light mode weights distinguish forward-only, NEE-only, and power-
heuristic MIS. Analytic lights currently have no forward intersection
technique, so their NEE estimator intentionally carries full weight.
When the light tree is disabled, the Luisa NEE path applies Cycles'
`film_exposure / light_sampling_threshold` roulette to the unshadowed light
sample and compensates surviving samples by the reciprocal probability.
The versioned `integrator_clamp_direct` probe uses a white emission of 10 and
Blender direct clamp 2; both Cycles and Luisa/fallback produce linear RGB
`(2, 2, 2)`, with zero RMSE and maximum absolute error in all recorded passes.

Adaptive sampling and denoising are exported and diagnosed but are not part of
the path-integrator estimator. Psycles renders fixed-count, un-denoised linear
passes; authoritative Cycles differential renders disable both. A connected
Displacement root and an enabled Cycles light tree remain hard scene
capability errors rather than silently ignored settings. A connected Volume
root is preserved through Blender import and material compilation, then
explicitly rejected by scene compilation until the Luisa volume-stack
integrator is enabled.

The following integrator work remains explicit and is not considered Cycles
compatible yet:

- Cycles' emitter importance distribution and environment importance map;
- light-tree construction and traversal;
- volume transport and path-kernel integration of volume stack state;
- forward intersections for analytic area and distant lights;
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
