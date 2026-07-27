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
2. The Blender adapter walks the active Surface root recursively. It inserts
   explicit Cycles socket conversions and emits a single topologically ordered
   typed-value instruction stream. Volume and Displacement roots remain
   release-gated work.
3. Closure-producing nodes remain an Add/Mix tree. They are not flattened into
   a fixed-size closure array.
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

Noise Texture now lowers the Blender 4.5.10 Cycles hash, 1D–4D Perlin
gradients, coordinate precision correction, five fractal recurrences,
normalization, distortion, and color seeds directly into Luisa DSL. Focused
256 spp probes currently measure Combined RMSE `0.000924` for 3D Color,
`0.00357` for 2D Fac, and `0.00179` for Noise-to-Bump; the latter's Normal
RMSE is `0.000401`. Unaffected data passes are exactly zero and all probe
pixels are finite. The node remains `device_implemented_unverified` until
focused probes cover every dimension, fractal mode, normalization state, and
socket combination; finite-sample Combined differences also include the
known random-dimension sequence gap. Each static dimension/fractal/output
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
Volume or Displacement root and an enabled Cycles light tree are hard import
errors rather than silently ignored settings.

The following integrator work remains explicit and is not considered Cycles
compatible yet:

- Cycles' emitter importance distribution and environment importance map;
- light-tree construction and traversal;
- volume transport and volume stack state;
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
