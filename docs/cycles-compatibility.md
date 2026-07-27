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
2. The Blender adapter walks each active Surface, Volume, and Displacement
   root recursively. It inserts explicit Cycles socket conversions and emits a
   single topologically ordered typed-value instruction stream.
3. Closure-producing nodes remain an Add/Mix tree. They are not flattened into
   a fixed-size closure array.
4. Image pixels, mesh attributes, parameter blocks, and geometry are uploaded
   to Luisa resources. No host-side shader evaluator is used.
5. `GraphSurface` traces the typed program while constructing Luisa DSL. Bump
   evaluates its height dependency subgraph at the center and two ray
   differential offsets.
6. `Polymorphic<Surface>` performs device-side material dispatch. The same
   generated program runs on Luisa `fallback` and GPU backends.

Unknown nodes, modes, sockets, or properties produce named coverage
diagnostics. A material with no usable surface root is rendered with an
explicit magenta coverage material; it is never silently replaced with a
plausible BSDF.

`docs/cycles-shader-nodes-4.5.10.json` is the versioned Blender RNA inventory.
`tools/check_cycles_shader_node_coverage.py --require-complete` is deliberately
red until every Cycles-applicable node has a verified Luisa implementation.
An implementation used by Lone Monk is still classified as partial or
unverified until a focused Cycles linear-EXR probe covers its modes and socket
semantics.

## Integrator contract

The Blender scene package carries these Cycles settings into
`RenderSettings::integrator`:

- total, minimum, diffuse, glossy, transmission, volume, and transparent
  bounce limits;
- transparent minimum bounces;
- direct and indirect sample clamp;
- reflective and refractive caustics controls;
- direct-light sampling mode;
- light-tree enablement and light-sampling threshold.

The Luisa path state has separate regular and transparent depth. At a regular
bounce limit it follows Cycles' terminate-after-transparent behavior: emitter
evaluation remains active and mixed closure selection is filtered and
renormalized to transparent closures only. Transparent bounces preserve the
previous non-transparent ray flags and forward-MIS PDF. Russian roulette is
evaluated after the next surface hit and after surface emission, matching
Cycles' continuation placement. Contribution clamping uses the Cycles
sum-of-absolute-RGB rule per contribution.

Direct-light mode weights distinguish forward-only, NEE-only, and power-
heuristic MIS. Analytic lights currently have no forward intersection
technique, so their NEE estimator intentionally carries full weight.

The following integrator work remains explicit and is not considered Cycles
compatible yet:

- Cycles' emitter importance distribution and environment importance map;
- light-tree traversal and light-sampling threshold behavior;
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
