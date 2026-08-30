# Cycles path-trace oracle

Image-space comparisons tell us that two renderers differ, but they do not
identify the first semantic divergence. Psycles therefore has a diagnostic
per-path trace whose only oracle is the current official Cycles source.

The trace does not replace Cycles with a host reference implementation. A
small, diagnostic-only Cycles instrumentation writes already-computed kernel
state to color AOVs. It never requests another random dimension and never
changes a transport branch. The same instrumentation is compiled into Cycles
CPU and HIP kernels.

## Versioned contract

`tools/cycles_path_trace_schema.py` is the indexed contract shared by:

- the Blender AOV setup and one-pixel render harness;
- the OpenEXR decoder;
- the field-by-field comparator;
- the Luisa fallback/HIP/Vulkan trace implementation.

The Luisa device-side indices are generated into
`include/psycles/luisa/path_trace_schema.h`. A regression compares the
committed header byte-for-byte with the Python generator, so a schema edit
cannot silently leave the JIT kernel on an older layout. The raw 3D random
order is the Cycles order `(u, v, selection)`; `PRNG_LENS_TIME` is
`(time, lens_u, lens_v)`.

Schema version 3 contains 344 RGB records. It preserves all 328 version-2
indices and appends a separate forward-emission tail:

- 8 camera/global slots;
- 4 path events with 72 slots per event;
- up to 8 raw Cycles closures per event, with meta, weight, and normal records;
- 4 shadow events with 8 slots per event for the exact ray interval,
  source/light identities, first eligible backend hit, and transmittance;
- 4 forward-emission events with 4 slots per event for the evaluated emission,
  effective side policy and discrete selection PDF, BSDF/light PDFs and MIS
  weight, and the contribution after Cycles sample clamping;
- reserved event slots for compatible schema growth.

The trace includes camera RNG and ray state, absolute RNG dimensions, path
flags and bounce counters, intersection and surface state, raw closure
weights, closure selection and rescaled random values, direct-light selection
and PDFs, BSDF sample/evaluation/PDF/event label, and the resulting path state.
Every 32-bit RNG hash or flag field is stored as two exact 16-bit values so an
EXR float cannot round away integer bits.

Instrumentation writes are enabled only when the film contains the complete
344-AOV range. A normal render, or a render with only some similarly named
AOVs, performs no trace writes. This guard prevents a diagnostic build from
writing past a smaller film buffer.

## Comparison semantics

`tools/compare_cycles_path_traces.py` applies policy by field rather than one
global epsilon:

- sampled random values and all discrete state are exact gates;
- continuous geometry, BSDF, light, PDF, and throughput values use
  `1e-6 + 2e-6 * max(abs(reference), abs(actual))`;
- a triangle ID or barycentric mismatch at a shared edge is equivalent only
  when object, primitive type, shader, surface position, geometric normal, and
  shading normal describe the same surface event;
- reserved fields are not compared.

This is deliberately stricter than an image tolerance while remaining
invariant to accelerator-specific tie-breaking that Cycles CPU and HIP
themselves do not define identically.

## Render and compare

Coordinates use Cycles film convention, with `(0, 0)` at the lower-left of the
uncropped image. The render remains one pixel in the EXR, but camera projection
and RNG hashing use the requested full image dimensions.

The oracle also renders exactly one **absolute** sample from a complete Cycles
sampling sequence. `--total-samples 128 --sample 6` leaves the scene's Samples
value at 128 and uses Cycles' official sample-subset scheduler for `[6, 7)`.
This distinction is required for Tabulated Sobol: changing Samples to one or
substituting a seed would no longer describe sample 6 of the production render.
The generated `.render.json` records the total, absolute offset, and subset
length explicitly.

Blender stores normalized render borders as float32 and forms the integer crop
by truncating each stored border coordinate multiplied by the full extent.
Exact rational boundaries such as `1041 / 1152` can round below the intended
edge and silently produce a 2x1 or 1x2 oracle. The harness therefore places
the lower and upper coordinates one quarter pixel inside the truncation
intervals for `p` and `p + 1`, respectively. The regression checks every pixel
of representative extents through 8192 after an explicit float32 round-trip.
The original Barbershop failure at Cycles pixel `(1041, 254)` was also rerun at
1152x480 with the instrumented Blender build; `oiiotool --info` reports the
result as exactly 1x1 with the complete trace/pass channel set.

Only the `PsyTraceNNN` AOVs are a per-path oracle. Blender's cropped Render
Result applies a render-border normalization to ordinary filtered passes. On
the 2048x858 benchmark at pixel `(1150, 607)`, absolute sample 508, both a 1x1
crop and the center of a 3x3 crop stored Combined as exactly 0.75 times the
Cycles raw film contribution. A full-frame sample-subset render stored the raw
value exactly. The device trace, the post-clamp contribution, the raw Combined
buffer, and the value passed through `BlenderOutputDriver` all agreed before
that crop-only conversion. Consequently cropped Combined or light-pass values
must never be used to infer a transport, MIS, clamp, or material difference.

```bash
TRACE_BLENDER=/home/mike/Projects/blender-install-psycles-trace/blender
SCENE=build/diagnostics/minimal-point/point_light.blend

"$TRACE_BLENDER" "$SCENE" --background --python-exit-code 1 \
  --python tools/render_cycles_path_trace.py -- \
  /var/tmp/psycles-trace/cpu.exr \
  --width 32 --height 32 --pixel-x 17 --pixel-y 16 \
  --total-samples 128 --sample 6 \
  --cycles-device CPU

"$TRACE_BLENDER" "$SCENE" --background --python-exit-code 1 \
  --python tools/render_cycles_path_trace.py -- \
  /var/tmp/psycles-trace/hip.exr \
  --width 32 --height 32 --pixel-x 17 --pixel-y 16 \
  --total-samples 128 --sample 6 \
  --cycles-device HIP --device-name "RX 9070 XT"

python tools/compare_cycles_path_traces.py \
  /var/tmp/psycles-trace/cpu.exr \
  /var/tmp/psycles-trace/hip.exr \
  /var/tmp/psycles-trace/cpu-vs-hip.json
```

The normal Psycles scene renderer can optionally capture the same indexed
buffer. Its trailing arguments are output JSON, full-film Cycles `x/y`, and
absolute sample:

```bash
TMPDIR=/var/tmp/psycles-compiler-tmp \
  build/bin/psycles_render_blender_scene \
  build/diagnostics/minimal-point/export \
  /var/tmp/psycles-trace/fallback.ppm \
  fallback 32 32 1 1 \
  /var/tmp/psycles-trace/fallback.raw.json 17 16 0

python tools/compare_cycles_path_traces.py \
  /var/tmp/psycles-trace/cpu.exr \
  /var/tmp/psycles-trace/fallback.raw.json \
  /var/tmp/psycles-trace/cpu-vs-fallback.json
```

The renderer serializes the fixed RGBA slot array with the existing yyjson
dependency. `decode_cycles_path_trace.py` accepts either the Cycles multipart
EXR or this raw Psycles JSON. Trace capture is observational: it targets one
full-film pixel/sample and does not alter sample dispatch partitioning or
request another random value.

### Absolute sample ranges and progressive pixel chunks

The same renderer exposes absolute subranges of the complete sampling
sequence after the trace arguments:

```text
[sample-first=0] [sample-count=samples-sample-first]
```

`samples` remains the whole-sequence total used to construct the Tabulated
Sobol table. Splitting a render therefore preserves the absolute Cycles sample
identity instead of silently generating a shorter random sequence. Pass `-`
as the path-trace filename when only the later diagnostic arguments are
needed.

Three additional arguments capture one film pixel after every progressive
sample chunk:

```text
[sample-chunk-pixel.json] [probe-chunk-size=1] [probe-full-frame=0]
```

For example, this records 32 four-sample chunks while keeping the production
full-frame dispatch shape:

```bash
build/bin/psycles_render_blender_scene \
  export /var/tmp/unused.ppm hip 1152 480 128 4 \
  - 1047 253 0 0 128 \
  /var/tmp/chunks.json 4 1
```

Coordinates retain the lower-left Cycles film convention. With
`probe-full-frame=0`, the render window is reduced to the requested pixel but
camera projection and RNG hashing still use the full extent. Value arrays use
the explicit relation

```text
delta(progressive_output * rendered_sample_count)
```

This is an exact additive chunk contribution for linear passes such as
Combined. A pass divided by another accumulated pass is nonlinear; its record
retains the stated scaled-output-delta meaning and must not be mislabeled as
per-sample radiance. `ProgressivePixelAccumulator` implements and validates
that relation independently of a renderer backend, and
`psycles.progressive_pixel_probe` covers pixel extraction, progressive
reconstruction, pass-layout rejection, and transactional failure behavior.

Blender 5.3 writes each AOV as a separate OpenEXR multipart subimage. The
decoder enumerates every subimage and also accepts older single-part,
multi-channel files. `tests/test_cycles_path_trace_decoder.py` locks the
multipart behavior.

## Current oracle build

The current diagnostic branch is
`82186b01ad2e79435e67a02de93b178bfbe0f6c4`, refreshed onto official Blender
main commit `29ccd5e2e824128c86fc6174c9c502c02212434a` on 2026-08-06. Its dedicated
worktree is `/home/mike/Projects/blender-cycles-trace`; the unmodified reference
checkout remains `/home/mike/Projects/blender-cycles`. The installed oracle is
`/home/mike/Projects/blender-install-psycles-trace/blender` and reports Blender
5.3 Alpha with build hash `82186b01`.

The original version-1 instrumentation was committed as
`7fa06e0a26f9b20b91005705a2ef8cef3df52562` against official commit
`ff404d072bb4bae52c578d2be3aeeea2a057ab63` and is retained as the historical
standalone patch
[`tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch`](../tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch).

Both the CPU kernels and the `gfx1201` HIP fatbin compile the same schema.
Build and install use all 32 hardware threads:

```bash
TMPDIR=/var/tmp/psycles-compiler-tmp \
  cmake --build /home/mike/Projects/blender-build-psycles-trace \
  --target blender --parallel 32
TMPDIR=/var/tmp/psycles-compiler-tmp \
  cmake --build /home/mike/Projects/blender-build-psycles-trace \
  --target install --parallel 32
```

The point-light CPU/HIP checkpoint is recorded under
`docs/validation/2026-07-30/cycles-path-trace/`. All 43 discrete fields and 16
random fields match exactly. The 84 continuous fields pass their float32
bounds; maximum absolute error is `4.76837158203125e-7`.

The first Luisa checkpoint is recorded under
`docs/validation/2026-07-30/luisa-path-trace/`. The initial differential run
found that Psycles represented camera clipping as nonzero ray `tmin/tmax`,
while Cycles advances `ray.P` to the near plane, starts at zero, and stores
`far - near` (with perspective direction-cosine scaling). After correcting
that contract, every currently populated continuous field passes with maximum
absolute error `4.76837158203125e-7`, and all 13 currently populated random
fields are exact. The remaining failures are deliberately visible unpopulated
light, raw-closure, BSDF, and post-bounce records plus the Cycles shader ID;
they are the next implementation gates, not tolerated differences.

The following light checkpoint found a second representation mismatch:
Psycles had folded delta-point inverse-square falloff into radiance while
using a conditional light PDF of one. Cycles instead stores `distance²` in
`LightSample.pdf` and keeps the normalized point `eval_fac` at `1 / (4π)`.
Psycles now uses that same Luisa DSL contract, including the formal rule that
a zero-radius point has no competing forward-BSDF measure. The fallback, HIP,
and Vulkan device regression passes, the three Luisa path traces agree, and
all newly populated point-light trace fields pass against Cycles CPU. The
strict comparison now reports 24 rather than 30 outstanding gates.

The next gate identified a deterministic BSDF sample-mapping mismatch.
Psycles used a polar disk map and an arbitrary valid frame around the shading
normal. Those choices preserve cosine-weighted density, but they do not
preserve the mapping from a Cycles RNG pair to its world-space direction.
Camera aperture and diffuse closure sampling now share one Luisa DSL
implementation of the Cycles concentric disk map and fixed algebraic
orthonormal basis. The equality branch, disk-boundary branch, and absence of a
final direction renormalization are treated as part of the definition rather
than implementation details.

The device regression locks the first-bounce oracle sample
`random=(0.82080835, 0.67639267)` to
`wo=(0.601930976, -0.222150967, 0.767025411)` and
`pdf=0.244151756`; it passes on fallback, HIP, and Vulkan. The three fresh
Luisa path records still compare with zero failures. The strict Cycles
comparison remains at 24 outstanding gates because closure/BSDF output slots
are intentionally not marked written until the raw closure inventory and
lobe-selection trace are connected.

That closure gate is now connected through a trace-only surface ABI. The
ordinary renderer keeps its compact `SurfaceSample`; an oracle build requests
the post-shader closure array and the selected-closure record from the same
GraphSurface implementation that performs sampling. No closure selection is
reimplemented in the integrator.

For the point-path diffuse closure, all newly written fields pass against
Cycles CPU: closure count/index/type/sample weight, raw weight and normal,
rescaled selection dimension, BSDF label, `wo`, weighted evaluation, PDF,
unguided PDF, sampled roughness, and eta. In particular, Cycles reports
Diffuse sampled roughness as `(1, 1)`, independently of the Diffuse node's
model-selection Roughness input. The strict failure count falls from 24 to 12
with no comparator changes or waivers.

The device regression executes the entire raw-closure and sample path on
fallback, HIP, and Vulkan. All 32 project tests pass. Aggregated Principled is
intentionally exposed with the Cycles virtual closure identity until it is
expanded into the same physical closure list as Cycles; complex materials
therefore cannot falsely pass this gate.

The post-bounce gate is now driven by a single Cycles path-state transition,
not by independent trace-field assignments. Given the previous flags,
visibility, counters, closure label, runtime flags, and scene-synchronized
bounce limits, the Luisa DSL transition advances:

- regular and transparent bounce counters;
- the absolute Sobol RNG offset;
- Cycles `PathRayFlag` and `PathRayVisibilityFlag` bit layouts;
- diffuse, glossy, singular, transmission, and transparent termination
  semantics.

The production path consumes the same transition to derive its object
visibility mask. Secondary BSDF directions are normalized at the same point
as Cycles, before the conditional triangle-origin construction. Surface
runtime flags are accumulated from the actual post-shader closure inventory.
Closure allocation now also uses Cycles' exact `1e-5` cutoff and
`fabs(average(weight))` sample weight; the device regression covers values
below and at the allocation boundary.

On the point-path oracle, `post_depth`, throughput, ray origin/direction,
flags, MIS state, runtime flags, and post visibility all pass against Cycles
CPU. In particular:

```text
post bounce / transparent / rng = (1, 0, 32)
post path flag                 = 266513
post visibility                = 4
surface runtime flag           = 12
mis ray / minimum pdf          = (0.244151756, 0.244151756)
```

The strict failure count falls from 12 to 4. The only remaining point-probe
gates are Blender/Cycles synchronization identities: surface shader ID,
light object/group ID, and light shader ID. Fallback versus HIP and fallback
versus Vulkan both pass with zero failures, and all 33 project tests pass.

## Explicit synchronization identity and shadow-origin checkpoint

The four remaining identity gates are now populated from explicit Blender
synchronization metadata. They are not reconstructed later from Psycles array
indices. The exporter mirrors the relevant `BlenderSync` ordering:

- the five Cycles default shaders retain their fixed leading slots, including
  the world shader at index 3;
- light shaders are assigned in dependency-graph object order;
- material shaders follow in dependency-graph material order;
- object, light-group, and full 64-bit shader identities travel through the
  scene contract and GPU records without truncation.

This also removes the former lexical sort of analytic lights. That sort was
visually harmless in a one-light probe but changes which physical emitter is
selected for a fixed random number in a multi-light scene. The exporter
regression deliberately names two lights in reverse lexical order and locks
the dependency-graph order, shader IDs, object IDs, and light-group IDs.
Cycles' shader flags are represented by a formal source-identity composition
function with a host regression for the exact ABI values.

The resulting fallback, HIP, and Vulkan point-path records each pass the
official Cycles CPU oracle with zero failures:

```text
exact fields        = 43 / 43
random exact fields = 16 / 16
float32 fields      = 84 / 84
topology checks     = 3 / 3
maximum abs error   = 4.76837158203125e-7 (fallback)
                      7.152557373046875e-7 (HIP, Vulkan)
```

Visual inspection then exposed a separate backend-dependent self-shadow that
the single-pixel trace did not cover: fallback showed a diagonal dotted band
and Vulkan showed a dark corner, while HIP was clean. The missing operation
was Cycles' second surface-ray offset stage, not a tunable epsilon. Psycles now
composes the same two predicates:

1. construct the light shadow origin with the shadow-terminator offset;
2. while the source primitive exclusion is active, transform the ray to object
   space and run the source-triangle self-intersection certificate;
3. retain the exact origin when the certificate succeeds, otherwise apply the
   robust ULP offset while retaining explicit source exclusion.

The shared-edge regression proves that an interior origin remains exact while
an ambiguous edge origin takes the robust branch. It runs on fallback, HIP,
and Vulkan. Fresh EXRs are pixel-identical between all three Luisa backends
under `oiiotool --diff`, and the post-fix triptych was also inspected
visually. The complete project suite passes 35/35 with 32-way scheduling.

The committed reports and triptych are under
[`docs/validation/2026-07-30/luisa-path-trace/`](validation/2026-07-30/luisa-path-trace/).

## Multi-emitter and spherical-rectangle light contract

The analytic-light trace is no longer restricted to a zero-radius point
light. A two-point scene locks both halves of a flat, non-light-tree
distribution: selection random `0.1931770742` selects dependency-graph emitter
zero and `0.8768947124` selects emitter one, each with selection PDF `0.5`.
Both complete paths pass Cycles CPU on fallback, HIP, and Vulkan with all
`43 + 16 + 84 + 3` comparison gates.

Full-spread rectangle lights now lower Cycles'
`area_light_rect_sample` measure contract directly into Luisa DSL. This is the
Ureña spherical-rectangle parametrization, including the
cancellation-resistant four-`asin` solid-angle expression and Cycles' planar
fallback at tiny solid angles and grazing normalized edges. Rectangle
sampling produces a solid-angle PDF directly; it is never subjected to a
second area Jacobian. Ellipse lights use Cycles' concentric disk mapping.

Center and grazing complete-path oracles pass on all three Luisa backends. The
grazing case also produced a backend-level regression: strict Vulkan float32
`asin` was not reproducible through either native SPIR-V or the HLSL fallback.
Luisa `next` commit `0d2ea3f6e` implements one no-contraction, range-reduced
strict contract, exercises both code-generation routes, and bumps the native
Vulkan shader-cache tag so stale SPIR-V cannot survive the semantic change.
No application-side trigonometric approximation was retained.

Full-frame comparison revealed and corrected a separate oracle configuration
bug. `render_cycles_golden.py` now pins Tabulated Sobol, scrambling distance
one, and disabled automatic scrambling, matching both the path oracle and
Psycles. A Blender regression locks this configuration. On the one-sample
rectangle diagnostic, this reduces Combined RMSE from `0.01826` under
Blender's unrelated `AUTOMATIC` sampler to `0.002610`.

The remaining sparse image residual is expected but not waived. At the
maximum-error pixel, all sampled NEE and BSDF fields pass; the BSDF secondary
ray intersects the rectangle inside its extents. Cycles therefore evaluates a
forward analytic-light hit, while Psycles currently cannot intersect analytic
lights. Production NEE must remain full-weight until that complementary
technique exists. Consequently:

1. sampled rectangle position, direction, identity, PDF, evaluation factor,
   and theoretical MIS inputs are aligned;
2. actual analytic-light MIS is not aligned until forward light
   intersection/evaluation is implemented;
3. narrow-spread area lights still require the formal Cycles spread-clamping
   construction before sampling.

The reports, image metrics, and inspected reference/actual/difference
triptych are in the
[`luisa-path-trace` validation record](validation/2026-07-30/luisa-path-trace/README.md).
