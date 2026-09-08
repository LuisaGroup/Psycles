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
TRACE_BLENDER=/home/mike/Projects/blender-install-psycles-trace-5.2/blender
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
  fallback 32 32 128 1 \
  /var/tmp/psycles-trace/fallback.raw.json 17 16 6 6 1

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

## Current Cycles 5.2.1 oracle

The authoritative checkout is
`/home/mike/Projects/blender-cycles-trace-5.2`, branch
psycles-path-trace-5.2, revision
cb168525138fecc792cc393f94afc39582b0103c. The installed trace binary is
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender`, reporting
Blender 5.2.1 LTS / cb168525138f. The checkout retains diagnostic edits in
scene/light.cpp and scene/svm.cpp; preserve and review them rather than
assuming every local source byte is represented by the installed build hash.

The production performance reference is a different binary:
`/home/mike/Projects/blender-install-5.2-hiprt/blender`,
5.2.1 LTS / 9e2066aef7ef. Never benchmark the instrumented trace build as if
it were the production Cycles baseline. Verify source, build, schema and
device identities before generating an oracle. Rebuild with all 32 threads
when instrumentation changes; keep toolchains and temporary files local.

Current surface trace records come from the native SVM closure allocator
and its production sampling consumers in path_tracer_cycles_svm_surface.cpp.
They do not use a separate GraphSurface reference sampler or CPU renderer.
The original Cycles CPU and HIP kernels can both supply observational traces,
but the current large-scene performance target is HIP.

## Current evidence and unresolved path differences

The [sampler contract](validation/2026-09-07/sampler-contract/README.md)
pins Cycles' actual automatic-scrambling property. Scene frame, authored seed,
animated-seed behavior, effective seed and full-sequence sample count must
match. Do not force an obsolete diagnostic sampler setting into every scene.

The [Monk residual](validation/2026-09-07/lone-monk-residual/README.md)
identifies a visibility divergence at coincident original leaf geometry.
Keep duplicate primitives; this is not permission for deduplication or
slower bit-matching intersections.

The [Barbershop shared-closure correction](validation/2026-09-08/shared-closure-weights/README.md)
restores additive contributions from shared graph branches. At the diagnosed
pixel, the first four surface events and all 45 sampled random fields match;
a later NEE event still selects an adjacent emitter triangle. This is local
evidence, not proof of global RNG, visibility or path parity.

Forward analytic-light and volume consumers are part of the native runtime;
old progress claims that those paths do not yet exist are not current
guidance. Consult [compatibility status](cycles-compatibility.md) for actual
remaining features. The [equal-pass full-scene campaign](validation/2026-09-08/matched-pass-hip/README.md)
records unresolved image errors and performance gaps independently of focused
trace passes. Historical gate counts and custom-executor trace checkpoints
remain in dated reports/Git history, not as present completion claims.
