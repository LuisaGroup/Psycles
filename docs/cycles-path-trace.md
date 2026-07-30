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

Schema version 1 contains 296 RGB records:

- 8 camera/global slots;
- 4 path events with 72 slots per event;
- up to 8 raw Cycles closures per event, with meta, weight, and normal records;
- reserved event slots for compatible schema growth.

The trace includes camera RNG and ray state, absolute RNG dimensions, path
flags and bounce counters, intersection and surface state, raw closure
weights, closure selection and rescaled random values, direct-light selection
and PDFs, BSDF sample/evaluation/PDF/event label, and the resulting path state.
Every 32-bit RNG hash or flag field is stored as two exact 16-bit values so an
EXR float cannot round away integer bits.

Instrumentation writes are enabled only when the film contains the complete
296-AOV range. A normal render, or a render with only some similarly named
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

```bash
TRACE_BLENDER=/home/mike/Projects/blender-install-psycles-trace/blender
SCENE=build/diagnostics/minimal-point/point_light.blend

"$TRACE_BLENDER" "$SCENE" --background --python-exit-code 1 \
  --python tools/render_cycles_path_trace.py -- \
  /var/tmp/psycles-trace/cpu.exr \
  --width 32 --height 32 --pixel-x 17 --pixel-y 16 \
  --cycles-device CPU

"$TRACE_BLENDER" "$SCENE" --background --python-exit-code 1 \
  --python tools/render_cycles_path_trace.py -- \
  /var/tmp/psycles-trace/hip.exr \
  --width 32 --height 32 --pixel-x 17 --pixel-y 16 \
  --cycles-device HIP --device-name "RX 9070 XT"

python tools/compare_cycles_path_traces.py \
  /var/tmp/psycles-trace/cpu.exr \
  /var/tmp/psycles-trace/hip.exr \
  /var/tmp/psycles-trace/cpu-vs-hip.json
```

Blender 5.3 writes each AOV as a separate OpenEXR multipart subimage. The
decoder enumerates every subimage and also accepts older single-part,
multi-channel files. `tests/test_cycles_path_trace_decoder.py` locks the
multipart behavior.

## Current oracle build

The initial diagnostic build is based on official Blender main commit
`ff404d072bb4bae52c578d2be3aeeea2a057ab63` (2026-07-30). Its dedicated
worktree is `/home/mike/Projects/blender-cycles-trace`; the unmodified reference
checkout remains `/home/mike/Projects/blender-cycles`. The instrumentation is
committed locally as `7fa06e0a26f9b20b91005705a2ef8cef3df52562` and exported
as
[`tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch`](../tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch).
`git apply --check` passes against the exact official base.

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
