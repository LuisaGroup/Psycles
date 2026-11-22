# Cycles Path-Trace Oracle

Image-space EXR comparisons show *which* pass diverges, not *where* along a path two renderers first split. Psycles provides a deterministic 344-AOV per-path diagnostic trace whose sole oracle is official Blender Cycles source compiled with the observational patch in [`tools/cycles_path_trace/README.md`](../tools/cycles_path_trace/README.md).

---

## 1. Contract & Schema (`v3`, 344 RGB Slots)

`tools/cycles_path_trace_schema.py` defines the indexed layout shared across:

- Cycles Blender trace rendering (`tools/render_cycles_path_trace.py`)
- OpenEXR multipart/channel decoding (`tools/decode_cycles_path_trace.py`)
- Field-by-field comparison (`tools/compare_cycles_path_traces.py`)
- Luisa device capture (`include/psycles/luisa/path_trace_schema.h`, verified byte-identical to the schema generator by CTest)

Schema version 3 allocates **344 RGB records** (`PsyTrace000`–`PsyTrace343`):

- **8** camera/global slots (`PRNG_LENS_TIME = (time, lens_u, lens_v)`, RNG hash/seed, camera ray)
- **4 path events × 72 slots:** bounce RNG `(u, v, selection)`, path flags, bounce counters, intersection/surface geometry, up to 8 raw Cycles closures (meta, weight, normal), closure selection, direct-light selection & PDFs, BSDF eval/sample/PDF/label, and post-event state
- **4 shadow events × 8 slots:** exact ray interval, source/light IDs, first eligible hit, and transmittance
- **4 forward-emission events × 4 slots:** evaluated emission, side policy, discrete selection PDF, BSDF/light PDFs, MIS weight, and post-clamp contribution

All 32-bit integer RNG hashes and flags split cleanly across two 16-bit floats so EXR storage introduces zero integer rounding loss. Trace capture is strictly observational: writes occur only when all 344 AOVs are enabled (or single-pixel CLI trace mode is explicitly requested), and never request extra RNG dimensions or alter control flow.

---

## 2. Field-Level Comparison Policy

`tools/compare_cycles_path_traces.py` enforces tiered invariants rather than one global epsilon:

- **Exact gates:** sampled random dimensions, bounce counters, flags, object/shader IDs, closure types, and event labels.
- **Float32 tolerances:** continuous geometry, BSDF, light, PDF, and throughput quantities allow `1e-6 + 2e-6 * max(|reference|, |actual|)`.
- **Shared-edge tie breaking:** triangle ID / barycentric differences at shared edges are accepted when object, primitive type, shader, surface point `P`, and geometric/shading normals describe the same physical intersection.

---

## 3. Capturing & Comparing One Sample

Coordinates use Cycles lower-left `(0, 0)` pixel convention over the **full-frame extent** while isolating a single pixel `(x, y)` and a single **absolute sample** from the complete Sobol sequence (`--total-samples N --sample K`):

1. **Render instrumented Cycles trace EXR:**

   ```bash
   TRACE_BLENDER=/path/to/instrumented-cycles/blender
   SCENE=/path/to/scene.blend

   "$TRACE_BLENDER" "$SCENE" --background --python-exit-code 1 \
     --python tools/render_cycles_path_trace.py -- \
     /tmp/psycles-trace/cycles.exr \
     --width 32 --height 32 --pixel-x 17 --pixel-y 16 \
     --total-samples 128 --sample 6 \
     --cycles-device HIP --device-name "RX 9070 XT"
   ```

   *(Note: To avoid float32 crop rounding boundaries in Blender render borders such as `1041 / 1152`, the Python harness offsets window bounds by one-quarter pixel inside the target pixel box. Always inspect `PsyTraceNNN` AOVs rather than cropped `Combined` values, because Blender's crop border normalizes ordinary filtered passes.)*

2. **Capture matching Psycles single-sample trace JSON & compare:**

   ```bash
   ./build/bin/psycles_render_blender_scene \
     /path/to/export-package \
     /tmp/psycles-trace/psycles.ppm \
     hip 32 32 128 1 \
     /tmp/psycles-trace/psycles.raw.json 17 16 6 6 1

   python tools/compare_cycles_path_traces.py \
     /tmp/psycles-trace/cycles.exr \
     /tmp/psycles-trace/psycles.raw.json \
     /tmp/psycles-trace/diff.json
   ```

### Progressive Sample-Chunk Capture

To capture film deltas for one pixel across multi-sample chunks `[chunk_json] [chunk_size] [full_frame=0|1]` without writing trace slots (`-`):

```bash
./build/bin/psycles_render_blender_scene \
  /path/to/export-package /tmp/unused.ppm hip 1152 480 128 4 \
  - 1047 253 0 0 128 \
  /tmp/chunks.json 4 1
```

Recorded deltas represent `delta(progressive_output * rendered_sample_count)` (`ProgressivePixelAccumulator`), tested across all backends by `psycles.progressive_pixel_probe`.

*Note: Never run performance timing benchmarks with an instrumented Cycles oracle binary — keep oracle and release Cycles builds strictly separate.*
