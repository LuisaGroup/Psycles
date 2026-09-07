# Zero-BSDF path termination

This is a control-flow correction, not a floating-point precision workaround
or a change to default scalar/vector initialization.

## Root cause and transfer rule

Cycles 5.2.1 `kernel/integrator/shade_surface.h`,
`integrate_surface_bsdf_bssrdf_bounce`, rejects an ordinary BSDF sample when
`bsdf_pdf == 0` **or** `bsdf_eval_is_zero`. Its BSSRDF branch returns earlier
and does not execute this ordinary-BSDF guard. A rejected sample neither
updates bounce/RNG state nor schedules another closest intersection. The
current vertex's independently prepared direct-light contribution is retained.

Psycles previously accepted any sample with a valid label and positive PDF.
That is insufficient: Ashikhmin velvet can have a positive uniform-hemisphere
density and exactly zero scattering contribution over a broad region. At
normal incidence, small sigma makes the original native exponential underflow;
this is not a marginal one-ULP decision or a fabricated zero-weight closure.

`path_kernel_surface_scatter.cpp` now applies the missing all-RGB-zero guard
before changing path state. The per-path observer uses the same eligibility
test, so it does not record an ordinary BSDF/post-bounce event that Cycles
never reaches. The BSSRDF route keeps its original separate semantics.
`any(f != 0)` has no epsilon or energy cutoff. Native arithmetic, inlining
and fast-math remain enabled; no software float or strict-FP path is added.

## Original-code witnesses

`tools/cycles_zero_bsdf_oracle.hip` directly invokes the original Cycles velvet
setup/sample functions on HIP, compiled with `-O3 -ffast-math`. For normal
incidence and random pair `(0.1, 0.5)`, both sigma values return label 6 and
PDF 0.159154937:

| Sigma | Original BSDF value per channel |
| --- | ---: |
| 0.01 | 0 |
| 0.5 | 0.000455386529 |

`tools/create_cycles_zero_bsdf_probe.py` builds a three-panel Blender scene:
zero velvet, finite velvet and red-only diffuse. The ordinary scene exporter
supplies the committed JSON and exact geometry bytes. Original Cycles HIP
path-trace AOVs supply `tests/data/cycles_zero_bsdf_trace.txt`. No Psycles host
reference renderer or substitute shading formulas are used.

The permanent regression renders absolute sample 0 of the 16-sample sequence
at pixels `(2,2)`, `(6,2)` and `(10,2)` of the 12x4 camera. It runs the complete
renderer in both megakernel and staged-coroutine modes, checking original
closure observations, BSDF observations and post-bounce state. Discrete state
and observation presence are exact; continuous values allow normal fast-math
error. The red-only control prevents confusing "any zero channel" with "all
channels zero".

Before the correction both schedulers incorrectly record PDF 0.159155,
label 6, bounce 1 and RNG offset 32 on the zero-contribution panel. Cycles
does not reach those observation points. Both nonzero controls already agree
before the fix. HIP, fallback and strict native Vulkan focused regressions
pass after the fix.

Source oracle: `/home/mike/Projects/blender-cycles-trace-5.2`, commit
`cb168525138fecc792cc393f94afc39582b0103c`; installed observer:
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender`.
The observer records already-computed state and does not change sampling or
transport branches. Regeneration starts with:

```sh
blender --background --threads 32 --python-exit-code 1 \
  --python tools/create_cycles_zero_bsdf_probe.py -- /tmp/zero-bsdf.blend
blender /tmp/zero-bsdf.blend --background --threads 32 --python-exit-code 1 \
  --python tools/export_psycles_scene.py -- /tmp/zero-export
for x in 2 6 10; do
  blender /tmp/zero-bsdf.blend --background --threads 32 --python-exit-code 1 \
    --python tools/render_cycles_path_trace.py -- /tmp/zero-cycles-$x.exr \
    --width 12 --height 4 --pixel-x $x --pixel-y 2 \
    --total-samples 16 --sample 0 --cycles-device HIP --device-name '9070 XT'
done
```

The trace fixture stores `pixel_x, slot_index, RGBA` for slots 32..45 and 47
(slot 46 is the separately covered ShaderData-flag observation). The geometry
fixture is the unchanged 1456-byte export encoded with `od -An -v -tx1`.
Raw evidence is in `/var/tmp/psycles-path-structure-3ahR3G`.

## Relation to the remaining image discrepancy

This proves and fixes a real extra-path case. It is not evidence that zero
BSDFs explain Lone Monk's remaining Diff Ind difference. The new Lone Monk
trace after the preceding camera correction matches all 94 checked discrete
fields and 25 random components at pixel `(720,360)`, sample 7/256. Its
remaining comparison failures are continuous second-hit light-shader values;
they are not being addressed with bit-matching arithmetic.

Full-scene HIP canaries retain the unchanged exports and fast-math:

| Scene | Extent / samples | Sampling seconds | Main frame | Combined relative RMSE | Diff Ind relative RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Lone Monk | 1440x1080 / 256 | 13.536 | 220 B | 0.01240450 | 0.12881587 |
| Monster | 1080x1080 / 256 | 14.849 | 284 B | 0.00547921 | 0.02552784 |

Both have zero invalid pixels. These are single-run canary timings, not a
measured speedup. No profiler, trace, histogram, concurrent build or test is
active during sampling. Both frames are unchanged. The nearly unchanged
Diff Ind errors confirm that this fix is not the main source of those scene
differences. The preceding three-run Lone Monk comparison remains the current
repeated baseline: Psycles 13.5629 s versus Cycles 13.408873 s (1.15% slower).

## Validation and patch isolation

The 32-thread all-target build succeeds. Full active-worktree CTest passes
HIP 166/166 and fallback 168/168. Core is 143/144, with only the existing
source-size failure on four untouched files; the already failing
`blender_export_render_settings` test remains excluded from that core run.

The exact staged application code is exported to `source` and built in
`candidate/build` inside the evidence directory. It excludes the inherited
closure/feature work, the three protected path-tracer files, the unrelated
light-name import/export changes and the dirty Luisa gitlink. It uses clean
Luisa `origin/next` headers at `8e2b0ac78` and active runtime libraries; this
does not claim an independent clean SDK rebuild. Isolated HIP, fallback and
strict native Vulkan regressions pass both renderer modes and all three
original scene cases. Vulkan requires all three gates:
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and
`LUISA_VULKAN_DISABLE_DXC=1`; primary shader-cache reuse is disabled.

The frame sizes and full-scene timings above are from the active worktree,
including its inherited, still-uncommitted frame/feature work. They are not
advertised as properties of this small termination-only commit. Monster also
uses the inherited light-name import fix; no unrelated hunk is included here.
