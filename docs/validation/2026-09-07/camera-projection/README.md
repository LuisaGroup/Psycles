# Camera projection and actual surface work

This checkpoint aligns static, non-stereo perspective/orthographic cameras
with Cycles 5.2.1. It does not require floating-point bit identity. Motion,
stereo and additional panorama models are not newly implemented here.

## Structural mapping

| Boundary | Previous implementation | Cycles and this change |
| --- | --- | --- |
| Camera source | Reconstructed projection from sensor angles | Preserve optional lens/sensor dimensions and pixel aspect; retain legacy angle-only bundles |
| AUTO fitting | Used vertical sensor angle in portrait | AUTO selects sensor width regardless of fitted axis |
| Raster input | Top-down pixel-to-screen arithmetic per ray | Bottom-left raster through uploaded `rastertocamera` |
| Perspective DOF | Normalize before constructing focus position | Unnormalized projected point, focus plane, lens offset, normalize |
| Differentials | Reconstructed camera-space neighboring rays | World-space center and uploaded `dx/dy`, independent of DOF |
| Orthographic DOF | Disabled by the application | Original lens/focus and near-plane intersection sequence |
| Aperture blades | Any nonzero count used a polygon | Counts below three use a disk |
| Window coordinates | Per-node reconstruction and divisions | Uploaded `worldtondc`, including perspective-background camera-position adjustment |

The camera pose exposed to existing SVM services remains in the original
Blender -Z convention. Ray projection uses Cycles' +Z camera basis; the
change does not silently invert unrelated camera-space inputs.

All inversions and matrix composition are host scene setup work. The GPU
uses native float arithmetic and the ordinary optimizing compiler. The
camera/Window regression inspects the actual production XIR for absence of
device matrix inversions. No noinline, software float, precision annotation,
changed scalar/vector initialization, or disabled fast-math is introduced.

The main rendering option and HIP JIT already enable fast-math by default.
Existing strict exceptions were inspected: the volume-majorant prepass
protects a probabilistic upper bound, motion-mesh preprocessing validates
finite values and swept bounds, and HIP `frem` preserves range reduction for
large quotients. These are not last-bit matching paths. Existing affine FMA
helpers use native operations, not software rounding.

## Independent regressions

Inputs are in `tests/cycles_camera_projection_fixture.h`. There are 24
camera configurations and six raster positions each: landscape/portrait,
AUTO/horizontal/vertical fitting, non-square pixels, shifted viewplanes,
circular/polygonal/anamorphic apertures, DOF on/off, perspective/orthographic,
and translated/rotated camera poses. Fewer-than-three blade cases are explicit.

`tools/cycles_camera_projection_oracle.cpp` invokes the original Cycles host
projection/Transform utilities. `tools/cycles_camera_ray_oracle.hip` invokes
the original GPU `camera_sample_perspective`, `camera_sample_orthographic`
and `camera_world_to_ndc`. No Psycles CPU camera/shading reference is used.

The original implementation fails the new ray regression before the fix.
Host matrices are tested for both physical and legacy angle-only inputs.
Ray origin/direction, compact differentials, clipping, foreground Window and
background Window coordinates are tested on HIP, fallback and strict native
Vulkan. Numeric comparisons use `5e-6 * max(1, abs(expected))`, not ULP or
bit-equality gates. A Window service sample is taken along the ray at a
surface distance, independently of the ray-origin test: reprojecting a
rounded world point exactly at a 0.001 near plane magnifies normal float
roundoff and is not a reason to disable fast-math or add a precision path.

The exporter/importer regressions preserve sensor/lens/pixel-aspect inputs,
accept legacy manifests and reject nonpositive physical lens/sensor values.

Oracle source: `/home/mike/Projects/blender-cycles-trace-5.2`,
`cb168525138fecc792cc393f94afc39582b0103c`. Example generation commands:

```sh
c++ -std=c++20 -O2 -msse4.2 -I include \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_camera_projection_oracle.cpp \
  /home/mike/Projects/blender-build-psycles-trace-5.2/lib/libcycles_util.a \
  /home/mike/Projects/blender-build-psycles-trace-5.2/lib/libbf_intern_guardedalloc.a \
  -o /tmp/camera-projection-oracle
/tmp/camera-projection-oracle > tests/data/cycles_camera_projection.txt
hipcc -parallel-jobs=32 --offload-arch=gfx1201 -DHIPCC -std=c++20 \
  -O3 -ffast-math -I include \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_camera_ray_oracle.hip -o /tmp/camera-ray-oracle
/tmp/camera-ray-oracle tests/data/cycles_camera_projection.txt \
  > tests/data/cycles_camera_ray.txt
```

## Lone Monk work census

For the unchanged frame-4 export at 1440x1080, 256 spp, seed 0:

- Psycles actual SVM material evaluations before this camera change:
  **819,154,817**, execution histogram `exact: true`.
- Original production Cycles HIP queued surface shading work items:
  **818,667,845**, 1,396 dispatches.
- Difference: **486,972**, or **0.05948%**.

The Cycles count is the unrounded `work_size` at `HIPRTDeviceQueue::enqueue`,
observed read-only with GDB. For sorted surface kernels,
`PathTraceWorkGPU::enqueue_path_iteration` assigns `work_size = num_queued`;
HIP launch rounding happens later. These are not padded lanes. Psycles
records only `material_evaluated()` executions. The histogram's additional
host-projected legacy instruction estimates are not native SVM work counts
and are deliberately not used here.

The counts do not support a large excess of aggregate surface evaluations
on this scene. They do not establish per-hit instruction parity, identical
path populations by bounce/material, or absence of other structural errors.
The diagnostic runs are not performance measurements.

## Images and performance status

On the unchanged Lone Monk export, the first full camera-ray canary changes
Diff Ind relative RMSE from 0.14609184 to **0.12881620**; Combined is
0.01239393. Diff Ind mean ratio is 1.0014943 and invalid pixels are zero.
Its unprofiled single-run sampling time is 13.4653 s; the main coroutine frame
remains 55 fields / 220 B. This is not a repeated speedup claim and does not
establish that the remaining image discrepancy is a floating-point issue.

The new optical fields were also exported from the original Blender scene.
Full-image A/B uses the unchanged older bundle to avoid conflating camera
changes with other exporter evolution.

Three alternating, unprofiled HIP runs after the complete camera/Window
change (1440x1080, 256 spp, identical export/source settings, fast-math on):

| Run | Cycles sampling wall seconds | Psycles sampling wall seconds |
| --- | ---: | ---: |
| 1 | 13.408571 | 13.5341 |
| 2 | 13.408873 | 13.5721 |
| 3 | 13.427984 | 13.5629 |
| Median | 13.408873 | 13.5629 |

The current median difference is **1.15% slower for Psycles**. This compares
sampling-phase wall times, not pure GPU time or application startup: Cycles'
own `Rendered 256 samples in ... seconds` scheduler report and Psycles'
render-session timing. Export/import, scene compilation, JIT and EXR writing
are excluded. No GDB, profiler, per-path trace, execution histogram, concurrent
build or other test run is active during these measurements. Psycles shader
cache reuse is disabled for its primary compilation; generic scheduler
helper caches remain allowed. The full command is retained in
`camera-benchmark.sh` in the evidence directory. This is a current comparison,
not a controlled before/after speedup measurement for the camera patch alone.

The final full-patch Lone Monk image has Diff Ind relative RMSE 0.12881724,
Combined 0.01241008 and zero invalid pixels. Monster at 1080x1080/256 spp
changes Diff Ind from 0.02752061 to **0.02552784** and Combined from
0.00590316 to **0.00547921**, also with zero invalid pixels. Monster's main
frame remains 71 fields / 284 B. Its canary overlaps validation work and is
not used for a performance claim.

## Completed validation and isolation

The all-target build uses 32 threads. Active-worktree tests pass HIP 165/165
and fallback 167/167. Core is 143/144; the only failure is the pre-existing
source-size check on four untouched files (`cycles_svm_nodes.cpp`,
`test_cycles_svm_compiler.cpp`, `test_luisa_compact_surface_preparation.cpp`
and `test_luisa_cycles_svm.cpp`). The previously failing
`blender_export_render_settings` test remains excluded from that core run.

The exact staged application source is separately exported to
`camera-source` and built in `camera-candidate/build` under the evidence
directory. It excludes the inherited closure/feature/ABI-assumption work,
all three protected path-tracer files, the unrelated light-name import/export
hunks and the dirty Luisa gitlink. It uses clean Luisa `origin/next` public
headers at `8e2b0ac78` with the active runtime libraries, not a claim of an
independent clean SDK rebuild.

That isolated candidate passes the four focused HIP/core/import tests,
two fallback tests and two native Vulkan tests. Whole-renderer canaries
pass Lone Monk 32x32/4 spp on HIP, followed by the original Map Range bundle
16x16/4 spp on fallback, and finally on Vulkan with all three gates:
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`,
`LUISA_VULKAN_DISABLE_DXC=1`. The Vulkan log records successful native
SPIR-V generation through the complete renderer. Primary shader-cache reuse
is disabled in these canaries.

The isolated candidate's Lone Monk frame is 535 fields / 2140 B, unchanged
from the preceding isolated checkpoint. The 220 B / 284 B frame sizes and
full-image performance numbers above are from the active worktree with its
still-uncommitted frame/feature work; they are not advertised as properties
of this camera-only commit. Likewise Monster uses the active light-name
import fix; the inherited alias collision is not bundled into this checkpoint.

Evidence is retained in `/var/tmp/psycles-native-surface-candidate-eqsvzP`:
original-code probes, read-only Blender camera observations, exact work
census, red/green tests, EXRs, comparison JSON and benchmark commands/logs.
