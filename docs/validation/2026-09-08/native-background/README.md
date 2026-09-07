# Native Cycles background evaluation and importance-map baking

World shader execution now uses the same native Cycles 5.2.1 word image as
surface, shadow and lamp shading. The old background SurfacePoint construction,
SurfaceProgram replay and EnvironmentCallables have been removed. The
transitional material library still exists for other consumers; this is not
completion of the renderer-wide legacy deletion.

## Source semantics and corrections

Authority: `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`.

| Entry | Original source | Required state |
| --- | --- | --- |
| Forward background | `kernel/integrator/shade_background.h` | Actual ray origin/direction/differential, static-scene time 0.5, incoming visibility/flags plus EMISSION; SURFACE_BACKGROUND mask |
| Sampled background | `kernel/light/sample.h`, `kernel/geom/shader_data.h` | Sampled shadow-ray origin/direction/time, clamped compact differential; visibility NONE, EMISSION, SURFACE_LIGHT mask |
| Importance map | `kernel/bake/bake.h` | Origin zero, time 0.5, finite-difference equirectangular ray differential; EMISSION plus IMPORTANCE_BAKE; SURFACE_LIGHT without RAYTRACE/LIGHT_PATH state features |

The common setup preserves P=D, ray_P=origin, N=Ng=wi=-D, the decorated
background shader ID, KernelShader flags, no object/primitive, FLT_MAX ray
length, the original orthonormal basis and compact/parametric differentials.
Camera rays are not clamped to the importance-map differential. Other
forward rays and NEE rays are. LEAVE/bump and unrelated SVM semantics are
unchanged.

Raw path counters reach NODE_LIGHT_PATH. Its emission-ray increment occurs
exactly once. Removing the LIGHT_PATH *feature* for bake is not permission to
erase that opcode: it still reports Ray Depth 1 and visibility-derived
outputs while not reading integrator counters. Tests supply nonzero counters
to distinguish these meanings. Emission evaluation initializes the same
Cycles LCG seed; bake uses its original zero seed.

The native production KernelGlobals adapter was missing
`background.use_sun_guiding`. The new regression was red with its inherited
false value. It now projects the configured sampling state. Existing native
Sky code suppresses the solar disc only for guided IMPORTANCE_BAKE rays;
ordinary background evaluation still sees the disc. Authored native world
nodes, including their SkyLoader images, are authoritative. Environment
sampling metadata cannot replace the graph, append a duplicate sun or cause
a second world LUT to be uploaded/precomputed. Standalone contract panoramas
without a world shader retain their explicit image input, without invoking
SurfaceProgram.

Forward invisible worlds skip shader evaluation before accumulating zero,
as in Cycles. Constant emission uses KernelShader, including black and signed
values, and does not execute a node stream. Importance baking still executes
the original shader even for constant output and ensures finite components.

## Render-time sampling data ownership

The old loader baked a world before render extent and camera projection were
known. That is insufficient for a world that reads Window coordinates.
Baking now occurs after RenderKernelParameters are finalized. Each render
session holds its own immutable CDF buffers; preparing a second session
does not change the first one's sampling data. This is Cycles' ordinary
importance-map evaluation, not profiling/pre-rendering to size local arrays.
Native SVM stack capacity remains the compiler's static high-water mark.

## Independent regression

`tests/cycles_svm_background_fixture.h` provides typed *inputs* shared by
the original HIP oracle and native interpreter. No shader output is computed
by a CPU reference renderer. `tools/cycles_svm_background_oracle.hip` invokes
original `shader_setup_from_background`, `surface_shader_eval` and
`kernel_background_evaluate`.

```sh
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_background_oracle.hip -o /var/tmp/background-oracle
/var/tmp/background-oracle
```

`tests/data/cycles_svm_background.txt` is the resulting output, SHA-256
`452dcc1d02d6db653d0afa708391f11f993ea48a9bd146f15f854b30e8fb94d5`.

`test_luisa_cycles_svm_background.cpp` checks:

- Eighteen entry/ray combinations: all 32 setup float lanes, eight integer
  state lanes and native SVM emission, plus diagnostic PC/NODE_END status.
- Actual production sampled-light setup, not an extracted sibling.
- Camera/indirect differential clamp, zero and nonzero derivatives, varying
  ray origin/time, poles, seam and equal-component normal basis branches.
- KernelGlobals sun-guiding enabled and disabled, and native-graph authority
  despite deliberately conflicting environment-sun metadata.
- Constant black/signed records with no valid SVM jump entry.
- Both original bake output and two camera-dependent Window CDFs, retained
  simultaneously and read back after both sessions have prepared their data.

Fast math stays enabled. Floating comparisons tolerate small native
arithmetic differences; no software floating-point path or inlining control
was added.

## Validation

Evidence directory: `/var/tmp/psycles-native-volume-svm-06XnDX`.
This checkpoint builds on Psycles `e0d2a163` and the already-published Luisa
`fb315d27d`; it makes no Luisa, allocation, inlining or fast-math changes.

- Full 32-thread all-target build passes (`native-background-final-build.log`).
- Original Cycles HIP oracle was rebuilt with the command above, executed on
  the GPU, and its complete stdout matches the checked-in data byte for byte
  (`background-oracle-build.log`, `background-oracle.txt`).
- Focused HIP passes 10/10 (`native-background-focused-hip.log`); the complete
  HIP selection passes 172/172 (`native-background-all-hip.log`). After the
  final volume-shadow differential correction, all 27 volume/background HIP
  tests pass again (`native-background-volume-final-hip.log`).
- Focused fallback passes 10/11 (`native-background-focused-fallback.log`).
  The final complete fallback selection passes 173/174
  (`native-background-all-fallback.log`, 282.30 s). The sole failure in both
  runs is the already-recorded sample-dispatch film trace, described below;
  the new background regression and whole-render consumers pass.
- Strict native-XIR Vulkan passes 5/5: background, zero-BSDF whole render,
  background-sun sampling, sampled-light setup and Sky
  (`native-background-strict-vulkan.log`). A separate uncached background
  regression with `LD_DEBUG=libs` records successful SPIR-V compilation and
  loads neither `libdxcompiler` nor `libdxil`
  (`native-background-vulkan-loader.log`).
- The environment-texture world scene renders 64x64 / 4 spp in staged
  Vulkan with native-XIR required and DXC disabled. All 46 EXR channels are
  finite (`environment-native-background-vulkan.log`, `.exr`). This is a
  world-only scene, not a geometry performance benchmark.
- Host selection passes 148/149 (`native-background-host.log`). Its sole
  failure remains the existing source-size policy: `cycles_svm_nodes.cpp`,
  `test_cycles_svm_compiler.cpp`, `test_luisa_compact_surface_preparation.cpp`
  and `test_luisa_cycles_svm.cpp` exceed 2,000 lines. No new file exceeds the
  limit and the policy has not been relaxed.

The fallback allocator no longer aborts above 4 MiB. The existing later
wavefront/megakernel trace comparison still differs at slot 30, component 2
(`light_ng.z`): expected `-0.62323 (0xbf1f8bfd)`, actual
`-0.623204 (0xbf1f8a50)`, exactly the previously recorded pair. This is not a
new background failure; it also has not been fixed, hidden by a tolerance
change or attributed to a compiler bug without reduction.

Strict Vulkan selection:

```sh
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 ctest --test-dir build --output-on-failure \
  -R '(cycles_svm_background|cycles_svm_nee_setup|background_sun_sample|cycles_svm_sky|cycles_zero_bsdf)_vk$'
```

## Large HIP canaries

Device: AMD Radeon RX 9070 XT, `gfx1201`, HIP 7.2.53211. Both runs use native
SVM by default, native fast math, the staged wavefront scheduler, the
direct-light queue, fixed absolute samples [0, 256) of 256 and a 1,048,576
main-frame capacity. Main path shader caching is disabled; auxiliary/CDF
compilation retains its default cache policy. Times below are render-only,
excluding scene compilation and JIT. All 46 output channels are finite.

| Scene | Resolution / spp | Render time | Main subroutines / fields / bytes | Combined / DiffInd relative RMSE |
| --- | --- | --- | --- | --- |
| Lone Monk | 1440x1080 / 256 | 13.5612 s | 4 / 55 / 220 | 0.01240469 / 0.12881537 |
| Monster (SSS) | 1080x1080 / 256 | 14.9770 s | 6 / 71 / 284 | 0.00547924 / 0.02552799 |

Scene bundles:

- `/var/tmp/psycles-fullres-hip-20260830/exports/lone-monk-current`
- `/var/tmp/psycles-native-volume-svm-06XnDX/monster-export`

Original Cycles HIP references:

- `/var/tmp/psycles-shadow-queue-cycles-aBpy7U/cycles.exr`
- `/var/tmp/psycles-multiscene-52-Vw3BkE/monster/cycles1080/cycles.exr`

`tools/compare_cycles.py` validates matching Blender build identity
`9e2066aef7ef` using each reference's `cycles.json` and each bundle's
`scene.json`, without an unverified-identity override. Logs and EXRs are
`{monk,monster}-native-background-hip.*`; comparison reports are
`{monk,monster}-native-background-compare.json` in the evidence directory.

These are single canaries against existing references, not a new paired
performance campaign. Relative to the previous native-default checkpoint,
frame sizes and image residuals are essentially unchanged. The small timing
movement does not establish a speedup. In particular, this migration does
not resolve Monk's remaining DiffInd/visibility divergence. Native
stacked-volume/displacement consumers, remaining SVM opcodes, complete
legacy deletion and the full current-revision multi-scene benchmark remain
open.
