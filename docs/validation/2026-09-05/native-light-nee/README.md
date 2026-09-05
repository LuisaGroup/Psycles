# Native Cycles SVM at SHADE_LIGHT_NEE

## Scope and state mapping

The native-SVM renderer now evaluates sampled emitter shaders at the
`SHADE_LIGHT_NEE` cut, following Cycles 5.2.1 `integrate_light_nee` in
`kernel/integrator/shade_light.h`. The legacy surface provider no longer
evaluates those shaders before the cut when the native migration gate is on.
The legacy renderer route remains available and unchanged in this respect.

`DirectLightTaskCall` carries the final shadow ray, emitter identity, time,
sample index, RNG hash/offset, and pre-scatter bounce counters. The new host/JIT
component reconstructs emitter ShaderData without transporting the receiving
surface's closures or ShaderData across coroutine suspension.

| Emitter | Cycles operation reproduced |
| --- | --- |
| Triangle | `triangle_light_uv`, then `shader_setup_from_ray`; final offset ray, flat/smooth/corner normals, transforms, orientation and compact derivatives |
| Point/spot/area/sun | `light_normal_uv_from_position`, then `shader_setup_from_sample`; original lamp UV conventions and zero sample derivatives |
| Background | `shader_setup_from_background`; P=D, N=Ng=wi=-D, NONE object/primitive, clamped angular differential, background tangent basis |

The shared unbounded-ray value is now `FLT_MAX`, matching both the sun setup
branch and the observable Light Path Ray Length. The old `1e30` sentinel caused
six fields in the sun setup oracle to disagree. LCG initialization uses the
original `hash_uint3(rng_hash ^ 0xb4bc3953, rng_offset, sample)` mapping.

All setup branches merge into one original SVM PC loop with
`KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT`, `PATH_RAY_EMISSION`, the uploaded
word image, and no closure storage. Light Path applies its emission-ray depth
increment itself; the task's bounce counters are not incremented twice.
Non-END diagnostic status is rejected. The implementation uses the already
committed diagnostic interpreter API, not the separately pending production
API or stack-size changes elsewhere in the dirty worktree.

The runtime feature field is a necessary included dependency: shader features
are installed after compilation; finalization transactionally adds hair shape
features only for instanced geometry, as Cycles does. A six-case regression
covers ribbon/thick/thick-linear datablocks with and without instances, plus
failed transaction checks. Finalization is single-use.

No application `noinline`, altered scalar/vector default initialization,
software floating-point path, alternate SVM, or Luisa gitlink change is part
of this patch. Triangle emitter geometry remains in the existing static
production domain; this work does not claim moving-triangle support.

## Minimal regressions and external oracles

`tests/test_luisa_cycles_svm_nee.cpp` invokes the actual production evaluator
factory on 16 layered Principled emission cases, starting with zero cached
emission. It reuses the independently generated Cycles light-emission oracle
from the preceding change. Before integration, 13 nonzero-emission cases fail;
after integration all 16 agree. It tests the actual stage, not another reference
shader implementation.

`tests/test_luisa_cycles_svm_nee_setup.cpp` checks 19 setup cases, each exposing
32 floats and eight integer lanes. `tools/cycles_svm_nee_setup_oracle.hip`
executes the external Cycles setup and light geometry functions on HIP. Shared
fixtures contain only inputs. Coverage includes flat/smooth/corner normals,
backfacing, applied and instanced negative scale, nonuniform transforms,
both alternate derivative projection axes, a thin triangle, point sphere/disk/
zero radius, spot sphere/disk, area, finite/delta sun, and both background
differential clamping cases. Integer state is exact. Geometry float comparisons
use relative 5e-5 with a 1e-5 scaling floor.

UV coordinates use the same 5e-5 tolerance with a unit-coordinate scaling floor,
for both UV lanes, every case and all backends. This is not a changed golden
or a production precision workaround. A standalone uncached native-XIR probe
isolated the former Vulkan failure to `acos(0.8f)`:

| Native operation | HIP fast | Vulkan fast |
| --- | ---: | ---: |
| acos | 0.643501103 | 0.643529117 |
| 0.75 - (1 - acos/pi) | -0.045167245 | -0.045158327 |

The approximately 8.9e-6 UV difference is amplified in relative terms by the
barycentric subtraction near zero. Vulkan's GLSL.std.450 precision rules
derive acos accuracy from atan2 and arithmetic, with a 4096-ULP atan2 allowance,
not HIP libm equivalence. See the
[Khronos precision table](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#spirvenv-precision-operation).
The test's unit-coordinate budget is still stricter than the full range of
errors that this specification permits. The strict-math diagnostic also
confirmed the source of the discrepancy; production remains backend-native
fast math. No Luisa source changes were necessary.

## Whole-scene failure exposed a selected-sampler PDF bug

The first native integration passed focused tests but failed the original
1440x1080/256-sample scene: Combined relative RMSE became 94.31%, with sparse
extreme bright samples. Single-pixel reduction at film coordinate (639,799)
isolated sample 5. The same error occurs with and without deferred surface
scatter, so it is not explained by corruption across that coroutine cut.
Its NEE PDF was 0.05567218 and red contribution 87814.9.

Formal cause: for a selected mixture component j, Cycles retains the PDF
returned by its sampler and evaluates only the other components:

`p_mix = weight_j * sampled_pdf_j + sum(k != j, weight_k * eval_pdf_k(D))`.

Psycles discarded `sample_uniform_cone`'s PDF and instead re-tested the rounded
direction with `sun_pdf`. A correctly generated edge sample can fail that
strict angular membership test, dropping the selected sun density to zero.
The remaining map PDF is orders of magnitude smaller. Genuine SVM sky emission
made this visible as a bright sample where the earlier separate sky evaluator
could reject the sun at its own boundary.

The fix propagates direction AND PDF from the existing cone sampler through
sun-only, sun/map, and portal-capable background sampling. Evaluated sun PDF is
still used for directions generated by a different component. It does not
widen the cone, change trigonometry, clamp radiance, or special-case the pixel.

`tools/cycles_background_sun_sample_oracle.hip` calls the actual external Cycles
`background_light_sample` on HIP. Its 18 input RNG pairs cover the square's
edges, the float immediately below one, and interior/center samples, in sun-only
and sun/map mixtures. Before the fix, 26 of 36 PDF comparisons fail: sun-only
returns zero instead of 4179.79883; mixed sampling returns roughly 0.017 instead
of 3343.856. After the fix both the direct helper and the production
portal-capable route agree with the same 36-row oracle. The latter has no
eligible portal, exercising the original sun/map mixture in that control path.
The previously failing scene sample now has red contribution 11.2353201.

## Reproduction

Authoritative external checkout:
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`.

SHA-256, relative to `intern/cycles`:

| Source | SHA-256 |
| --- | --- |
| kernel/integrator/shade_light.h | `25500a32676b857b8395713d3c18c37ae2f568a1d58f56e8c89926c210ac06db` |
| kernel/geom/shader_data.h | `d96bbb1cc712df9a5862943a9ee20fac726204e629f809610b8525bda19de666` |
| kernel/light/background.h | `d1017a24a797395ae8d66bf80a9e76ec02396476c6012aefff45052d5905f515` |
| kernel/sample/mapping.h | `3a7e9b2cfbbb0f22654874e23524b2965ac65ad527a9faf8e59667b76ae8adad` |

From the Psycles root:

```sh
probe_dir=$(mktemp -d /var/tmp/psycles-native-nee-XXXXXX)
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++17 -O3 -ffast-math -I include \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_svm_nee_setup_oracle.hip -o "$probe_dir/setup"
"$probe_dir/setup" > "$probe_dir/setup.txt"
diff -u tests/data/cycles_svm_nee_setup.txt "$probe_dir/setup.txt"
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++17 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_background_sun_sample_oracle.hip -o "$probe_dir/sun"
"$probe_dir/sun" > "$probe_dir/sun.txt"
diff -u tests/data/cycles_background_sun_sample.txt "$probe_dir/sun.txt"
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure -R '_hip$'
ctest --test-dir build --output-on-failure -R '_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 ctest --test-dir build -V \
  -R '^psycles[.]luisa_(background_sun_sample|cycles_svm_nee_setup|cycles_svm_nee|cycles_svm_light_emission|cycles_svm_geometry_runtime|direct_lighting_plan)_vk$'
```

## Validation status

Final full active-tree build passes using all 32 threads. All 148 HIP tests,
150 fallback tests, and 51 compiler/host tests pass. The six focused native
Vulkan tests pass with USE_XIR, REQUIRE_NATIVE_XIR_SPIRV and DISABLE_DXC all
enabled; the sun regression log records fresh SPIR-V generation (8072 to 2111
and 10443 to 4204 words). No DXC is loaded. After the final source updates the
external sun sampler exactly reproduces its checked-in golden.

The isolated candidate builds fully with 32 threads and passes the six focused
tests on HIP, fallback, and strict native Vulkan. Its full original Lone Monk
1440x1080/256-sample render also completes with zero nonfinite pixels in all
eight compared passes. Evidence is in `/var/tmp/psycles-native-nee-K9h6XY` and
`/var/tmp/psycles-native-nee-clean-emvq04`. The isolated source candidate starts
at `a6cd3ce0` and excludes unrelated Psycles changes, including the pending
microfacet albedo source, interpreter API, and three protected path files.
It uses current Luisa headers/libraries, so it is not a clean-Luisa validation.

The isolated candidate's image has Combined relative RMSE 2.217102% and render
elapsed 24.3037 s. To distinguish the omitted dirty fixes from a new regression,
the preceding isolated source at
`/var/tmp/psycles-light-emission-VBHAdP/source` was checked with
`git diff a6cd3ce0 -- src include cmake` (empty), built with all 32 threads, and
used to render the same complete scene. That clean pre-change baseline has
Combined relative RMSE 2.225858%, elapsed 23.3823 s, and no nonfinite pixels.
All eight pass comparisons remain at the baseline error level. These isolated
runs are not profiled benchmarks; their timings also do not establish a speedup.
They establish that the candidate does not require the excluded Psycles code
to render successfully or to preserve its own baseline accuracy. Results are
`compare-cycles.json` in the candidate directory and
`clean-baseline-compare.json` in the main evidence directory.

## Whole-renderer HIP result and remaining gap

Final image/profiler evidence: `/var/tmp/psycles-native-nee-fixed-TU5xTS`.
The original scene, not an extracted sibling, was rendered at 1440x1080,
256 fixed samples, frame 4, seed 0, native SVM surface, staged wavefront,
surface sorting enabled. No other GPU tests or builds ran during profiling.
The Cycles HIP reference is
`/var/tmp/psycles-lone-monk-cycles-hip-profile-ulQYKD`; its metadata verifies
adaptive sampling and denoising disabled. Although the imported source scene
requests adaptive sampling, Psycles renders the explicitly requested fixed
sample count, matching the reference run.

| Measurement | Psycles | Cycles HIP |
| --- | ---: | ---: |
| Render elapsed | 20.0004 s | 16.450413 s |
| Surface kernel | 12.021888 s | 5.949119 s |
| Closest intersection | 3.236058 s | 3.305630 s |
| Light NEE | 0.774932 s | 0.491862 s |
| Shadow intersection | 0.843501 s | 0.934577 s |
| Shadow shading | 0.333138 s | 0.373949 s |

Psycles surface is `kernel_3426ed6d53470369`, with 256 VGPRs and 5072 B private
scratch; native NEE is `kernel_df8055d98e371de0`, with 256 VGPRs and 1472 B
scratch. These are distinct from persistent coroutine state.

Combined relative RMSE is 1.114406%, Normal 0.238444%, DiffCol 0.122990%,
and all eight compared passes have zero nonfinite pixels. DiffInd and GlossInd
remain noisier at 14.1061% and 16.1976%; restoring the baseline is not a claim
of exact image identity. The preceding validated render was 19.7822 s with
Combined relative RMSE 1.115790%. This change is a semantic/stage-alignment
improvement, not an established end-to-end speedup.

The generated coroutine frame has 88 fields, 448 B AoS size, and 444 B actual
SoA storage per slot (416 B user payload plus 28 B scheduler fields). Cycles'
HIP SoA layout for this scene's feature mask uses 172 B main state
(104 path + 44 ray + 24 intersection) and 224 B shadow state
(76 path + 52 ray + 96 intersection array). Equal-capacity main/shadow slots
therefore require 396 B, not a single 396 B AoS object. Psycles is 48 B (12.1%)
larger. Queue/sort buffers, register usage and private scratch are excluded.
This size difference alone does not explain execution time.

Forward/volume emitter migration, other renderer coverage and the overall
performance goal remain unfinished. The next performance investigation should
compare the surface and NEE kernels' actual structure and state traffic; it
must not turn the measured gap into an assumed frame-size cause.
