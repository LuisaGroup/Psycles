# Native shader metadata owns scene consumers

## Formal boundary and counterexample

For native Cycles execution, shader-dependent host decisions must be functions
of the current compiled Cycles used-shader domain. They must not depend on
whether a legacy `MaterialLibrary` is absent, retained, or contradictory.
The previous source-authority fix made the word image independent of that
library, but several host consumers still consulted its `SurfaceProgram`:

- Analytic-light contribution classification and emission estimates.
- Mesh and background light-tree emission estimates.
- Emitter membership, by iterating legacy material-cache entries.
- Geometry binding flags, BSSRDF support, and background spatial variation.

This is a structural source-of-truth difference, not a numerical one. A minimal
scene with two native shaders, emission `(1, 2, 4)` and zero emission, shows it:
without legacy entries both point lights were treated as contributing. Retain
opposite legacy graphs and the wrong light survives instead. A triangle of
area 2 likewise acquires the old cache's energy or the fallback white estimate.

The external reference is Cycles 5.2.1 at
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`:

- `scene/shader.cpp::Shader::estimate_emission`: estimates, constant-emission
  eligibility and the strict AUTO threshold are shader-owned facts.
- `scene/shader.cpp::ShaderManager::device_update_common`: the corresponding
  flags and estimate populate the native `KernelShader` record.
- `scene/light.cpp::Light::has_contribution` and `test_enabled_lights` use the
  shader estimate and `has_surface_spatial_varying` for light enablement.
- `scene/light_tree.cpp`: triangle energy is
  `area * average(abs(shader->emission_estimate))`; analytic lights also use
  the shader-owned estimate.

These analytic expectations come from Cycles itself. No CPU reference
renderer, alternate SVM evaluator, synthetic word oracle, or precision
workaround is added.

## Implementation

`CompiledShaderTable` now retains `ShaderCompileMetadata` in the same dense
source-shader index domain as its word and `KernelShader` images, including
inert holes. No new facts are inferred from opcodes and no graph analysis is
reimplemented in the scene builder.

Checked native-material lookup rejects a missing native table or material;
it cannot silently fall back to a legacy cache. Analytic/mesh/background
emission estimates use this lookup. Emitter membership iterates finalized
material bindings rather than unrelated cache residency.

The native geometry-binding adapter projects `KernelShader` flags directly:
emission, constant emission, transparent shadow, volume, BSSRDF bump,
bump-map correction, MIS side selection and volume sampling. The renderer's
surface tag, parameter-block address and material identity are preserved.
The shader index is the actual compiled source identity, including an index
assigned by the native compiler for renderer-authored materials.

Native BSSRDF geometry planning and background enablement now read the same
retained source facts. The non-native execution route retains its existing
legacy analysis. Native SVM words, stack addresses, PC control flow, feature
masks and device closure state are unchanged by this patch.

## Permanent regression and validation

`tests/test_luisa_cycles_svm_scene_metadata.cpp` exercises the actual production
scene compiler, light packer and mesh light-tree builder, with absent and
contradictory legacy caches. It also covers dense source holes, storage-address
preservation, all effective MIS sides, zero emission, AUTO's exact threshold,
transparent-shadow policy, bump-correction policy, all three volume sampling
modes, dynamic Camera Data and missing-native-state diagnostics.

The existing `test_cycles_svm_scene.cpp` additionally checks retention of
constant emission, transparency, volume connectivity/attribute dependence and
BSSRDF-bump facts across linking. Its complete external word-image regressions
remain enabled.

Evidence directory: `/var/tmp/psycles-native-material-metadata-tG07wM`.

- `hip-red.log`: all four initial production-consumer counterexamples fail
  before the fix; `hip-green.log`: all four pass after the fix.
- `metadata-final-hip.log`: the expanded seven scenarios pass, including
  their policy/threshold subcases.
- Full build with all 32 threads passes. `full-build-final.log` and
  `full-build-validation.log` cover the final implementation and expanded
  tests; `final-test-config.log` verifies the permanent strict Vulkan test
  environment. The earlier `full-build.log` records a test-only unqualified
  `Vec3f` compilation error, corrected before validation.
- `all-hip.log`: all 153 HIP tests pass, 9.45 s with parallel/cached execution.
- `core-adapter.log`: 100 Cycles/Blender compiler/import tests pass, 2.35 s.
  The unchanged `psycles.blender_export_render_settings` probe-inventory
  failure documented in `../native-shader-input/README.md` remains excluded;
  this is not a claim that every registered CTest is green.
- `all-fallback.log`: all 155 fallback tests pass, 10.42 s.
- `strict-vulkan.log`: nine strict native-XIR Vulkan canaries pass, 0.52 s.
- `native-vulkan-recompile.log`: forced fresh AST-to-XIR-to-SPIR-V compilation
  and validation pass; loader tracing shows no DXC/DXIL library load.
- `original-camera-native.log`: the unchanged original Camera Data export
  passes the full native word-tail comparison and device upload regression.

Strict Vulkan checks set `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and `LUISA_VULKAN_DISABLE_DXC=1`.
The forced compilation additionally uses `LUISA_DUMP_SOURCE=1`,
`LUISA_VULKAN_PROFILE_COMPILATION=1`, and `LD_DEBUG=libs`.

## Original Lone Monk HIP canary

Evidence: `/var/tmp/psycles-native-metadata-lone-monk-Ymn8uG`.
Original frame-4 export, RX 9070 XT, 1440x1080, 256 fixed spp, seed 0,
native SVM and staged wavefront with Cycles shader-locality sorting. The
render ran after HIP/core tests finished and before fallback/image analysis;
no concurrent build or test was launched during the timed rendering interval.

The renderer reports **16.0064 s**, versus 16.0102 s before this change:
effectively neutral, not a significant speedup. Actual kernel measurements:

| Kernel | Calls | GPU seconds | VGPRs | Scratch bytes |
| --- | ---: | ---: | ---: | ---: |
| Surface | 1271 | 8.121585 | 256 | 3424 |
| Closest intersection | 1868 | 3.210569 | 128 | 240 |
| Shadow intersection | 1194 | 0.840347 | 144 | 400 |
| NEE | 1199 | 0.734585 | 256 | 1472 |
| Shade shadow | 1194 | 0.333638 | 128 | 16 |

Surface and NEE final LLVM IR are byte-identical to the previous canary:

- Surface SHA-256:
  `f09c834825072e7911b64aeeda8e562652915d4649ea5c751bc08d1e9192ff07`.
- NEE SHA-256:
  `91bf6424b243b39cfa8d809de864597cdfd3cd1cf2420bab132cf946257040c3`.

Dump numbering shifted: these are final modules 4 and 5 in this run, not
3 and 4. Frame storage is unchanged: 88 fields, 416 B payload plus 28 B
coroutine metadata, 444 B SoA / 448 B aligned AoS, capacity 1,048,576.

All eight linear passes in `compare.json` have zero non-finite pixels.
Combined relative RMSE is 0.0111262245, within the preceding ~0.01112--0.01114
range. DiffInd remains 0.141060732 and GlossInd 0.161977980. Comparison-tool
success alone is not a pass/fail threshold or an assertion of image parity.
The comparable Cycles internal interval remains 13.507438 s: Psycles is still
about 18.5% slower, so the performance objective remains unmet.

## Remaining full-scene gate

`full-camera.log` still exits 1 because the complete scene builder requires
legacy SurfaceProgram lowering before native shader compilation. No full
Camera Data render is claimed. This patch removes actual metadata-consumer
dependencies but does not ignore that diagnostic or insert dummy programs.

The remaining migration includes attribute residency, AO and closure-budget
planning, legacy auxiliary shader execution (transparent shadow, volume,
displacement and BSSRDF-normal routes), and legacy resource/program creation.
The imported camera-basis discrepancy also remains unresolved. These are
continuation work toward the original native Cycles SVM renderer objective,
not reasons to narrow its supported node or control-flow domain.
