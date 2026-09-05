# Native shader source authority and Blender input order

## Reproduced failures

`build_cycles_svm_runtime(scene, snapshot)` used `scene.materials` as its first
source of shader graphs. For missing entries it constructed another legacy
`MaterialLibrary`, which requires `SurfaceProgram` lowering. Consequently:

- A source graph accepted by the native Cycles compiler could still be
  rejected by the unrelated legacy lowerer (`Camera Data` is the reproducer).
- A retained same-ID graph silently overrode the current snapshot.
- A valid retained graph could hide an invalid current source graph.

The required invariant is that the native shader image is a function of the
current snapshot and its Cycles used-shader domain, independent of the state
of the legacy material cache. The fix normalizes and validates precisely that
domain with `ShaderCompiler` and passes those programs directly to the native
scene linker. Original shader indices, compilation order, resources, typed
payloads, feature masks and surface/volume/displacement entries are retained.
No new interpreter or CPU reference evaluator is introduced.

After that boundary was fixed, the imported Camera Data graph exposed six
word differences in the complete shader tail. Two independent scalar Math
branches were reversed, including their stack allocations and Combine XYZ
consumers. This was not a floating-point precision difference.

The reference implementation is Cycles 5.2.1 at
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`:

- `source/blender/nodes/intern/shader_nodes_inline.cc::ensure_node_inputs`
  pushes available inputs in socket order onto a LIFO stack. The inliner
  prepares producers before copying a consumer.
- `intern/cycles/blender/shader.cpp::add_nodes_inlined` imports the resulting
  node creation order.
- `intern/cycles/scene/svm.cpp::generate_svm_nodes` uses node ID to break ties
  between equal Sethi-Ullman scheduling keys.

Psycles instead let each node-lowering component recurse in its own `bind`
order, generally forward input order. The importer now prepares linked inputs
in reverse socket order before ordinary-node lowering. Muted, reroute and
group forwarding retain their own socket semantics. This is not a Camera
Data-specific scheduling rule and does not change the SVM scheduling algorithm.

The wider importer tests then exposed a second boundary issue: caching a raw
output by the consumer's requested type can duplicate its producer. The cache
now identifies the natural source output/type; consumer conversions happen
after lookup. Surface and volume closure projections remain separate contract
domains. The existing `blender_curve_family_import` regression catches the
duplicate-producer counterexample.

## Permanent external oracle

`tests/test_luisa_cycles_svm_scene_compilation.cpp` covers absent legacy state,
stale legacy state, invalid current source, an unreachable invalid material,
and a missing used-material reference. It also uploads the production native
word buffer and reads it through a device shader.

`tests/data/cycles_camera_data_scene.json` retains the exact exported Camera
Data material and socket payloads. Unrelated geometry is omitted from this
small compiler fixture; a material slot establishes the native used-shader
domain. The same executable accepts the unmodified original Blender export
as an optional second argument after the backend.

`tests/data/cycles_camera_data_words.txt` contains all 143 words from the
external six-shader Cycles dump. Shader 5 (`Camera Data`) begins at global
word 89 and has a 54-word tail. Its 58-word local image has entries
`(NODE_SHADER_JUMP, 4, 56, 57)`. The regression normalizes only shader-tail
relocation; every word within the tail must match exactly. Expected words
were not adjusted to match Psycles output.

Original evidence:

- `/tmp/psycles-camera-svm.IsQZVs/camera_data.svm52`, SHA-256
  `4a27083b3f0595823472b6778412774b058b17372b7eb9ed5f2e4e9f8cb8d2a4`.
- `/tmp/psycles-camera-svm.IsQZVs/camera_data-export/scene.json`, SHA-256
  `bb9505fe9ab0ceb1b217de0ac5f378fc972117de0d553a417c3c4c785869305e`.
- The fresh, unmodified exported camera scene is at
  `/var/tmp/psycles-camera-basis-audit-ALcdKA/export`.

The binary dump format is the original `PSYSVM52` header, version 1,
64-bit word count, 32-bit shader count, shader ID/name records, followed by
little-endian 32-bit SVM words. The text fixture is a lossless decoding of
that word array, not a second compiler's output.

## Validation evidence

Evidence directory: `/var/tmp/psycles-native-shader-input-Tb4KcJ`.

- `hip-red.log`: four of five source-authority scenarios failed before the
  production fix. The missing-material diagnostic already passed.
- `word-diagnostic.log`: source authority fixed, six exact Camera tail words
  still differed because of importer input order.
- `hip-order-green.log`: all five scenarios passed after the ordering fix.
- `original-camera-hip.log`: the same test passed against the complete,
  unmodified fresh Blender export rather than only its small fixture.
- `bump-order.log`: the independent BOTH bump-state word oracle passed.
- `all-hip-tests.log`: all 152 HIP tests passed for the initial ordering fix.
- `core-adapter-tests.log`: expanded validation found the producer-cache
  issue described above and one unrelated stale Python probe inventory.

Final validation includes the natural-output cache correction:

- Full all-target build with all 32 available threads: passed
  (`full-build-final.log`, 122 build steps).
- 100 Cycles/Blender compiler and importer tests: passed
  (`core-adapter-final.log`). The one excluded test is
  `psycles.blender_export_render_settings`: its unchanged Python expectation
  omits the already-registered `color_to_scalar`, `math_clamp_svm_oracle`, and
  `svm_vector_displacement_nested_both_oracle` probes. Its independent rerun
  still fails (`preexisting-probe-inventory.log`). Neither that test nor the
  probe registry is modified by this patch; a completely green full CTest
  inventory is not claimed.
- All 152 HIP tests: passed (`all-hip-final.log`, 203.84 s).
- All 154 fallback tests: passed (`all-fallback-final.log`, 47.35 s).
- Eight strict native-XIR Vulkan canaries: passed
  (`strict-vulkan-final.log`, 0.60 s).
- Forced fresh native Vulkan compilation of the new regression: passed
  (`strict-native-recompile.log`), with AST-to-XIR-to-SPIR-V compilation and
  validation, and no DXC/DXIL library loaded under `LD_DEBUG=libs`.
- The unchanged original Camera Data export still passes the new native
  compilation regression (`original-camera-hip-final.log`). Its complete
  renderer invocation still exits 1 at the separate legacy scene gate
  (`full-camera-still-blocked.log`); no Camera Data render is claimed.

The Vulkan checks set all three strict flags:
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`. The forced compilation additionally uses
`LUISA_DUMP_SOURCE=1` and `LUISA_VULKAN_PROFILE_COMPILATION=1`.

## Whole original Lone Monk canary

Evidence is at `/var/tmp/psycles-native-input-lone-monk-ABYBPY`: original
frame-4 export, RX 9070 XT HIP, 1440x1080, 256 fixed samples, seed 0, native
SVM, staged wavefront and Cycles shader-locality sorting. The timed run was
scheduled after the HIP suite finished and before fallback tests or image
analysis; no concurrent builds were started.

The complete rendering interval is **16.0102 s**, compared with the previous
16.0200--16.0446 s. This is effectively neutral, not evidence of a significant
speedup. Surface GPU time is 8.134590 s (1271 launches, 256 VGPRs, 3424 B
scratch); NEE is 0.728268 s (1199 launches, 256 VGPRs, 1472 B scratch).
The actual final surface and NEE LLVM IR remain byte-identical to the previous
retained camera-precompute run:

- Surface: `f09c834825072e7911b64aeeda8e562652915d4649ea5c751bc08d1e9192ff07`.
- NEE: `91bf6424b243b39cfa8d809de864597cdfd3cd1cf2420bab132cf946257040c3`.

The legacy-dependent shade-shadow kernel changes from 144 to 128 VGPRs,
but its measured time is also neutral: 0.333531 s previously, 0.333149 s now.
Persistent state remains 88 fields, 416 B payload plus 28 B coroutine metadata:
444 B SoA / 448 B aligned AoS, capacity 1,048,576.

All eight linear passes were compared against the established real Cycles HIP
image. Each has zero non-finite pixels. Combined relative RMSE is 0.01114202,
consistent with the preceding ~0.01112--0.01114 results. DiffInd 0.1410593 and
GlossInd 0.1619785 remain unresolved; a successful comparison-tool exit is not
an assertion of image parity. The matching Cycles internal rendering interval
is 13.507438 s, so renderer performance parity also remains unmet.

## Remaining native integration boundary

This change does **not** claim that the full Camera Data scene now renders.
`LuisaPathTracerBackend::compile_scene` still requires legacy material
lowering for scene metadata and remaining auxiliary execution paths.
Simply ignoring its errors or inserting dummy SurfacePrograms would silently
break those consumers and is not an acceptable fix.

The audited remaining consumers include:

- `path_tracer_scene.cpp`: BSSRDF/transparent/emission/volume metadata,
  attribute residency, shader resources and world spatial variation.
- `path_tracer_geometry.cpp`: transparent shadow shader evaluation.
- `path_tracer_displacement_scene.cpp`: mesh displacement evaluation.
- Volume segment/shadow/majorant paths: legacy volume evaluation.
- `path_tracer_surfaces.cpp`: emission and BSSRDF-normal auxiliary callables.

Those must be migrated to the same native Cycles graph metadata and evaluator,
not supported by adding a second Camera implementation to the legacy model.
The separate imported-camera basis conversion discrepancy documented in
`../camera-transform/README.md` remains open as well.
