# Native Holdout: state, word stream and integrator boundary

This repair addresses the object/node Holdout boundary (**S3**) of the
[whole-surface audit](../surface-semantic-audit/README.md). Twelve original
surface shader images match exactly, and twelve complete surface scenes match
the original HIP films under both megakernel and staged scheduling. A separate
original-GPU state regression exposed an extra generic allocator clear; that
clear is removed in the implementation. Three further original word images
cover volume cleanup and a shared surface/volume Holdout node. These results
do **not** certify the complete renderer or a performance gain.

Current status (2026-09-10): the [SDK 98f integration](../next-integration/README.md)
passes complete host 183/183, HIP 192/192, fallback 194/194 and strict native
Vulkan 12/12, including the four previously failing compiler tests. The
separate SDK native Vulkan selection passes 17/17. The frozen SDK-6e evidence
below remains 7/11 with four CFG verifier aborts; its failed gate and absent
completed archive are not relabeled. Fresh Barbershop/Monk canaries are finite,
but Classroom's historical failure, indirect-light differences and a new full
four-scene paired benchmark remain open.

## Original contract and implementation

The source reference is `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`. The original functions are called by
the GPU state observer, not reimplemented in a host reference evaluator.
Source-file hashes, including the retained observer checkout's dirty-file
identity, are recorded in the fixture provenance below.

| Boundary | Exact contract |
| --- | --- |
| Compiler projection | `scene/shader_nodes.cpp::HoldoutNode` declares internal `SurfaceMixWeight`, then `VolumeMixWeight`, and one `Holdout` output. It emits `NODE_CLOSURE_SET_WEIGHT` with RGB one, then `NODE_CLOSURE_HOLDOUT` with the typed mix-offset payload. The node is linear and reports `CLOSURE_HOLDOUT_ID` for closure budgeting. |
| Node dispatch | The one-word payload contains the byte-sized stack offset. Invalid offset uses the current closure weight; a valid exactly-zero mix returns before allocation or flag mutation. Other weights, including signed and zero closure weights, are not clamped or filtered. `SD_HOLDOUT` is set even when allocation fails. |
| Allocation | `kernel/closure/alloc.h::closure_alloc` writes only type and weight, increments `num_closure`, and decrements `num_closure_left` on success. It does not initialize sample weight, normal, or the rest of the slot. Failure changes none of these fields. |
| Object Holdout with transparency | If the object mask is set, `SD_TRANSPARENT` is set, and `SD_HAS_ONLY_VOLUME` is clear, return `1 - closure_transparent_extinction`. Set every nontransparent type in the initialized prefix to `NBUILTIN_CLOSURES`; do not compact the prefix or refund capacity. Apply the original flags arithmetic below. |
| Other Holdout evaluation | An object mask outside that transparent branch returns RGB one without changing closures. Without an object mask, return the ordered sum of weights whose type is exactly `CLOSURE_HOLDOUT_ID`; ignore other types. |
| Integrator | At ordinary surface events, prepare closures, apply eligible Holdout, then emission, termination/roulette, data passes, NEE and continuation. Eligibility is `(SD_HOLDOUT || SD_OBJECT_HOLDOUT_MASK) && PATH_RAY_TRANSPARENT_BACKGROUND`. A BSSRDF exit skips Holdout. |
| Film and termination | Accumulate `average(holdout_weight * throughput)` as transparency, without a clamp. Terminate before emission iff the weight is exactly RGB one, irrespective of throughput. The existing transparent-film operation also handles primary-volume-transmit accumulation; shadow-catcher support is not certified here. |

`surface_shader_apply_holdout` preserves closure count/left/order, weights,
sample weights, normals, payloads, emission and transparent-extinction
aggregates. Only the specified type and flag writes occur. In particular,
Cycles 5.2.1 spells the mask as
`sd.flag &= ~(SD_CLOSURE_FLAGS - (SD_TRANSPARENT | SD_BSDF))`.
`SD_CLOSURE_FLAGS` excludes `SD_TRANSPARENT`, so integer subtraction is **not**
equivalent to a set-difference rewrite. Preserving this affects the Normal pass
even for a fully transparent object Holdout.

The compiler input-order declaration was initially missing from the new node's
projection. Adding a handler or node tag alone is insufficient: the projector
checks the exact declared input set and order. The repair adds both internal
inputs to `cycles_svm_closure_inputs.h` and their graph projection; it does not
alter captured expected words to accept a different graph.

The production consumer now retains native ShaderData through Holdout and
constructs its preparation result afterwards. An eager preparation cache would
retain pre-Holdout emission/closure reductions. There is still one material
evaluation and one production preparation call. The fully-Holdout `$break`
exits the containing path loop; the helper's closure loops have already ended.

Diagnostic closure traces expose sample weight/normal only for types defining
those fields. Portal is in the native BSDF interval. Volume-scatter records
define sample weight but contain phase payload rather than a surface normal.
Holdout/invalid records retain diagnostic defaults instead of motivating
production initialization of otherwise undefined fields.

## Permanent oracles and frozen focused evidence

Surface fixtures are in [tests/data/cycles_holdout](../../../../tests/data/cycles_holdout/manifest.json).
The authoring and capture tools are `tools/create_cycles_holdout_scenes.py`,
`tools/capture_cycles_holdout.py`, and `tools/extract_cycles_holdout_fixture.py`.
The manifest records 12 original inputs, 36 successful export/render/observer
processes, all 48 fixture-file hashes, producer binary hashes and source hashes.

- Complete films come from the production HIP Blender build `9e2066aef7ef`
  at `/home/mike/Projects/blender-install-5.2-hiprt/blender`.
- SVM words come from the observer-enabled HIP build `cb168525138f` at
  `/home/mike/Projects/blender-install-psycles-trace-5.2/blender`.
- Local shader extraction relocates only the three ShaderJump entry offsets;
  all other original typed payload words are copied unchanged. JSON and decoded
  geometry reproduce the exported input bytes.
- Every film is 16x16, one sample, seed zero, with the same 15-pass/46-channel
  contract. Each scene/scheduler compares all 11,776 lanes with
  `2e-6 + 2e-6 * abs(original)` tolerance and finite-value checks. No pixels or
  channels are excluded. These are correctness probes, not large-scene timings.

The twelve scenes cover ordinary emission, object Holdout with opaque emission,
transparent and colored-transparent materials, transparent mixtures with
emission/diffuse, opaque-film and secondary-ray controls, and node Holdout with
constant and position-dependent mix weights. Disconnected nodes remain in the
authored input; they must not introduce reachable SVM cases.

The state oracle is [cycles_holdout_state.json](../../../../tests/data/cycles_holdout_state.json),
with input definitions in `tests/cycles_holdout_state_fixture.h`, original GPU
observer `tools/cycles_holdout_oracle.hip`, and 36 recorded rows in
`tests/data/cycles_holdout_state.txt`. It observes before-node, after-node and
after-Holdout state: 132 float lanes and 24 integer lanes per case. The original
observer uses HIP `gfx1201`, `-O3 -ffast-math`, and 32 compiler jobs. Its exact
command and binary/source hashes are in the JSON.

Every physical test slot receives sample-weight/normal sentinels before logical
count/left is established. Thus the observer checks preservation of defined
storage, not arbitrary values of fresh production Holdout fields. It covers
signed weights, zero/negative-zero mix, invalid/high stack offsets, no free
slots, full prefixes, excluded types, object/transparent/volume-only flags and
the native subtraction mask. Direct function-boundary controls are not all
claimed to arise from admitted scenes. This state test checks the handler's
typed-payload cursor advancement, not a complete `eval_nodes` PC/status trace.

Frozen evidence directory: `/var/tmp/psycles-holdout-CXlJR7`.

| Log | Observed result and scope |
| --- | --- |
| `red-hip-2.log` | Before repair, ordinary emission agrees; object Holdout disagrees and connected Holdout nodes are not lowered. Compiler comparison records missing node words. |
| `green-compiler-1.log` | 12 surface shader images, 267 words in total, zero word mismatches. |
| `green-render-hip-1.log` | 12 scenes x 2 schedulers = 24 distinct observations, 282,624 compared film lanes, zero mismatches. Focused CTest passes; 192.85 s is test-process wall time, not renderer performance. |
| `state-red-hip.log` | Original-GPU sentinel regression fails with 20 float mismatches across 36 cases/3 snapshots. The extra `sample_weight=0` store erases defined sentinels, e.g. `node-unlinked`: 0 instead of 0.125 after allocation and after Holdout. |
| `state-green-hip.log` | The same 36 cases/3 snapshots have zero mismatches after removing the generic clear. |
| `volume-red.log` | Pure-volume Holdout agrees at 7 words; Holdout+Emission and cross-domain sharing have 8 and 20 word mismatches before the typed alias repair. |
| `final-host.log` | 183/183 host tests pass, including all 15 exact original word images, per-entry usage masks and the three provenance guards. |

The allocator correction removes that generic store. All other direct
production callers (BSDF, BSSRDF, Transparent and Portal setup) explicitly
initialize sample weight before consuming it. Ordinary scalar/vector default
initialization is unchanged. The red witness was captured before removal;
the earlier green film log does not substitute for the post-removal state and
full-suite gates.

### Volume projection and static pruning

[Three original volume images](../../../../tests/data/cycles_holdout_volume/manifest.json)
come from a separate original-Cycles capture. They contain 7, 22 and 31 words:
pure Holdout on Volume is removed by native graph cleanup; a Holdout mixed with
valid Emission remains; a single Holdout shared by Surface and Volume retains
both declared input dependencies and one native node. All three now match
word-for-word. The contract-only `volume_holdout` socket alias projects back to
the same native HoldoutNode before cleanup; no new SVM node or generic type
conversion rule is introduced. The regression also checks the native closure
budget implied by these graphs. This is volume **compiler/admission** coverage,
not a full-volume transport or mixed-phase compaction certification.

The AST regression now observes 100 implemented handlers in the 110-tag
catalog: eight semantic gaps and two sentinels remain. Holdout has no native
node-feature/domain guard, including under a zero feature mask. In the twelve
surface fixtures, all authored graphs contain a Holdout node but only four
connect it: the other eight must not generate a Holdout case. Volume and
displacement entries of those surface fixtures retain only ShaderJump/End.
Static entry usage, not a render profile, determines these cases and stacks.

Log SHA-256, in the order compiler-green, film-green, state-red:

```text
dc68ad9ff46e64d52a48b6f9d368bceeeb6f77e142b42b3aac7b2074bf285d4d
8fbbf15e5bd6468bd637e47d2bc029b95631b808920b22fed8c0ce387791dfd6
fdbab06912f39ebaefa036a1ad010970d01e9ae72dbbb28422492444bb5d7fe7
```

## Historical full validation checkpoint (frozen SDK 6e)

The tested implementation starts from Psycles `f3f852aedbb4` with the Holdout
changes described here, on the unchanged published Luisa `6e58928d8460`.
The newer upstream tile merge is not part of this frozen checkpoint. The
all-target build uses all 32 hardware threads; host tests use 32 workers and
device suites run sequentially. Suite wall times are not renderer timings.

| Gate | Result | Evidence |
| --- | --- | --- |
| All-target build | Passed | `full-build.log`, recorded exit status 0 |
| Complete host selection | 183/183 | `final-host.log`, 4.67 s |
| Complete HIP selection | 192/192 | `full-hip.log`, 1029.85 s |
| Complete fallback selection | 194/194 | `full-fallback.log`, 617.61 s, recorded exit status 0 |
| Strict native Vulkan selection | **7/11, failed** | `strict-native-vk.log`, four compiler aborts, 84.28 s, recorded exit status 8 |
| Large-scene finite gate | **5/6, failed** | All six renders and all 15 pass comparisons complete; runner exits 2 |

The final HIP and fallback suites each include the post-allocator-removal
state regression and all 24 surface film observations, with zero mismatches.
The first
fallback attempt was interrupted without a terminal CTest result; its log is
`full-fallback.interrupted-215304.log`, not a successful suite observation.
The successful restart has its own `full-fallback.log`.

The failed Vulkan tests are `lamp_routing`, `zero_bsdf`, `holdout_render` and
`ray_portal_render` (the `psycles.luisa_cycles_*_vk` CTest names). Each aborts
with the same observed diagnostic from `restructure_cfg` output verification:
`Instruction operand does not dominate its use`, with `invalid=1`. This is
a compiler-verifier failure, not a completed image comparison. A shared
diagnostic does not by itself establish the formal root cause or a fix.

All eleven tests set `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and `LUISA_VULKAN_DISABLE_DXC=1`.
The `LD_DEBUG=libs` log contains 67 successful native SPIR-V compilation
messages and zero DXC/DXIL loader matches; this verifies the required route
but does **not** turn the 7/11 result into a pass. Reduction/repair and the
required rerun were outstanding at this frozen checkpoint; the later
[integration report](../next-integration/README.md) records the separate green
gate. No completed `results.json`
archive or all-backend-success claim is produced for this snapshot.

Terminal log SHA-256:

```text
full-fallback.log  17b44fd6e59865e160ed36a19b8580008c2b8413c939f17342db35763f18704f
strict-native-vk.log  06c502b2526b9be76bb6b05381f91efc5c42d45383d2b5c2443327a23908ec4e
```

### Four-scene, six-render HIP follow-up

The campaign is retained in
`/var/tmp/psycles-holdout-CXlJR7/canaries-pAJ8cR/canaries.json`. All renders
use 256 spp, staged scheduling and the unchanged 15-pass/46-channel contract
on the RX 9070 XT. Source geometry/image bytes match the earlier campaign;
socket metadata comes from its pinned controls. The comparison images are
retained equal-pass Cycles references, **not fresh timing pairs**. No build,
other device test, concurrent render or profiler overlapped these canaries.

| Scene / extent | Render seconds | Session init seconds | Main stages / fields / bytes |
| --- | ---: | ---: | ---: |
| Lone Monk / 1440x1080 | 12.4505 | 43.9887 | 4 / 50 / 200 |
| Monster / 1080x1080 | 13.7470 | 52.6500 | 6 / 68 / 272 |
| Classroom / 1920x1080 | 17.2434 | 38.3884 | 5 / 63 / 252 |
| Barbershop / 2048x858, run 1 | 37.8379 | 58.1651 | 6 / 92 / 416 |
| Barbershop, run 2 | 37.8313 | 19.1775 | 6 / 92 / 416 |
| Barbershop, run 3 | 37.8524 | 20.5389 | 6 / 92 / 416 |

Barbershop's median is **37.8379 s**, 49.1% above the retained Cycles median
of 25.3775 s. It is 0.91% above the previous S2 median of 37.4951 s; this is a
temporal comparison, not an isolated causal regression estimate. None of the
three observations is discarded. This repair establishes no renderer speedup.
The main coroutine frames are unchanged; their sizes alone do not measure
SVM local storage, register spills or frame traffic.

Session initialization includes shader JIT, setup and baking, not render time
or a compiler-only interval. Main shader caching is disabled; downstream and
OS caches retain normal policy. Barbershop's main HIPRTC link produces an
862,656-byte code object and takes 27,000.95 ms initially, then 92.18 and
108.99 ms. These observations localize the first/repeat delay but do not
establish a controlled cold-JIT comparison or a specific cache mechanism.

| First run versus retained Cycles | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: |
| Lone Monk | 0.01240804 | 0.12881600 |
| Monster | 0.00547855 | 0.02552525 |
| Classroom | 0.00353320 | 0.17820339 |
| Barbershop | 0.01052115 | 0.07126171 |

All 46 channels are finite in five renders. Classroom retains 18 invalid
channel values at eight unique pixels: seven invalid pixels each in DiffDir
and GlossDir, two each in DiffInd and GlossInd. Combined is finite. Original
Classroom has 25/27 invalid DiffDir/GlossDir pixels at different coordinates;
affected comparisons explicitly exclude and report the invalid union. A
completed comparison on that finite domain is not a passed all-finite gate.

`image-intervention.json` compares the S2 and S3 captures without rerendering.
Classroom's NaN/+Inf/-Inf masks are exactly unchanged; its Combined RMSE
between captures is 1.09e-9. Monster's corresponding RMSE is 6.48e-10, and
both Normal passes are unchanged. Barbershop's Combined RMSE is 2.17e-6.
Monk retains localized changes: Combined RMSE 5.49e-4 / maximum error 0.283,
Normal RMSE 1.47e-4 / maximum error 0.0535. The earlier S1 same-binary repeat
also exposed local Monk variation. Its cause remains unresolved: this is
neither proof of a Holdout-induced change nor a one-ULP/numerical-noise waiver.
No RNG, precision, inlining or sampling policy is changed in this repair.

## Reproduction and remaining work

From `/home/mike/Projects/Psycles-surface-svm`:

```bash
cmake --build build --parallel 32 --target \
  psycles_cycles_svm_holdout_tests \
  psycles_cycles_svm_holdout_volume_tests \
  psycles_luisa_cycles_holdout_state_tests \
  psycles_luisa_cycles_holdout_render_tests
ctest --test-dir build --parallel 1 --output-on-failure \
  -R '^psycles\.(cycles_svm_holdout(_volume)?|cycles_holdout(_state|_volume)?_fixture|luisa_cycles_holdout_(state|render)_hip)$'
```

The complete HIP/fallback suites and large-scene campaign are recorded above.
The frozen failed strict native XIR-to-SPIR-V Vulkan gate blocked completion;
its original-module replay and subsequent green gate are documented in the
[integration report](../next-integration/README.md). Vulkan must retain all three native guards from
[DEVELOP.md](../../../../DEVELOP.md); no DXC fallback is an acceptable pass.

`capture_results.py` archives only complete evidence: it reconciles exact
CTest selections, deduplicates repeated log observations, verifies original
fixture/source/binary identities and retains the failing finite gate.
`validate_results.py` recomputes that archive from the retained evidence.
Neither helper generates shader results, substitutes expected words or renders.
The frozen failed Vulkan gate is not eligible for that completed archive.

S5 remains open: the current preparation still computes eligible AOV reductions
before termination and does not reproduce every original pass-eligibility
predicate. The Barbershop performance/indirect-light gap, W1 ownership work and
the legacy displacement bridge are not solved by this Holdout repair. The
[previous continuation campaign](../native-surface-continuation/README.md)
and this campaign both retain Classroom separated-pass nonfinite values.
Whole-renderer correctness, repeatability and efficiency goals remain open.
