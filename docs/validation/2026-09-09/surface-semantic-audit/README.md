# Whole-surface SVM structural audit

The native interpreter is **not yet an end-to-end Cycles-identical surface
implementation**. This audit finds three production rendering failures beyond
node tests: premature termination at volume-only boundaries, missing Ray Portal
continuation, and missing object-holdout application. It also finds lossy path
label transport, incomplete feature integration, and several remaining
work-placement differences. Fixing another small arithmetic expression would
not close these contracts.

This is an audit and reproducible red baseline, **not a production repair or a
new performance result**. The audit captures use Psycles `287bc520` / implementation
`9e3ba165`, Luisa `8911828eb`. Source comparison uses Cycles
`cb168525138fecc792cc393f94afc39582b0103c`; the full-render witnesses use installed
Cycles 5.2.1 HIP build `9e2066aef7ef`. Neither original source nor either installed
Blender was modified. Its two inherited dirty source files retain their hashes.
[results.json](results.json) contains the source identities, all opcode rows,
four-scene census, commands and original/actual observations. Full local evidence:
`/var/tmp/psycles-surface-structural-audit-J9tiiH`.

## What was checked, and what the counts mean

The review follows the entire boundary sequence: scene admission and graph
compilation; geometry and ShaderData; SVM population; closure preparation;
holdout, emission and termination; data passes; NEE; closure selection and
sampling; path/ray updates; main/shadow scheduling. A handler's existence,
its unit-test result, and its production consumer are separate obligations.

The catalogs match all 110 native numeric tags. There are **99 implemented
dispatch handlers, nine missing semantic opcodes, and two sentinels**. This
does not mean 99 handlers or all scene contexts have been semantically proved.
`audit_sources.py` records case locations and called handler names; those are
navigation/presence evidence, not a fabricated coverage percentage.

Missing operations are `NODE_CLOSURE_HOLDOUT`, `NODE_RADIAL_TILING`,
`NODE_BEVEL`, `NODE_AMBIENT_OCCLUSION`, `NODE_RAYCAST`, `NODE_AOV_START`,
`NODE_AOV_COLOR`, `NODE_AOV_VALUE`, and `NODE_SCENE_TIME`. `NODE_NONE` and
`NODE_PAD1` are sentinels, not missing shaders. Object holdout below does not
use the missing Holdout node and therefore exposes a separate consumer gap.

All four current control bundles were recompiled with the production
used-shader collector and compiler. The diagnostic was rebuilt with 32 threads.
Counts below are serialized nodes by typed ShaderJump interval, including
END-only dense holes. They are **not executed node counts**, invocation counts,
or profile-derived allocation bounds.

| Scene | Compile units / dense slots | Words | Peak stack floats / closure slots | Surface opcode kinds / serialized nodes |
| --- | ---: | ---: | ---: | ---: |
| Barbershop | 280 / 490 | 895,648 | 33 / 12 | 45 / 9,729 |
| Lone Monk | 35 / 39 | 54,308 | 25 / 24 | 39 / 613 |
| Monster | 29 / 32 | 10,948 | 22 / 24 | 42 / 445 |
| Classroom | 90 / 91 | 68,504 | 23 / 12 | 34 / 1,428 |

ShaderJump itself is excluded from the last column. Bump fallthrough belongs
to the surface interval. Monster contains one ENTER/LEAVE pair; Barbershop
contains 58 SET_NORMAL nodes but no ENTER/LEAVE pair. Source-tree nodes and
exported images are separately recorded; disconnected authoring nodes and
unbound images must not be counted as reachable execution.

Barbershop's fresh dump is byte-identical to the
[resource-identity audit](../resource-identities/README.md), SHA-256
`1d179d2dad2cc91237f4301a1be6d0f1dd4357dcb817a476cecd3578fea03d10`.
That audit joins 279 used original/actual shader images: 115 raw-exact, 164
different only in declared resource-ID fields whose identities all match.
It checks all 600 image and 795 attribute references, without word
normalization. Compile units, dense slots and joined original images are
different denominators. These results constrain compiler structure, not
decoded resource values, every shader-state transition or every rendered path.
No fresh same-session original word/resource capture was done for the other
three scenes in this audit; their table is a current inventory, not that proof.

## Three full-render counterexamples

These use unchanged production HIP and original Cycles HIP, 16x16, one sample,
native fast math, and staged main/shadow queues. Inputs are generated with
Blender, exported normally and rendered by both engines. No expected shader,
sampler, volume integrator or pixel value is calculated by a host reference.
Source blends are hashed before/after; all outputs and process logs are kept.
These are deliberately tiny correctness witnesses, not timing benchmarks.

| Scene | Original Combined RGB | Actual Combined RGB | Result |
| --- | ---: | ---: | --- |
| One absorbing slab, ordinary/transparent limits zero | 0.939413 | 0.939413 | Control agrees |
| Two absorbing slabs, same limits | 0.882497 | 0 | **Fails** |
| Two slabs, transparent limit raised to two | 0.882497 | 0.882497 | Budget control agrees |
| Ordinary white emission | 1 | 1 | Control agrees |
| Same emission, object holdout enabled | 0 | 1 | **Fails** |
| Ordinary transparent plane, white world | 1 | 1 | Control agrees |
| Ray Portal plane, white world | 1 | 0 | **Fails** |

Each reported failure affects all 768 RGB lanes. Both final runners complete
their processes successfully but return **2** for the parity gate: boundary
controls 2/3 and surface integration 2/4. Both gate families reproduce the
initial failures. The gate uses 2e-6 absolute/relative tolerance, not bitwise
floating-point comparison. This is not an all-green renderer certificate or
an already-integrated CTest suite; the replay scripts preserve the red cases.

### S1. Independent volume-boundary lifetime is truncated

Actual [outer pipeline](../../../../src/luisa/path_kernel_pipeline.cpp) loops
over `max_path_steps`. Its pure-volume branch performs the native boundary
transition and `$continue`, consuming one outer iteration without advancing
ordinary or transparent bounce counters. However,
[cycles_path_step_limit](../../../../src/luisa/cycles_integrator_limits.h)
derives only `2 * synced_maximum + max(transparent_maximum, 1) + 1`, capped at
1024. The term for pure-volume crossings is absent.

Original `kernel/integrator/path_state.h::path_state_volume_next` advances an
independent `volume_bounds_bounce`, allows up to `VOLUME_BOUNDS_MAX == 1024`,
and advances RNG dimensions. With authored ordinary/transparent limits zero,
the actual outer limit is four: two closed slabs consume four boundaries and
the terminal background is lost. Raising only the transparent limit to two
allows five outer iterations and restores the image, although neither
renderer performs a transparent BSDF bounce. The original image is unchanged.
This isolates the cross-counter bound, not a second-box absorption failure.

The current host limit test repeats the same incomplete formula. The existing
single-boundary runtime test cannot prove composition with this outer loop.
A repair must derive termination from all original state domains, including
SSS entry/exit and terminal work; another scene/profile-specific cap is not a
valid fix. This counterexample does **not** demonstrate bound exhaustion in
the current Barbershop render.

### S2. Portal population has no portal bounce consumer

Original `integrate_surface_bsdf_bssrdf_bounce` dispatches selected BSSRDF,
then Ray Portal, then ordinary BSDF. `integrate_surface_ray_portal` owns portal
P/D, selection-mass weighting, self identity, ray range/differentials and
`LABEL_TRANSMIT | LABEL_RAY_PORTAL`; `path_state_next` also updates the portal
counter when SD_RAY_PORTAL is present.

Actual [surface population](../../../../src/luisa/path_tracer_cycles_svm_surface.cpp)
handles BSSRDF versus ordinary BSDF only. The ordinary
[BSDF dispatcher](../../../../src/luisa/cycles_svm_bsdf.cpp) deliberately has
an empty portal case, since native ordinary sampling must never receive it.
The production path nevertheless sends it there and ends without reaching
the world. The path-state projection also lacks persistent `portal_bounce`;
the surface SVM context receives zero.

The existing standalone portal test checks original word evaluation, closure
payload and PC/status, **not a selected portal's production continuation**.
The new witness is admitted without error and fails after that boundary.
None of the four benchmark programs contains a portal closure, so this is
not evidence for their current timing or DiffInd gap.

### S3. Packed object holdout is never applied to the surface

`use_holdout` is exported, imported and packed into native KernelObject flags.
The witness's exported object flag is explicitly true; this is not a missing
export socket or use of an unsupported Holdout node.

Original `integrate_surface_holdout` runs after closure preparation and before
emission. `surface_shader_apply_holdout` consumes SD_OBJECT_HOLDOUT_MASK,
preserves transparent contributions, invalidates other closures, writes film
holdout, and terminates a fully held-out path. The actual surface performs
none of that operation. It writes the plane's emission, producing white where
the original produces zero. The scene is silently admitted.

All four benchmark bundles have zero holdout **geometry instances**. This
defect is not an explanation of their current discrepancy. Adding only the
missing Holdout opcode would leave this object-level defect intact.

## Other confirmed structural gaps and incomplete boundaries

| ID / boundary | Original contract versus current production | Current four-scene exposure / proof limit |
| --- | --- | --- |
| S4: Native label roundtrip | `events_from_label` maps seven native bits into SurfaceEvent; `label_from_events` cannot restore TRANSMIT_TRANSPARENT or RAY_PORTAL. The former controls TRANSPARENT_BACKGROUND after ordinary transmission. KernelGlobals declares the transparent-roughness service, but the production subclass does not override its -1 default and export does not carry the glass/roughness controls. | Source-proved lossy mapping and missing service; not an additional full-render witness here. All four original scenes have transparent-glass disabled, including transparent-film Classroom, so this cannot currently explain their DiffInd. |
| S5: Data-pass eligibility and placement | Original film data passes use TRANSPARENT_BACKGROUND, SHADOW_CATCHER_PASS, PASS_ANY and SINGLE_PASS_DONE, after termination. Actual preparation uses ordinary depth==0 and happens before emission/termination; the normal writer selects zero instead of guarding the atomic. Roughness is also constructed even when not requested. | The depth predicate coincides for current ordinary paths with transparent-glass disabled, but is not the native general contract. Normal/color work is present; whether unused roughness survives final DCE is **not proved**. |
| S6: Shadow-catcher integration | The object bit is packed and some visibility fields exist. A complete native shadow-catcher path split/film consumer is absent from the reviewed production pipeline. This must not be described as implemented from the flag alone. | No geometry instance uses shadow catcher in these four bundles. Some **lights** carry this flag; they are not shadow-catcher geometry. No dedicated full-render catcher witness in this audit. |
| S7: True displacement | Production surface, volume, world and light use native SVM, but the geometry displacement prepass still consumes a private SurfaceProgram/legacy value graph with an inert surface root. Native displacement opcodes do not remove that bridge. | Confirmed dependency in `path_tracer_scene.cpp` / `path_tracer_displacement_scene.cpp`; removal remains required. Monster exercises BOTH bump-state, which is not proof of a fully native geometry prepass. |
| S8: Image/geometry services | The nominal UDIM service performs ordinary 2D sampling without a tile mapping. 3D image interpolation is unreachable in the production provider. Static ribbon curves are admitted; motion objects are explicitly rejected. Point helper availability is not point-cloud scene support. | Original four scenes contain no TILED images, volume-grid objects, point-cloud objects or motion blur. No general UDIM/point-cloud admission claim is made; explicit motion and light-linking rejection are distinguished from silent consumer gaps. |
| S9: Other native configurations | Closure preparation implements blur, not the native filter_closures mask used by baking controls. Native MNEE-specific shading/state and full native film/AOV/denoising interfaces are not established here. Path guiding's CPU-only domain must not be charged as missing HIP work. | Native filter-mask producers inspected are baking operations, not ordinary benchmark rendering. Four original scenes have no caustic caster/receiver/light flags and no Fast GI/guiding. Authored AO passes and Monk denoising do not define the equal-pass workload: that runner explicitly disables source-only passes/denoising. |

These are obligation groups, not disjoint counts of bugs or an exhaustive
proof of every accepted scene configuration. In particular, the nine missing
opcodes, portal **consumer** failure and partial object integration must not
be collapsed into a single node-coverage percentage.

## Work-placement differences relevant to Barbershop

The following are source-level operation/ownership differences. They need
original execution predicates, production-AST/final-IR confirmation, and
whole-scene controls before attributing time. An eager expression can be
eliminated or sunk by the compiler; a source branch count is not GPU traffic.

| ID | Actual production versus original Cycles | Required next evidence |
| --- | --- | --- |
| W1: Geometry roundtrip | `path_kernel_cycles_svm_surface_geometry.cpp` builds native ShaderData, reads additional vertices/normals/material metadata and fills SurfacePoint; population reads shader/object flags and reconstructs ShaderData. Native integrates from one shader setup. Native ray-offset/terminator consumers fetch some geometry only when needed. | Per-field ownership/liveness map and final load use-def chains; do not assume every source read survives CSE. |
| W2: Data reductions | Preparation computes data-pass reductions before the original termination boundary, shares one include_aov predicate for colors/normal/roughness, and later writes a zero normal when SINGLE_PASS_DONE. | Original predicates and actual AST side effects; distinguish genuine normal atomics from dead unused roughness. Native color passes intentionally repeat through transparent surfaces. |
| W3: First-bounce weights | Scatter calculates both BSDF component ratios before `records_first_surface`; native ordinary bounce computes them inside bounce==0. BSSRDF has a separate native route. | Production AST control dependence and later-bounce observations. Keep the native zero-only division rule; no restored epsilon cutoff. |
| W4: Transparent ray continuation | Actual scatter normalizes a sampled direction, calculates the triangle self-intersection/ray offset and differential widening, then selects the unchanged transparent ray. Native's transparent branch only updates tmin and skips that construction. | Predicate-aware AST/IR; Barbershop contains transparent closures, but their dynamic cost is unmeasured. |
| W5: Random-tuple lifetime | Actual surface BSDF random tuple is constructed before NEE and before the sampler's SD_BSDF/SD_BSSRDF rejection. Native ordinary bounce rejects before requesting its tuple. Light-termination random is also constructed for every eligible NEE context. | Actual optimized control dependence and lifetime; sampling is counter-based, so movement alone is not changed RNG sequence. |
| W6: Forward MIS | The SD_EMISSION/SSS gate is repaired, but the inner forward-emission operation still calculates selection/triangle PDFs before selecting no competition for primary or MIS_SKIP paths. Native forward MIS rejects those paths first. | Original boundary states, side-effect-free work dominance and full original shader validation. |
| W7: NEE proposal finalization | A constant-proposal finalizer is called before selecting by proposal kind. Native constant/non-constant shader and light-tree modes have distinct termination predicates. | Prove which calls remain after JIT specialization before asserting redundant device execution. |
| W8: Lamp inverse ownership | Actual analytic consumers rebuild inverse linear transforms with cross products/determinant/reciprocal. Native lamp inverse access reads the packed object transform. | Native host table plus original GPU transform states; no isolated inverse microbenchmark presented as a renderer gain. |

Separate numerical-domain differences also remain: the analytic proposal PDF
has a `1e-20` floor, its direction length uses a related cutoff, inverse
validity has an epsilon determinant test, and actual scatter rejects negative
or NaN throughput beyond native ordinary PDF/zero-BSDF checks. These need
original boundary-domain analysis. They are not reasons to add bit-matching,
slow division, or changes to native fast math, nor proved causes of current
Barbershop DiffInd.

The existing AOV loop fusion is not itself a defect: matching several native
reductions in one ordered traversal need not mean duplicating their loops.
Likewise, native pointer identities represented by exact closure indices are
not an alternate closure model. The obligations are allocation, initialized
prefix, ordering, control predicates and state consumers.

## Aligned boundaries and rejected performance hypotheses

| Boundary | Evidence / limitation |
| --- | --- |
| Word/typed stack/PC | Native scalar-uint variable-size payload stream and ShaderJump/PC loop are retained. Stack loads use native typed-input/invalid-offset rules, not per-node bounds. The old uint4 comment in Cycles is not a different current wire format. |
| Static pruning | Host `if (node_types_used[tag])` occurs before `$case`; native feature masks further gate bodies and closure consumers. Release surface uses `eval_nodes_assume_valid`, END return, and no diagnostic active/status lane. Light/shadow diagnostic adapters still differ and need their own contract review. |
| Storage | Scene compiler supplies static stack and native closure bounds. Retained Barbershop optimized LLVM has `[33 x float]` stack and `[60 x [4 x i32]]` closure storage: twelve 80-byte slots. Local lifetime-only storage is not cleared; ordinary scalar/vector zero initialization is unchanged. |
| One material population | NEE and continuation consume the same populated surface; they do not independently re-run the SVM. SSS exit skips material evaluation without bump and replaces closures after its required bumped evaluation. Existing original regressions constrain this domain, not portal/holdout integration. |
| Ordinary mixture sampling | Ordered closure mass/pick, random rescaling, seeded MIS complement reduction, BSSRDF mass and shared case groups match the reviewed native formulas. Extra consumer validity guards and portal dispatch remain separate findings. No closure deduplication is proposed. |
| Bump / resource / handler families | Existing original-word/GPU fixtures cover bump state, derivatives, attributes, procedural and image families, Principled/standalone closure families and volume. They are scoped historical evidence, not re-certified exhaustive semantics from this inventory. Full catalog navigation is in results.json. |
| Main/shadow scheduling | The measured Barbershop log has `direct_light_queue=true`; surface NEE publishes a detached task before BSDF continuation. It is not waiting for serial shadow completion. The six-stage / 93-field / 416-byte main frame is distinct from SVM scratch and from the auxiliary shadow task. Physical stage counts need not equal native kernel count. |
| Inlining | Main SVM, 3D Noise and microfacet bodies were already inlined. The retained object has ordinary 1D/4D Noise and OCML tan helpers. Existing inlining/launch controls were reverted when slower; no new inline/noinline policy is introduced. |
| Texture path | HIP already emits native image-sample intrinsics with device-resident descriptors; ordinary BYTE4 is not expanded to FLOAT4. Bicubic uses four native bilinear samples. Generic packed-format branches can add static code, but are not the ordinary Barbershop sampler path. No ROCclr hack is justified by this audit. |

The [static code audit](../surface-static-audit/README.md) explains why comparing
only native and actual main-function sizes is invalid: their outlined work and
scene specialization differ. Static scratch sites, register counts and
function-address materializations do not establish dynamic traffic, occupancy
loss or instruction-cache stalls. The previous unsupported hardware-counter
attempt supplies no such attribution.

## Performance and correctness status remain open

The latest [controlled film-routing experiment](../film-routing/README.md)
still has Barbershop surface about **1.83 times** retained original Cycles,
and the six-render 256-spp follow-up has a 38.4822-second Barbershop median
versus retained native 25.3775 seconds. These are previous observations, not
new timings obtained by this audit. Surface visit counts are already nearly
equal; that does not prove identical execution within a visit or global
RNG/path parity.

The three reproduced defects are important conformance failures, but none
has been shown to explain the current Barbershop gap. There are no portal or
holdout instances there, and no observed outer-bound exhaustion. Its fog is
emission-only with zero volume bounces. Raw source node/image counts and
flags on lights must not be used to argue those missing features are active.

Residual DiffInd remains unresolved. So does Classroom's latest separated-pass
finiteness failure: 18 invalid lanes at eight pixels. The original tiny-weight
GPU observer supports a shared fastmath arithmetic boundary, not per-path
equivalence or a waiver of that gate. Nothing in this audit replaces those
NaN/Inf values, changes tolerances for a full-scene pass, or claims parity.

## Completion order and replay

Work is organized around native boundaries, not whichever expression is easy
to shrink:

1. Preserve these full-render reds as permanent original-oracle regressions;
   repair independent path lifetimes and native portal/holdout consumers.
2. Remove lossy native-state projection and bind missing scene controls;
   test labels, flags, counters and film predicates through production callers.
3. Audit one ShaderData/closure lifetime through geometry, prepare, NEE and
   scatter; restore native work predicates in W1-W8 with proof of the actual
   generated operations, not forced inlining or guessed compiler behavior.
4. Close the nine opcode/admission gaps and remove the private legacy
   displacement consumer. Disabled or unsupported domains must be explicit.
5. At each compiler change: formal cause, minimal red, permanent regression,
   generic fix, full original module, HIP then fallback and strict native
   Vulkan. At each performance claim: isolated full-scene controls, followed
   by the original multi-scene / 15-pass campaign and same-path diagnosis.

The two replay commands below each use a **fresh** output directory. Until
the corresponding fixes, both correctly exit 2. They launch only original
HIP and current production HIP; inspect results rather than calling successful
process completion a pass.

```bash
python docs/validation/2026-09-09/surface-semantic-audit/run_boundary_probe.py \
  /home/mike/Projects/Psycles-surface-svm /var/tmp/psycles-boundary-fresh \
  --blender /home/mike/Projects/blender-install-5.2-hiprt/blender

python docs/validation/2026-09-09/surface-semantic-audit/run_boundary_probe.py \
  /home/mike/Projects/Psycles-surface-svm /var/tmp/psycles-surface-integration-fresh \
  --blender /home/mike/Projects/blender-install-5.2-hiprt/blender \
  --surface-integration
```

`read_blend_settings.py` opens original files read-only in background Blender;
it does not render, save or mutate their settings. `audit_sources.py` takes the
repository, original `intern/cycles`, four control-bundle root, native typed
layout JSON and evidence root. It produces the full mechanical inventory.
`archive_audit.py` takes the first two roots and evidence root and emits this
compact publication snapshot. Layout/decoder hashes, complete commands and
source locations are retained. No production source changes or new backend
fixes are included in this audit checkpoint; prior complete-suite results
remain prior results, and the new full-render gates are explicitly red.

## Publication SDK integration check

Upstream advanced while this audit was being prepared. The publication retains
Psycles `4ad112d1` and its Luisa `03a0f5158` gitlink, including the separately
published CFG loop-epoch and swizzle-reference fixes. No SDK fix is attributed
to this audit. The original seven witness captures above remain pinned to
`8911828eb`; they were not relabeled as new-SDK measurements.

The rebuilt SDK was checked separately, without overlapping GPU workloads:

| Gate | Result |
| --- | --- |
| All-target build, 32 threads | Passed |
| Complete host CTest, 32 jobs | 176/176, 1.28 s |
| Focused HIP, serial | 8/8, 10.42 s |
| Focused fallback, serial | 8/8, 7.58 s |
| Full-input Barbershop HIP, 2048x858 / 256 spp | Completed; all 46 channels finite; main frame 6 stages / 93 fields / 416 B |
| Strict native Vulkan, serial, loader audit | 8/8, 14.85 s; native SPIR-V compilation observed, no DXC/DXIL library initialization |

The focus set is film routing, surface emission, shared closure, subsurface
exit, native shadow, bump state, closure pool and standalone Ray Portal.
Vulkan uses `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`,
`LUISA_VULKAN_DISABLE_DXC=1`, and `LD_DEBUG=libs`.
These are SDK integration gates, **not a new full backend suite or a repair
of the three full-render failures**. The standalone portal test is still
subject to the consumer limitation documented in S2.

The single Barbershop canary observed 17.705 s shader initialization and
38.4109 s rendering with the main shader cache disabled. It is neither a
fresh original/actual timing pair nor an estimate of an SDK speedup. Its
commands, binary identities, input controls and gate logs are archived in
[sdk-integration.json](sdk-integration.json). `run_sdk_canary.py` replays this
check from the reviewed baseline and socket-control roots, using a fresh
empty output directory.

The previous six measured binaries were copied before rebuilding, into
`/var/tmp/psycles-surface-structural-audit-J9tiiH/pre-integration-bin-Kt9aOf`.
This is an identity snapshot, not a complete relocatable SDK. The integrity
validator rechecked **158 captured file identities and all seven image pairs**:

```bash
python docs/validation/2026-09-09/surface-semantic-audit/validate_audit.py \
  --live --captured-binary-root \
  /var/tmp/psycles-surface-structural-audit-J9tiiH/pre-integration-bin-Kt9aOf
```

An integrity pass deliberately preserves the four agreeing controls and
**three renderer parity failures**; it does not turn a red renderer gate green.
