# Independently scheduled main and shadow paths

## Scope and reference

Reference: Cycles 5.2.1 source at
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`.

The previous coroutine path retained the shadow task and its traversal batch
while the main path waited for the entire shadow loop. The existing optional
side queue released the main path, but fused NEE, traversal, all transparent
shading, and film output into one consumer kernel. Neither represented Cycles'
independent main state and independently scheduled shadow stages.

This change replaces that optional fused consumer with three stages sharing
one shadow-state allocation. `staged_direct_light_queue` remains the selection
flag; its enabled path now means staged independent work, not a fused consumer.
No SVM word, dispatch, feature-mask, closure, or initialization semantic changes.
No application `noinline` annotations are introduced.

## State and ownership

Each valid surface proposal appends exactly one `DirectLightTaskCall`. The
original main coroutine continues independently; the side task holds its own
pixel, path state, RNG, ray, and throughput, never a pointer to a main frame.

- Nonconstant lights enter SHADE_LIGHT_NEE. Rejection terminates; acceptance
  commits finalized throughput and transitions to INTERSECT_SHADOW.
- Constant lights bypass NEE and enter INTERSECT_SHADOW directly.
- Opaque traversal terminates in INTERSECT_SHADOW without scheduling shading.
- Unblocked traversal publishes its complete typed four-hit batch and queues
  SHADE_SHADOW, including the empty-hit film-output case.
- SHADE_SHADOW uses the existing production evaluator. A full transparent batch
  writes back ray minimum, throughput, transparent depth, RNG offset, and volume
  bounds before returning to INTERSECT_SHADOW. Visible termination atomically
  writes film and light passes once. Opaque shading writes no contribution.

The persistent task and the edge-only traversal batch use separate SoA storage.
Each stage writes only state it actually modifies. Only a SHADE_SHADOW-owned
slot has a live batch; compaction never reads batch storage from other stages.
Buffer allocation does not clear task or batch payloads: full producer writes
and the ownership token establish validity, while ordinary DSL scalar/vector
default initialization is unchanged.

Stage counts sum to live occupancy. Transitions retain a slot; termination
releases it. The Luisa stage-aware auxiliary protocol, published as
`a6bb5a7e7` on `origin/next`, admits complete main producer batches against the
shared capacity and competes individual shadow queues against main queues.
Capacity-pressure priority is NEE, INTERSECT_SHADOW, SHADE_SHADOW.

The Psycles submodule gitlink is deliberately not staged from the dirty Luisa
checkout. This checkpoint requires the published stage-aware Luisa API above;
the old pinned Luisa revision alone does not provide it. Completing the clean
dependency pin remains part of integrating the outstanding Luisa work, not an
implicit inclusion of those unrelated changes here.

Append allocation uses in-place compaction: live suffix items move into holes
below the live-prefix bound. The two address ranges are disjoint, so no second
payload array or inter-thread ordering is needed. Counts and semantic states
are unchanged by relocation.

## Remaining scheduling differences

This checkpoint compacts when an admitted producer would exhaust the append
tail. Cycles' host scheduler instead applies its factor-two / 32-slot compaction
heuristic before testing append-space admission. Main/auxiliary equal-count
ties still follow the existing Luisa registration policy, not a globally
declared Cycles DeviceKernel ordering. These are explicit remaining alignment
items; this checkpoint does not claim exact host scheduling parity.

Renderer-specific policy belongs in Psycles' Coro Ext/Handler integration.
Future queue priorities, compaction thresholds, and phase-state bindings must
not become Cycles-specific branches or settings inside Luisa's generic
scheduler/compiler. The shared-capacity stage protocol is renderer-neutral;
it does not encode Cycles stage names, device-kernel IDs, or compaction ratios.

## Regression and validation

Evidence directory: `/var/tmp/psycles-independent-shadow-cGjf52`.

`tests/test_luisa_cycles_shadow_pipeline.cpp` applies the genuine Cycles HIP
queue oracle in `tests/data/cycles_shadow_queue.txt` to the production side
pool. Prepared hit batches isolate scheduling from geometry/SVM, which retain
their separate native-shadow oracle regression.

The test covers 67 main paths through 19- and 32-slot pools, two publications per
path, reset/reuse, constant-light bypass, deferred NEE acceptance/rejection,
opaque traversal, shading extinction, transparent four-hit retraversal, exact
main and per-shadow-stage counts, and exact film/pass contributions.

A forced relocation case starts with 19 allocated slots, terminates six opaque
paths, leaves four NEE and nine SHADE_SHADOW tasks live, compacts, and publishes
six new tasks. Checking every final pixel detects loss of either invariant
state or the shade-owned traversal batch.

- Baseline HIP test failed because the old consumer exposed only one stage.
- New production pool and forced-relocation test passed HIP and fallback.
- Cache-disabled Vulkan passed with all three strict native-XIR/no-DXC guards
  and `LD_DEBUG=libs`; the log contains native SPIR-V compilations and no DXC or
  DXIL library loads. Optional libudev/loader symbol probes in that diagnostic
  log are unrelated to shader compilation and did not fail the test. The
  cache-disabled run recorded 35 native SPIR-V compilations.
- Before the renderer integration, the generic scheduler change passed the
  full original module: HIP 156/156 and fallback 158/158.

## Lone Monk HIP canary

Same frame-4 export, 1440x1080, 256 fixed spp, seed 0, RX 9070 XT. All runs use
the normal runtime library path; no relocated `LD_LIBRARY_PATH` harness. Each
run disables shader caching and records kernel traces, final LLVM IR, and ISA.
Only the last CLI argument (`staged_direct_light_queue`) changes.

| Sequence | Independent shadow | Render seconds | Evidence directory |
| --- | --- | ---: | --- |
| B | enabled | 14.3667 | `/var/tmp/psycles-independent-shadow-monk-8ateSs` |
| A | disabled | 16.7837 | `/var/tmp/psycles-independent-shadow-control-ESgOPk` |
| B | enabled | 14.3797 | `/var/tmp/psycles-independent-shadow-repeat-UEztNi` |

Mean enabled time is 14.3732 seconds: 14.36% less render time, or 16.77% higher
throughput, than the intervening control. This is an initial B/A/B measurement,
not a broad benchmark suite. Relative to the latest matched Cycles HIP sample
interval, 13.831431 seconds in `/var/tmp/psycles-shadow-queue-cycles-aBpy7U`, it
is still 3.92% slower. The performance goal is not yet reached.

Fresh paired control versus the second enabled run (GPU kernel duration):

| Stage | Control dispatches | Independent dispatches | Control seconds | Independent seconds | Independent VGPR / scratch / block |
| --- | ---: | ---: | ---: | ---: | --- |
| surface | 1495 | 1402 | 8.467394 | 6.863300 | 256 / 3328 / 512 |
| closest intersection | 2048 | 1403 | 3.273518 | 3.049294 | 128 / 240 / 32 |
| light NEE | 1258 | 1072 | 0.798302 | 0.371960 | 256 / 1360 / 32 |
| shadow intersection | 1213 | 1071 | 0.957553 | 0.818636 | 160 / 400 / 32 |
| shadow shading | 390 | 384 | 0.264568 | 0.220418 | 256 / 1856 / 32 |

The repeated enabled surface kernel is `kernel_f2d9275be2883246`; its launched
lanes are 819522048 versus 819547136 in the paired control. Most of the gain is
not eliminating surface work: the main path no longer waits for shadow stages,
its batches coalesce, and its continuation state is smaller. This architectural
change also changes generated surface IR, unlike the preceding opaque-shadow
termination checkpoint. No attribution to one machine instruction is claimed.

The main coroutine has 55 fields / 220 B, including its seven reserved fields;
its payload is 192 B. Previously it had 89 fields / 448 B. The separate shadow
pool uses 192 B task SoA plus 108 B traversal SoA per capacity slot, plus queue
storage. Therefore equal-capacity main+shadow state is **520 B**, not 220 B and
not half the previous total allocation. Cycles' measured 172+224 B is still
smaller; shadow phase-state reuse and native field layout remain to be aligned.

All images are compared against the same Cycles multilayer EXR with exact
Blender build metadata verification. Combined relative RMSE is 0.0111218351
and 0.0111332182 for the two enabled runs, versus 0.0111417695 in the control.
The second enabled run's DiffInd / GlossInd relative RMSE remains
0.1410585097 / 0.1619815470. Existing indirect-light discrepancies are not fixed
by scheduling. Full pass metrics are retained in each run's `compare.json`.

All eight compared passes have zero invalid actual pixels in each run. The
two enabled surface IR dumps are byte-identical, SHA-256
`3007b17ff2bd66282206d5215d6fd36bd99936b53bb8dce8797614a216d83f73`.
The files are `hip_kernel_final_12.ll` in the first run and
`hip_kernel_final_9.ll` in the repeat: map dumps by kernel name, never by the
parallel compilation sequence number.

## Final checkpoint validation

- Full 32-thread build: `post-integration-build.log` passed.
- Full HIP backend CTest: **157/157**, 257.02 seconds.
- Full fallback backend CTest: **159/159**, 47.55 seconds.
- Core/adapter selection: **100/100**, 11.74 seconds. The pre-existing
  `blender_export_render_settings` failure is excluded, exactly as in the
  preceding checkpoint; this is not a claim that every repository CTest is
  green.
- Strict native-XIR Vulkan selection: **13/13**, 1.38 seconds.
- Isolated candidate contains only these Psycles files on 15cfa4c9, and built
  359 steps with all 32 threads. Its shadow-pipeline, shadow-queue, native-shadow,
  direct-lighting-plan, and surface-queue tests passed **5/5 HIP + 5/5 fallback**.
  `ldd` confirms its Psycles core/runtime libraries come from the isolated build;
  Luisa headers and libraries come from the active workspace. It does not
  validate an unchanged old Luisa gitlink or silently include pending Psycles
  path-tracer changes.

The tested active runtime library SHA-256 is
`35a81ef1273199441fddd8fa22389f767d1fa689cb09d6005a76e9e133c31504`.
The complete SVM/renderer and performance goals remain active, including
indirect-image parity, native phase-state packing, exact scheduler priorities
and compaction policy, and outstanding domains/geometry support.
