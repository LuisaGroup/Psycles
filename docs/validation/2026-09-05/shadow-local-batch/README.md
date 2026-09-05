# Coroutine-owned shadow hit batches

## Storage and lifetime proof

Production shadow traversal previously wrote its nearest four hits into an
invocation-indexed external SoA, returned a three-word summary, materialized
the hits in the caller, and copied the batch into the coroutine frame before
`SHADE_SHADOW`. The external SoA also allocated three summary words that its
read/write methods never accessed. Both allocations were sized to the full
wavefront capacity.

The existing local collector and the external collector use the same
`ShadowBatchReducer`, candidate callbacks, identity filtering, transparent
bounce bound, and nearest-hit replacement rule. The local collector now also
implements the split traversal component. Its batch is fully defined before
returning: accepted hits occupy the retained prefix and the other entries
are explicit misses. The caller still suspends with the complete batch as a
value. Coro lowering alone owns its cross-stage storage, so compaction and
queue reordering cannot change its owner. No physical-lane pointer or index
escapes the producer invocation.

The storage policy is selected in C++ while recording DSL. There is no new
device branch, `noinline`, altered scalar/vector default initialization, or
SVM/closure/feature-mask change. The optional external-storage component and
its original ABI/ownership tests remain available. Production coroutine
executors no longer allocate, retain, or bind the duplicate buffer.

Cycles 5.2.1's `kernel/integrator/intersect_shadow.h` and
`kernel/integrator/shade_shadow.h` retain the intersection batch in the
integrator state across these stages. This change removes a second Psycles
storage layer; it does not introduce a new shadow algorithm or CPU oracle.

## Regression

`psycles_luisa_scene_traversal_tests <backend> local` uses the production
local component. Before the change this fails on HIP because the factory
returns no split intersection component without external storage. The
ordinary invocation continues to verify the compact external-summary ABI.

Both variants exercise the same 257-lane, 256-thread-block coroutine, with
even/odd logical IDs first entering different queues. Each queue contains
empty, partial, full, and overflowing batches (0/2/4/6 intersections). Tests
check the pre-suspend summary, retained hit identities and distances, miss
entries, and logical ID after compaction and resumption. Existing traversal
tests additionally cover source/light exclusion, opaque termination, budget
termination, nearest-four selection, and continuation. No assertion in the
external-storage regression is removed.

## Matched HIP measurements

Lone Monk frame 4, 1440x1080, 256 spp, fixed seed/sample interval, native
Cycles SVM, staged wavefront. The experimental A/B switch only selected the
storage policy and was removed after comparison. No other GPU test or build
ran during these measurements. Timings include the pre-existing uncommitted
renderer/compiler work; they do not describe the isolated commit alone.

| Measurement | External storage | Local batch, run 1 | Local batch, run 2 |
| --- | ---: | ---: | ---: |
| Render elapsed | 20.0096 s | 19.7419 s | 19.7875 s |
| Surface | 12.422039 s | 12.427999 s | 12.425433 s |
| Shadow traversal | 0.990622 s | 0.831334 s | 0.833869 s |
| Shadow shading | 0.381634 s | 0.329826 s | 0.329983 s |
| Shadow traversal VGPR | 176 | 144 | 144 |
| Shadow traversal private scratch | 288 B | 400 B | 400 B |
| Shadow traversal code object | 48,688 B | 43,312 B | 43,312 B |
| Coro frame fields / AoS size | 105 / 440 B | 90 / 456 B | 90 / 456 B |
| Coro frame actual SoA per slot | 436 B | 452 B | 452 B |
| Separate hit buffer per slot | 108 B | 0 B | 0 B |
| Combined persistent state per slot | 544 B | 452 B | 452 B |

Shadow traversal and shading each launch 1,194 dispatches. Their rounded
lane totals are 412,421,408 for external storage and 412,421,184 for local
storage, a difference below 0.0001%. The local version exchanges more private
scratch for fewer VGPRs and no external hit-buffer traffic. Traversal is
about 16% faster and end-to-end rendering about 1.1-1.3% faster in these runs.
The saved 92 B per slot is exactly 92 MiB at the current 1,048,576-slot
capacity. Frame field count alone would have been a misleading metric.

The final default-path canary, after removing the experimental switch and
external-buffer lifetime machinery, completed in 19.6794 s. Its surface,
shadow traversal, and shadow shading totals were 12.318937 s, 0.833398 s, and
0.330694 s. The shader identities and frame layout match the local A/B
variant. Combined relative RMSE versus Cycles is 1.1151456%, with zero
nonfinite pixels. The additional surface timing variation should not be
attributed to this shadow-only change; about 1% is the conservative overall
gain supported by the matched comparison.

Cycles' corresponding main+shadow SoA fields use 396 B per equal-capacity
slot pair for this feature configuration (172 B main, 224 B shadow). Queues,
sorting storage, registers, and private scratch are not included in either
persistent-state comparison. The remaining state difference is 56 B, not
evidence of a proportional speed difference.

The main performance gap remains surface shading: Cycles launches
819,379,200 lanes in 5.949119 s; Psycles launches 819,494,400 in 12.425433 s.
The surface workload is essentially the same size; its execution cost is
still about 2.09 times Cycles. The overall performance goal is not reached.

Local run 1 versus the Cycles HIP oracle has Combined relative RMSE
1.1146785%, with zero nonfinite pixels. Local run 2 versus the matched
external-storage render has Combined relative RMSE 0.05453%, Normal 0.02660%,
and DiffCol 0.000586%, with zero nonfinite pixels. Exact pixel equality is not
claimed.

Evidence: `/var/tmp/psycles-shadow-local-profile-L70lIa` contains the three
profiles, code objects, local LLVM IR, EXRs, and comparison JSON. The isolated
candidate is `/var/tmp/psycles-shadow-local-clean-mr4sF3/source` and contains
only this change, excluding all previous dirty renderer edits and the Luisa
gitlink. It builds against the existing Luisa headers/libraries.

## Validation

All builds use 32 threads.

- Complete active-tree build passed, including a final rebuild after cleanup.
- All 40 compiler/SVM and scheduler host tests passed.
- Active-tree HIP suite: 142/143 passed. Fallback suite: 144/145 passed.
  Both fail only the legacy graph-surface `luisa_principled_thin_wall` test
  (thick/thin subsurface dispatch/AOV). Native SVM thin-wall tests pass.
  The legacy test passes in both the isolated pre-change tree and this
  isolated candidate; the failure occurs with the existing dirty worktree
  changes, which are not included in the candidate. Full-suite green is
  explicitly not claimed.
- Isolated candidate: all production libraries and the renderer built in
  352 steps. Three focused HIP tests passed (external and local traversal,
  plus legacy thin-wall); four fallback tests passed, additionally including
  the film/path-trace comparison across every scheduler.
- Both traversal tests passed with `LUISA_VULKAN_USE_XIR=1`,
  `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
  `LUISA_VULKAN_DISABLE_DXC=1`.
- Both HIP traversal tests passed again after the final default-path render.
- The normally fallback-only all-scheduler film test was also invoked on HIP.
  It passes in both the isolated pre-change tree and the isolated candidate.
  In the active dirty tree it reports
  pass 8 value 3 as 0.644753 versus 0.644790, exceeding the unchanged 2e-5
  tolerance. This additional diagnostic is not silently treated as passing.
- The Cycles/Psycles/absolute-difference triptych was inspected. Geometry and
  large-scale illumination align; amplified residual differences remain.

Suite logs are `/var/tmp/psycles-shadow-local-suite-{core,hip,fallback}.log`;
the native Vulkan log is
`/var/tmp/psycles-shadow-local-test-native-vulkan.log`. Isolated results are
in the candidate directory's `ctest-hip.log`, `ctest-fallback.log`, and
`all-schedulers-hip.log`. The active-tree manual HIP diagnostic is
`/var/tmp/psycles-shadow-local-all-schedulers-hip.log`.
