# Native Holdout and joint surface batching: first integration evidence

The integrated renderer now batches compatible incoming surface routes into
one sorting/resume queue. The complete Barbershop diagnostic records **1,062
to 798 surface dispatches**, while the number of surface executions is almost
unchanged: **332,307,665 to 332,307,666**. This repairs a scheduling difference;
it does not demonstrate that excess shading or the remaining Cycles image
difference has been eliminated.

Both first HIP 256-spp canaries completed and have **zero nonfinite values in
all 46 channels**, including Combined alpha. Full suites pass 183/183 host,
192/192 HIP and 194/194 fallback. This is still a partial integration
checkpoint: strict native Vulkan now passes 12/12 and the SDK scheduler
selection passes 17/17, but the other full-scene canaries and a new repeated
four-scene paired benchmark are not certified by this capture.

## Scope and identity

The rendered root source is Psycles `f3f852aedbb4c8389b0204c49e251ffd703b6005`
plus the captured native Holdout implementation and explicit surface-Handler
batching opt-in. Luisa is
`98f4667ca678a3a1425ff4467e0d7803a0a0d12e`: latest shared-callable integration,
the CFG prepare-to-update bypass repair, generic logical-priority/joint-Handler
batching, and the genuine guarded-projection negative regression. This is
not a queue-only isolated A/B intervention.

Upstream `1a2c3ea922c2` already fixes the complete previously failing zero-BSDF
XIR module. The additional local CFG repair addresses a separate, permanently
reproduced prepare backedge that must bypass update payload; it is not a claim
that `fixup_construct_exits` caused the original failure. See the SDK's
`docs/validation/2026-09-09/prepare-update-bypass/README.md` for the original
replay, semantic counterexample and complete host XIR validation.

[First-canary data](first-canaries.json) contains the exact renderer and
comparison commands, all pass metrics, all-channel finite masks/counts,
old/new image comparisons, EXR/log SHA-256 identities and eight post-build
binary hashes. The renderer SHA-256 is
`83ede06f45502b78df7ff6888d389306ea1ca763110232654eb799f1420a99e9`.
The plan records the dirty source snapshot; the captured binary hashes,
not later worktree modifications, identify these executions.

Evidence root:
`/var/tmp/psycles-holdout-CXlJR7/next-integration-YpUTb4`.
Both engines use the retained equal 15-pass / 46-channel workload. The Cycles
references are the original Blender 5.2.1 HIP captures, not new paired renders.
Both comparison reports verify their Blender-build metadata.

The older Luisa `6e58928d8460` [Holdout checkpoint](../native-holdout/README.md)
remains historical evidence: its four strict native Vulkan aborts and
Classroom finite-value failure are not overwritten or relabeled by this run.

## First 256-spp timings

These are one fresh observation per scene, not medians or a paired speedup
claim. Main shader caching is disabled, fast math and staged shader sorting
are enabled, frame capacity is 1,048,576 and maximum sample batch is 64.

| Scene / extent | Scene compile | Session setup / JIT | Render | Frame stages / fields / AoS bytes |
| --- | ---: | ---: | ---: | ---: |
| Barbershop / 2048x858 | 14.8769 s | 50.0167 s | 36.6909 s | 6 / 92 / 416 |
| Lone Monk / 1440x1080 | 4.53941 s | 31.8320 s | 12.5366 s | 4 / 50 / 200 |

The retained old-S3 Barbershop median is 37.8379 s and retained Cycles median
is 25.3775 s. This new single render is numerically 3.03% lower than old S3
but still 44.58% above retained Cycles. Monk is 0.69% above its old-S3
12.4505 s observation and 5.33% below the retained Cycles 13.2425 s median.
Different revisions, cache histories and unpaired observations prevent
attributing these changes to batching alone. The frame summaries did not
shrink in this integration.

The logged timing boundaries are important:

- Scene compile surrounds `renderer.compile_scene`, after scene import and
  device construction and before session construction.
- The message called “shader JIT” times `create_session`. It includes DSL/XIR
  construction, compilation and scheduler preparation, but also allocations,
  uploads and optional guiding setup. It is not an isolated HIPRTC timer.
- Render times `render_samples`, including Sobol-table preparation/upload,
  dispatches and synchronization, film readback and output-sink population.
  The later PPM/PFM/EXR file serialization is outside this interval. It is not
  a sum of GPU kernel durations. The retained Cycles number is its logged
  main-loop wall interval, not the enclosing Blender process time.
- `PSYCLES_DISABLE_SHADER_CACHE=1` controls the main path shader policy, not
  every auxiliary or driver/OS cache. These logs still contain 46 / 29 HIP
  cache-load markers for Barbershop / Monk. Session times are not certified
  globally cold JIT measurements.

The timing definitions are implemented in
[`render_blender_scene.cpp`](../../../../examples/render_blender_scene.cpp),
[`path_tracer_session.cpp`](../../../../src/luisa/path_tracer_session.cpp)
and [`path_tracer_kernel.cpp`](../../../../src/luisa/path_tracer_kernel.cpp).

## Image checks and remaining differences

The independent inspection requires exactly one layer and all 46 expected
channels. It finds 0 nonfinite scalars and 0 affected pixels in the fresh,
old-S3 and Cycles images for both scenes. Barbershop has 1,757,184 pixels /
80,830,464 channel values; Monk has 1,555,200 / 71,539,200. Old/new finite
masks match exactly and Combined alpha is numerically unchanged.

| Scene / pass | Fresh-vs-Cycles RMSE | Relative RMSE | Luminance mean ratio |
| --- | ---: | ---: | ---: |
| Barbershop Combined | 0.002310416 | 1.0521% | 0.999947 |
| Barbershop DiffInd | 0.008526775 | 7.1262% | 0.999925 |
| Barbershop GlossInd | 0.024908952 | 11.4033% | 1.000056 |
| Lone Monk Combined | 0.019848544 | 1.2398% | 1.000678 |
| Lone Monk DiffInd | 0.052106779 | 12.8816% | 1.001535 |
| Lone Monk GlossInd | 0.051769618 | 13.5965% | 1.002054 |

Every pass uses the full pixel count here because both images are finite.
Generally, the comparison denominator is the intersection of pixels with
all three finite RGB/XYZ components in both images. RMSE averages all three
components over those valid pixels; relative RMSE divides by reference RMS.
Mean ratios near one do not establish matching RNG dimensions, path counts
or bounce/shading predicates. Normal's signed luminance ratio is not a
radiometric acceptance metric.

`compare_cycles.py` returning zero means the comparison completed; it does
not reject nonfinite pixels and its RGB/XYZ metrics omit alpha. The separate
46-channel inspection is therefore necessary. The capture independently
checks per-pass finite denominators and recomputes Combined, DiffInd,
GlossInd and Normal RMSE using float64 accumulation. This is analysis of
already rendered images, not a CPU renderer or a renderer arithmetic change.

Old-S3 to fresh Combined RMSE / maximum absolute component difference is
`2.80799e-6 / 0.00281888` for Barbershop and
`5.92509e-4 / 0.282721` for Monk. Monk's 99th-percentile pixel RMSE is zero,
while its Normal RMSE / maximum difference is `1.32622e-4 / 0.0522835`:
the change is localized, not absent. Its cause remains unresolved. Neither
this local variation nor the Cycles DiffInd gap is waived as 1-ULP error or
ordinary noise, and these checks do not certify structural parity.

## Complete 64-spp queue diagnostic

The [strict counter capture](barbershop-queue-stats.json) reads the old and
fresh diagnostic logs independently. Both render 2048x858 at 64 spp in seven
dispatch ranges of `1, 1, 2, 4, 8, 16, 32` samples, summing to exactly
112,459,776 generated primary paths. The last 32-spp table is not the whole
render. These instrumented runs are not promoted to a performance pair.

| Counter over all seven ranges | Old SDK 6e | Integrated SDK 98f | Difference |
| --- | ---: | ---: | ---: |
| Generated primary paths | 112,459,776 | 112,459,776 | 0 |
| Resumed main-stage entries | 874,893,895 | 874,893,902 | +7 |
| Surface entries / Handler entries | 332,307,665 | 332,307,666 | +1 |
| Surface dispatches / Handler dispatches | 1,062 | 798 | -264 |
| Scheduler iterations | 3,697 | 3,245 | -452 |
| Gather-scan entries | 2,680,142,538 | 2,340,822,341 | -339,320,197 |
| Compaction-scan entries | 108,421,318 | 107,769,504 | -651,814 |

Surface dispatches fall 24.86%, iterations 12.23% and gather-scan entries
12.66%. These are diagnostic counter changes, not frame-traffic byte counts
or end-to-end speedups. An extension Handler can launch multiple sort kernels;
Handler dispatch count is not GPU kernel-launch count. Auxiliary work also
changes batch shape: NEE dispatches are 432 to 436, shadow-intersection 249 to
254, shadow-shading 143 to 119. It would be wrong to claim every stage launches
less often.

The old finalized graph has six incoming annotation boundaries for the same
surface continuation: entry, closest, volume, forward light, background and
surface itself. These are six source-edge registrations, not six different
surface shaders. Only closest and volume executed in the old full 64-spp
capture: queue 7 / boundary 2 had 710 dispatches and 271,308,737 entries;
queue 8 / boundary 10 had 352 and 60,998,928. The other four were unobserved.
Observation alone cannot prove those four edges unreachable; eliminating
them requires static control-flow proof, not scene profiling or hardcoding.

In the integrated log, queues 7–11 (boundaries 2, 10, 12, 14, 16) explicitly
alias queue 6 / boundary 1 / continuation 5. Only queue 6 executes, with the
same count and dispatch count as the surface continuation in every batch.
The six edge identities survive for routing, but compatible inputs now join
one physical before-resume batch before sorting. The prior captured frame
audit and its finalized field mappings remain [revision-pinned old-SDK
evidence](../barbershop-coroutine-audit/README.md); this log verifies the
unchanged 6-stage / 92-field / 416-byte summary, not a fresh complete
normalized-graph field-I/O capture.

The +7 total resumed entries include +5 closest, +1 light and +1 surface;
volume and background totals are unchanged. Queue order and compiler/backend
changes can affect floating-point/path decisions, but these counters do not
identify the cause. They support reduced scheduling overhead, not a claim
that unnecessary SVM evaluations have been removed or that native path
decisions already match.

## Reproduction and checkpoint limits

From the repository root, the two capture scripts read retained logs/images
only; neither builds code, launches a backend or renders a CPU reference.
Choose new output names because both refuse to overwrite evidence:

```sh
/usr/bin/python docs/validation/2026-09-09/next-integration/capture_first_canaries.py \
  /var/tmp/psycles-holdout-CXlJR7/next-integration-YpUTb4 \
  /tmp/psycles-next-first-canaries-review.json
/usr/bin/python docs/validation/2026-09-09/next-integration/capture_queue_stats.py \
  /var/tmp/psycles-holdout-CXlJR7/barber-frame-OFalxg/barbershop.log \
  /var/tmp/psycles-holdout-CXlJR7/next-integration-YpUTb4/barbershop-next-stats.log \
  /tmp/psycles-next-queue-review.json
```

The first script also checks the current SDK revision and all eight current
binary hashes against the post-build manifest; a later build will correctly
prevent recertifying this snapshot from changed binaries. `first-canaries.json`
records parent-confirmed renderer process exits and independent comparison
exit files. Both renderer exit files were subsequently read and confirmed 0.

## Terminal backend checkpoint (2026-09-10)

All root gate exit-status files contain zero. These are correctness-suite
wall durations, not renderer performance measurements. The all-target build
used all 32 hardware threads; device suites ran sequentially.

| Gate | Result | Wall time | Retained log under the evidence root |
| --- | ---: | ---: | --- |
| Complete host | 183/183 | 8.30 s | `host.log` |
| Focused HIP | 3/3 | 182.65 s | `focused-hip.log` |
| Complete HIP | 192/192 | 540.44 s | `full-hip.log` |
| Complete fallback | 194/194 | 549.07 s | `full-fallback.log` |
| Strict native Vulkan | 12/12 | 726.28 s | `strict-native-vk.log`, `strict-native-vk-detail.log` |
| SDK scheduler strict native Vulkan | 17/17 | 20.55 s | `sdk-regression/native-vk-ctest.log` |

The focused HIP tests cover Holdout state, Holdout rendering and the surface
queue. All twelve root Vulkan tests set `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1` and `LUISA_VULKAN_DISABLE_DXC=1`.
The complete `LD_DEBUG=libs` trace has 456 successful SPIR-V compilations and
zero DXC/DXIL loader matches. The previously failing lamp-routing, zero-BSDF,
Holdout-render and Ray-Portal-render tests all pass in this new snapshot.

The separate SDK selection has 163 unit cases / 21,349 assertions across
seventeen complete binaries. Each per-test loader trace is retained under
`sdk-regression/build/logs/`; all are free of DXC/DXIL references and run with
the three native guards. This includes the later test-only relocation
coverage expansion, not a change to the captured renderer libraries or a
validation of the subsequent source-only Boolean-reachability candidate.
See the [SDK batching evidence](../../../../third_party/LuisaCompute/docs/validation/2026-09-09/resume-queue-batching/README.md).

```text
840ce39249b8c78b5e4597c7686c21b87aca8658a7e7a8175fd20cfd887b4e8c  strict-native-vk.log
20d447eea1097ad57a5cf40102716b708212f3283ecc7f192b96cb84ce56f110  strict-native-vk-detail.log
207e77b4c9d510564dca3a7c8d038f3d711b2312504c900eb4a1d3bcee551c0a  sdk-regression/native-vk-ctest.log
```

Still required outside this first-canary capture: the remaining
scenes and timing repeats, investigation of Classroom nonfinite values and
Monk local variation, and continuing Cycles work-placement/path alignment.
Two finite images and successful backend suites do not complete those goals.
