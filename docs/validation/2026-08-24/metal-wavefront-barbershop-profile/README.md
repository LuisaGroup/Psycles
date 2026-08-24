# Metal staged-wavefront Barbershop profile

Date: 2026-08-24

## Outcome

This checkpoint isolates the 1920x1080 Barbershop staged-wavefront slowdown on
an Apple M1 Max. It distinguishes two separate issues:

- the one-sweep radix look-back had a Metal device-visibility bug that could
  produce multi-second tails or wait forever; and
- after that bug is fixed, the largest path sub-kernel is `shade_surface`, but
  the dominant end-to-end structure is the staged scheduler's per-iteration
  host/GPU synchronization.

The latest-main 1-spp diagnostic completed in 7.89694 seconds. The radix
`wavefront_hint_sort` command buffers consumed 67.427458 ms in total, while
`shade_surface` consumed 1425.297208 ms. The host spent 5410.155384 ms inside
1635 `MetalStream::synchronize` calls. Synchronization time includes waiting
for queued GPU work and therefore must not be added to the GPU command-buffer
totals.

## Revision and test shape

- Initial isolation: Psycles `d476a69184092c72e5146945e8a5f129d5426a53`,
  LuisaCompute `a407116fb`
- Latest-main profile: Psycles `aef5e8b`, LuisaCompute `4dcadeea6` (`next`)
- Device: Apple M1 Max, Metal
- Scene: Blender Barbershop Interior export
- Resolution: 1920x1080
- Nominal sample range: 64 samples
- Diagnostic sample range: absolute `[0, 1)`
- Samples per dispatch: 1
- Scheduler: `wavefront-staged`, execution block 32, surface sorting enabled,
  direct-light queue disabled, frame capacity 1048576

Both compact renderer experiments were enabled:

```text
PSYCLES_COMPACT_SURFACE_VALUES=1
PSYCLES_POPULATE_SURFACE_ONCE=1
```

This is a bounded 1-spp profiling run, not the pending formal 64-spp
Psycles/Cycles benchmark row.

## Failure isolation and root cause

Before the fix, a same-size 1-spp staged run completed in 49.2742 seconds. Its
process-wide profiler attributed 40.197570 seconds to unlabeled command
buffers, including a 5.506164-second maximum. Because that profiler also
included setup, a second run reset profiling at the first
`wavefront_initialize`. That run stopped making progress in the first render
chunk.

A live sample placed the main thread in:

```text
WavefrontCoroScheduler::_dispatch
  -> Stream::Synchronize
  -> MetalStream::synchronize
  -> MTLCommandBuffer waitUntilCompleted
  -> AGX / IOGPU command-queue semaphore wait
```

There was no memory pressure, swap activity, Metal recovery message, or IOGPU
device-fault report. The behavior instead matched the chained look-back in the
one-sweep radix kernel: one threadgroup published a status word and other
threadgroups spun until observing it.

The status buffer previously used volatile reads and writes. Volatile prevents
compiler reordering but does not establish device-wide visibility between
Metal threadgroups. Luisa now publishes status with atomic exchange and polls
it with an atomic zero-add read. A scheduler-scale regression sorts 130560
items, large enough to exceed normal GPU residency and exercise the chained
publication protocol rather than only resident two-block cases.

The same Luisa change also labels the radix command buffers and allows Metal
profiling to start at a named stage, so setup and render intervals are no
longer conflated.

## Latest-main Metal integration

Pulling Psycles through `aef5e8b` exposed a second Luisa Metal backend bug
before rendering began. The new fixed-shape surface runtime correctly marks
its bindless buffer slots as block-uniform, producing
`UNIFORM_BINDLESS_BUFFER_READ`. Metal AST codegen handled only the ordinary
bindless operation and aborted at its generic `Not implemented` case.

Luisa `4dcadeea6` lowers ordinary, uniform, typed, and typed-uniform bindless
resource operations to the same Metal descriptor helpers. These variants have
identical value semantics on Metal; uniform and typed are backend optimization
and resource-typing information, not different resources. Bindless writes now
derive their template type from the written value rather than the void call's
result type. The generic failure also reports the unsupported CallOp by name.
A real Metal regression executes both uniform and typed-uniform buffer reads
and writes.

## Corrected latest-main profile

The sum of the listed command-buffer GPU intervals is 2352.676623 ms.

| Metal command-buffer stage | Dispatches | GPU total | Share of profiled GPU time |
| --- | ---: | ---: | ---: |
| `wavefront_resume_5/shade_surface` | 433 | 1425.297 ms | 60.582% |
| `wavefront_resume_1/intersect_closest` | 414 | 544.813 ms | 23.157% |
| `wavefront_resume_2/shade_volume` | 241 | 90.395 ms | 3.842% |
| `wavefront_resume_7/intersect_shadow` | 115 | 81.717 ms | 3.473% |
| `wavefront_hint_sort` | 433 | 67.427 ms | 2.866% |
| `wavefront_generate/<entry>` | 16 | 38.653 ms | 1.643% |
| `wavefront_publish_resumed_count` | 1585 | 32.472 ms | 1.380% |
| `wavefront_resume_8/shade_shadow` | 115 | 18.410 ms | 0.782% |
| `wavefront_gather_selected` | 1585 | 14.502 ms | 0.616% |
| unlabeled | 1635 | 12.643 ms | 0.537% |
| all remaining labeled stages | 200 | 26.346 ms | 1.120% |

The scheduler processed the frame in 16 Metal-safe chunks, generated exactly
2073600 path states, resumed 20672900 continuations, and executed 1585
iterations. The summed scheduler interval was 7.513713 seconds. Renderer-only
time was 7.89694 seconds; scene compilation was 11.5192 seconds and the cold
shader JIT after the latest Psycles and Metal-codegen changes was 149.759
seconds.

Host synchronization measured:

| Operation | Calls | Total | Average | Maximum | Share of render wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `synchronize_wait_until_completed` | 1635 | 5410.099 ms | 3.309 ms | 83.278 ms | 68.510% |
| callback drain | 1159 | 0.057 ms | 0.000049 ms | 0.000750 ms | <0.001% |

`shade_surface` is therefore the concrete sub-kernel to optimize first if the
goal is GPU shader work. It is 2.62 times the GPU time of
`intersect_closest` and 21.14 times the GPU time of hint sorting. However, the
larger scheduler-level opportunity is to stop copying continuation counts to
the host and synchronizing once per iteration. The current staged path does
this after every queue-count publication; batching or device-side scheduling
is the appropriate next experiment.

Relative to the first post-radix-fix baseline, latest Psycles main reduced
`shade_surface` GPU time by 12.514%, total profiled GPU time by 7.993%, and
renderer-only time by 4.650%. Hint sorting changed by only +1.898%, consistent
with run noise and confirming it is not the steady-state throughput hotspot.
The latest corrected render was 6.24 times faster than the earlier
49.2742-second pathological observation. These single observations isolate
the cause but are not a formal throughput benchmark.

## Validation

Luisa Metal validation passed:

| Test | Result |
| --- | ---: |
| `test_coro_radix_sort metal` | 34 assertions / 8 tests |
| `test_coro_wavefront metal` | 783 assertions / 28 tests |
| `test_metal_codegen_regressions metal` | suite passed / 7 assertions |

The Psycles renderer rebuilt from latest main against `4dcadeea6` and completed
the real Barbershop render above. The production images and temporary profiler
artifacts remain outside the source tree.

## Reproduction

From `build-macos-shader-translation-plan`:

```bash
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
LUISA_CORO_WAVEFRONT_STATS=1 \
LUISA_CORO_SHADER_MAP=1 \
LUISA_METAL_COMMAND_BUFFER_PROFILE=1 \
LUISA_METAL_COMMAND_BUFFER_PROFILE_START_STAGE=wavefront_initialize \
./bin/psycles_render_blender_scene \
  ./benchmarks/2026-08-21/barbershop-interior-1080p64-compact-metal/export \
  /tmp/psycles-barbershop-metal-profile-latest-main-fixed.ppm \
  metal 1920 1080 64 1 - 960 540 0 0 1 - 1 0 \
  wavefront-staged 32 32768 32 1 1 0 1 1 4096 131072 1 0 1 1048576
```
