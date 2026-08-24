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

The corrected 1-spp diagnostic completed in 8.28202 seconds. The radix
`wavefront_hint_sort` command buffers consumed 66.171750 ms in total, while
`shade_surface` consumed 1629.167833 ms. The host spent 6054.234098 ms inside
1635 `MetalStream::synchronize` calls. Synchronization time includes waiting
for queued GPU work and therefore must not be added to the GPU command-buffer
totals.

## Revision and test shape

- Psycles parent revision: `d476a69184092c72e5146945e8a5f129d5426a53`
- LuisaCompute revision: `a407116fb` (`next`)
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

## Corrected profile

The sum of the listed command-buffer GPU intervals is 2557.049584 ms.

| Metal command-buffer stage | Dispatches | GPU total | Share of profiled GPU time |
| --- | ---: | ---: | ---: |
| `wavefront_resume_5/shade_surface` | 433 | 1629.168 ms | 63.713% |
| `wavefront_resume_1/intersect_closest` | 414 | 544.858 ms | 21.308% |
| `wavefront_resume_2/shade_volume` | 241 | 91.273 ms | 3.570% |
| `wavefront_resume_7/intersect_shadow` | 115 | 81.344 ms | 3.181% |
| `wavefront_hint_sort` | 433 | 66.172 ms | 2.588% |
| `wavefront_generate/<entry>` | 16 | 39.176 ms | 1.532% |
| `wavefront_publish_resumed_count` | 1585 | 32.966 ms | 1.289% |
| `wavefront_resume_8/shade_shadow` | 115 | 17.914 ms | 0.701% |
| `wavefront_gather_selected` | 1585 | 14.263 ms | 0.558% |
| unlabeled | 1635 | 12.852 ms | 0.503% |
| all remaining labeled stages | 200 | 27.064 ms | 1.058% |

The scheduler processed the frame in 16 Metal-safe chunks, generated exactly
2073600 path states, resumed 20672903 continuations, and executed 1585
iterations. The summed scheduler interval was 7.914823 seconds. Renderer-only
time was 8.28202 seconds; scene compilation was 11.4525 seconds and cached
shader JIT was 6.11373 seconds.

Host synchronization measured:

| Operation | Calls | Total | Average | Maximum | Share of render wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `synchronize_wait_until_completed` | 1635 | 6054.178 ms | 3.703 ms | 87.039 ms | 73.101% |
| callback drain | 1155 | 0.056 ms | 0.000049 ms | 0.000334 ms | <0.001% |

`shade_surface` is therefore the concrete sub-kernel to optimize first if the
goal is GPU shader work. It is 2.99 times the GPU time of
`intersect_closest` and 24.62 times the GPU time of hint sorting. However, the
larger scheduler-level opportunity is to stop copying continuation counts to
the host and synchronizing once per iteration. The current staged path does
this after every queue-count publication; batching or device-side scheduling
is the appropriate next experiment.

The corrected render was 5.95 times faster than the earlier 49.2742-second
diagnostic observation. This single before/after observation confirms removal
of the pathological tail but is not a formal throughput benchmark.

## Validation

Luisa Metal validation at `a407116fb` passed:

| Test | Result |
| --- | ---: |
| `test_coro_radix_sort metal` | 34 assertions / 8 tests |
| `test_coro_wavefront metal` | 783 assertions / 28 tests |

The Psycles renderer rebuilt against the corrected submodule and completed the
real Barbershop render above. The production images and temporary profiler
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
  /tmp/psycles-barbershop-metal-profile-atomic.ppm \
  metal 1920 1080 64 1 - 960 540 0 0 1 - 1 0 \
  wavefront-staged 32 32768 32 1 1 0 1 1 4096 131072 1 0 1 1048576
```
