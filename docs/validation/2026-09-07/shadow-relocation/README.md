# Dense shadow-state relocation

## Scope and cause

This is a queue-work optimization, not a fix for the Lone Monk image residual.
The pre-change implementation passes the same output/transition regression.
No path, SVM node, RNG dimension, closure, floating-point expression, or default
scalar/vector initialization changes. Fast math remains enabled.

Cycles 5.2.1 `PathTraceWorkGPU::compact_paths` first collects two index lists:
live sources above the live-count boundary, and holes below that boundary.
`integrator_compact_shadow_states` then copies the indexed states. Psycles
previously gathered holes but dispatched its payload-copy kernel over the
entire allocated suffix, filtering empty slots inside that relatively heavy
kernel. This launches more payload-kernel lanes than necessary.

Authority: `/home/mike/Projects/blender-cycles-trace-5.2/`,
`intern/cycles/integrator/path_trace_work_gpu.cpp:718` and
`intern/cycles/kernel/device/gpu/kernel.h:671`. No CPU renderer is used.

## Invariants and implementation

Let E be the append extent, L the number of live states, and H the number of
empty slots in `[0,L)`. There are L-H live prefix states, so exactly H live
states remain in `[L,E)`. Therefore:

- `H <= min(L,E-L)`.
- Source indices fit in `[0,H)` of the existing index buffer.
- Hole indices fit in `[L,L+H)`; `L+H <= E <= capacity`.
- Source and destination slot sets are disjoint and individually unique.

A lightweight token-only collector builds both dense lists in one launch,
using two counters. The copy kernel dispatches the host-known bound
`min(L,E-L)`, guards on the actual H, and performs no allocation atomic or
suffix search. It copies invariant task state and, only for token 3, the
shade-owned traversal batch. No scratch state is read at NEE/intersection.

The host compaction/admission policy remains the Cycles policy: no relocation
below extent 32 or unless the extent can be halved. No additional host readback
is introduced. Unlike Cycles' exact-H host launch, this uses the proven upper
bound: the current HIP indirect-dispatch implementation would synchronize the
host to read GPU-authored records. Queue storage grows by two uint counters
(8 bytes), without an additional payload allocation.

## Regression and evidence

`tests/test_luisa_cycles_shadow_pipeline.cpp` runs the real production queue
with the existing independently generated Cycles HIP shadow-transition oracle.
Its existing cases cover capacity-19 drain-before-append and capacity-32
compaction of mixed NEE payloads and shade-owned batches.

New deterministic slot-order cases have E=32, L=8 and H=0,4,8. Live tasks own
two-hit transparent traversal batches. After relocation, 24 new tasks fill the
reclaimed suffix; stage counts, complete drainage, and every pixel's Combined
and Diffuse Direct result must remain correct. The power-of-two fixture
assertions check state preservation, not renderer bitwise parity.

Evidence directory: `/var/tmp/psycles-shadow-relocation-m8T359/`.

- `test-baseline-build.log`, `test-baseline-hip.log`: new cases pass the old
  implementation on HIP with shader cache disabled.
- `candidate-full-build.log`: all-target build with all 32 threads.
- `test-candidate-hip.log`: new implementation passes on HIP, cache disabled.
- `benchmark.sh`: alternating old/new Lone Monk 1440x1080, 256 spp, fast math
  enabled, independent shadow queue, no shader cache or observer.
- `profile.sh`: separate kernel tracing; profiled wall time is not used as an
  uninstrumented performance result.

The baseline libraries were copied before the production edit and their loader
resolution was checked with `ldd`. Both variants use the same executable,
Luisa libraries, scene bundle and render settings. The baseline is Psycles
5a16f2c6 plus the pre-existing dirty worktree; this is not a clean-checkout
performance claim. No inherited changes or Luisa gitlink belong to this patch.

## Results

RX 9070 XT (gfx1201), HIP, identical 1440x1080 / 256-spp Lone Monk settings:

| Alternating pair | Baseline sampling seconds | Dense relocation seconds |
| --- | ---: | ---: |
| 1 | 13.5195 | 13.4914 |
| 2 | 13.5229 | 13.5064 |
| 3 | 13.5354 | 13.5233 |
| Median | 13.5229 | 13.5064 |

The observed median difference is only 0.12%; this is not evidence of a large
end-to-end speedup. No profiler, trace, build, or other render runs concurrently
with these six measurements.

Separate kernel profiles identify the payload-copy launches by their function
hashes: baseline `f8695c7d096ad4fa`, candidate `f1fe8254e17016de`.
The baseline LLVM dump also identifies its five captured buffers and three
runtime integer parameters. The new copy and collector log their own hashes.

| Payload-copy metric | Baseline | Dense relocation |
| --- | ---: | ---: |
| Launches | 1090 | 1090 |
| Padded GPU lanes | 429660928 | 79248640 |
| GPU time, ms | 205.291 | 189.149 |
| VGPRs / scratch bytes | 160 / 8 | 160 / 8 |

Payload-kernel lanes fall by 81.56% and its measured GPU time by 7.86%.
The new token-only collector (`2419a03e347dac0b`) costs 6.283 ms and uses
8 VGPRs. That cost is not hidden in the copy-only comparison. Padded lanes
are a launch-size statistic, not an estimate of traced paths or shaded nodes.

New-versus-old Combined relative RMSE is 0.0003464, versus 0.0005419 between
two old-implementation repeats. Diff Ind is 0.0010830, versus 0.0013255 for the
old repeats. New implementation repeats are also not bitwise identical
(Combined 0.0003504, Diff Ind 0.0010027). No numerical-alignment workaround is
introduced to eliminate these repeat variations.

Unchanged original Cycles HIP references give:

| Scene / resolution / spp | Combined relative RMSE | Diff Ind relative RMSE | Main frame |
| --- | ---: | ---: | ---: |
| Lone Monk / 1440x1080 / 256 | 0.01240402 | 0.12881606 | 220 B |
| Monster / 1080x1080 / 256 | 0.00547921 | 0.02552785 | 284 B |

All compared passes have zero invalid pixels. Monster's single 14.8044-second
canary is not a repeated performance result. These residuals are essentially
unchanged; dense relocation does not explain or fix the Diff Ind discrepancy.

## Completed validation

- Active-worktree all-target build: 32 threads, passed.
- Full HIP CTest: 166/166; full fallback CTest: 168/168.
- Focused policy/shadow HIP: 3/3; fallback shadow: 2/2; native Vulkan: 2/2.
- Direct cache-disabled queue regressions: HIP, fallback, strict Vulkan passed.
- Isolated application snapshot: HEAD 5a16f2c6 plus only the two modified
  implementation/test files, 358 build steps with 32 threads, passed.
- The isolated queue regression passes HIP, fallback, and strict Vulkan.
- The isolated complete renderer renders the Map Range scene at 16x16 / 4 spp
  on HIP, then fallback, then native Vulkan, capacity 64, fast math enabled,
  shader cache disabled. All output values are finite.

Isolation uses clean Luisa 8e2b0ac78 headers and active SDK libraries, not an
independent clean SDK rebuild. Vulkan sets `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, `LUISA_VULKAN_DISABLE_DXC=1`;
the full renderer's `LD_DEBUG=libs` log contains native SPIR-V compilations
and no DXC/DXIL library loads. Logs and EXRs remain in the evidence directory.
