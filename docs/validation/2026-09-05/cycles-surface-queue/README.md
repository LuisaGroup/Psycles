# Cycles shader identity and frame-local surface sorting

## Structural cause and source contract

Cycles 5.2.1 is the sole reference, at
`/home/mike/Projects/blender-cycles-trace-5.2`, commit
`cb168525138fecc792cc393f94afc39582b0103c`. The following source files have
no local modifications:

- `kernel/bvh/util.h`: `intersection_get_shader_from_isect_prim()` reads
  `tri_shader[prim]` or `curves[prim].shader_id`, then removes shader flags.
- `device/queue.h`: HIP inherits `DeviceQueue::num_sort_partitions()`.
- `integrator/path_trace_work_gpu.cpp`: `alloc_integrator_sorting()` converts
  that partition count into a ceiling-divided path-slot divisor.
- `kernel/integrator/state_flow.h`: `INTEGRATOR_SORT_KEY` puts path-slot
  locality before shader identity.

For capacity C, shader-table extent S, and stable state index i:

```
partitions = S < 300 ? max(floor(C / 65536), 1) : 1
divisor    = ceil(C / partitions)
sort_key   = shader_index + S * floor(i / divisor)
```

The previous native SVM route instead reused the legacy expanded-graph
topology tag and sorted it across the entire frame pool. Lone Monk had
23 topology tags, versus 39 entries in the source shader identity domain;
its 1,048,576-frame capacity should have 16 locality partitions.

The native queue now reads the canonical, already uploaded Cycles geometry
shader image. Triangle offsets and curve-to-source-curve mappings preserve
the same primitive identities used by the renderer. Instance material
overrides are resolved by native geometry-image construction; incompatible
shared-geometry shader arrays are rejected there, not silently guessed by
the queue. The shader-table extent preserves holes and non-surface entries.
The legacy diagnostic evaluator retains its topology key.

Both staged wavefront and selective graph-wavefront configuration use the
source partition policy. Luisa's existing `hint_partition_size` API performs
the composite sort. It was already published in `origin/next` by
`e6ecbbbe7` (`coro: preserve frame locality during hint sorting`). No new
Luisa modification or submodule gitlink is part of this patch.

## Independent regression

`tools/cycles_wavefront_sort_oracle.hip` includes the original Cycles HIP
headers and executes the real shader lookup and sort-key macro. Its output
exactly matches `tests/data/cycles_wavefront_sort.txt`:

```sh
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_wavefront_sort_oracle.hip -o /tmp/wavefront-sort-oracle
/tmp/wavefront-sort-oracle
```

`tests/test_luisa_cycles_surface_queue.cpp` independently exercises the
production queue stage with triangle-only, curve-only and mixed hit domains,
both native and legacy evaluators. It covers shared topology with distinct
shader identities, flags, per-instance resolved primitive ranges, and a
non-identity curve index mapping. The initial HIP run failed semantically:
source shaders 12 and 19 both became topology key 1; the curve keys also
disagreed. The legacy route already passed.

Host policy boundary checks cover capacities around 64K/128K, non-divisible
capacities, and the exact 299/300 shader threshold. These are explicit
checks of the source formula, not outputs falsely attributed to the HIP
oracle's host policy. The HIP fixture additionally checks the composite
keys at actual source state indices.

A complete Luisa wavefront coroutine runs 131,073 frames with the source
policy (divisor 65,537). Physical resume order must be a complete permutation
sorted first by frame partition, then by the oracle shader identity. It uses
greedy, incremental scheduling. Fused counter publication is orthogonal to
this regression and is exercised by the full renderer runs; the test does
not require a newer fused-count API than the recorded submodule revision.
No CPU reference renderer is introduced.

## Full-scene factorial isolation

All runs use RX 9070 XT HIP, Lone Monk frame 4, 1440x1080, 256 spp, seed 0,
adaptive sampling and denoising disabled. No GPU tests or builds ran during
the timed integration intervals. Each variant executes the full original
renderer, not an extracted shader benchmark.

| Queue key | Slot partition | Psycles render interval | Surface GPU time |
|---|---:|---:|---:|
| Legacy topology, fresh control | none | 18.8774 s | 10.944312 s |
| Cycles shader identity only | none | 19.5359 s | 11.595561 s |
| Legacy topology, partition diagnostic | 65,536 | 15.9729 s | 8.051088 s |
| Cycles shader identity, retained policy | 65,536 | 16.1076 s | 8.151486 s |
| Retained policy, repeat | 65,536 | 16.1354 s | 8.178585 s |

The locality partition is the demonstrated dominant improvement. Shader
identity alone is not a speedup. The legacy-key partition diagnostic was
slightly faster in its single observation; it is not the retained native
policy because it does not reproduce Cycles' shader identity domain.
The retained source-aligned policy reduces the renderer's own interval
about 14.5--14.7%, and surface time about 25.3--25.5%, versus the fresh old
control. This is a measured Lone Monk result, not a general-scene guarantee.

Every variant has the identical surface final LLVM IR SHA256:
`efe377767ce1134ae4e96bdb8db5dd35be048b54acd93fc04993e830de27e47d`.
The retained repeat's actual linked surface code object is byte-identical
to the pre-change object, SHA256:
`ead3a3c8d6f0dd6ed726873eda245d51d8b19ae3f022c983fa3e45aed6f13fb4`.
Surface remains `kernel_fe586f1dd5609578`, 256 VGPRs, 3424 B scratch,
and block size 512. Persistent state is still 88 fields, 448 B AoS / 444 B
SoA. No SVM words, feature masks, stack/closure layout, inlining flags, or
default scalar/vector initialization semantics changed.

## Correct timing scope and remaining gap

The previous Cycles script value around 16.45 s must not be directly compared
with Psycles' `render_samples()` interval as proof of parity. The script
times `bpy.ops.render.render(write_still=True)`, including scene preparation
and EXR output. Psycles reports its rendering-to-memory interval after scene
preparation/JIT and before EXR output.

A fresh debug-log Cycles control reports **13.507438 s** for its internal
256-sample rendering loop, while its enclosing script reports 16.434719 s.
Its surface GPU time is 5.924705 s. Thus the retained Psycles policy is still
about 19% slower on the corresponding rendering-loop intervals, and surface
alone still costs roughly 2.24 s more. NEE is another remaining difference:
about 0.79 s in Psycles versus 0.49 s in Cycles. Performance parity and full
renderer equivalence are not established.

Against the established Cycles HIP image, the two retained runs' Combined
relative RMSE is 0.01113805 / 0.01114613; all eight compared passes have zero
non-finite pixels. Existing indirect-pass differences remain (DiffInd about
0.141, GlossInd about 0.162). They are not reclassified as solved by this
scheduling change.

The direct retained-policy/old-control image comparison has Combined
relative RMSE 0.00040692, with zero non-finite pixels in all eight passes.
The full images are not asserted to be bit-identical.

## Evidence and validation

Test/build evidence: `/var/tmp/psycles-cycles-surface-sort-JC4LHP`.

- Full build with all 32 available threads: passed (`full-build-final.log`).
- All 150 HIP backend tests: passed (`full-hip-final.log`, 289.82 s).
  The queue test was then rebuilt without the orthogonal, newer fused-count
  API dependency and re-passed HIP (`queue-final-build.log`,
  `queue-final-hip.log`). No production code changed after the full sweep.
- All 152 fallback backend tests: passed (`full-fallback-final.log`, 67.96 s).
- All 51 Cycles core tests: passed (`core-final.log`).
- Seven strict native-XIR Vulkan canaries: passed
  (`strict-native-vk-final.log`, 1.96 s), including the new complete queue test.

Vulkan required `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`. The additional
`strict-queue-native-recompile.log` uses `LUISA_DUMP_SOURCE=1` to force
actual native recompilation, with compiler-phase logging and `LD_DEBUG=libs`.
It records successful native AST-to-XIR-to-SPIR-V compilation and no loaded
DXC/DXIL library; this is not only a cached-shader canary.

- Old control: `/var/tmp/psycles-cycles-sort-old-control-7HzONq`.
- Identity only: `/var/tmp/psycles-cycles-sort-key-only-GhsTC8`.
- Partition only: `/var/tmp/psycles-cycles-sort-partition-only-dkMaoI`.
- Retained policy: `/var/tmp/psycles-cycles-sort-lone-monk-WyBHBN`.
- Retained repeat: `/var/tmp/psycles-cycles-sort-lone-monk-repeat-8kfoPI`.
- Cycles control: `/var/tmp/cycles-sort-control-HcaPwO`.
- Cycles phase-scope control: `/var/tmp/cycles-sort-phase-control-34x586`.

Each renderer directory contains its full command/log, `render_results.db`
and image (Cycles uses `cycles_results.db`). The retained repeat also contains
the live code objects; its surface is `hip_isa_11.co`. LLVM dump sequence
numbers change with sorting kernels, so locate the surface by its function
name, not by a fixed `_3.ll` filename.

All diagnostic source toggles were removed before the final full-thread
build. The existing executor logging, block-size override and fused-count
changes are pre-existing dirt and are excluded from this selective commit,
as are the other pending path-tracer changes and Luisa gitlink.
