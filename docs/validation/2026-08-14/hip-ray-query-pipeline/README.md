# HIP synchronous RayQuery pipeline validation

This checkpoint optimizes LuisaCompute's HIP RayQuery implementation without
embedding renderer-specific self-intersection or alpha-test policy in the
backend. The optimized path is selected from XIR semantics: a lowered
`RayQueryPipelineInst` whose traversal cannot escape its handlers executes as
one synchronous operation. Explicit `proceed`/dispatch control, motion queries,
3-D workgroups, and handlers that issue nested ray tracing retain the reentrant
software path.

## Formal model

The gfx12 traversal has two disjoint frontier representations:

1. the ordinary hardware frontier `H`, stored by `ds_bvh_stack`; and
2. at most one parent-link subtree continuation
   `S = (root, node, completed_child)`.

Before expanding a BVH8 node, the implementation switches from `H` to `S` when
the packed hardware entry count exceeds `capacity - 8`. The eight-entry reserve
is sufficient for one complete BVH8 expansion, so the hardware operation cannot
overflow before the switch. While `S` is active, descending clears
`completed_child`; ascending follows the node's parent link and supplies the
completed child as the next skip token. Exhausting `root` resumes the unchanged
hardware frontier. A nested BLAS traversal saves the TLAS continuation and
restores it only after the BLAS root is exhausted.

The resulting invariant is that every reachable candidate belongs to exactly
one of `H` or `S`, and every candidate transition removes it from that frontier
before invoking user code. A callback is therefore neither replayed nor
omitted. An unexpected hardware overflow traps as a violated invariant instead
of returning a false miss.

The callback environment is compacted with an interprocedural least-fixed-point
demand analysis. For Callable argument `(f, i)`, a normal instruction use seeds
`live(f, i)`. A use that only forwards the value to local callee argument
`(g, j)` adds the equation `live(f, i) |= live(g, j)`. Propagating backwards
from the seeds makes forwarding-only SCCs dead. The environment retains the
union of arguments demanded by the surface and procedural handlers; all other
formal operands are replaced by unobservable poison values. Calls to external
or interposable functions and arguments with ABI attributes are conservative
live seeds.

Candidate handlers may themselves trace rays. Since the gfx12 LDS stack is
lane-local but not reentrant, XIR call-graph capability analysis detects that
case before code generation and selects the generic software-stack symbol
family. The regression includes an outer RayQuery whose surface callback issues
a nested `intersect_any`.

HIPRT's RTIP 3.1 triangle-pair traversal also received a state-machine fix. It
now chooses the nearest unprocessed pair member, marks it processed before the
filter callback, and evaluates the farther member only after rejection or a
later resume. This removes both eager side effects and duplicate callbacks.
The generic synchronous path contracts the active traversal interval through
HIPRT's new `contractRayMaxT` operation; that operation is monotone by
construction, so Luisa no longer relies on the private traversal-object layout
and cannot accidentally re-expand an already-pruned frontier.

## Performance

Machine: Radeon RX 9070 XT (`gfx1201`), ROCm 7.2.53211. Benchmark:
`example_path_tracing_cutout hip --offline --spp 64`, 1024×1024, HIP register
limit 176. FPS is the example's render-only metric; higher is better.

| Configuration | Five/seven-run median FPS | Relative to legacy |
|---|---:|---:|
| Legacy resumable HIP RayQuery | 333.310 | 1.000× |
| Safe synchronous pipeline, final | 447.626 | 1.343× |
| Earlier unsafe short-stack control | 446.173 | 1.339× |
| Ordinary no-RayQuery trace baseline | 916.774 | 2.751× |

The safe implementation is within 0.4% of the earlier unsafe control while
preserving exact overflow continuation. Ordinary trace remains at its original
8-entry LDS configuration; raising every kernel to 16 entries reduced its
median to about 790 FPS, while the final split configuration restores roughly
917 FPS. The remaining cutout-to-direct ratio is about 2.05× in time and still
includes real candidate callback/alpha work.

The final cutout runs were 446.674, 448.155, 449.641, 451.026, 446.064,
447.626, and 444.584 FPS. The direct runs were 912.170, 916.774, 911.640,
917.720, and 949.644 FPS; medians are reported to reject the last-run outlier.

Kernel resource metadata changed as follows:

| Metric | Before environment projection | After |
|---|---:|---:|
| Private segment | 480 B | 336 B |
| VGPRs | 176 | 173 |
| VGPR spills | 4 | 0 |
| SGPRs | 107 | 107 |
| SGPR spills | 10 | 2 |
| LDS | 32 KiB | 32 KiB |

A formally safe 12-entry stack was also measured. Its additional parent-link
continuation work reduced the median to about 429 FPS, so the selected query
configuration remains 16 entries. A lifetime-marker-only environment change
left both the 480-byte private segment and performance unchanged and was not
retained.

## Correctness and visual checks

The final HIP suite passes 1482 assertions in eight tests. Coverage includes
surface/procedural commits, `RayQueryAny` termination, paired-triangle resume,
coincident and reconstructed hits, bounded-stack BLAS/TLAS backtracking,
implicit Callable arguments forwarded through two Callable levels, and nested
handler tracing. The deep traversal regression uses 1024 overlapping BLAS
triangles and 256 overlapping TLAS instances and proves that every callback is
observed exactly once.

Ten related HIP ABI/XIR verifier, interchange, lowering, and pass-pipeline tests
also pass. The same runtime test executable passes its fallback baseline (42
assertions; HIP-specific cases are explicitly skipped).

The independently configured Vulkan canary keeps native XIR-to-SPIR-V enabled
and the LLVM-SPIR-V compatibility route disabled. Its strict native route guard
passes 3 assertions, the native/compatibility route contract passes 15
assertions, and the complete Vulkan SPIR-V code-generation path passes 2101
assertions in 93 tests. The latter intentionally exercises both routes in
separate cases; only those explicit compatibility cases load DXC.

The 1024×1024 cutout scene was rendered at 256 spp and inspected at full image
resolution. Both patterned boxes retain their expected open slats, silhouettes,
and projected shadows; there are no solidified cutouts, missing surfaces,
duplicate bands, or isolated traversal artifacts. The result is consistent
with the repository's [canonical cutout image](../../../gallery/test_path_tracing_cutout.png).

Commands:

```bash
cmake --build build-tests --target \
  luisa-compute-backend-hip test_hip_ray_query_pipeline \
  example_path_tracing example_path_tracing_cutout -j32

./build-tests/bin/test_hip_ray_query_pipeline hip

./build-tests/bin/example_path_tracing hip --offline --spp 64
./build-tests/bin/example_path_tracing_cutout hip --offline --spp 64
./build-tests/bin/example_path_tracing_cutout hip --offline --spp 256
```

The next performance step is to measure the same backend in Psycles production
scenes and separate candidate-handler cost from traversal cost per kernel. This
checkpoint does not claim that the remaining 2.05× cutout/direct gap is closed.
