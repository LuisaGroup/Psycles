# Metal graph-wavefront XIR exit convergence and 1080p/64 spp validation

Date: 2026-08-24

## Outcome

This checkpoint closes the Barbershop graph-wavefront XIR non-convergence,
pins the failure with an order-adversarial XIR regression, and validates the
current Metal scheduler policy on an Apple M1 Max.

- LuisaCompute `c45da5fc9` fixes terminal selection-exit targets being placed
  behind a forwarding fallback and is pushed directly to `next` through the
  separate SSH remote.
- Psycles `9e367fd` advances the submodule on `main` and is pushed directly to
  `origin/main`.
- The real Barbershop coroutine now completes XIR restructuring, Metal JIT,
  and a 1920x1080/64 spp render instead of exhausting every post pass.
- The automatic graph tail resolves to three eighths of active capacity. For
  the 131072-state Metal chunks here this is 49152 states.
- Graph wavefront is the fastest working Psycles scheduler tested on this
  machine. On Barbershop it is 2.247x faster than sorted staged wavefront.
- Material sorting is beneficial, not the source of the slowdown. The sort
  scatters frame indices, but the reduction in material divergence and shader
  working set is much larger than that cost.

All formal rows below use Metal only, 1920x1080, 64 spp, and one sample per
dispatch. Both compact surface experiments are enabled:

```text
PSYCLES_COMPACT_SURFACE_VALUES=1
PSYCLES_POPULATE_SURFACE_ONCE=1
```

`frame_buffer_compaction` remains a host-side structural value because it
changes coroutine layout and control flow. Queue capacity, worker count, and
the nonzero graph-tail magnitude remain runtime policy and do not specialize
the scheduler kernels.

## XIR root cause and fix

A multi-target selection-exit protocol is lowered to a selector ladder. Every
target except the final one is a direct conditional arm; the final target is
reached through a forwarding fallback. Stable basic-block order could put a
`Return` sink last while an ordinary in-construct continuation occupied a
direct arm. On the next post round, the forwarding edge to `Return` looked
like a fresh illegal selection exit, so an equivalent six-block dispatch was
created again. The observed Barbershop CFG therefore grew by six blocks and
eleven instructions per round until the iteration limit was exhausted.

The selector now uses semantic classes before stable ownership order:

1. legal loop or switch boundaries;
2. terminal `Return`/`Unreachable` sinks;
3. ordinary continuations, one of which is deliberately left as the ladder
   fallback.

A generated raw dispatch marker is released only while the header still owns
a raw `ConditionalBranchInst`. If that dispatch has already become a physical
structured `IfInst` loop-boundary guard, the marker is retained so later
construct-exit validation does not treat the guard as a new ordinary
selection.

Two counters expose coverage in pass reports:

- `selection_exit_terminal_target`; and
- `selection_exit_terminal_fallback_reorder`.

## XIR regression design

The new
`restructure_exit_dispatch_sink_priority_converges_for_target_orders` fixture
constructs the same nested loop/selection graph twice. One graph allocates the
terminal sink before the ordinary continuation; the other deliberately
allocates it after every block in that continuation's local construct. It
checks all of the following:

- a multi-target state dispatch was actually exercised;
- a terminal target was observed and stable order alone would have selected
  the wrong fallback in one ownership order;
- neither main nor post iteration budget is exhausted;
- no raw conditional remains except a canonical loop prepare;
- CFG block growth is bounded;
- strict unique-merge and canonical break/continue verification passes;
- both ownership orders translate to the same AST hash; and
- a second restructure pass is a no-op with an unchanged block count.

An A/B run with terminal semantic priority removed fails the cross-order AST
polarity assertion. With the fix restored, the complete XIR suite passes:

```text
Suite 'global': all tests passed (1647 asserts in 83 tests)
```

The real exported Barbershop 64x64/1 spp integration smoke also passes with
the production coroutine: 9 subroutines, 176 frame fields, an 848-byte frame,
4096 active frames, 5.245 s warm shader JIT, and 0.258 s render-only time.

The Metal-only surface-population fixture, which previously was not generated
by the project CMake, is present and passes in 69.72 s including its first
Metal JIT:

```text
psycles.luisa_surface_population_metal ... Passed
```

## Sorting and coherence diagnosis

There was a genuine Luisa Metal visibility bug in the chained look-back radix
sort: volatile status-buffer accesses did not establish device-wide ordering
between threadgroups. That was fixed earlier by publishing and polling the
status through atomics, with a 130560-item scheduler-scale regression. The
current controlled sorted/unsorted profile does not show another coherence
failure.

The paired Barbershop diagnostic uses the full 1920x1080 frame and one sample
to make per-continuation attribution practical:

| Staged surface order | Render-only | `shade_surface` GPU | Hint sort GPU | Host synchronize wait |
| --- | ---: | ---: | ---: | ---: |
| Unsorted | 10.6911 s | 3756.236 ms | 0 | 8216.948 ms |
| Sorted by material | 8.53636 s | 1544.838 ms | 70.945 ms | 6053.421 ms |

Sorting makes the renderer 1.252x faster and makes `shade_surface` 2.431x
faster. The 70.945 ms sort cost is only 3.21% of the 2211.398 ms saved inside
`shade_surface`. Although sorted indices make the coroutine frame's SoA loads
less contiguous, material/control-flow coherence dominates on this scene.

The corrected detailed profile independently attributes 1425.297 ms to
`shade_surface`, 544.813 ms to `intersect_closest`, and 67.427 ms to sorting.
The host spends 5410.099 ms in 1635 `waitUntilCompleted` synchronizations.
Thus `shade_surface` is the slowest GPU sub-kernel, while staged scheduler
readback/synchronization is the larger end-to-end problem.

## Graph scheduler optimization

The graph-tail threshold sweep uses the same 1920x1080 Barbershop frame with a
warm cache. Median render-only time for the most relevant repeated rows is:

| Tail threshold | Median 1-spp render |
| ---: | ---: |
| 4096 | 5.91402 s |
| 32768 | 4.41970 s |
| 49152 | 4.23854 s |
| 65536 | 4.38148 s |

49152 is 28.33% lower than 4096 and is the best measured point. It is exactly
three eighths of the 131072 active frame capacity, so Luisa now resolves the
`auto` sentinel on the host for each logical dispatch, rounds down to a full
execution block, and caps explicit values to active capacity. The sentinel's
printed value (`UINT_MAX`) is configuration metadata; the resolved dispatch
threshold is 49152. The magnitude no longer changes a shader cache key.

Cross-scene 1920x1080/1-spp checks show that 49152 improves every tested scene
relative to 4096: Classroom 8.54%, Lone Monk 12.80%, Monster 3.39%, and
Barbershop 28.33% by the repeated median above.

## Formal 1920x1080/64 spp performance

Times are renderer-reported render-only intervals. Scene export, scene
compilation, cold shader JIT, image conversion, and comparison are excluded.
Cycles uses Blender 5.2.0 LTS on the Apple M1 Max Metal device; Psycles uses
Metal graph wavefront, selective scheduling, material hint sorting, an active
capacity/worker count of 131072, and automatic tail 49152.

| Scene | Cycles Metal | Psycles graph | Psycles / Cycles | Previous graph | Graph speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classroom | 13.720 s | 87.741 s | 6.395x | 199.188 s | 2.270x |
| Lone Monk | 15.416 s | 86.160 s | 5.589x | 203.364 s | 2.360x |
| Monster Under the Bed | 21.047 s | 95.024 s | 4.515x | 203.652 s | 2.143x |
| Barbershop Interior | 44.104 s | 242.894 s | 5.507x | XIR did not converge | n/a |

The geometric-mean Psycles/Cycles slowdown is 5.460x. This remains well behind
Cycles Metal, but the graph-tail policy more than doubles Psycles throughput
on the three scenes with a prior formal result and makes Barbershop measurable
for the first time on this XIR path.

The current Barbershop scheduler comparison is:

| Renderer / scheduler | Scene compile | Shader JIT | Render-only | Relative to graph |
| --- | ---: | ---: | ---: | ---: |
| Cycles Metal | n/a | n/a | 44.201 s | 0.182x |
| Psycles graph, auto tail | 11.743 s | 5.305 s warm | 242.894 s | 1.000x |
| Psycles staged, sorted | 11.875 s | 171.408 s cold | 545.745 s | 2.247x |

Cold JIT is listed separately and is not mixed into throughput. Persistent was
also tried earlier on this exact machine; even its 64x64/1-spp smoke caused the
Metal compiler service to abort while building the giant persistent pipeline,
so it has no valid performance row. It is not silently counted as a timeout or
replaced by another backend.

## Image validation

Every graph output has 2073600 valid Combined pixels and no invalid values.
The classroom inspection shows the clock face and the window/transom above
the door shaded normally; the earlier black surfaces and purple window-line
artifact are absent.

| Scene | Combined RMSE | Relative RMSE | Luminance mean ratio | p99 pixel RMSE |
| --- | ---: | ---: | ---: | ---: |
| Classroom | 0.009092 | 2.1022% | 0.999381 | 0.045915 |
| Lone Monk | 0.032040 | 1.7134% | 1.000015 | 0.124715 |
| Monster Under the Bed | 0.021641 | 13.5429% | 1.003331 | 0.089521 |
| Barbershop Interior | 0.018772 | 9.4698% | 1.003502 | 0.074307 |

Monster and Barbershop relative RMSE are inflated by dark, high-variance
finite-sample paths. Their mean energy and visible scene structure agree.
Graph and staged Barbershop outputs agree much more closely with each other:
RMSE `4.62721e-5`, mean absolute error `5.87430e-8`, and luminance ratio
`1.00000018`. The scheduler speedup therefore is not an image-changing shortcut.

## Artifacts

Formal graph manifests, multilayer EXRs, logs, and Cycles/Psycles triptychs are
under:

```text
build-macos-shader-translation-plan/benchmarks/2026-08-24/graph-auto-tail-1080p64/
```

The Barbershop staged comparison is under:

```text
build-macos-shader-translation-plan/benchmarks/2026-08-24/scheduler-comparison-1080p64/barbershop-staged-sorted/
```

Generated renders remain outside the source tree and are not committed.

## Validation commands

```bash
cmake --build build-metal-cache-audit \
  --target test_xir_pass_restructure_cfg -j 8
./build-metal-cache-audit/bin/test_xir_pass_restructure_cfg

cmake --build build-macos-shader-translation-plan \
  --target psycles_render_blender_scene \
           psycles_luisa_surface_population_tests -j 8
ctest --test-dir build-macos-shader-translation-plan \
  -R '^psycles\.luisa_surface_population_metal$' --output-on-failure
```

The formal runner invocation uses `tools/run_scene_benchmark.py` with
`--width 1920 --height 1080 --samples 64
--max-samples-per-dispatch 1 --psycles-backends metal`, the scheduler named in
the row, and the two environment variables shown above.
