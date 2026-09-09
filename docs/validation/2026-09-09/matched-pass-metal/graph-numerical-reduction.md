# Metal4 graph numerical failure: reduction log

The first film-aligned 1920x1080 / 256-sample graph run has Combined luminance
0.775731912 of Cycles and relative RMSE 0.230399849. It is retained as a failed
correctness result, not a performance score. The reduction below identifies an
undefined packed-word dependency in generic Luisa coroutine splitting. Its
correction passes minimal XIR, Metal/Metal4 runtime, and original-scene replay
checks and the full 1080p/256 Cycles gate. Unless explicitly marked diagnostic
or corrected, historical checks below use published Luisa `03a0f5158` and
unchanged upstream Psycles film implementation `9e3ba165`.

## Independent negative controls

Build-local `atomic_film_probe.cpp` checks exact integer counts and dyadic
floating sums at entry, a suspended surface, a variable-trip-count suspended
loop and termination. Versions exercise strided and contiguous destinations,
floating values surviving suspension, hint sorting (39 keys and frame locality
partitions), disabled/automatic tail, and changing runtime scalar arguments.

Metal and Metal4 initially pass the million-invocation constant-value controls.
Metal4 also passes runtime floating-payload and sorting controls, then the
complete 1920x1080x256 logical domain: 530841600 invocations, with all 2073600
pixels checked. Both a single invocation and four consecutive 64-sample
invocations of the same scheduler pass. A further four-invocation test changes
the scalar argument and carried floating weight each time and also passes.
These results do not reproduce or repair the original renderer failure.

## Original-module sample-range controls

The original renderer is run at 1920x1080 with the authored total sample count
fixed to 256, but a restricted absolute sample range. Keeping the total fixed
preserves the sampler configuration. Each row below is a fresh process; it is
not a checkpoint downloaded during the failed continuous run.

| Absolute range | Maximum batch | Diffuse Color mean / valid Metal 256-sample output |
| --- | ---: | ---: |
| [0, 8) | 8 | 1.0000335 |
| [0, 64) | 64 | 1.0000294 |
| [64, 128) | 64 | 0.9999828 |
| [128, 136) | 8 | 0.9999999 |

None of these controls reproduces the coherent energy loss. They have
different sample ranges and substantial residual sampling error, so the
cross-backend ratios above are diagnostic observations, not Cycles-oracle
compatibility gates. Their elapsed times are not performance results.
The production output path verifies every pixel's integer count against the
number of rendered samples before writing these images.

## Full repeat and progressive isolation

The exact 256-sample repeat fails again: Combined luminance / Cycles is
0.783365179 and relative RMSE is 0.226245049. Its render-only time of 294.441 s
(paired Cycles 50.6957 s) is an invalid-output observation, not a performance
score. All eight actual EXRs across the three completed matrices have exactly
46 finite channels. The repeat used no profiler, source changes or overlapping
device work/builds. The implementation is unchanged; later upstream and local
commits only change documentation.

The upstream CLI's full-frame progressive pixel probe then renders all
1920x1080x256 samples, downloading passes after each 64-sample batch. It
captures film pixel (960, 540), and records differences of accumulated sums,
not four independent renders. Its normalized per-chunk values are:

| Absolute range | Normal Y delta / 64 | DiffCol R delta / 64 |
| --- | ---: | ---: |
| [0, 64) | -0.998743236 | 0.362441063 |
| [64, 128) | -0.780146658 | 0.283830285 |
| [128, 192) | -0.764528930 | 0.277497441 |
| [192, 256) | -0.639871180 | 0.233006090 |

The first chunk matches the independent [0,64) output at this pixel exactly
for these fields. The independent [64,128) process gives Normal Y
-0.998647392 and DiffCol R 0.362267792 instead. PFM orientation is explicitly
accounted for: the OIIO PFM reader exposes this pixel at row 540, while the
renderer's raster sink uses row 539. Every downloaded integer sample count
still passes the production check. Thus additional per-chunk downloads do not
remove the failure, and the observed loss starts after the first batch at
this pixel. This is not evidence that a particular atomic or frame operation
is responsible.

A further full-domain negative control uses non-dyadic floating contributions
with relative tolerance 2e-5 plus absolute tolerance 1e-6, while integer counts
remain exact. Direct, graph-no-tail and graph-auto-tail all pass every pixel.
A separate two-stream timeline/readback probe passes 64 rounds. These probes
do not reproduce the original fault and do not justify a synchronization fix.
The two-batch, eight-sample-per-batch original-scene control also reproduces
the loss at 1080p: the captured pixel's Normal Y delta / 8 changes from
-0.998951852 in the first batch to -0.749210417 in the second. The total sampler
setting remains 256. This is a size reduction, not a timing measurement.

## Same-sample replay: tail and sorting interaction

A build-local copy of the upstream CLI, `render_replay_probe.cpp`, only changes
host orchestration: it can replay exactly the same sample range and writes
intermediate full-frame passes. It links the unchanged production renderer
libraries; it does not implement a second renderer, sampler, SVM or scheduler.
At 64x64, two consecutive renders of [0,64), total sampler setting 256, expose
the same failure without a sample-index change. Both use the same session.

For normalized accumulated outputs C0 and C1, the second batch's normalized
contribution is D1 = 2*C1 - C0. Identical samples should reproduce C0 within
floating accumulation tolerance. The following controls differ only in the
named upstream scheduler option:

| Surface sorting | Tail | DiffCol mean D1 / C0 | DiffCol relative RMSE D1 vs C0 | Normal relative RMSE |
| --- | --- | ---: | ---: | ---: |
| on | auto | 0.8591860 | 0.14249086 | 0.14682795 |
| on | off | 1.0000000 | 3.4552e-7 | 5.7944e-7 |
| off | auto | 1.0000000 | 3.4708e-7 | 5.8326e-7 |

Thus this reduced workload fails with sorting and tail together; neither
disabling an option nor the smaller workload is accepted as a repair or as
a substitute for the original 1080p/256 gate. First-batch auto/no-tail Combined
relative RMSE is only 8.75e-8. A generic Luisa probe that repeatedly exports a
changing hint inside a suspended loop still passes, including subsequent
reuse. It is another negative control, not a permanent failing regression.

A temporary, explicitly diagnostic SDK build downloads the complete frame
pool immediately before and after tail. Both batches leave all 14942208
32-bit words unchanged, while DiffCol second/first remains 0.85907555. This
does not support an unexpected tail write to the pool; it does not audit all
scene inputs or establish that all frame reads are semantically correct.
The check includes extra synchronization and is never used for performance.
A further entry-plan diagnostic finds matching live/store sets (41/41 and
48/48 fields) on the two entry edges, with no missing field. Temporary pool
initialization is a diagnostic only, not a proposed fix.
With full pool initialization before each invocation, the same-sample replay
passes (Diffuse Color relative RMSE 3.5142e-7, Normal 5.7845e-7). This is evidence
of a dependency on prior pool contents, not permission to add initialization
to the production scheduler. An all-but-one-field initialization sweep is
used to identify the stale field. Its per-case outputs are diagnostic only;
no valid timing is inferred from this sweep.

## Identified cause and correction

The all-but-one sweep identifies exactly physical field 90 as the failing
case: DiffCol replay ratio 0.8591442 and relative RMSE 0.1432196. This is
payload slot 83, a uint packing five logical Boolean variables into two
interfering bit lanes. `_reg_101849` is its physical diagnostic name, not a
unique logical variable and not the sorting key.

The split callable always read-modified the old packed word, including entry
edges defining every live bit. A fresh CoroFrame deliberately has undefined
payload. Metal4 lowers that to LLVM poison, so the initial word store can be
eliminated. Direct readback after the first entry in two successive dispatches
finds low-two-bit histograms `[131072,0,0,0]` and
`[29290,12377,89405,0]`: the second fresh entry retains prior state.

The generic correction seeds the word from zero unless live, unstored bits
must pass through. In that case it reads only those bits. One common mask
calculation drives code generation and scheduler input metadata. It adds no
slots, pool clear, backend conditional, or Psycles implementation change.
See the [Luisa proof and permanent regressions](../../../../third_party/LuisaCompute/docs/validation/2026-09-09/coro-packed-word-definedness/README.md).

The minimal XIR case fails before correction and passes afterward. Split,
materialize, distill and dataflow suites pass 1009 assertions in 111 tests.
Both Metal and Metal4 pass the 20-test / 100-assertion existing scheduler suite
and the new 40-dispatch packed-word replay regression. With every temporary
SDK diagnostic removed and sorting/tail both still enabled, the original
64x64 replay passes: DiffCol relative RMSE 3.4740e-7 and mean ratio 1.000000006;
Normal relative RMSE 5.7967e-7; Combined relative RMSE 4.3722e-7. These clear
the reduced failure. The full original 1080p/256 gate also passes: all 46
channels finite, Combined relative RMSE 0.007764527 / luminance ratio
0.999891523, DiffCol relative RMSE 0.000850013 / luminance ratio 1.000004020.
The full-resolution triptych no longer has coherent darkening. Observed
render-only time is 328.193 s versus fresh Cycles Metal 50.778 s; this is a
single correctness-gate observation, not a repeated performance conclusion.
The frame remains 91 fields / 456 B. Evidence is in
`packed-word-original-gate/graph/run-1` and
`packed-word-original-gate-audit.json` beneath the directory below.

## Evidence

The ignored build-local directory is
`build-macos/benchmarks/2026-09-09/lone-monk-metal-schedulers`:

- `atomic-film-probe-*.log`: independent negative controls.
- `graph-numerical-reduction/*/command.json`, `render.log`, and linear images:
  exact original-module control commands and outputs.
- `film-aligned-published/graph/run-1`: the original failed full run.
- `film-aligned-published/graph/run-2`: exact full-size repeated pair.
- `film-aligned-repeated-failure-audit.json`: hash-checked, all-channel audit
  of both failed graph observations and completed controls.
- `graph-numerical-reduction/full-progressive-64/pixel-chunks.json`: the
  unmodified upstream CLI's four full-frame batch checkpoints.
- `metal-event-probe-metal4.log`: passing independent timeline negative control.
- `graph-numerical-reduction/replay-64px-same-first*`: same-sample host replay,
  full-frame checkpoints, and explicitly named diagnostic variations.
- `atomic-hint-loop-probe-metal4.log`: passing repeated-hint negative control.

All original failed data is retained. Small/isolated controls never replace the
full failing gate. The final valid Metal4 staged Combined triptych has also
been inspected at its original resolution; its per-pass residuals remain in
the campaign report.
