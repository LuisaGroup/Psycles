# Metal4 graph numerical failure: reduction log

The first film-aligned 1920x1080 / 256-sample graph run has Combined luminance
0.775731912 of Cycles and relative RMSE 0.230399849. It is retained as a failed
correctness result, not a performance score. No corrective implementation has
been made for this numerical failure yet. All checks below use the published
Luisa `03a0f5158` and unchanged upstream Psycles film implementation `9e3ba165`.

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
The cause remains open; a two-batch, eight-sample-per-batch original-scene
control is being used to reduce execution size.

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

All original failed data is retained. Small/isolated controls never replace the
full failing gate. The final valid Metal4 staged Combined triptych has also
been inspected at its original resolution; its per-pass residuals remain in
the campaign report.
