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

The evidence still admits an original-workload state/reuse issue or a
non-deterministic compilation/execution failure. The cause is not established.
An exact 256-sample repeat with a fresh Cycles Metal reference is in progress,
without profiler, source changes or overlapping device work/builds.

## Evidence

The ignored build-local directory is
`build-macos/benchmarks/2026-09-09/lone-monk-metal-schedulers`:

- `atomic-film-probe-*.log`: independent negative controls.
- `graph-numerical-reduction/*/command.json`, `render.log`, and linear images:
  exact original-module control commands and outputs.
- `film-aligned-published/graph/run-1`: the original failed full run.
- `film-aligned-published/graph/run-2`: exact full-size repeated pair.

All original failed data is retained. Small/isolated controls never replace the
full failing gate. The final valid Metal4 staged Combined triptych has also
been inspected at its original resolution; its per-pass residuals remain in
the campaign report.
