# Equal-pass HIP benchmark: Barbershop remains the outlier

## Technical summary

The corrected four-scene campaign completed all 12 pairs at Psycles c5bf9247
and Luisa 9ea3b720f. At original full-frame extents and 256 fixed samples,
Psycles takes 3.3% / 5.9% / 4.1% / 54.8% more render time than original
Cycles for Monk / Monster / Classroom / Barbershop. The efficiency goal
is not complete. Barbershop needs a surface-kernel diagnosis, not a general
claim that volumes explain the slowdown.

Both renderers now produce exactly the same 15 linear passes / 46 channels.
This replaces the unequal-pass performance baseline, including the withdrawn
Classroom speed-lead interpretation. Original Cycles still owns numerical
reference images; fallback is not an independent rendering oracle.

## Matched rendering leaves one large performance gap

Each row contains three fresh Cycles HIP -> Psycles HIP pairs on the same
RX 9070 XT. Times are seconds: Cycles' logged main-loop wall interval versus
Psycles' render-only wall interval. Relative time divides their respective
three-run medians, not enclosing-process durations or summed GPU kernels.

| Scene / extent | Cycles median [min, max] | Psycles median [min, max] | Psycles / Cycles |
| --- | ---: | ---: | ---: |
| Lone Monk / 1440x1080 | 13.2425 [13.2291, 13.2542] | 13.6826 [13.6656, 13.7207] | 1.0332 |
| Monster / 1080x1080 | 14.2865 [14.2778, 14.2915] | 15.1328 [15.0723, 15.1502] | 1.0592 |
| Classroom / 1920x1080 | 18.0319 [18.0290, 18.0608] | 18.7626 [18.4019, 19.0408] | 1.0405 |
| Barbershop / 2048x858 | 25.3775 [25.3208, 25.3852] | 39.2753 [39.1701, 39.3267] | 1.5476 |

Barbershop's 13.90-s median gap is well outside its observed repetition
range and much larger than the other scenes' gaps. These are descriptive
observations on one workstation, not confidence intervals or a universal
performance estimate. Three repeats do not establish long-run variability.
Exact observations and per-pair ratios are in [summary.json](summary.json).

## Compilation and frame storage are separate costs

The CLI field shader_jit_seconds times the complete session initialization,
including JIT plus setup/baking. It is not an isolated compiler measurement.
Scene compilation is a separate host interval. Cycles loads precompiled GPU
kernels, so the two engines do not have symmetric JIT timing scopes.

| Scene | Session init median s | Scene compile median s | Coroutine stages / fields / bytes |
| --- | ---: | ---: | --- |
| Monk | 18.7327 | 5.10951 | 4 / 55 / 220 |
| Monster | 22.1471 | 1.49333 | 6 / 71 / 284 |
| Classroom | 18.1196 | 2.11660 | 5 / 66 / 264 |
| Barbershop | 25.6044 | 15.43720 | 6 / 93 / 416 |

Frame layouts are unchanged across repetitions. Their smaller sizes are
useful implementation results, but are not proof of performance parity.
The scheduler uses a shared 1048576-entry frame pool, not per-thread malloc.
Main Psycles shader caching is disabled; auxiliary, downstream and OS caches
retain their normal policy. This is not a fully cold whole-toolchain campaign.

## Image residuals remain after the workload correction

All 46 Psycles channels are finite in every render, and all 15 pass reports
complete for all pairs. Below are first-pair relative RMSE values (fractions,
not percentages). They compare linear RGB before display transforms.

| Scene | Combined | DiffCol | DiffInd |
| --- | ---: | ---: | ---: |
| Monk | 0.01241440 | 0.000545920 | 0.12881733 |
| Monster | 0.00547872 | 0.000110133 | 0.02552496 |
| Classroom | 0.00353306 | 0.000102311 | 0.17820333 |
| Barbershop | 0.01079848 | 0.001597578 | 0.07454033 |

Original Cycles Classroom retains 25 invalid DiffDir and 27 invalid GlossDir
pixels per run. The comparator explicitly excludes the union of invalid
pixels; these counts are not silently folded into a finite-only accuracy
claim. Every pass's absolute error, reference RMS and invalid counts are
archived beside its relative error. All four first-pair Combined triptychs
were inspected at the viewer's resized resolution; this is not an exhaustive
pixel-level visual certification. Full-resolution images remain at the paths
recorded in each report.

The residuals are not waived as sampling noise or one-ULP error. DiffInd alone
does not count extra paths: it is an albedo-demodulated output, so diagnostic
work must also check contribution-weighted errors and same-sample events.
Matching aggregate energy does not establish matching path structure.

## Reproducible scope and controls

Production Blender is 5.2.1 LTS build 9e2066aef7ef, with exactly the named HIP
RX 9070 XT device. The source build is not the instrumented path-trace oracle
build. Original blends, exported geometry/materials and source frame settings
are retained. Authored seed / frame / effective seed are Monk 0 / 4 / 0,
Monster 0 / 1 / 0, Classroom 1 / 1 / 1 and Barbershop 0 / 1 / 1267069554.
Barbershop's animated-seed setting is preserved.

Native fast math is enabled. Psycles uses wavefront-staged, a 32-thread
execution block, surface sorting and the separate direct-light queue, with
64 samples per dispatch. Exact remaining settings and commands are in all
12 manifests. No profiler, concurrent render or build overlapped these pairs.
Scene order rotates between rounds; renderer order is always Cycles then
Psycles, not randomized. All ten executable/library/script hashes were
verified unchanged after every pair and again after the campaign.

The [pass-contract regression](../pass-contract/README.md) explains the prior
measurement defect, the permanent negative tests and actual EXR validation.
Schema-v1 whole-render-call ratios and schema-v2 unequal-pass ratios must not
be mixed into this schema-v3 baseline. The failed header-alias attempt was
excluded; no incomplete pair was reused as a timing observation.

## Surface work remains the next diagnostic target

The [homogeneous-volume work fix](../volume-work/README.md) removes verified
unnecessary scatter/phase work, but its before/after 64-spp profile has no
measurable volume-kernel gain. It does not explain the Barbershop result here.
Updated equal-pass kernel profiling is a separate instrumented workload;
it must not replace the unprofiled numbers above.

The new sequential 2048x858 / 64-spp rocprofv3 pair uses the corrected pass
contract on both engines. Kernel identities come from the recorded Cycles
names and Psycles stage/hash map, not row order. Summed trace durations and
call counts independently match the profiler statistics:

| Stage | Cycles GPU s / calls | Psycles GPU s / calls | Additional GPU s |
| --- | ---: | ---: | ---: |
| shade_surface | 2.962829 / 797 | 5.726934 / 1120 | 2.764105 |
| intersect_closest | 1.366894 / 910 | 1.546774 / 992 | 0.179880 |
| shade_volume | 0.353596 / 381 | 0.433407 / 429 | 0.079812 |

The surface difference is much larger than the complete Psycles volume
budget, even before subtracting Cycles' volume cost. This localizes the
dominant measured gap to surface work but does not identify its mechanism.
The three rows are selected semantic stages, not an additive decomposition
of whole-render wall time. Setup/BVH kernels in the raw profiler statistics
are excluded from this table. Launch counts are not active shading-event
counts, and different queue capacities/tail behavior affect launches.

Cycles' surface kernel reports 192 VGPRs / 6976 scratch bytes versus
Psycles' 256 / 2432. Lower scratch allocation alone plainly does not establish
faster execution. These are resource allocations, not measured spill traffic
or occupancy. [Kernel evidence](barbershop-kernel-profile.json) retains exact
duration sums, resources, actual pass inventories and source hashes.
The initial Psycles profile command had total samples 64 but sample-count 256;
CLI validation rejected it before rendering. It is preserved separately and
excluded. The corrected command uses the absolute range [0, 64) of 64.

Next, distinguish extra surface events from extra cost per event, then
reduce any identified semantic discrepancy against original Cycles. Dispatch
counts, padded grids, VGPR counts or scratch allocation alone cannot prove
active-path counts, occupancy or spill traffic. Static node/stack pruning,
sampler dimensions and Cycles control-flow invariants remain mandatory;
profiling must not determine scene-local allocation bounds.

Remaining questions include the dominant surface-kernel mechanism, later
same-path indirect-light divergences, unsupported native opcodes and legacy
displacement removal. No inlining policy, software bit-matching path or
scene-specific shortcut was introduced to make these measurements pass.
