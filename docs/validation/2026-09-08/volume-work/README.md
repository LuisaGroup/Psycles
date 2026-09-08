# Homogeneous volume rejection: corrected work, no demonstrated Barbershop speedup

Psycles base 253767f5 / Luisa 9ea3b720f. The homogeneous component had a
real work-placement defect, but fixing it does not measurably resolve
Barbershop's slowdown. Its surface kernel remains the larger outstanding
difference. This checkpoint does not claim renderer/Cycles efficiency parity.

## What changed and why

Original Cycles 5.2.1 integrates emission and attenuation, then returns for
PATH_RAY_TERMINATE or zero sigma_s. It chooses either indirect scattering
or transmission, and invokes direct-light and phase continuation only for
the corresponding sampled scatter event. Psycles masked the final results
with select while recording channel/distance/PDF operations and light/phase
consumers before that mask. A failed collision-point light resample also
needs to return before emission and receiving-phase evaluation.

The component now records these original branches. Equiangular initialization
is preserved before coefficient shading, including on emission-only media;
the light proposal is not skipped using knowledge of the fog material.
Only the selected sampling technique is evaluated, and a competing PDF is
computed when MIS actually needs it. Emission, attenuation, random dimensions,
closure equations and native fast math are preserved. No inlining decision
or coroutine policy is changed.

The independent GPU work regression failed 24/72 checks before the fix and
passes after it. The final test covers all three direct-sampling techniques,
216 checks in total, using production transport/phase code and counters at
the existing light-provider interface. No separate CPU estimator is used.
See [source-notes.md](source-notes.md) for original function mappings, the
Barbershop material/seed audit, exact profile queries and regression scope.

## Completed renderer observations

Each row is one new 256-spp HIP canary against the unchanged original Cycles
run-1 image, not a newly paired or repeated benchmark. The six executable /
library hashes and source/export/reference identities were verified. No build,
profiler or other render overlapped them. Main shader caching is disabled;
auxiliary/downstream caches retain their normal policy. Session initialization
includes JIT plus setup/baking and is not a fully cold compiler-only timer.

| Scene / extent | Render s | Session init s | Frame B | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monk / 1440x1080 | 13.6432 | 18.7690 | 220 | 0.01239223 | 0.12881415 |
| Monster / 1080x1080 | 15.0614 | 21.9577 | 284 | 0.00547890 | 0.02552521 |
| Classroom / 1920x1080 | 18.9022 | 18.0410 | 264 | 0.00353278 | 0.17820334 |
| Barbershop / 2048x858 | 39.3508 | 27.8893 | 416 | 0.01079812 | 0.07450917 |

All 46 actual channels are finite, all 15 common-pass comparisons complete,
and all four Combined triptychs were visually inspected. Existing image
residuals remain; original Classroom retains 25 invalid DiffDir and 27 invalid
GlossDir pixels, excluded explicitly by the invalid union. Exact commands,
all pass metrics, timings, frame layouts and hashes are in
[canaries.json](canaries.json).

Barbershop changed from the previous single 39.6390-s observation to
39.3508 s (0.73% lower), which is not enough evidence of stable improvement.
The independent before/after 2048x858 / 64-spp rocprofv3 observations are:

| Stage | Before GPU s | After GPU s | Calls before / after |
| --- | ---: | ---: | ---: |
| shade_surface | 5.684112276 | 5.688461864 | 1120 / 1120 |
| intersect_closest | 1.626827636 | 1.554720358 | 992 / 992 |
| shade_volume | 0.427889843 | 0.430448010 | 429 / 429 |

The surface and closest kernel hashes are unchanged. The volume structural
hash changes from 7fca871f3dc049b0 to 859d7ba000078b22, but both report
256 VGPRs and 452 scratch bytes. This does not prove identical machine code;
it also does not support an occupancy or spill-traffic improvement claim.
Profiled render wall time is 10.3439 / 10.2883 s. Kernel durations are not
interchangeable with that wall interval or with the 256-spp observations.

The earlier original Cycles surface / volume observations are 2.935582003 /
0.359619786 s. Their extra AO output workload prevents an equal-pass speed
ratio. Even without using such a ratio, the measured volume budget is far
too small to explain the outstanding surface difference. Extra active paths,
duplicate surface evaluations and sampler divergence remain hypotheses to
measure, not conclusions from padded dispatch sizes.

## Validation gate

| Gate | Result | Qualification |
| --- | --- | --- |
| Full build | Passed | All 32 threads |
| Full HIP | 179/179, 332.37 s | No failed selected tests |
| Full fallback | 180/181, 301.51 s | Existing film light_ng.z mismatch only |
| Strict native Vulkan | 5/5, 142.53 s | Homogeneous/stacked volume, work counter, volume path, original emission film |

These include actual volume renders and the original 5.2.1 volume-emission
film fixture, not only work counters. Vulkan uses all three native-XIR,
require-native-SPIR-V and disable-DXC guards. Its verbose loader log records
SPIR-V compilation and no loaded DXC/DXIL library. The fallback mismatch
remains expected 0xbf1f8bfd versus actual 0xbf1f8a50, with no tolerance change.
Raw logs use the volume-work prefix at the evidence root in source-notes.md.

The diagnostic report distinguishes verified control flow, callback counts,
whole-render correctness, one-run timings and unresolved causality. Original
Cycles remains the only rendering oracle; no tolerance was relaxed to make
these checks pass.
