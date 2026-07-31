# Heterogeneous-volume candidate walk

This checkpoint implements the candidate free-flight state machine from
`volume_integrate_advance()` in Blender/Cycles main
`0ae970969e3f37eb63dc5be5701dbc93885fbfae`. There were no `intern/cycles`
changes between the previous `b82c3f0d` pin and this latest fetched main
revision. Cycles remains the only renderer oracle; this regression executes
the Luisa implementation on device and does not add a Psycles CPU reference
renderer.

## Component boundary

`HeterogeneousVolumeCandidateWalk` is a host-stage OOP component that records
the state machine into the Luisa AST. It consumes two abstract components:

- a monotone `VolumeMajorantSegmentSequence`, implemented in production by
  the ordered multi-root hierarchy traversal; and
- a random-access `HeterogeneousVolumeTrackingRandomSource`, which maps an
  explicit copied RNG offset to Cycles scatter-distance and 2D shade-offset
  samples.

The walker owns only free flight and boundary continuation. Raw closure
evaluation, real/null coefficients, reservoir selection, VSPG, phase
closures, and direct-light MIS remain independent components for the
production segment composer.

## Exact transition

For current segment `[t, t_max]`, majorant `sigma`, and uniform `r`, the
walker applies:

```text
dt = -log(1 - r) / sigma
candidate_t = t + dt
```

`sigma == 0` advances without evaluating the quotient. A candidate exactly at
`t_max` belongs to the current segment because Cycles advances only when
`candidate_t > t_max`. When the candidate lies beyond the segment, the same
uniform variate is reused after removing the already consumed optical depth:

```text
r' = saturate(1 - (1 - r) * exp((t_max - t) * sigma))
```

Every boundary crossed before that candidate receives the shade sample from
the current copied tracking offset. A successful candidate advances the
offset by one 16-dimension bounce block; an exhausted traversal does not.
The primary-ray optical-depth statistic accumulates the unscaled hierarchy
majorant, while the free-flight rate alone receives the VSPG majorant scale.

Cycles tests the step limit as `step++ > 1024`. Consequently, candidate
attempts numbered 1 through 1025 are accepted and the next call terminates
with observable step value 1026. The implementation preserves that domain
rather than replacing it with the superficially similar `>= 1024`.

## Device regression

The first fixture crosses a zero-majorant segment and two nonzero segments
using one offset. It then crosses another boundary using the next offset:

| candidate | distance | local `dt` | rate | residual `r` | optical depth |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `2.30471896` | `0.304718956` | `2.0` | `0.456343634` | `5.0` |
| 2 | `5.04292184` | `1.04292184` | `0.5` | `0.406347364` | `7.0` |

The successful boundary shade sequence is `0.25, 0.25, 0.75`, proving that
multiple crossings within a candidate retain one offset and the next
candidate observes `+16`. A third attempt exhausts the traversal at distance
`14.2532622` without consuming another bounce block. A separate zero-distance
fixture pins 1025 successful candidates, final failed step 1026, and final
offset `1025 * 16`.

Fresh focused results:

| backend | wall time | result |
| --- | ---: | --- |
| fallback (LLVM) | `0.06 s` | pass |
| HIP | `0.14 s` | pass |
| Vulkan/RADV | `0.06 s` | pass |

The complete build used `--parallel 32` and completed in `8.64 s`. Serial
CTest passed `108/108` tests in `9.36 s`, including the source-size gate.

This checkpoint changes an internal stochastic state transition rather than a
complete pixel estimator, so no image is presented as visual evidence.

## Reproduction

```sh
cmake --build build \
  --target psycles_luisa_heterogeneous_volume_candidate_tests \
  --parallel 32
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_heterogeneous_volume_candidate_(fallback|hip|vk)$' \
  --parallel 3
```
