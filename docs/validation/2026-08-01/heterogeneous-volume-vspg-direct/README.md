# Heterogeneous-volume VSPG and direct lighting

This checkpoint composes Cycles' heterogeneous volume-scattering probability
guiding (VSPG), independent direct-light estimator, and distance/equiangular
MIS into the production Luisa path AST. Blender/Cycles
`main@6f7add4a791e69f23bcc7ff0bdf4ea0307b002c5` is the only renderer oracle.
There were no `intern/cycles` changes between this pin and the previously
inspected `0ae970969e3f` revision. Psycles adds no CPU reference renderer and
does not bake, fit, or preprocess Blender material closures.

## Host-stage composition

The implementation uses ordinary host-language polymorphism to generate one
fused Luisa device AST. The former segment monolith now composes typed
components in real `.h + .cpp` translation units:

- `HeterogeneousVolumeCollisionProvider` evaluates the original stacked
  `GraphSurface` closure tree at every requested point;
- `HeterogeneousVolumeScatterProbability` owns the scalar history-dependent
  VSPG probability and majorant scale;
- `HeterogeneousVolumeCandidateWalk` owns free flight and exposes a
  non-mutating observer boundary to coupled estimators;
- `HeterogeneousVolumeReservoir` owns streaming scatter candidates and the
  final defensive scatter/transmit resampling;
- `HeterogeneousVolumeTransmittance` owns Cycles' randomized telescoping
  residual-ratio estimator; and
- `HeterogeneousVolumeSegmentComponent` composes those pieces with raw phase
  recovery and direct distance/equiangular MIS.

No device virtual dispatch is introduced. Host virtual calls run while Luisa
traces the kernel and append the corresponding typed AST. The source-size
gate passes, and the largest implementation file in this checkpoint is
`src/luisa/heterogeneous_volume_segment.cpp` at 858 lines.

## Cycles measure and random domains

The composed estimator preserves the current Cycles control flow:

1. direct-technique choice reads the original unscrumbled
   `PRNG_VOLUME_SCATTER_DISTANCE` sample;
2. the copied tracking offset is then Hash-Prospector-scrambled with
   `0xe35fad82`;
3. VSPG derives the requested scatter probability from accumulated
   scatter/transmit radiance, computes the minimum majorant scale, and applies
   the same `0.75` defensive mixture at final event selection;
4. scatter candidates stream through one reservoir using Cycles' conditional
   uniform remapping, while transmission is added only in final VSPG
   resampling;
5. distance NEE reuses the chosen scatter candidate; equiangular NEE observes
   the same candidate walk and never duplicates traversal;
6. equiangular transport without MIS evaluates residual-ratio transmittance
   independently over each immutable majorant segment;
7. equiangular transport with MIS records the null-event product and scalar
   majorant at the exact crossing point;
8. both MIS arms use the same local scatter probability as indirect tracking
   and apply Cycles' `2 * power_heuristic` one-sample weight; and
9. every selected direct or indirect point re-evaluates the original closure
   graph to recover raw phase closures. No coefficient grid or pre-baked
   material replaces that evaluation.

The residual-ratio estimator pins the biased sample count
`k = clamp(round(sigma_range * length), 1, 1024)`, a truncated geometric
power-of-two expansion count, paired lower-order estimators, and the
single-term telescoping correction. A production audit found and fixed one
random-domain error: Cycles deliberately samples
`PRNG_VOLUME_EXPANSION_ORDER` with `rng_hash = 0` so all pixels share the
expansion order and GPU work remains synchronized. The named
`sample_volume_expansion_order()` boundary and Sobol regression now prevent a
caller from substituting the ordinary pixel hash.

## Regression evidence

The segment fixture now pins 26 float records against the current Cycles
equations. It covers weighted-delta recursion, empty-history VSPG, both final
scatter and transmit selections, raw phase recovery, non-MIS distance NEE,
and both distance and equiangular MIS arms. The dedicated transmittance fixture
forces `k=2`, `N=4`, `PMF=0.009`, and eight raw closure evaluations, then
separately pins the `sigma_range == 0` biased-only branch.

Warm focused results:

| boundary | fallback | HIP | Vulkan |
| --- | ---: | ---: | ---: |
| heterogeneous segment | `0.26 s` | `0.31 s` | `1.42 s` |
| residual-ratio transmittance | `0.03 s` | `0.11 s` | `0.04 s` |
| production majorant scene | `0.09 s` | `0.11 s` | `0.08 s` |
| production volume path | `0.18 s` | `0.40 s` | `1.22 s` |

The focused set passed `12/12` tests in `4.27 s`. The complete build used
`--parallel 32` and completed in `24.93 s`. Parallel CTest passed all
`114/114` tests in `53.30 s`, including fallback, HIP, Vulkan, source-size,
OpenEXR, Blender export, and production path coverage.

## Honest scene boundary

The camera-path heterogeneous estimator now includes VSPG and direct lighting,
but heterogeneous shadow rays still use the existing homogeneous volume-shadow
component. The spatial-volume scene capability gate therefore remains closed.
Opening it before residual-ratio shadow traversal is connected would produce
biased full-scene pixels.

Consequently this checkpoint has no honest Cycles/Psycles full-frame image or
triptych. The next acceptance boundary is to reuse the raw collision and
residual-ratio components for heterogeneous shadow-stack traversal, remove the
gate only after three-backend regressions pass, and then generate the requested
Lone Monk/Splash/Classroom EXRs, visual triptychs, numerical comparisons, and
Cycles CPU/HIP versus Psycles fallback/HIP/Vulkan timing matrix.

## Reproduction

```sh
cmake --build build --parallel 32
ctest --test-dir build --parallel 8 --output-on-failure
ctest --test-dir build --output-on-failure -j1 \
  -R 'psycles\.luisa_(heterogeneous_volume_segment|heterogeneous_volume_transmittance|volume_majorant_scene|volume_path)_(fallback|hip|vk)$'
```
