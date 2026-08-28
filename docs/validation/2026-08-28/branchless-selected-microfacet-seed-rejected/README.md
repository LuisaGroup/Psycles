# Rejected branchless selected-microfacet mixture seed

## Result

This experiment removed the independent regular reflection-microfacet replay
from the selected BSDF path and seeded the existing mixture fold with the
conditional contribution computed while the sampled half-vector was live. It
was rejected and removed: the complete Barbershop `shade_surface` continuation
was 0.053% slower in the fresh interleaved comparison, render-only time was
0.249% slower, and the HIP surface object grew by 0.564%. These differences
are noise/slight regression rather than a useful speedup.

The negative result is stronger than the earlier per-iteration skip attempts.
The competitor traversal here had no divergent `if (index != selected)` arm:
it used one formally bijective rank projection and evaluated exactly `n - 1`
closures. The local conditional regular-GGX probe fell from roughly 13.52 ms
for sample plus independent replay to 10.8 ms for the sampled witness, but
that arithmetic is not a dominant fraction of the complete production
continuation. The next surface/SVM work should target material value execution
and closure population rather than adding more selected-lobe state.

## Formal model

Let a populated closure sequence have indices `I_n = {0, ..., n - 1}` and let
`s` be the selected index. The compact competitor rank `r` is projected by

```text
phi_s(r) = r + [r >= s],       0 <= r < n - 1.
```

For `r < s`, `phi_s(r) = r`; for `r >= s`, `phi_s(r) = r + 1`. The two image
ranges are disjoint, ordered, and their union is `I_n \\ {s}`. Thus `phi_s` is
a bijection from `I_(n-1)` to every competitor closure, with neither omission
nor duplication. Since exclusion was enabled only after a valid positive-
measure selection, `n >= 1` and the `n - 1` loop bound could not underflow.

For closure weights `a_i`, conditional densities `p_i(w)`, and contributions
`g_i(w)`, the baseline mixture fold computes

```text
P(w) = sum_i a_i p_i(w) / sum_i a_i
G(w) = sum_i g_i(w).
```

The candidate initialized the fold with `(a_s p_s(w), g_s(w), a_s)` returned
by the selected regular-reflection sampler, then folded `phi_s(r)` for all
competitor ranks. Commutativity and associativity of the mathematical sums
give the same mixture. The sampler and evaluator shared one microfacet-witness
formula, so this was not a second hand-maintained approximation.

This proof establishes semantic coverage; it does not establish profitability.
The production measurement shows that transporting and selecting the seed,
while retaining the generic competitor evaluator for every other domain, did
not reduce fixed resources or end-to-end work enough to offset its code cost.

## Structure and HIP profile

Both builds used commit `0a8412f` plus the uncommitted candidate where noted,
LuisaCompute `ca29d37d`, ROCm 7.2.4, the RX 9070 XT, 640x480, 64 fixed samples,
Tabulated Sobol, compact surface values, one surface population, and the staged
wavefront scheduler. Adaptive sampling and denoising were disabled.

| Metric | Exact baseline | Candidate | Change |
|---|---:|---:|---:|
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |
| Final surface LLVM | 53,989 lines / 3,008,050 B | 54,396 lines / 3,030,257 B | +0.738% bytes |
| HIP surface object | 340,184 B | 342,104 B | +0.564% |
| Fixed private / VGPR | 3,096 B / 256 | 3,096 B / 256 | unchanged |
| Fresh `shade_surface` | 26.7574 ns/item | 26.7736, 26.7695 ns/item | +0.053% mean |
| Render-only | 2.51794 s | 2.51925, 2.52917 s | +0.249% mean |

The fresh baseline agrees with the three retained exact-baseline samples
`26.7558`, `26.7645`, and `26.8457 ns/item`. Normalizing each profile by its
actual `grid_x * grid_y * grid_z` work leaves the candidate inside noise and
on the slower side in both runs.

## Correctness and visual inspection

The microfacet anisotropy, physical-closure, and surface-population suites
passed on fallback and HIP (6/6). The candidate was rejected at this gate, so
strict native-XIR Vulkan was not run for code that would not be retained.

All 15 named Barbershop film passes contain zero invalid pixels. Combined has
RMSE `3.69554e-4`, mean absolute error `2.92355e-6`, and mean-luminance ratio
`1.00000598`; Normal has RMSE `1.11459e-8`. Glossy Indirect has a sparse
high-energy outlier (RMSE `4.44583e-3`) but a mean-luminance ratio of
`1.00019876`.

I inspected the three triptychs at native resolution. Geometry, UV/texture
placement, floor, ceiling, cabinets, material regions, lighting, and normal
orientation agree. The amplified panels show sparse stochastic/atomic-arrival
differences, not a coherent rendering change.

- [All-pass numerical report](all-pass-report.json)
- [Combined triptych](triptychs/combined.png)
- [Glossy Indirect triptych](triptychs/glossind.png)
- [Normal triptych](triptychs/normal.png)

## Reproduction

```sh
cmake --build build --parallel 32 --target \
  psycles_render_blender_scene \
  psycles_benchmark_surface_closures \
  psycles_luisa_microfacet_anisotropy_tests \
  psycles_luisa_surface_closure_physical_tests \
  psycles_luisa_surface_population_tests

ctest --test-dir build --output-on-failure -j2 \
  -R '^psycles\\.luisa_(microfacet_anisotropy|surface_closure_physical|surface_population)_(fallback|hip)$'

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
LUISA_CORO_SHADER_MAP=1 \
rocprofv3 --kernel-trace --scratch-memory-trace --stats \
  -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene BARBERSHOP_EXPORT out.exr hip \
    640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
    32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```

The candidate source and its now-unused mapping helper are intentionally
absent from production. Machine-readable metrics remain in
[`metrics.json`](metrics.json).
