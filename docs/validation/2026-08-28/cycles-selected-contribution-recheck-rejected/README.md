# Cycles selected-contribution recheck

## Outcome

The refined candidate was rejected and fully reverted. On the current
`cd69bb8` surface baseline it preserved the 177-field, 864-byte coroutine
frame, but enlarged the HIP surface object by 1.25%, enlarged the main entry
by 1.51%, added 16 bytes of private storage per work-item, and made normalized
Barbershop `shade_surface` time 0.51% slower. No renderer change or test
tolerance from the candidate is retained.

This is a second, newer-baseline confirmation of
[`selected-sample-contribution-rejected`](../../2026-08-27/selected-sample-contribution-rejected/README.md).
The recheck was still useful: it reused the sampled microfacet half-vector
directly, covered every currently represented regular closure family, and
therefore rules out half-vector reconstruction alone as the reason the former
candidate lost.

## Reference identity

- Retained Psycles baseline: `cd69bb85ef5ee488b78aff66dfbd4a9254ffe5ef`.
- LuisaCompute: `eeda4b154fcf43e8709d1b42478e958677b9c6ae`.
- Cycles source: Blender 5.2 release at
  `9e2066aef7ef7e20c142ad7bd3303138a4304c93`.
- Device: AMD Radeon RX 9070 XT (`gfx1201`), HIP backend.
- Scene: official Blender 5.2 Barbershop export, 640x480, 64 fixed samples,
  Tabulated Sobol, adaptive sampling disabled, staged wavefront scheduler.

Cycles' reference implementation is
`intern/cycles/kernel/integrator/surface_shader.h`. Its selected closure
sampler returns the conditional value and PDF. The MIS evaluator initializes
the mixture with that result and visits all closures except the selected
`ShaderClosure *`.

## Formal model

Let retained closures be the ordered finite sequence
`C = (c_0, ..., c_(n-1))`. Closure `c_i` has categorical mass `a_i`,
conditional directional density `p_i(w)`, and transport contribution `g_i(w)`.
If categorical inversion selects `s` and its conditional sampler produces
direction `w`, the existing all-closure fold is

```text
G_old(w) = sum_i g_i(w)
P_old(w) = sum_i a_i p_i(w) / sum_i a_i.
```

The candidate formed

```text
G_new(w) = g'_s(w) + sum_(i != s) g_i(w)
P_new(w) = (a_s p'_s(w) + sum_(i != s) a_i p_i(w)) / sum_i a_i.
```

Thus equivalence follows from the local sampler contract
`g'_s(w) = g_s(w)` and `p'_s(w) = p_s(w)`. The implementation established
that contract separately for diffuse, translucent, rough translucent, sheen,
regular reflection microfacet, thin glass, and dielectric glass/refraction.
For regular microfacets it evaluated the sampled VNDF witness directly instead
of reconstructing `H = normalize(I + O)`. Delta, transparent, and BSSRDF
domains continued to use their existing categorical denominator and special
transport numerator in `finish()`; they were not reinterpreted as
finite-direction densities.

The proof is over real arithmetic. Different floating-point addition order and
sampled-witness versus reconstructed-witness rounding need not be bitwise
identical. Existing differential tests use their established scale-aware
tolerance for scattering values; exact ABI/hash tests were not weakened.

## Implementation shape tested

The candidate added no device ABI field. It temporarily reused the existing
conditional-sample value/PDF lanes for regular closures, constructed the
selected closure's complete fold contribution after sampling, seeded the
accumulator with that contribution, and guarded the generic closure evaluator
with `index != selected_index`. The remaining closure evaluator always ran in
the non-selected mode.

This shape is semantically complete, but it simultaneously keeps the selected
result live and adds a skip diamond around the generic typed handler. The HIP
compiler did not turn that representation into a smaller surface program.

## Backend regression gate

The candidate was built with all 32 host threads. The following 15 existing
differential tests passed sequentially on fallback, HIP, and strict native
XIR-to-SPIR-V Vulkan:

```text
psycles.luisa_microfacet_anisotropy_{fallback,hip,vk}
psycles.luisa_surface_closure_collection_{fallback,hip,vk}
psycles.luisa_surface_population_{fallback,hip,vk}
psycles.luisa_compact_surface_preparation_{fallback,hip,vk}
psycles.luisa_surface_closure_physical_{fallback,hip,vk}
```

In particular, the population test compares the populated-array sampling path
against the independent authored-closure fold, so the test observes the whole
selected seed plus competitor sum rather than only one helper in isolation.

## Cold generated structure

Both rows use the same one-sample Barbershop shader AST and a disabled Psycles
shader cache. Static metadata came from `llvm-nm` and `llvm-readelf` on the
dumped HIP object.

| Metric | Exact baseline | Candidate | Change |
|---|---:|---:|---:|
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |
| HIP surface object | 357,880 B | 362,360 B | +4,480 B (+1.252%) |
| Main entry | 302,604 B | 307,188 B | +4,584 B (+1.515%) |
| Private storage | 3,296 B | 3,312 B | +16 B |
| VGPR | 256 | 256 | unchanged |
| Reported VGPR spills | 363 | 357 | -6 |
| Surface kernel hash | `048185b34d9b6ba1` | `c31d9c401698b1a8` | changed |

The six fewer reported spill sites do not offset the larger private allocation
or executed program.

## HIP measurement

Each trace rendered the same 640x480 image at 64 fixed samples. Kernel time is
normalized by the actual `grid_x * grid_y * grid_z` sum so the scheduler's two
extra dispatches in the candidate do not bias the comparison.

| Build / run | Calls | Work-items | GPU duration (ns) | ns/work-item |
|---|---:|---:|---:|---:|
| Baseline 1 | 293 | 53,658,304 | 1,460,832,118 | 27.224716569 |
| Baseline 2 | 293 | 53,658,336 | 1,459,120,568 | 27.192803146 |
| Candidate 1 | 295 | 53,659,008 | 1,467,428,992 | 27.347300047 |
| Candidate 2 | 295 | 53,658,976 | 1,467,475,565 | 27.348184300 |
| Baseline mean | | | | 27.208759858 |
| Candidate mean | | | | 27.347742174 |
| Candidate change | | | | **+0.511%** |

Candidate render-only wall times were 2.54250 s and 2.54142 s. The normalized
surface result is the decision metric because it isolates the modified stage.

## Consequence

The Cycles estimator is not the problem; the current Luisa representation of
that estimator is. Another local seed/skip variation would repeat an already
falsified hypothesis. A profitable next step must change the translation
shape itself: reduce typed-handler duplication and live ranges, or execute a
compact closure/SVM program whose sample operation naturally returns the
selected conditional value without transporting an additional aggregate
through the surrounding surface CFG. That work still requires the same
per-family equivalence law and backend differential gates.
