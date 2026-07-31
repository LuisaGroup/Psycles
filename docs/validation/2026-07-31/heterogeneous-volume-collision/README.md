# Heterogeneous volume collision measure

This checkpoint implements the local null/real collision transition used by
the current Cycles heterogeneous volume integrator. Cycles main `b82c3f0` is
the sole oracle. Psycles does not contain a CPU reference integrator, and this
checkpoint does not pre-bake or replace a Blender volume graph.

## Formal boundary

For a sampled scalar majorant `m` and extinction `sigma_t`, the component
constructs

```text
M       = max(m, max_channel(sigma_t))
sigma_n = M - sigma_t
```

The preprocessing contract is `max_channel(sigma_t) <= m`. Its failure is
returned as `majorant_exceeded`; it is never interpreted as a supported
approximation. With a sound bound, `M == m` and every candidate shares the
normalization `throughput / m`.

For a scattering closure, the event measure is the current Cycles
throughput/albedo channel mixture:

```text
sigma_c = sigma_s + sigma_n
albedo  = safe_divide(sigma_s, sigma_t, fallback = 1)
q       = normalize(abs(throughput * albedo))
p_s     = dot(sigma_s / sigma_c, q)
p_n     = 1 - p_s
```

The two continuations are emitted together:

```text
real throughput = normalized_throughput * sigma_s / p_s
null throughput = normalized_throughput * sigma_n / p_n
```

Absorption-only candidates do not sample a zero-contribution absorption
event. They deterministically take the null branch and leave the scalar
null-event probability product unchanged, matching
`volume_integrate_step_scattering()`.

The same component pins exponential free flight
`-log(1-random)/majorant`. `cycles_sampler` now also exposes Cycles'
Hash Prospector functions and the four-aligned path-offset scramble used
before the variable-length collision walk.

## Regression

The device fixture covers:

- a sound RGB majorant with nontrivial real and null probabilities;
- an absorption-only medium, including a zero-extinction channel;
- an intentionally violated majorant and its defensive Cycles correction;
- real/null random-number rescaling;
- zero-rate and nonzero-rate exponential free flight;
- exact `hash_hp_uint`, seeded hash, and aligned offset bits.

Run:

```sh
cmake --build build \
  --target psycles_luisa_heterogeneous_volume_tests \
  --parallel 32
ctest --test-dir build --output-on-failure \
  -R 'psycles\.luisa_heterogeneous_volume_(fallback|hip|vk)' \
  --parallel 3
```

Result: all three backends pass. This is an internal measure test, not an
image comparison, so no triptych is applicable at this checkpoint. The first
heterogeneous image triptych will accompany the production path connection
and official Blender/Cycles EXR oracle.

## Remaining connection

This component deliberately does not invent a density bound. The subsequent
majorant-hierarchy checkpoint reproduces Cycles' sampled hierarchy and
single-root traversal. Its finite shader samples estimate rather than prove a
bound, while this component keeps the formal violation visible. Raw Luisa
shader evaluation, overlapping-root reduction, candidate RNG advancement,
VSPG reservoir selection, phase recovery, and direct-light MIS still have to
be composed. Until those pieces are present, the existing scene capability
diagnostic continues to reject heterogeneous volume graphs.
