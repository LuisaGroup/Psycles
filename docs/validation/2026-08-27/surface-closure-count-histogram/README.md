# Runtime surface-closure count census

## Question and Cycles oracle

Cycles 5.2 avoids categorical closure selection when
`ShaderData::num_closure <= 1` and only rebuilds the full BSDF mixture after a
sample when `num_closure > 1`. Psycles currently performs a second closure
program traversal for inverse-CDF selection for every non-BSSRDF surface,
regardless of the post-population count. A static material census cannot prove
how often a runtime fast path would apply because graph control flow, closure
weights, visibility, and path depth all change the retained closure sequence.

The source oracle is the local Blender 5.2 tree at
`9e2066a5d571f50afdf84c6f509a94c08e329f55`, specifically
`intern/cycles/kernel/integrator/surface_shader.h`. This diagnostic measures
Psycles' corresponding post-population count at the actual surface consumer;
it does not pre-bake a Cycles result or add a CPU reference renderer.

## Event and storage model

Let `C(e)` be the number of retained physical closures after population for a
surface-shading event `e`. Both Cycles and Psycles cap this sequence at 64, so
the histogram map is injective:

```text
H[i] = |{ e : C(e) = i }|, 0 <= i <= 64.
```

Exactly one event is recorded after successful surface preparation. A BSSRDF
exit records the synthetic one-closure local domain consumed by directional
sampling, rather than the original material population that is not consumed
at that point. Thus the measurement is hit- and bounce-weighted, not a count
of materials or authored nodes.

The feature is an optional host/JIT policy. When it is absent, the production
shader AST contains neither an atomic increment nor a device-side enable
branch. When present, bin `i` is stored in one `float4`; the lane is the
deterministic projection

```text
lane = (pixel_id + absolute_sample_index) mod 4.
```

Every increment is exactly one. The host accepts a result as exact only when
each lane is finite, nonnegative, integral, and strictly below `2^24`, the
largest consecutive integer boundary for binary32. It then widens and sums
the four lanes into a `uint64_t` bin. This makes overflow or lost-unit
precision explicit instead of silently presenting an approximate census.

The histogram reuses the diagnostic buffer with the disjoint layout

```text
[ optional path-trace prefix ][ 65 histogram bins ].
```

The path-trace reader and histogram reader both use bounded subviews. During
development, the combined path-trace-plus-histogram regression exposed that
the old path-trace reader copied the full extended buffer into a prefix-sized
host span. The resulting fallback heap corruption was fixed at the
representation boundary: the reader now copies exactly the path-trace schema
view. This is required for every future diagnostic suffix, not a fallback
special case.

## Regression coverage

`psycles_luisa_sample_dispatch_film_tests` enables path tracing and the
histogram simultaneously. Its fixture has exactly three retained closures and
checks:

- one count-3 event for each of the 48 `(pixel, absolute sample)` pairs;
- exact equality between a single request and requests split at sample 3;
- an exact histogram result on fallback, HIP, and Vulkan;
- the existing Combined, Normal, Albedo, and light-pass comparisons across
  megakernel, per-sample, wavefront, graph-wavefront, staged-wavefront, direct
  light queue, and persistent scheduling paths.

Commands:

```sh
cmake --build build --parallel "$(nproc)"
./build/bin/psycles_luisa_sample_dispatch_film_tests fallback
./build/bin/psycles_luisa_sample_dispatch_film_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
  ./build/bin/psycles_luisa_sample_dispatch_film_tests vk
```

All three passed. The Vulkan log reported native SPIR-V optimization and
compilation throughout; DXC was disabled.

The same environment ran the complete suite: 289/295 passed. The six failures
are the unchanged numeric-oracle set already recorded by the anisotropy
checkpoint: `stacked_volume_fallback`, `homogeneous_volume_fallback`,
`area_light_forward_vk`, `volume_path_fallback`, `volume_path_vk`, and
`volume_triangle_fallback`. No histogram or sample-dispatch regression failed.

## Hit-weighted scene results

The official Blender 5.2 exports were rendered on the Radeon RX 9070 XT with
HIP staged wavefront, 640x480, fixed sampling, and the diagnostic enabled.
Barbershop used 16 spp; Classroom and Monster Under the Bed used 8 spp. These
instrumented render times are not performance measurements because every
surface event performs an atomic increment.

| Scene | Surface events | count 0 | count 1 | count 2 | count 3+ |
|---|---:|---:|---:|---:|---:|
| Barbershop | 13,412,983 | 1.5513% | 55.1439% | 40.3069% | 2.9978% |
| Classroom | 5,822,695 | 4.3091% | 66.6933% | 28.9968% | 0.0007% |
| Monster Under the Bed | 5,060,902 | 2.6165% | 19.1189% | 39.7939% | 38.4706% |

Every artifact reports `exact: true`. Full bins are preserved in
[`barbershop-hip.json`](barbershop-hip.json),
[`classroom-hip.json`](classroom-hip.json), and
[`monster-hip.json`](monster-hip.json).

The Barbershop command was:

```sh
./build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 \
  /var/tmp/psycles-barbershop-closure-hist-20260827.exr \
  hip 640 480 16 16 - 0 0 0 0 16 - 1 0 \
  wavefront-staged 32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  /var/tmp/psycles-barbershop-closure-hist-20260827.json
```

Classroom and Monster used the same command topology with their Blender 5.2
export directories and `8 8 ... 8` sample fields.

## Optimization consequence

The necessary and sufficient bypass condition is the runtime retained count
`C(e) == 1`; authored node count, material identity, positive-weight guesses,
and backend names are not equivalent predicates. The next implementation can
therefore retain the unique candidate during the population/measure traversal
and bypass only the second categorical traversal for `C(e) == 1`. Zero and
multi-closure events keep the existing path. The sampler/evaluator ABI remains
unchanged, avoiding the previously rejected selected-contribution transport
experiment and its code-object growth.

The three scenes also show why this must remain a runtime structural
optimization rather than a Barbershop specialization: singleton coverage is
55.14%, 66.69%, and 19.12% respectively, while Monster retains materially
larger mixtures on many paths.
