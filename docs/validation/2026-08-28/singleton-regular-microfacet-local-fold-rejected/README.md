# Rejected singleton regular-microfacet local fold

## Outcome

This experiment tested the narrowest form of reusing the regular reflection
microfacet contribution produced while the sampled half-vector is live. The
implementation was **rejected and is not present in the source tree**: two HIP
traces made `shade_surface` 0.83% slower, enlarged the main entry by 2,104 B,
and increased fixed private storage by 16 B.

The result does not reject the Cycles sampling model. It rejects adding a
second runtime property/count branch around the existing Psycles mixture
evaluator for only one closure family. Cycles instead makes every BSDF sampler
return its conditional value and PDF, initializes the mixture with that value,
and evaluates only the other closures. That branch-free algebraic form remains
the relevant next experiment.

## Formal domain

Let the retained closure measure contain exactly one closure `c`, and assume
that `c` is a regular reflection microfacet closure. Its categorical weight is
`a > 0`, conditional density is `p_c(w)`, and weighted contribution is
`g_c(w)`. The balance mixture is therefore

```text
P(w) = a p_c(w) / a = p_c(w)
G(w) = g_c(w).
```

The sampler already has the exact sampled half-vector `H`, `I.H`, Fresnel,
distribution, masking-shadowing terms, and Jacobian. The candidate computed
`g_c(w)` and `p_c(w)` there and transported them through the existing delta
value/PDF lanes; it added no device-ABI fields. Only when the retained count
was one and the selected sample was a non-delta glossy sample did the final
mixture use these lanes. Every other closure domain and every multi-closure
mixture retained the established evaluator.

This is semantically valid, but its generated control-flow cost is not free.
In particular, the candidate introduced a dynamic conjunction over retained
count and sampled properties, followed by a merge of the full
`SurfaceEvaluation` tuple. It still retained the original evaluator for the
other arm.

## Validation

The existing microfacet anisotropy, physical-closure, closure-collection, and
surface-population suites passed on fallback, HIP, and strict native-XIR
Vulkan: 12/12 tests. A cold 640x480 one-sample Barbershop build showed that the
sampled half-vector did not escape into the coroutine frame.

| Metric | Exact baseline | Candidate | Change |
|---|---:|---:|---:|
| Coroutine frame | 177 fields / 864 B | 177 fields / 864 B | unchanged |
| HIP surface object | 357,880 B | 359,928 B | +2,048 B |
| Main entry | 302,604 B | 304,708 B | +2,104 B |
| Fixed private storage | 3,296 B | 3,312 B | +16 B |
| VGPR count | 256 | 256 | unchanged |

The retained comparison uses fixed 640x480, 64 spp, Tabulated Sobol, adaptive
sampling disabled, compact surface values, single population, and the staged
wavefront scheduler. Kernel time is divided by the actual launched work; call
count and queue packing therefore cannot bias the comparison.

| Run | `shade_surface` ns/item |
|---|---:|
| exact baseline A | 27.224716569 |
| exact baseline B | 27.192803146 |
| candidate A | 27.405217591 |
| candidate B | 27.465925775 |
| baseline mean | 27.208759858 |
| candidate mean | 27.435571683 |

The candidate regresses the normalized stage by 0.8336%. Both candidate
samples are slower than both baseline samples, consistent with the increased
private allocation and entry size. The code was reverted rather than hidden
behind a scene/backend special case.

## Reproduction

```sh
cmake --build build --parallel "$(nproc)" --target \
  psycles_render_blender_scene \
  psycles_luisa_microfacet_anisotropy_tests \
  psycles_luisa_surface_closure_physical_tests \
  psycles_luisa_surface_closure_collection_tests \
  psycles_luisa_surface_population_tests

ctest --test-dir build --output-on-failure -j1 \
  -R 'psycles\.luisa_(microfacet_anisotropy|surface_closure_collection|surface_population|surface_closure_physical)_(fallback|hip|vk)$'

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
LUISA_CORO_SHADER_MAP=1 \
rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene BARBERSHOP_5_2_EXPORT out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576
```
