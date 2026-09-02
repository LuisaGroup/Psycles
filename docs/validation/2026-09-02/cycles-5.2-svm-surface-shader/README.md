# Cycles 5.2 SVM surface-shader validation

## Result

Psycles now has a direct Luisa projection of the non-guided closure-array
consumer in Blender Cycles 5.2.1
`kernel/integrator/surface_shader.h`. It consumes the exact retained
`ClosurePool`; it does not translate entries into the legacy Psycles surface
closure model.

This checkpoint completes the collection-level boundary needed above the
already copied `kernel/closure/bsdf.h` dispatch:

- `surface_shader_bsdf_bssrdf_pick`;
- `surface_shader_bssrdf_sample_weight`;
- `_surface_shader_bsdf_eval_mis`;
- `surface_shader_bsdf_eval`;
- `surface_shader_bsdf_sample_closure`;
- `film/light_passes.h::BsdfEval` diffuse/glossy/sum classification;
- exact ClosureType interval predicates used by those routines.

The pinned source is Blender Cycles commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`.

## Formal model

Let `E` be the ordered retained indices whose type satisfies Cycles'
`CLOSURE_IS_BSDF_OR_BSSRDF`, and let `s_i >= 0` be each entry's
`sample_weight`. Closure selection is inverse transform sampling over the
finite measure

```text
S = sum(i in E, s_i)
r = random.z * S
i = first index such that r < prefix_i + s_i
random.z' = (r - prefix_i) / s_i
```

The strict comparison, retained-array order, and z-coordinate reuse are
observable Cycles semantics. No alias table or reordered reduction is
substituted at this boundary. Cycles' `num_closure <= 1` fast path returns
entry zero and leaves the random vector unchanged; the Luisa copy preserves
that condition exactly.

The returned pick contains only that index and the rescaled random vector,
matching Cycles' closure-pointer-plus-in/out-random interface. Closure type,
weight, selection sum, and validity are read from the retained pool at their
point of use instead of being duplicated into live transport state.

For a direction `w`, every BSDF entry has directional density `p_i(w)` and
unweighted value `f_i(w)`. BSSRDF entries have selection mass but no
directional density. The one-sample balance-heuristic mixture is

```text
p(w) = sum(i in BSDF, s_i * p_i(w)) / sum(i in E, s_i)
f(w) = sum(i in BSDF, weight_i * f_i(w))
```

The sampled-BSDF path seeds both reductions with the selected closure and
executes the same fold over the complement. Thus the partition theorem

```text
fold({selected}) + fold(E \\ {selected}) = fold(E)
```

guarantees that evaluating the sampled direction independently produces the
same mixture value and PDF. The regression checks this identity on-device;
the implementation does not duplicate the mixture algebra.

Light-link exclusion affects `f(w)` only. The excluded closure remains in
both selection mass and PDF sums, matching Cycles. A light without
`SHADER_USE_MIS` receives the same value but a returned BSDF PDF of zero.
Average squared roughness is weighted by `s_i * p_i(w)` and is undefined only
when that total is zero, where Cycles returns zero.

For a selected BSSRDF closure `j`, Cycles' transport weight is

```text
weight_j * S / s_j
```

when more than one retained closure exists. This exactly cancels the
selection probability `s_j / S`; the regression freezes the cancellation
with two BSDF entries and one BSSRDF entry.

## Permanent regression

`tests/test_luisa_cycles_svm_surface_shader.cpp` covers:

- all ClosureType predicate values from 0 through 43 against the exact source
  intervals;
- the three inverse-CDF regions of a `(0.25, 0.75, 0.5)` selection measure;
- exact random-z rescaling and retained closure indices;
- BSSRDF selection and `S / s_j` transport weighting;
- analytic Lambert mixture value/PDF with BSSRDF denominator participation;
- sampled-direction versus independent-evaluation partition equivalence;
- `SHADER_USE_MIS` and diffuse light exclusion semantics;
- the single-closure fast path;
- a non-scattering closure preceding the only eligible BSDF.

The analytic case has identical normal-aligned Diffuse closures with weights
`(0.2, 0.4, 0.6)` and `(0.3, 0.2, 0.1)`. At cosine `c`, the frozen oracle is

```text
f = (0.5, 0.6, 0.7) * c / pi
p = (c / pi) * (0.25 + 0.75) / (0.25 + 0.75 + 0.5)
```

Ordinary backend floating-point differences use the existing scattering-test
tolerance; no ULP-specific slow path is introduced.

## Commands and results

```bash
cmake --build build --parallel 32 --target \
  psycles_luisa_cycles_svm_surface_shader_tests

ctest --test-dir build --output-on-failure -R \
  '^psycles\.luisa_cycles_svm_surface_shader_(fallback|hip|vk)$'

cmake --build build --parallel 32
ctest --test-dir build --output-on-failure
```

Result: fallback, HIP, and Vulkan passed 3/3. The Vulkan registration requires
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`; therefore the Vulkan result is the native
XIR-to-SPIR-V route.

The complete project suite passed 540/540 in 143.01 seconds after the full
32-thread rebuild.

No full-scene image or performance claim is made at this isolated consumer
checkpoint. Production `shade_surface` still needs a scene-backed exact
`KernelGlobals` projection and an explicit route switch before a Cycles /
Psycles / difference triptych would measure this code.

## Remaining boundary

The next structural step is to construct the production `KernelGlobals`
adapter over the already uploaded native SVM scene tables, populate one exact
`ShaderData`/`ClosurePool` per hit, and map this collection result into the
unified path-kernel ABI. Only after whole-program equivalence tests pass can
the legacy `SurfaceClosureEvaluator` route be made unreachable and removed.
