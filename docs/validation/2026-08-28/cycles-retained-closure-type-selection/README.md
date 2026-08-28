# Cycles retained `ClosureType` selection ABI

## Outcome

Accepted as a structural Cycles-alignment change. A retained physical closure
now stores its exact post-setup Cycles `ClosureType`, and the categorical
measure and pick loops classify that type directly. They no longer reconstruct
the identity from Psycles' authoring-oriented `kind`, `lobe`, setup flag, and
allocation weight at every consumer.

The generated HIP selection loop falls from four scalar field loads per closure
to two (`type` and `sample_weight`). The physical record and coroutine frame do
not grow, and the main HIP entry is 184 B smaller. Two normalized profiles are
performance-neutral at +0.139% relative to the two-run baseline mean, so this
change makes no runtime-speedup claim. Its value is establishing the same
retained closure discriminant used by Cycles so subsequent evaluation and
sampling work can move to a compact type-directed ABI instead of accumulating
more projections around the old representation.

## Cycles source model

The reference is Blender 5.2 release commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93` in
`/home/mike/Projects/blender-cycles`:

- `intern/cycles/kernel/svm/types.h` defines the ordered `ClosureType` domain
  and its interval predicates.
- `intern/cycles/kernel/integrator/surface_shader.h` iterates the retained
  `ShaderClosure` array and forms the categorical measure from only
  `sc->type` and `sc->sample_weight`.
- failed setup is represented by `CLOSURE_NONE_ID`; a slot that was never
  allocated is absent from `sd->closure[0..num_closure)`.

Psycles keeps the source Blender closure graph and evaluates it in Luisa DSL.
This is an execution-ABI translation, not material baking or a textual copy of
Cycles kernels.

## Formal equivalence

Let `R = (r_0, ..., r_(n-1))` be the initialized prefix published by
`SurfaceClosureSet`. Append establishes

```text
allocation_weight(r_i) >= CLOSURE_WEIGHT_CUTOFF
and i < capacity
```

for every `r_i in R`. Let `T(r)` be the exact post-setup Cycles closure type.
`T(r) = NONE` if setup failed; otherwise it belongs to exactly one of the
represented diffuse, glossy, transmission, glass, transparent, or BSSRDF type
classes. Define the old event-eligibility predicate as

```text
E_old(r, mask) = allocated(r) and setup_valid(r)
                 and kind/lobe(r) is enabled by mask.
```

Over the reachable physical-closure domain, the ordered Cycles type intervals
form a disjoint partition, so there is a type predicate `Q` such that

```text
E_old(r, mask) = Q(T(r), mask), for every r in R and every event mask.
```

`NONE` belongs to no eligible interval, which discharges the setup-failure
case. Prefix membership discharges allocation. The transformation preserves
closure order, `sample_weight`, and the selected normal, therefore both the
total categorical mass and every partial sum used by inverse-CDF selection are
unchanged.

The permanent physical-closure regression checks this commuting relation for
every represented closure family and eight event masks:

```text
candidate record --old selection--> selection result
       |                                  ^
       | pack/unpack                      | equality
       v                                  |
retained (type, sample_weight) --new selection-->
```

It compares weight, returned type, sample weight, and normal. Existing
round-trip tests additionally require the stored type to equal a fresh exact
classification.

## Physical ABI

No matrix, buffer, or frame field was added. The existing identity row was
repacked as

```text
uint4(closure_type, kind, lobe, method_and_flags)
```

The low 29 bits of the last lane hold `SurfaceBssrdfMethod`; the three high bits
hold setup-valid, preserve-GGX-energy, and Beckmann flags. A compile-time bound
checks that the declared method domain fits. `kind` and `lobe` remain
temporarily for payload-union dispatch and will disappear only when every
physical family consumes the Cycles type-directed payload ABI.

## Generated HIP structure

Both cold dumps use the official Blender 5.2 Barbershop export at one sample
with shader caching disabled.

| Metric | Baseline | Retained type | Change |
| --- | ---: | ---: | ---: |
| Final LLVM IR | 53,903 lines / 3,001,514 B | 53,905 / 3,000,977 B | +2 lines / -537 B |
| Surface object | 339,800 B | 339,544 B | -256 B |
| Main entry | 318,104 B | 317,920 B | **-184 B** |
| Selection-loop scalar loads/closure | 4 | 2 | **-2** |
| Private storage | 3,096 B | 3,096 B | unchanged |
| VGPR / SGPR metadata | 256 / 107 | 256 / 107 | unchanged |
| Coroutine frame | 177 fields / 864 B | 177 / 864 B | unchanged |

The baseline dump is `/var/tmp/psycles-typed-region-b0.vSqMha`; the retained
type dump is `/var/tmp/psycles-cycles-type-selection.I9qgNS`. Final LLVM
inspection shows that the former loops load `kind`, packed setup flags,
`allocation_weight`, and `sample_weight`. The new loops load only
`closure_type` and `sample_weight` before applying the type intervals.

## HIP profile

Each run renders the 640x480 Barbershop export at 64 fixed samples with the
staged wavefront scheduler on the Radeon RX 9070 XT. Timing is normalized by
the actual launched work because the scheduler's queue prediction can vary the
number of calls and inactive work-items between processes.

| Build / run | Calls | Work-items | Total ms | ns/work-item |
| --- | ---: | ---: | ---: | ---: |
| Baseline 1 | 366 | 53,601,728 | 1,154.071 | 21.530472133 |
| Baseline 2 | 357 | 53,557,504 | 1,158.068 | 21.622885021 |
| Retained type 1 | 357 | 53,564,352 | 1,157.492 | 21.609367570 |
| Retained type 2 | 355 | 53,559,232 | 1,157.100 | 21.604109503 |
| Baseline mean | | | | 21.576678577 |
| Retained-type mean | | | | 21.606738536 |
| Change | | | | **+0.139316%** |

The result is within run-to-run noise and is treated as neutral. Candidate
traces are `/var/tmp/psycles-cycles-type-profile1.LjWWgY` and
`/var/tmp/psycles-cycles-type-profile2.y1hodE`.

## Numerical and visual inspection

All 15 Combined, data, light, and volume passes contain zero invalid pixels in
both comparisons. Full statistics are in
[`candidate-vs-baseline.json`](candidate-vs-baseline.json) and
[`candidate-repeat.json`](candidate-repeat.json).

The parallel per-sample path is already non-deterministic across processes.
For example, Combined RMSE is `0.0142921` between baseline and candidate and
`0.0144014` between the two candidate runs. Candidate-versus-baseline
Combined error is therefore below the same candidate binary's repeat error.
Environment and both zero-valued volume passes are exact. The formal selection
equivalence and backend regressions, rather than stochastic EXR equality, are
the semantic gate.

Native-resolution triptychs put the baseline on the left, retained-type build
in the center, and amplified absolute difference on the right:

- [Combined](triptychs/combined.png)
- [Normal](triptychs/normal.png)
- [Diffuse Color](triptychs/diffcol.png)

Visual inspection found identical geometry, UVs, material regions, normal
orientation, and lighting structure. Differences are stochastic edges,
subpixel details, and high-variance highlights; no coherent material,
transform, or closure-class change appears.

## Regression results

The project was built with 32 host threads. Fifteen physical surface tests pass
across fallback, HIP, and strict native Vulkan:

```text
psycles.luisa_microfacet_anisotropy_{fallback,hip,vk}
psycles.luisa_surface_closure_collection_{fallback,hip,vk}
psycles.luisa_surface_population_{fallback,hip,vk}
psycles.luisa_compact_surface_preparation_{fallback,hip,vk}
psycles.luisa_surface_closure_physical_{fallback,hip,vk}
```

The Vulkan cases set

```text
LUISA_VULKAN_USE_XIR=1
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1
LUISA_VULKAN_DISABLE_DXC=1
```

so they cannot pass through DXC.

## Reproduction

```sh
cmake --build build --parallel 32

ctest --test-dir build --output-on-failure -j1 \
  -R 'psycles\.luisa_(microfacet_anisotropy|surface_closure_collection|surface_population|compact_surface_preparation|surface_closure_physical)_(fallback|hip)$'

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure -j1 \
  -R 'psycles\.luisa_(microfacet_anisotropy|surface_closure_collection|surface_population|compact_surface_preparation|surface_closure_physical)_vk$'
```

The HIP profile command is the same as the preceding surface experiments,
with a 64-thread block and `wavefront-staged 64 32768 32 1 1 0 4 2 auto`.

## Consequence

Removing two header loads alone does not move a 256-VGPR, 3,096-byte-private
surface kernel. The next useful step is the larger Cycles translation:

1. make evaluation and sampling dispatch on retained `ClosureType` directly;
2. replace the wide common-plus-family projection with a compact
   `ShaderClosure`-style type-tagged payload;
3. execute the material bytecode as a compact SVM interpreter with only the
   reachable closure setup handlers, instead of expanding scene material
   topology into the surface stage.

That is where Cycles obtains bounded code size. This commit deliberately lays
the discriminant invariant needed for that change without pretending the
foundation alone closes the performance gap.
