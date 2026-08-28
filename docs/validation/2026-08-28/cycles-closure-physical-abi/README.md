# Exact Cycles physical-closure ABI

## Outcome

Accepted as an intermediate SVM representation milestone in Psycles commit
`0432c8f5e37510212b093e11d76264fc418435e9`. The retained hot-path closure
identity is now the exact post-setup Cycles pair
`(ClosureType, MicrofacetFresnel)`, rather than Psycles' authored
`kind/lobe` plus setup flags. Selection, runtime flag derivation, packed
storage, evaluation, and sampling consume that pair directly.

The physical dependency cut no longer contains allocation state, authored
mixture state, setup validity, AOV inputs, distribution booleans, or a BSSRDF
method. Mutually exclusive payloads are represented as a family-tagged sum
with two `float4x4` blocks and are demand-loaded only in the matching family
branch. No material input is evaluated or baked on the host; all values remain
Luisa expressions recorded into the shader AST.

This does **not** claim that the complete shade-surface SVM is finished or that
this Psycles A/B is a new Cycles differential result. One transitional layer
still maps immutable graph metadata `kind/lobe` to the exact setup result at
the final setup-to-retention boundary. The next structural step is to make
each setup component write its own exact type/Fresnel result and delete that
central reconstruction overload.

## Reference identity

- Parent Psycles baseline:
  `80ce864911fce24842cbff62b6c1d0067a2086c9`.
- Candidate Psycles commit:
  `0432c8f5e37510212b093e11d76264fc418435e9`.
- LuisaCompute:
  `9ae333d05ed028c20b369555c350fbfd6ac8ba43`.
- Cycles source oracle: Blender 5.2 release at
  `9e2066aef7ef7e20c142ad7bd3303138a4304c93`.
- Device: AMD Radeon RX 9070 XT (`gfx1201`), HIP 7.2.
- Scene: official Blender 5.2 Barbershop export, 640x480, 64 fixed samples,
  adaptive sampling disabled, staged wavefront scheduler.

The export emits the same unavailable-image warnings in both builds. This
experiment therefore proves parent-to-candidate semantic preservation; it is
not used as a Barbershop texture-fidelity claim against Cycles.

## Formal representation model

Let a retained closure be an element of the tagged sum

```text
Physical = Common x (Unit + General + Hair + Dielectric + BSSRDF).
```

Its discriminant is the exact pair

```text
tau = (Cycles ClosureType, Cycles MicrofacetFresnel).
```

`ClosureType::NONE` is the unique failed-setup state. Allocation failure is
absence from the retained prefix, so it needs no physical field. Define
`family(tau)` from exact retained types. Over the represented Cycles type
domain, `General`, `Hair`, `Dielectric`, and `BSSRDF` are pairwise disjoint;
all remaining successful types are `Unit`.

For each tag `tau`, let `Obs_tau(x)` project a logical closure onto only the
fields observable by Cycles evaluation and sampling for that exact tag. The
two-block encoding must satisfy

```text
Obs_tau(unpack_family(tau, pack(x))) = Obs_tau(x).
```

It also has a canonicality obligation:

```text
not observable(tau, lane) => pack(x)[lane] = 0.
```

Canonical zero is stronger than merely promising consumers will not read an
unused lane. It makes the packed representation a well-defined quotient by
observational equivalence: if `Obs_tau(x) = Obs_tau(y)`, hidden payload state
cannot distinguish their encodings. This audit found a real violation in the
BSSRDF payload: an unused lane retained generic `ior`. The lane is now zero,
and a permanent canonical-payload regression covers all five families.

The C++ staging API enforces non-interference. A common-only consumer cannot
name general, hair, dielectric, BSSRDF, or thin-film fields; every typed family
record exposes only its own payload. The common-only eliminator has no
`block_1` argument, so it cannot accidentally introduce a payload load into
the emitted device CFG. This is a host-side Luisa metaprogramming boundary,
not another device IR entity.

## Cycles 5.2 correspondence audit

The numeric `ClosureType` and `MicrofacetFresnel` ABI values were checked
against Blender 5.2's `intern/cycles/kernel/svm/types.h` and microfacet setup
implementation. In particular:

- virtual multi-GGX authoring types are overwritten by setup before a
  retained `ShaderClosure` is observed;
- GGX/Beckmann and reflection/refraction/glass distinctions are retained as
  exact closure types;
- dielectric, conductor, generalized-Schlick, and F82-tint remain distinct
  Fresnel families;
- BSSRDF method is represented by the exact post-setup closure type;
- the `2e-10` singular microfacet threshold and Cycles runtime-flag
  classification are preserved.

`DIELECTRIC_TINT` is present in the ABI but remains a known feature gap in the
consumer implementation. Cycles currently reaches it through OSL/MaterialX;
the Blender-node coverage exercised here does not generate it. It must be
implemented before claiming complete Cycles closure coverage.

## Permanent regressions

Compile-time negative contracts prove that `SurfaceClosurePhysicalRecord`
cannot expose allocation/setup state, AOV inputs, setup classification, or
authored mixture state. They also prove that each family record cannot name
another family's payload and that thin-film state exists only in the exact
families which can observe it.

Runtime tests cover:

- exact type/Fresnel identity for diffuse, translucent, GGX/Beckmann,
  reflection/refraction/glass, sheen, hair, transparent, BSSRDF, and thin
  glass;
- direct projection versus packed demand-decoding for eval and sample;
- successful and failed setup canonicalization;
- canonical zero in every unused row/lane of all payload families;
- thin-film tag gating and BSSRDF payload isolation;
- collection, reachability, population, caustics, anisotropy, and all affected
  closure-family behavior.

After a full 173-step rebuild with all 32 host threads, the focused matrix
passed 39/39 tests:

```text
13 fallback + 13 HIP + 13 Vulkan = 39/39
```

The Vulkan tests ran with `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`; no DXC compatibility route was permitted.

## HIP performance

The interleaved warm whole-scene order was A/B/B/A. Render-only time is
statistically unchanged:

| Build / run | Render-only time (s) |
|---|---:|
| Parent A1 | 2.34337 |
| Candidate B1 | 2.34642 |
| Candidate B2 | 2.34739 |
| Parent A2 | 2.35198 |
| Parent mean | 2.347675 |
| Candidate mean | 2.346905 (-0.033%) |

The staged coroutine stays at 177 fields / 864 B. The largest generated
AMDGPU body shrinks from 419,200 B to 417,680 B (-0.36%), while the linked
code object remains 336,984 B. These numbers support “no regression”; the
wall-time delta is measurement noise and is not claimed as a speedup.

The closure-only benchmark used 4,194,304 lanes, 128 iterations, nine measured
repeats, and the `selected_closure_sample` contract. Stable second-pair medians
were:

| Probe | Parent A2 (ms) | Candidate B2 (ms) | Change |
|---|---:|---:|---:|
| GGX glossy | 6.94363 | 6.94547 | +0.027% |
| GGX glass | 11.34490 | 11.34540 | +0.004% |
| Principled dielectric | 10.41580 | 10.41150 | -0.041% |

All checksums match. The sub-0.05% spread is noise-level evidence that exact
identity dispatch and family payload elimination did not add handler cost.

## Numerical and visual comparison

All 15 named passes have zero invalid pixels. Combined relative RMSE is
`3.6016e-4`; GlossDir, the most sensitive pass, is `4.3581e-4`. In every pass,
99% pixel RMSE is approximately `1e-8` or smaller. Repeating the unchanged
parent produces the same Combined and GlossDir residual magnitude, locating
the sparse maxima in nondeterministic floating-point accumulation order rather
than in the closure representation change. Full metrics are in
[`all-pass-report.json`](all-pass-report.json).

| Pass | Relative RMSE | Maximum absolute error | P99 pixel RMSE |
|---|---:|---:|---:|
| Combined | `3.602e-4` | `3.466e-2` | `4.302e-9` |
| Normal | `8.454e-7` | `4.173e-4` | `3.847e-8` |
| Diffuse Color | `3.223e-5` | `4.555e-3` | `3.441e-8` |

The following triptychs show the exact parent on the left, candidate in the
center, and amplified absolute difference on the right:

- [Combined](triptychs/combined.png)
- [Normal](triptychs/normal.png)
- [Diffuse Color](triptychs/diffcol.png)

Native-resolution visual inspection found no geometry, material, texture,
lighting, orientation, or other structural difference. The difference panels
require approximately `2.42e8`, `1.51e7`, and `3.02e7` amplification,
respectively. No exact-hash regression was weakened; numerical comparison
uses explicit tolerances only for the renderer's already non-deterministic
parallel accumulation.

## Reproduction

```sh
cmake --build build -j32

ctest --test-dir build --output-on-failure -j1 \
  -R '^psycles\.luisa_(standalone_sheen|legacy_hair|beckmann_glossy|microfacet_anisotropy|metallic_surface|subsurface_exit|surface_closure_collection|surface_closure_reachability|surface_population|surface_closure_physical|thin_film_surface|standalone_caustics|principled_transmission)_(fallback|hip|vk)$'

PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  BARBERSHOP_5_2_EXPORT out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576

build/bin/psycles_benchmark_surface_closures \
  hip glossy_ggx_regular 4194304 128 9 selected_closure_sample
```

`tools/compare_cycles.py` used `--allow-unverified-build-identity` only because
both inputs are explicitly identified Psycles A/B artifacts. That override is
not used to claim a Cycles compatibility result.
