# Cycles setup-owned closure identity

## Result

Psycles no longer reconstructs a retained Cycles `ClosureType` and
`MicrofacetFresnel` from its authoring `SurfaceClosureKind` / lobe after
closure setup. Each closure setup component now produces the exact final pair
that its Blender 5.2 Cycles counterpart leaves in `ShaderClosure`, and the
retention boundary only copies that pair.

The source oracle was Blender/Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. In particular, the audit followed
the setup sites in `kernel/closure/bsdf_*.h`, `kernel/closure/bssrdf.h`, and
`kernel/svm/closure.h`, including the Fresnel setup which runs after the base
microfacet setup.

## Formal model

Let `R` be a raw graph closure and let `S` be a closure which has completed one
Cycles-compatible setup. Let

```text
I = ClosureType x MicrofacetFresnel
I_none = (CLOSURE_NONE, NONE)
```

For each setup family `k`, the only legal transition is

```text
Setup_k : R -> S
identity(Setup_k(r)) = setup_valid ? exact_k(r) : I_none
```

with these invariants:

1. identity is absent on `R`;
2. exactly one setup transition assigns identity;
3. failed setup has the unique identity `I_none`;
4. retention is defined only for `S` and copies identity without
   reinterpretation;
5. authoring kind/lobe remains host-stage reachability metadata and cannot
   change device identity;
6. pack, select, evaluate, and sample consume only the retained exact pair.

`TracedClosure::cycles_identity` is an optional host-stage proof witness over
Luisa expressions. `set_cycles_closure_identity_after_setup` enforces single
assignment. Both the physical emitter and retention boundary assert that the
witness exists. It adds no field to the device closure ABI.

The previous centralized `finalize_cycles_closure_identity(kind, lobe)` switch
and its lower-level overload were deleted. There is consequently no fallback
path which can silently manufacture an exact device identity from an
authoring tag.

## Setup ownership audit

The following setup owners now assign final identity directly:

- Diffuse and Principled diffuse: Diffuse or Oren-Nayar after the exact
  roughness predicate.
- Translucent and thin subsurface: Translucent or Rough Translucent.
- Principled sheen and standalone sheen: Sheen or Ashikhmin Velvet, including
  the invalid-LTC `NONE/NONE` result.
- Principled coat: GGX plus Dielectric Fresnel.
- Principled and standalone metallic: GGX/Beckmann plus F82 Tint or Conductor
  Fresnel.
- Principled dielectric: GGX plus Generalized Schlick.
- Glass/refraction: GGX/Beckmann Glass or Refraction plus the final Fresnel
  family selected by the SVM setup sequence.
- Legacy hair reflection/transmission: their distinct Cycles closure types.
- BSSRDF: all four Blender 5.2 methods, plus the diffuse fallback and the
  failed-setup `NONE/NONE` state.
- Thin glass: reflection, thin-glass transmission, and transparency.
- Merged path transparency: Transparent.

The host reachability resolver was deliberately renamed to
`surface_closure_reachability_identity`: it only controls which family code
Luisa records. It does not participate in retained identity.

## Regression coverage

The existing closure-family tests are architectural regressions as well as
numeric tests. Recording any producer without a setup-owned identity triggers
the host assertion while its shader AST is constructed. Recording a producer
twice triggers the single-assignment assertion.

The focused matrix passed 39/39:

```text
13 fallback + 13 HIP + 13 Vulkan = 39/39
```

It covers standalone sheen (including invalid LTC), legacy hair, Beckmann,
anisotropy, both metallic models, BSSRDF success/failure and fallback,
collection/retention, reachability, population, thin film, caustics, and
Principled transmission. The Vulkan tests used the strict native route:

```text
LUISA_VULKAN_USE_XIR=1
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1
LUISA_VULKAN_DISABLE_DXC=1
```

`cmake --build build -j32` also completed for the whole project, and
`git diff --check` passed. A source audit found no remaining
`finalize_cycles_closure_identity` or
`canonical_surface_closure_identity` symbol outside historical validation
documents.

## HIP scene validation

The scene was the official Blender 5.2 Barbershop export, rendered at
640 x 480, fixed 64 spp, adaptive sampling disabled, using staged wavefront.
The exact parent was `0432c8f5e37510212b093e11d76264fc418435e9`.

| Build / run | Render-only time (s) |
|---|---:|
| Parent B1 | `2.34642` |
| Parent B2 | `2.34739` |
| Setup-owned B1 | `2.35392` |
| Setup-owned B2 | `2.33351` |
| Parent mean | `2.346905` |
| Setup-owned mean | `2.343715` (`-0.136%`) |

The change is noise-level and is not claimed as a render speedup. Coroutine
state remains 177 fields / 864 B. The largest generated AMDGPU body remains
417,680 B and the linked code object remains 336,984 B. This is expected:
host specialization and LLVM had already folded the old centralized switch,
so the refactor improves ownership and extensibility without changing the
optimized kernel.

The closure-only HIP benchmark used 4,194,304 lanes, 128 iterations, and nine
measured repeats:

| Probe | Median (ms) | Change from prior candidate |
|---|---:|---:|
| GGX glossy | `6.93739` | `-0.116%` |
| GGX glass | `11.3650` | `+0.173%` |
| Principled dielectric | `10.4590` | `+0.456%` |

All checksums are unchanged. The spread is scheduling noise and does not
support a handler-speed claim.

## Numerical and visual inspection

All 15 output passes have zero invalid pixels. The complete machine-readable
comparison is in [all-pass-report.json](all-pass-report.json). Representative
parent-to-candidate results are:

| Pass | Relative RMSE | Maximum absolute error | P99 pixel RMSE |
|---|---:|---:|---:|
| Combined | `3.604e-4` | `3.466e-2` | `4.302e-9` |
| Normal | `8.475e-7` | `4.187e-4` | `3.847e-8` |
| Diffuse Color | `1.528e-4` | `1.052e-2` | `3.441e-8` |

The same sparse high-amplitude outliers occur between unchanged repeated
runs because film/light passes use parallel floating-point accumulation. The
p99 errors and channel means are effectively unchanged, and no exact-hash
test was weakened.

The triptychs show the exact parent on the left, this implementation in the
center, and amplified absolute difference on the right:

- [Combined](triptychs/combined.png)
- [Normal](triptychs/normal.png)
- [Diffuse Color](triptychs/diffcol.png)

Native-resolution visual inspection found no geometry, material, texture,
lighting, orientation, or other structural difference. Difference panels
required approximately `2.42e8`, `1.51e7`, and `3.02e7` amplification.

## Reproduction

```sh
cmake --build build -j$(nproc)

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

The A/B report uses `--allow-unverified-build-identity` only because both
inputs are explicitly identified Psycles artifacts. It is not a new
Psycles-to-Cycles compatibility claim.
