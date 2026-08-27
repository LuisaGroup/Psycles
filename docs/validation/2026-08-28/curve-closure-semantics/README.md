# Cycles 5.2 curve closure semantics

## Outcome

Accepted. Psycles now preserves the primitive-kind predicate through the
post-population closure projection and applies the three curve-specific Cycles
laws at their actual semantic boundaries:

- a selected curve closure samples against its own authored normal rather than
  the triangle geometric normal;
- curve closures never receive bump-shadowing/terminator correction;
- curve closures retain their authored normal during specular-validity setup.

This is a primitive-domain correction, not a hair-material special case. It
therefore covers any raw closure graph evaluated on a curve/ribbon primitive,
including the hair and grass geometry used by Lone Monk. No closure value,
normal, or material parameter is pre-baked.

## Oracle identity

The only semantic oracle is Blender/Cycles:

- source: `/home/mike/Projects/blender-cycles`
- branch: `blender-v5.2-release`
- commit: `9e2066aef7ef7e20c142ad7bd3303138a4304c93`
- description: `v5.2.1-dirty` because the existing platform-library submodule
  is dirty

The implementation was checked directly against:

- `intern/cycles/kernel/closure/bsdf.h`
- `intern/cycles/kernel/closure/bsdf_util.h`

## Formal model

Let `p` be the shader point, `c` the selected closure, `K(p)` the predicate
that the primitive is a curve, `Ng(p)` the triangle geometric normal, and
`N(c)` the selected closure normal. Cycles defines the sampling support normal
as

```text
support(p, c) = K(p) ? N(c) : Ng(p).
```

The relation is deliberately evaluated after categorical closure selection:
`N(c)` is a field of the chosen member of the physical tagged union, not a
shader-wide field. Psycles projects a temporary closure-sampling point with
`geometric_normal = support(p, c)` before entering any family sampler. The
aggregate sample-validity boundary independently uses the same relation, so a
future sampler cannot silently restore triangle behavior.

For bump shadowing, with `B_triangle` denoting the existing Cycles triangle
relation,

```text
B(p, c, wo) = K(p) ? 1 : B_triangle(p, c, wo).
```

For specular-normal validity, with `V_triangle` denoting Cycles' correction,

```text
V(p, c) = K(p) ? N(c) : V_triangle(p, c).
```

These laws depend on `K(p)`, which cannot be reconstructed from equality or
orientation of `Ng(p)`, the smooth shading normal, and `N(c)`. The predicate
must therefore survive the exact `SurfacePoint -> SurfaceClosurePoint`
projection. It is packed into the existing flags word; the callable ABI stays
48 bytes and no closure payload or coroutine-frame field is added.

## Separating counterexample

The permanent regression uses

```text
N(c)  = ( 0.8, 0, 0.6)
Ng(p) = (-0.8, 0, 0.6)
Ns(p) = ( 0.0, 0, 1.0).
```

The center of the cosine/GGX random map selects the authored closure normal.
That direction lies below `Ng(p)` but above `N(c)`, separating triangle and
curve support without a floating-point tolerance boundary. The same point
also makes triangle bump softening non-identity and makes the singular glossy
reflection about the authored normal invalid for the triangle.

Before the correction, the curve diffuse sample was rejected and evaluation
was softened to `(0.180963, 0.119669, 0.0671316)` instead of the unattenuated
Lambert value `(0.197352, 0.130507, 0.0732113)`. After the correction, the test
proves all three laws:

1. triangle diffuse sampling rejects while curve diffuse sampling accepts;
2. curve diffuse evaluation and sampling both retain the unattenuated Lambert
   value while the triangle follows its bump relation;
3. curve singular GGX retains `N(c)` and produces reflection
   `(0.96, 0, -0.28)`, while triangle setup corrects the authored normal.

## Regression and backend gates

The predicate packing test covers all eight combinations of `is_curve`,
`use_bump_map_correction`, and `back_facing`, while retaining the 48-byte ABI
assertion. The semantic test compiles two original shader graphs, Diffuse and
singular GGX Glossy, through the normal typed SVM and physical-closure path.

The focused backend matrix passed:

```text
psycles.luisa_curve_closure_semantics_fallback  passed
psycles.luisa_curve_closure_semantics_hip       passed
psycles.luisa_curve_closure_semantics_vk        passed
```

The Vulkan test requires `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`; a DXC fallback cannot satisfy the gate.

The broader curve/hair filter passed `14/14`, including curve ribbon, curve
path, geometry upload, Blender curve import, particle-hair export, and all
three Luisa backends. The full suite passed `304/310`; the same six pre-existing
volume/area-light baseline tests failed, with no new failure:

```text
psycles.luisa_stacked_volume_fallback
psycles.luisa_homogeneous_volume_fallback
psycles.luisa_area_light_forward_vk
psycles.luisa_volume_path_fallback
psycles.luisa_volume_path_vk
psycles.luisa_volume_triangle_fallback
```

Commands:

```bash
cmake --build build -j32
ctest --test-dir build -j16 --output-on-failure -R 'curve|hair'
ctest --test-dir build -j32 --output-on-failure
```
