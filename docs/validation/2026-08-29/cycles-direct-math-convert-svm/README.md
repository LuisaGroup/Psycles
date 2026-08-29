# Direct Cycles surface Math/Convert SVM

This checkpoint replaces the first two surface value families with a direct,
typed-stack SVM evaluator. Math and Convert instructions no longer construct
`TracedValues`, `SurfaceValueExpression`, or host `ValueNode` objects while
recording the device AST. The remaining families deliberately retain the old
evaluator until their complete family semantics are replaced.

No rendering speedup is claimed here. This is a correctness-preserving
architectural checkpoint; the measured `shade_surface` time is statistically
flat and remains about 1% behind the retained pre-family baseline.

## Formal execution contract

Let an executable value record be

```text
r = (F, O, B, I, D, A)
```

where `F` is its SVM family, `O` the exact operation, `B` the typed result
bank, `I` the operation immediate, `D` the result address, and `A` the packed
operand-address sequence. The compiler interner proves before runtime
publication that:

```text
O is neither Parameter nor Passthrough
arity(A) = arity(O)
F = projection(O, I)
(F, O, B) identifies one static evaluator shape
each operand route and bank agrees with that shape
```

The direct evaluator is therefore a partial function over complete families,
not a material-specific patch:

```text
E_direct : {Math, Convert} x typed-stack -> typed-stack
```

For a reached supported family it decodes each packed operand word once,
reads through the statically proven scalar/vector route, evaluates the exact
Cycles-compatible operation, and writes `D` directly. Unsupported families
return to the retained evaluator. A supported family cannot silently return
to the old implementation.

The Convert implementation preserves the existing Cycles semantics,
including float-to-integer truncation for Scalar to Boolean, Rec.709 Color to
Scalar weights, and the three-component mean for Vector to Scalar. Generic
Math obtains its operation from the bytecode immediate and uses the same
finite immediate domain proven by the family interner.

## Regression coverage

A new dynamically connected graph forces all of the following operations to
survive lowering and become observable through a Principled closure:

```text
Geometry Normal
  -> Normal to Vector
  -> Vector to Scalar
  -> Math(MULTIPLY_ADD)
  -> Scalar to Color
  -> Color to Scalar
  -> Principled Roughness/Base Color

Math(MULTIPLY_ADD)
  -> Scalar to Boolean
  -> Principled Thin Wall with non-zero Transmission Weight
```

The test first checks that the five expected executable operations remain in
the compiled program. It then compares compact/direct and expanded/reference
execution for closure population, evaluation, and sampling on fallback, HIP,
and strict native XIR-to-SPIR-V Vulkan. The six compact preparation/tail tests
all pass.

The all-thread build completed successfully. Full CTest passed 312/318 in
39.08 seconds. The only failures are the same six pre-existing numerical
fixture drifts:

```text
psycles.luisa_stacked_volume_fallback
psycles.luisa_homogeneous_volume_fallback
psycles.luisa_area_light_forward_vk
psycles.luisa_volume_path_fallback
psycles.luisa_volume_path_vk
psycles.luisa_volume_triangle_fallback
```

## Barbershop pass and visual validation

The unchanged official Blender 5.2 Barbershop export was rendered through
HIP at 320x180 and one fixed sample. All 15 PFM payloads are byte-for-byte
equal to the retained family-ABI checkpoint: Combined, Normal, Albedo,
Emission, Environment, all Diffuse/Glossy/Transmission passes, and both
Volume passes.

Because the linear images are exact, the already checked-in baseline/family
triptychs remain the exact baseline/direct-evaluator triptychs as well; the
new middle image has the same bytes. I inspected them at native resolution.
Geometry, ceiling lights, salon furniture, material regions, and normals are
unchanged, and each absolute-error panel is black.

![Combined baseline/direct/difference](../cycles-surface-family-bytecode/triptychs/combined.png)

![Normal baseline/direct/difference](../cycles-surface-family-bytecode/triptychs/normal.png)

![Diffuse color baseline/direct/difference](../cycles-surface-family-bytecode/triptychs/diffcol.png)

## HIP profile

Two warm `rocprofv3` runs used the staged wavefront scheduler at 640x480,
64 fixed samples, and block size 64. The shader map identifies structural
hash `b52aa2326173e9bd` as `wavefront_resume_5/shade_surface`.

| measurement | pre-family baseline | family shell | direct Math/Convert | direct vs family |
|---|---:|---:|---:|---:|
| `shade_surface` ns/item | 25.7397 | 26.0082 | 25.9942 | -0.054% |
| `shade_surface` total | 1381.346 ms | 1395.754 ms | 1395.004 ms | -0.750 ms |
| static scratch/thread | 3364 B | 3380 B | 3380 B | unchanged |
| VGPR / SGPR | 256 / 128 | 256 / 128 | 256 / 128 | unchanged |
| largest loaded surface HIP cache entry | 390784 B | 312991 B | 312351 B | -640 B |

The two direct measurements are 25.9935 and 25.9949 ns/item over 53,665,984
items and 294 launches each. Their agreement makes the conclusion clear:
removing Math/Convert wrappers is correct but insufficient to change the hot
kernel materially. The direct build is still 0.989% slower than the retained
pre-family baseline.

The production census explains the next boundary. Barbershop contains 380
programs, 10,177 records, 8,808 value records, 34 reached SVM families, and
67 exact semantic variants. The dominant remaining old-evaluator traffic is
the coupled texture trunk: Mix Color, RGB Ramp, Mapping, and Image Texture.
Those families must be replaced together so their intermediate wrappers and
captured graph machinery disappear from the production AST.
