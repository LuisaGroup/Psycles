# Magic Texture validation

## Outcome

Psycles now imports Blender's original Magic Texture node and evaluates its
raw Vector, Scale, Distortion, depth, and requested Color/Factor output in the
generated Luisa shader. Blender and Cycles do not bake coordinates, texture
results, materials, or closures for Psycles. Every depth from 0 through 10,
both outputs, implicit Generated coordinates, signed and zero inputs, and the
large-coordinate range-reduction path match official Cycles on fallback, HIP,
and Vulkan.

The golden EXRs were rendered by official Cycles CPU in Blender 5.3.0 Alpha.
The local oracle build is commit `a29a0fec7adaba810aaf8540f2e47833efdd4291`
on the Psycles trace branch, based on Blender main
`6f7add4a791e69f23bcc7ff0bdf4ea0307b002c5`. Official Blender main was fetched
again before this checkpoint and is `61f93ccb`; the only intervening Cycles
change fixes CUDA/OptiX bicubic texture sampling on Blackwell and does not
touch Magic Texture, SVM math, shader-node lowering, or the CPU/HIP oracle
used here. No Psycles CPU renderer or CPU reference model exists or was
introduced.

## Formal implementation contract

The implementation is one bounded construction, not a list of scene-specific
cases:

1. The Blender adapter retains the original node graph. Unlinked Vector uses
   the same Generated-coordinate edge as the other Cycles procedural
   textures; linked Vector, Scale, and Distortion remain typed graph
   expressions.
2. Depth and output identity are finite static configuration in the typed
   `SurfaceProgram`. While Luisa traces the shader AST, the OOP
   `MagicValueNode` selects a cached typed callable for the authored depth.
   There is no weakly typed `float4` property table and no per-hit runtime
   depth switch.
3. The callable follows the operation ordering in Cycles `svm_magic`: three
   range reductions, the initial sine/cosine state, the depth-bounded
   recurrence, the distortion normalization branch, and final RGB mapping.
   Factor is exactly the arithmetic mean of that RGB result.
4. Only the requested Color or Factor instruction is emitted. All materials
   using the same depth share the callable AST instead of duplicating the
   recurrence for every graph branch.
5. Floating remainder remains a primitive in Luisa's AST/XIR. The former
   expansion `x - y * trunc(x / y)` is equal over real numbers but not over
   finite-precision floats: a large rounded quotient can discard every bit
   needed to recover the remainder.

For Vulkan binary32, the exact remainder implementation decomposes each
finite magnitude as `significand * 2^(scale - 149)`. For `|x| >= |y|` it
computes

```text
(significand_x * 2^(scale_x - scale_y)) mod significand_y
```

with integer exponentiation by squaring, then reconstructs the normal or
subnormal float bits while preserving the sign of zero. All significands are
at most 24 bits, so the integer products fit in 48 bits. Invalid operands,
infinite divisors, signed zero, subnormals, and `|x| < |y|` are handled by the
same IEEE contract rather than by Magic-specific branches. Fallback uses LLVM
floating remainder; HIP, CUDA, Metal, HLSL, SPIR-V, and C-family emitters now
retain their corresponding native or exact remainder operation.

## Cycles-oracle trap and regression

Blender's shader-node inliner evaluates a fully constant Magic node on the
host. That host helper does not perform the range reduction used by Cycles'
SVM kernel, so its result is not a valid kernel oracle for extreme
coordinates. The large-coordinate probe therefore links Geometry Position.x
through an Add node to `1e20`. The graph remains shader-varying and reaches the
real Cycles kernel, while float32 rounding keeps the authored X coordinate
exactly `1e20` over the small probe mesh.

The focused kernel oracle for Vector
`(1e20, -0.375, 0.8125)`, Scale `0.001`, Distortion `1`, and depth `2` is
`(0.555838704, 0.657886386, 0.937048197)`. Dedicated Psycles tests check that
value on fallback, HIP, and Vulkan. A separate Luisa regression checks scalar,
vector-vector, vector-scalar, and scalar-vector remainder, including ordinary
signs, the Cycles large-coordinate operand, maximum finite values,
subnormals, signed zero, zero/infinite divisors, and infinite dividends. The
native SPIR-V and HLSL-to-SPIR-V routes both pass exact bit comparisons against
`std::fmod`.

As an additional proof-oriented audit, 500,000 deterministic random binary32
operand pairs were compared with `libm fmodf`. The integer reconstruction
matched bit for bit for every non-NaN result; NaN payload bits were deliberately
ignored while NaN classification remained required.

The Luisa fix is published on `next` as
`73bfe5e9e` (`dsl: preserve exact floating-point remainder`). The full CUDA
backend target was also compiled with 32 jobs, even though this machine's
rendering comparisons use AMD HIP/Vulkan.

## Probe coverage and results

`magic_texture_matrix` contains 16 cells: all eleven depths, alternating Color
and Factor outputs, signed Scale, signed Distortion, zero Scale, zero
Distortion, a dynamic `1e20` coordinate, and one unlinked Generated-coordinate
case. Every run used 64x64, 16 spp, box filtering, the original node graphs,
and the same Cycles CPU EXR. Combined and Emit are hard-gated to luminance
ratio `0.9999..1.0001` and relative RMSE at most `1e-4`; all results are much
tighter and contain zero invalid pixels.

| Backend | Combined RMSE | Relative RMSE | Luminance ratio | Max error |
|---|---:|---:|---:|---:|
| fallback | 4.419885e-7 | 6.569065e-7 | 0.999999913 | 2.086163e-6 |
| HIP | 5.533496e-7 | 8.224173e-7 | 0.999999827 | 1.549721e-6 |
| Vulkan | 5.515036e-7 | 8.196737e-7 | 0.999999827 | 1.549721e-6 |

Machine-readable reports are in [`reports`](reports). The versioned shader
inventory now records Magic Texture as `cycles_verified`.

## Visual inspection

I opened all three 1552x582 triptychs at original resolution. Cycles and
Psycles have the same cell boundaries, colors, Factor values, depth recurrence,
signed-input behavior, Generated-coordinate variation, zero-distortion path,
and extreme-coordinate result on every backend. The difference panels are
automatically amplified by roughly 430,000--581,000 times; they show only
small floating-point residual patterns. There is no shifted cell, missing
branch, wrong output, orientation change, banding, NaN, or backend-specific
artifact.

- [fallback triptych](triptychs/fallback-combined.png)
- [HIP triptych](triptychs/hip-combined.png)
- [Vulkan triptych](triptychs/vk-combined.png)

## Compile diagnostics

On the cold full matrix, HIP spent about 3.03 s in LLVM code generation and
9.10 s linking the AMDGPU object, for 12.63 s total JIT. Vulkan spent about
5.95 s in SPIR-V optimization and 20.87 s total JIT; the remaining time is in
HLSL/DXC-to-SPIR-V generation and driver pipeline compilation. Render dispatch
itself took about 0.00345 s on HIP and 0.01436 s on Vulkan. These are shader
compile diagnostics, not full-scene throughput claims.

## Commands

```text
cmake --build build --parallel 32
build/bin/psycles_luisa_fmod_tests fallback|hip|vk
build/bin/psycles_luisa_magic_texture_tests fallback|hip|vk
python3 tools/run_cycles_shader_probes.py magic_texture_matrix --blender /path/to/blender --psycles-render build/bin/psycles_render_blender_scene --output-dir /tmp/magic --backend fallback|hip|vk --cycles-device CPU --width 64 --height 64 --samples 16
```

This checkpoint verifies Magic Texture itself. It does not claim final
Barbershop image parity: the full scene still needs a fresh render after the
hair, volume-emission, and remaining material-feature changes.
