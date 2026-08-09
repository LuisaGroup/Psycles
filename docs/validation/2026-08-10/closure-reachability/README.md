# Scene-union closure reachability

Date: 2026-08-10

## Outcome

Psycles now records only physical Principled closure families which can be
reached by at least one material instance sharing a compiled shader topology.
This is a host/JIT binding-time proof over raw Blender socket bindings. It does
not evaluate a linked shader input, replace a closure with a baked value, or
ask Blender/Cycles to precompute material results.

On the unchanged 37-material Lone Monk export, this reduces the complete cold
path-kernel JIT substantially on all three Luisa backends:

| Backend | Previous cold JIT | Current cold JIT | Speedup |
| --- | ---: | ---: | ---: |
| fallback | 207.306 s | 63.703 s | 3.25x |
| HIP | 100.147 s | 40.628 s | 2.47x |
| Vulkan native XIR/SPIR-V | 1,173.670 s | 569.378 s | 2.06x |

The measurements disable the render-shader cache and execute a real 1x1,
1-spp render after compiling the complete scene path kernel. Resolution and
sample count therefore do not weaken the compiled shader. These are compile
smokes, not image-quality checkpoints; no parity or throughput claim is made
from their one-pixel outputs.

## Formal specialization rule

For topology `T`, closure instruction `c`, physical family `f`, and the set of
material parameter blocks `M(T)` sharing that topology, generated code obeys

```text
generate(T, c, f) = OR over m in M(T) of may_reach(m, c, f).
```

`may_reach` may return false only from a direct, unlinked socket literal and
the exact Cycles/Psycles layer relation. A linked value is unknown. A
non-finite direct float is also unknown rather than being accepted as a proof.
Plans are unioned before the one shared `GraphSurface` branch is recorded, so
topology deduplication cannot accidentally specialize one material's callable
for another material's values.

Closure-tree reachability follows Add on both inputs and Mix according to a
direct clamped endpoint: factor `<= 0` selects A, factor `>= 1` selects B, and
an intermediate, linked, or non-finite factor retains both. Principled family
reachability follows the physical layer order:

```text
alpha -> sheen -> coat -> metallic -> transmission
      -> dielectric -> subsurface/diffuse
```

Attenuation is retained whenever a family may be requested, even if a runtime
caustics predicate later prevents allocation. Thick and Thin Wall transmission
are separate features. Thick and Thin Wall subsurface are also separate;
subsurface scale/radius cannot prove either family absent because thick BSSRDF
setup still produces Cycles' diffuse fallback for inactive radius channels,
while Thin Wall scattering does not consume scale/radius at all.

Standalone `GraphSurface` construction remains conservative. Production scene
compilation is the only path which supplies the complete parameter-block union.

## Lone Monk graph complexity

The export contains 35 source materials and 37 compiled material instances.
Whole-topology canonicalization reduces them to 24 shader topologies:

| Quantity across unique topologies | Count |
| --- | ---: |
| source/value programs | 24 |
| value IR instructions | 1,438 |
| parameter loads among value IR | 1,074 |
| non-parameter value computations | 364 |
| closure IR instructions | 38 |
| value opcode kinds used | 36 |

All 38 logical closure-tree instructions remain reachable after unioning the
37 instances. The large reduction instead comes from physical Principled
families: Lone Monk requires 3 metallic, 19 dielectric, and 18 diffuse family
occurrences across the unique topologies. Alpha, sheen, coat, thick/thin
transmission, and thick/thin subsurface are all provably absent.

The 36 used value operations are parameter, passthrough, scalar-to-color,
color-to-scalar, Math, clamp, Mix, Hue/Saturation, Invert, UV, Generated,
Object Position, Object Location, Object Random, Particle Random, Backfacing,
Random Per Island, Is Camera Ray, Layer Weight Fresnel, Mapping, Image Color,
Image Alpha, Attribute Color, Normal Map, Bump, Noise Factor/Color, Brick
Color, Gradient, Color Ramp, RGB Curves, Separate RGB, Combine Color, and
Nishita Sky. The small count of actual value computations compared with the
remaining backend module proves that value-graph interpretation alone cannot
remove the dominant residual code size.

The statistics are reproducible with:

```sh
build/bin/psycles_inspect_blender_material \
  /var/tmp/psycles-lone-monk-transmission-dbdcb17/export '*' \
  | sed -n '/^scene_summary/,$p'
```

## Generated-code measurements

The focused XIR regression records the same direct-zero Principled graph once
with a conservative topology-only plan and once with its material plan:

| Fixture | XIR instructions |
| --- | ---: |
| conservative all-family recording | 24,848 |
| reachable-family recording | 5,258 |

This is a 78.8% reduction. The regression requires at least a factor-of-two
reduction, in addition to host tests for topology union, mutually exclusive
thin/thick features, direct Mix pruning, linked-zero conservatism, non-finite
conservatism, and the Thin Wall/scale rule.

HIP code generation confirms the reduction survives backend lowering:

| HIP path shader | Previous | Current |
| --- | ---: | ---: |
| AMDGPU device code | 4.49 MB | 1,357,432 bytes |
| linked code object | 6.69 MB | 1,968,176 bytes |
| LLVM-to-AMDGPU | 27.859 s | 6.906 s |
| ROCm bitcode link | 66.082 s | 13.149 s |

Vulkan selected `AMD Radeon RX 9070 XT (RADV GFX1201)` and used native
XIR-to-SPIR-V throughout; DXC was not loaded:

| Vulkan path shader | Previous | Current | Reduction |
| --- | ---: | ---: | ---: |
| raw SPIR-V | 3,336,939 words | 1,431,985 words | 57.1% |
| optimized SPIR-V | 2,967,277 words | 1,116,158 words | 62.4% |
| process peak RSS | about 38.1 GiB | 8.98 GiB | 4.24x lower |

The path shader's native generation/compute-optimizer interval occupied about
eight minutes. The following RADV pipeline creation occupied approximately 87
seconds. Thus the current Vulkan bottleneck is primarily native SPIR-V
generation/optimization, with a still-material driver-lowering tail; it is
not DXC startup.

## Next structural step

A pure SVM replacement is not sufficient. Lone Monk has only 364
non-parameter value instructions after topology dedup, but still generates
1.43 million raw SPIR-V words. The next architecture combines:

1. a scene-specialized, strongly typed value SVM with separate scalar/vector
   register files, a runtime program bound, and only the 36 used opcode cases;
2. native direct expansion for small graphs selected by an explicit IR cost
   model, rather than material-name or node-type patches;
3. one material populate phase and shared physical closure-family
   eval/sample bodies, so 19 dielectric materials do not clone dielectric
   transport math 19 times;
4. backward value-dependency slicing per surface operation before either
   lowering route; and
5. GPU-coroutine scheduling for any later automatic wavefront execution,
   never a hand-written wavefront fork.

This keeps original typed closures and Cycles layer ordering intact while
placing a formal upper bound on scene-driven code growth. Full-resolution,
high-spp five-way rendering and triptychs remain required after the structural
work; the compile-only checkpoint intentionally does not substitute for that
quality gate.
