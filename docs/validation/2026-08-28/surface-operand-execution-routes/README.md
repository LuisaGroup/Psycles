# Surface SVM operand-route census

## Question

The compact surface evaluator had two plausible, independent costs:

1. an operand whose static variant can observe both a local and a material
   parameter retains a device-side storage-class branch; and
2. repeated material parameters issue repeated reads from the parameter
   buffers.

This validation adds an exact hit-weighted census before changing the bytecode
ABI. It does not infer a renderer speedup from static graph size.

## Model

For preparation instruction `i` and operand position `j`, let

- `c(i,j)` be the concrete bytecode storage class in `{local, parameter}`;
- `r(v(i),j)` be the scene-wide static evaluator route in
  `{local, parameter, dynamic}`; and
- `H(t)` be the exact number of surface populations of topology `t`.

Projecting `H(t)` through the immutable preparation program gives the four
execution cells `(r, c)`. A direct route that disagrees with the concrete class
is impossible in a valid executable; the diagnostic fails closed by setting
`exact=false` on such a disagreement, malformed operand, invalid range, arity
mismatch, float-counter loss, or host integer overflow.

For typed bank `b`, let `U(t,b)` be the set of distinct parameter indices used
by topology `t`. The reported unique-parameter count is

```text
sum over t: H(t) * |U(t,b)|
```

This is the exact semantic-load lower bound if every referenced parameter were
memoized once per surface hit. It is not a hardware transaction count and does
not assert that retaining all values is profitable; register lifetime, cache
behavior, and extra interpreter work still have to be measured.

## Regression fixture

The sample-dispatch film fixture now contains:

- a normal-to-vector and vector-math chain with direct local and direct
  parameter operands;
- two execution-equivalent Mix Color instructions whose first operand is
  parameter-backed in one instruction and local in the other, exercising both
  dynamic cells; and
- a shared color parameter consumed three times, proving that the distinct
  address count is smaller than the parameter reference count.

The histogram remains exactly equal across single and split sample dispatches.
Combined, Normal, Albedo, all light passes, and the absolute sample/RNG trace
remain covered by the existing fixture.

## Validation commands

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_sample_dispatch_film_tests \
           psycles_render_blender_scene

build/bin/psycles_luisa_sample_dispatch_film_tests fallback
build/bin/psycles_luisa_sample_dispatch_film_tests hip

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
build/bin/psycles_luisa_sample_dispatch_film_tests vk
```

All three passed. The Vulkan log reported native `SPIR-V optimization preset
'compute'` and `SPIR-V compilation successful` messages with DXC disabled.

The Barbershop diagnostic run used the 5.2 export and the compact populate-once
path:

```sh
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  /var/tmp/psycles-surface-route-weighted-20260828-v2/barbershop.exr \
  hip 320 240 16 16 - 0 0 0 0 16 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  - /var/tmp/psycles-surface-route-weighted-20260828-v2/barbershop-histogram.json
```

The run was exact and reported:

| Quantity | Count | Fraction |
| --- | ---: | ---: |
| Surface populations | 3,366,888 | 100% |
| All value operands | 701,567,158 | 100% |
| Direct-route operands | 407,162,195 | 58.0361% |
| Dynamic-route operands | 294,404,963 | 41.9639% |
| Parameter operands | 362,438,896 | 51.6613% |
| Unique parameter values per hit, weighted | 190,883,220 | 52.6663% of parameter operands |
| Repeated parameter references | 171,555,676 | 47.3337% of parameter operands |

The unique lower bound is 56.6943 parameter values per surface population,
split as 133,842,393 scalar, 43,128,504 vector, and 13,912,323 unsigned-integer
weighted values. Eliminating every repeated parameter reference would remove at
most 24.4532% of all ordinary value-operand reads, before accounting for the
cost of the cache itself.

The diagnostic render completed in 0.413381 s at 320x240/16 spp. This timing is
only a canary for the instrumented path; no performance improvement is claimed
by this change. The output EXR/PFM files were produced successfully. Renderer
semantics were not changed, so this validation does not introduce a new visual
alignment claim.

## Decision

The data rejects two indiscriminate transformations:

- removing only the route branch cannot address the separate 47.3% repeated
  parameter-reference opportunity; and
- preloading every unique parameter would keep roughly 56.7 typed values live
  per hit in this scene, risking more VGPR/private memory and a larger coroutine
  frame.

The next A/B must therefore use reuse distance and typed live intervals to
select profitable parameter materialization, or fuse values into a
material-specific data stream without making material values part of shader
AST/cache identity. Any candidate must preserve topology/handler sharing and be
judged by HIP object resources plus render-only time, not this upper bound.
