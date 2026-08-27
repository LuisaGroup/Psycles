# Microfacet anisotropy checkpoint

## Scope and oracle

This checkpoint closes the standalone Glossy and Principled
metallic/dielectric anisotropy gap against Blender 5.2 Cycles. The source
oracle is the local read-only Blender tree at commit
`9e2066a5d571f50afdf84c6f509a94c08e329f55`, especially:

- `intern/cycles/scene/shader_nodes.cpp` for constant-input specialization and
  default `LINK_TANGENT` behavior;
- `intern/cycles/kernel/svm/closure.h` for setup of tangent and distribution
  roughness;
- `intern/cycles/kernel/closure/bsdf_microfacet.h` for anisotropic GGX and
  Beckmann `D`, masking-shadowing, singular classification, and reported
  roughness;
- `intern/cycles/kernel/sample/mapping.h` for the tangent basis and visible
  normal mapping.

No Cycles SVM bytecode, closure response, or material value is pre-baked. The
Blender graph is imported as typed nodes and lowered to Psycles' own
topological surface IR; all setup, evaluation, and sampling remain Luisa DSL
expressions.

## Formal representation

Closure setup is the unique map from authored node parameters to the physical
state

```text
M = (T, alpha_x, alpha_y).
```

`T` is observationally dead when `alpha_x == alpha_y`. For perceptual
roughness `r`, Principled uses

```text
a = clamp(r, 0, 1)^2
s = sqrt(1 - 0.9 clamp(anisotropy, 0, 1))
alpha_x = clamp(a / s, 0, 1)
alpha_y = clamp(a s, 0, 1),
```

when Cycles retained the tangent input. Standalone Glossy first clamps
anisotropy to `[-0.99, 0.99]`, treats `abs(anisotropy) <= 1e-4` as isotropic,
and applies Cycles' signed reciprocal-axis parameterization. Before setup
saturation both parameterizations preserve `alpha_x * alpha_y`; the common
singular test is therefore the Cycles product threshold `2e-10`.

Principled rotates the authored tangent around the uncorrected shader normal;
standalone Glossy rotates it around the corrected physical reflection normal.
The zero-rotation case is a real device control-flow branch, so it neither
executes trigonometry nor perturbs the authored tangent. Sampling and
evaluation construct Cycles' deliberately unsafe tangent basis only when the
two axes differ.

## Storage proof and root-cause repair

The implementation initially exposed a schema-completeness bug rather than a
scattering bug. The state survived the compact program decoder but was lost at
two independent representation boundaries:

1. the four-matrix complete closure ABI; and
2. the branch-local `SurfaceClosureExpression` handle view.

The complete ABI now has one authoritative semantic-row inverse, shared by
full-record decoding and `SurfaceClosureSet` projections. This avoids both a
second lane decoder and a decode-then-repack AST expansion: the Beckmann
collection fixture fell from 3561 to 3290 pre-optimization XIR instructions
without raising its 3500-instruction ceiling. The physical ABI remains a
two-block tagged sum. Its general payload
uses the five previously spare lanes for `M`; dielectric and BSSRDF payloads
continue to own the same lanes under mutually exclusive tags. Program
compaction and bump expansion now share one canonical list of every
`ValueExpressionId` member, eliminating the duplicated remapping tables that
caused the first typed-bank failure during development.

For every closure family `k`, the retained invariant is

```text
P_k(unpack_k(pack(x))) = P_k(x),
```

where `P_k` is exactly the set of fields observable by that family. Complete
storage additionally checks every canonical closure field, including the new
microfacet state.

## Static specialization

The closure plan schedules anisotropy operands only when they are observable:

- Principled requires a reachable metallic or dielectric lobe and follows
  Cycles' constant `CLOSURE_WEIGHT_CUTOFF` rule; linked or non-finite values
  remain dynamic.
- Glossy follows Cycles' distinct constant isotropic interval
  `[-1e-4, 1e-4]`; linked or non-finite values remain dynamic.
- emission-only projections never carry the operands.

The resulting bit is part of the compact closure instruction identity. The
device decoder therefore does not read anisotropy, rotation, or tangent for a
specialized isotropic topology.

## Regression coverage

The permanent tests cover:

- Blender raw-graph import of unlinked Principled and Glossy tangent sockets,
  proving both become explicit `Geometry.Tangent` edges;
- typed value-bank assignment, dependency scheduling, compact bytecode
  control bits, and plan merging for isotropic versus anisotropic materials;
- complete closure storage, the two-block physical tagged sum, callable
  round trips, and branch-local expression transport;
- the exact Principled and signed Glossy alpha formulas, including negative
  anisotropy's reciprocal-axis branch;
- GGX and Beckmann quarter-turn tangent rotation, sample-direction rotational
  covariance, evaluation covariance, and geometric-mean reported roughness;
- fallback, HIP, and Vulkan execution, with Vulkan forced through native
  XIR-to-SPIR-V and DXC disabled.

Commands used from the build tree:

```sh
cmake --build build --parallel "$(nproc)"
./build/psycles_surface_closure_execution_plan_tests
./build/psycles_blender_import_tests
./build/bin/psycles_luisa_microfacet_anisotropy_tests fallback
./build/bin/psycles_luisa_microfacet_anisotropy_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
  ./build/bin/psycles_luisa_microfacet_anisotropy_tests vk
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
  ctest --test-dir build --output-on-failure --parallel 4
```

The closure collection and physical-ABI fixtures were run through the same
three-backend matrix. The strict complete suite passed 289/295; its six
failures are exactly the previously documented numeric-oracle set
(`stacked_volume_fallback`, `homogeneous_volume_fallback`,
`area_light_forward_vk`, `volume_path_fallback`, `volume_path_vk`, and
`volume_triangle_fallback`), with no new failure. Vulkan's fast-math
compilation produced a
maximum `2.34e-4` component difference between two bit-identical-input,
algebraically equivalent inlined thin-glass sample paths. The difference also
appeared with SPIR-V optimization disabled, while all scalar state was equal;
the regression therefore uses a backend-independent `5e-4` direction bound
(less than `0.05` degrees for unit vectors) and retains `1e-5` for transported
scalar state.

This is an analytic closure checkpoint, not a full-scene promotion. It has no
image triptych or renderer speed claim. Complex-scene EXR comparison, visual
inspection, and render-only profiling are the next validation layer and must
be recorded with their own Cycles CPU/HIP and Psycles fallback/HIP/Vulkan
artifacts.
