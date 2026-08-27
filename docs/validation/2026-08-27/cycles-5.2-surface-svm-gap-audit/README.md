# Cycles 5.2 surface/SVM gap audit

## Outcome

Psycles already has a real, data-driven surface SVM. It is not a material
graph demo and it does not expand one shader AST per material. The current
implementation deduplicates scene topologies, schedules a strict typed DAG,
colors typed value lifetimes, packs operands, interprets scene data, and emits
only the semantic handler variants reachable by the loaded scene.

The remaining gap to Cycles is split across two independent boundaries:

1. Blender 5.2 node and closure semantics are not complete. Of the 93
   constructible Cycles shader-node types, 67 have a source route in the
   Blender adapter or are structural output roots and 26 do not. Several of
   the routed nodes are intentionally partial.
2. Post-population physical closures are stored as a tagged sum, but the hot
   evaluation and sampling path reconstructs a Cartesian-product record before
   eliminating the tag. Cycles keeps the tag through consumption and loads a
   derived closure payload only after dispatch.

The second difference is a measured performance boundary. On the retained
640x480, 64-spp Barbershop HIP traces, Psycles `shade_surface` costs 28.844 ns
per launched item and Cycles costs 10.778 ns per item: Psycles is 2.676x the
normalized cost. The current Psycles entry uses 256 VGPRs and 3,152 B private
storage; Cycles uses 192 VGPRs and 6,976 B private storage. The result cannot
be explained by private bytes alone.

This audit does not change rendering and therefore does not create a new image
comparison or triptych. It identifies the semantic and compiler obligations
for the next change. Existing renderer changes remain subject to all-pass EXR,
visual, fallback/HIP, and native-XIR Vulkan validation.

## Reference identity

- Psycles: `717176eebc7b47e7d1b55db6dcc9e00f7285ff3b`.
- LuisaCompute: `eeda4b154fcf43e8709d1b42478e958677b9c6ae`.
- Cycles source: `blender-v5.2-release` at
  `9e2066aef7ef7e20c142ad7bd3303138a4304c93`.
- Render oracle: Blender 5.2.0 LTS, build hash `fbe6228777e7`.
- Device: AMD Radeon RX 9070 XT, `gfx1201`, ROCm 7.2.4.

The source checkout is newer than the exact release binary. Source structure
claims below use the checkout; render and profiler claims use the release
binary and its recorded build identity.

## What is already aligned

### Graph and bytecode structure

Cycles uses one sequential interpreter over scene `svm_nodes`, a 255-float
stack, typed instruction records, closure-tree jumps, and scene feature masks:

- `intern/cycles/kernel/svm/svm.h` owns `svm_eval_nodes`;
- `intern/cycles/kernel/svm/types.h` defines `SVM_STACK_SIZE = 255`;
- `intern/cycles/scene/svm.cpp` schedules the graph and allocates stack slots;
- `intern/cycles/scene/shader_graph.cpp` folds, simplifies, and deduplicates
  graph nodes.

Psycles implements the same essential data/algorithm separation without
copying the fixed untyped stack:

- `include/psycles/compiler/surface_program.h` defines a closed typed value IR
  and a separate typed closure IR;
- `include/psycles/compiler/surface_execution_plan.h` defines strict
  topological schedules, exact interval coloring, compact operand addresses,
  and executable scene images;
- `src/luisa/path_tracer_surface_values.cpp` interprets the value and closure
  programs and populates original physical closures in device code;
- host/JIT reachability retains only handler variants possible in the loaded
  scene, while program ids, parameters, and authored values remain runtime
  data.

For the Blender 5.2 Barbershop export, the current full surface runtime reports:

| Quantity | Value |
|---|---:|
| programs | 378 |
| value instructions | 10,301 |
| packed operand words | 9,628 |
| metadata records | 4,209 |
| semantic variants | 67 |
| scalar/vector/uint local slots | 8 / 9 / 0 |
| maximum program length | 230 |

The preparation domain has 189 unique material topologies and 6,272 runtime
instructions. This is already an SVM; replacing it with Cycles' untyped stack
would not address the measured hot path.

### Closure population

Both renderers preserve original closure graphs and allocate physical closures
at the shading point. Psycles does not ask Cycles to evaluate a material and
does not bake a BSDF response. Its current physical kinds include diffuse,
translucent, rough translucent, Principled lobes, glossy, glass, refraction,
thin-glass transmission, transparent, and BSSRDF. Principled sheen, coat,
metallic, transmission, and dielectric lobes retain distinct lobe tags.

Psycles also preserves Cycles' 64-closure capacity and stable Add/Mix order.
The two-matrix physical representation is a lossless tagged projection, not a
quality shortcut.

## Blender 5.2 functional coverage

The inventory was generated from the exact Blender 5.2 binary with:

```sh
/home/mike/Projects/blender-install-5.2/blender \
  --background --factory-startup \
  --python tools/cycles_shader_node_inventory.py -- \
  /var/tmp/psycles-blender-5.2-node-inventory-20260827.json
```

Blender exposes 93 constructible nodes applicable to Cycles. The disjoint
source/status classification is:

| Status | Count | Meaning |
|---|---:|---|
| present in the focused-probe registry | 42 | still-present node types whose checked-in probe status is `cycles_verified` |
| explicitly partial | 13 | a Luisa route exists but exposed Cycles semantics remain incomplete |
| source-routed, not certified by the registry | 9 | lowering exists, but the versioned coverage registry still says pending |
| structural output roots | 3 | Material, World, and Light Output select roots and emit no device instruction |
| no Psycles route | 26 | the active output falls back with a named diagnostic |

The checked-in probe baseline is stamped Blender 4.5.10. The 42-node number is
therefore evidence already accumulated for node semantics, not a false claim
that a Blender 5.2 release gate has passed. Blender 5.2 removes five old node
types from that inventory and adds `MATERIAL_RAYCAST` and Radial Tiling. A new
5.2 probe baseline is still required.

### Routed but known partial nodes

The coverage registry explicitly tracks these 13 nodes as partial:

```text
ATTRIBUTE, BSDF_GLOSSY, BSDF_PRINCIPLED, LIGHT_PATH, NEW_GEOMETRY,
OBJECT_INFO, PARTICLE_INFO, TEX_BRICK, TEX_COORD, TEX_IMAGE, TEX_SKY,
UVMAP, VERTEX_COLOR
```

Three material closures have concrete Blender 5.2 socket omissions:

- Principled BSDF does not lower `Anisotropic`, `Anisotropic Rotation`,
  `Tangent`, `Thin Film Thickness`, or `Thin Film IOR`.
- Glossy BSDF does not lower `Anisotropy`, `Rotation`, or `Tangent`.
- Glass BSDF has a route but does not lower `Thin Film Thickness` or
  `Thin Film IOR`.

The remaining partial statuses cover documented mode/output domains rather
than one shared failure. Examples include arbitrary Attribute domains,
Particle Info fields, Geometry Tangent/Parametric, complete Sky modes, and
texture source/mapping cases.

The nine routed node types not yet represented accurately by the coverage
registry are:

```text
BSDF_GLASS, DISPLACEMENT, HAIR_INFO, LIGHT_FALLOFF, PRINCIPLED_VOLUME,
SUBSURFACE_SCATTERING, VOLUME_ABSORPTION, VOLUME_COEFFICIENTS,
VOLUME_SCATTER
```

Some of these already have strong focused implementation regressions, notably
subsurface and volume closures. Their presence in this category is coverage
registry debt, not proof that their algorithms are absent.

### Nodes with no route

The seven absent surface-closure families are:

```text
BSDF_HAIR, BSDF_HAIR_PRINCIPLED, BSDF_METALLIC, BSDF_RAY_PORTAL,
BSDF_SHEEN, BSDF_TOON, HOLDOUT
```

Principled Sheen is implemented as a Principled lobe; that does not implement
the standalone Sheen BSDF node. Similarly, Principled metallic does not
implement Blender 5.2's standalone Metallic BSDF and its conductor/Fresnel
inputs.

The other 19 missing Cycles node routes are:

```text
AMBIENT_OCCLUSION, BEVEL, CAMERA, CURVE_FLOAT, CURVE_VEC,
MATERIAL_RAYCAST, NORMAL, OUTPUT_AOV, POINT_INFO, SQUEEZE,
ShaderNodeRadialTiling, TANGENT, TEX_GABOR, TEX_IES,
VECTOR_DISPLACEMENT, VECTOR_ROTATE, VECT_TRANSFORM, VOLUME_INFO,
WIREFRAME
```

This list is source-derived. A node is not counted as supported merely because
the generic unsupported-output path can return its exported socket default.

### Barbershop exposure

The raw Barbershop JSON contains two Hair BSDF nodes, but both are dead. The
two `shaving_brush_strands` materials connect Hair Info through a Color Ramp to
a Diffuse BSDF, and the active Material Output root consumes that Diffuse BSDF.
Running `psycles_inspect_blender_material SCENE '*'` reports no reachable
unsupported-node diagnostic and finds 189 unique active topologies.

Both Cycles and Psycles report the same missing external
`generic_scratches.png` and `guilder_ornament.png` assets for the retained
official file. Those warnings are an asset property, not an SVM semantic
difference. Consequently the measured Barbershop surface gap cannot be
attributed to an active missing Hair closure or to a Psycles-only missing
texture.

## The post-population representation gap

### Cycles

Cycles stores a fixed-size `ShaderClosure` base carrying weight, type,
sample weight, and normal. Derived closure structs duplicate that base layout.
`bsdf_sample` and `bsdf_eval` switch on `sc->type`; only the selected case casts
to `MicrofacetBsdf`, `HairBsdf`, `SheenBsdf`, or another derived payload and
loads its fields. `surface_shader_bsdf_eval` loops over the closure array and
accumulates the much smaller evaluation result.

This is elimination of a tagged sum:

```text
C = Common x (General + Dielectric + BSSRDF + ...)

consume(C) = case tag(C) of
  general    -> consume_general(common, general_payload)
  dielectric -> consume_dielectric(common, dielectric_payload)
  bssrdf     -> consume_bssrdf(common, bssrdf_payload)
```

### Psycles today

`pack_surface_closure_physical` stores the same logical sum in two
`float4x4` blocks. However,
`unpack_surface_closure_physical_payload` reads payload lanes for all three
families and selects canonical defaults into one
`SurfaceClosurePhysicalRecord`. Evaluation and sampling then derive family
predicates from that wide record.

The effective form is:

```text
C' = Common x General x Dielectric x BSSRDF

product = materialize_all_payloads(blocks, tag)
result  = consume_after_tag_tests(product)
```

The two representations are semantically equivalent under the documented
pack/unpack laws, but they are not equivalent for liveness. `C'` exposes every
family field to SSA construction before the family is known. It increases the
number of simultaneously available values and the optimizer's select/phi
problem even when only one family is dynamically observable.

The accepted Barbershop final LLVM module is 57,932 lines / 3,373,035 B and
contains 2,842 textual `phi` occurrences and 2,609 `select` occurrences. The
surface code object is 356,760 B, of which `.text` is 353,152 B. Its entry
symbol alone is 297,436 B. Disassembly contains 846 scratch loads and 647
scratch stores.

Cycles' complete HIP object is 7.24 MB because it contains the whole kernel
set and shared helpers, so whole-object size is not comparable. Its
`kernel_gpu_integrator_shade_surface` entry symbol is only 2,552 B and calls
shared device functions. This is useful structural evidence, not a claim that
all Cycles surface code occupies 2,552 B.

## HIP comparison

Both traces use 640x480, 64 fixed samples, Tabulated Sobol, adaptive sampling
off, and the Radeon RX 9070 XT. Work is the sum of
`grid_x * grid_y * grid_z` over the named dispatches.

| Renderer/stage | Calls | Work items | GPU time | ns/item | VGPR | Private bytes |
|---|---:|---:|---:|---:|---:|---:|
| Psycles `shade_surface` | 293 | 53,659,296 | 1,547.762 ms | 28.844 | 256 | 3,152 |
| Cycles `integrator_shade_surface` | 296 | 54,061,056 | 582.671 ms | 10.778 | 192 | 6,976 |

The work counts differ by less than 0.75%, so the 2.676x normalized ratio is
not a queue-count illusion. Cycles spills a larger private object yet runs much
faster; Psycles' 256-VGPR ceiling, large inlined entry, scratch instruction
traffic, and wider dynamic control/data flow are the relevant evidence.

This table does not assert that every instruction in the two stages is
identical. Both stages own material evaluation, closure setup, emission/data
passes, NEE preparation, and continuation sampling, while their scheduler
state layouts differ. Per-handler and per-closure probes remain necessary to
separate SVM dispatch, individual node algorithms, and BSDF consumption.

## Rejected tagged-elimination experiments

Several superficially plausible rewrites were measured and rejected:

| Experiment | Surface object | Render-only | Result |
|---|---:|---:|---|
| accepted product decoder | 356,760 B | 2.631 s profiled sample | baseline |
| family branches returning a unified record | 418,208 B | 5.103 s | reject |
| immutable direct record construction | 418,208 B | 5.097 s | reject |
| expression-view facade | 418,208 B | 5.106 s | reject |
| explicit `Expression*` matrix references | 418,208 B | 5.099 s | reject |

The failed final IR grew from 57,932 to 64,358 lines; `phi` occurrences grew
from about 2.8k to about 4.9k, and loads/stores grew correspondingly. Mutable
`Var<T>` construction was therefore not the root cause: an expression-only
view and explicit matrix-expression references produced the same final object.

The regression came from recording payload decoders separately inside every
family branch at every closure-loop consumption site. Branch-local payloads
were narrower, but the surrounding Luisa/LLVM loop state was cloned across
families and merged. A correct implementation cannot merely spell the current
decoder with more `$if` statements.

## Required next invariant

Let `A(k)` be the payload fields observable for runtime closure kind `k`, and
let `R_j` be the result tuple of consumer `j` (evaluation, selection, sampling,
roughness, or BSSRDF transport). The next representation must satisfy:

```text
semantic adequacy:
  result_j(k, blocks) = Cycles_result_j(k, decode_k(blocks))

demand loading:
  dynamically executed payload reads are a subset of A(k)

inactive-family non-liveness:
  no field outside A(k) is live across the family dispatch merge

bounded construction:
  generated control/data size is O(sum_f |handler_f| + |loop_j|),
  not O(number_of_sites * number_of_families * loop_state_width)
```

The tagged eliminator must pass common fields and a family-specific block view
directly to a family-specific consumer. It must merge only `R_j`, never return
a universal closure record. If Luisa's structured AST cannot preserve that
scope through the closure loop, the fix belongs in XIR aggregate/liveness or
control-flow lowering rather than in another renderer-side ad hoc branch.

No callable should receive blanket `noinline` or `alwaysinline`. Cycles leaves
ordinary profitability decisions to the backend, and the accepted Luisa HIP
policy now does the same. Any new callable boundary must be justified by final
IR, object, register/private metadata, and HIP time rather than source size.

## Priorities

1. Implement the non-product tagged eliminator or the compiler scope/liveness
   support required to express it without cloning loop state. Add a structural
   regression proving inactive payloads do not enter the merged SSA state.
2. Add the missing high-use closure semantics: Principled/Glossy anisotropy
   and tangent, thin film, standalone Metallic/Sheen, then Hair families.
3. Add typed multi-result SVM instructions where a Cycles handler performs one
   expensive operation and writes multiple live outputs. Barbershop currently
   has no exact fusable Image or Vector Math pairs, so this is a general
   completeness/other-scene task rather than its present hotspot.
4. Implement the remaining ray-tracing value nodes through backend-neutral
   RayQuery semantics and keep HIP traversal optimization in the HIP backend.
5. Regenerate a Blender 5.2 coverage baseline and run the release gate, then
   repeat complex-scene fallback/HIP/native-XIR Vulkan correctness and HIP
   per-stage profiling.

## Reproduction snippets

Current Barbershop executable-image census:

```sh
build/bin/psycles_inspect_blender_material \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 '*'
```

Psycles trace aggregation:

```sql
select name,
       count(*) as calls,
       sum(duration) / 1000.0 as milliseconds,
       sum(grid_x * grid_y * grid_z) as work,
       sum(duration) * 1.0 / sum(grid_x * grid_y * grid_z) as ns_per_work
from kernels
where name = 'kernel_23a343c4e8b2218d'
group by name;
```

The stage/hash association is printed by the renderer as:

```text
wavefront_resume_5/shade_surface -> 23a343c4e8b2218d
```

Static object inspection used `llvm-readelf`, `llvm-nm`, and `llvm-objdump` on
the dumped HIP code objects. No profiler counter result is claimed: rocprof's
PMC collection on this host currently aborts in its vector assertion, while
ordinary kernel tracing is reliable.
