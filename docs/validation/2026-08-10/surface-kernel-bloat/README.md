# Surface-operation reachability and shared shader primitives

Date: 2026-08-10

## Outcome

This checkpoint removes seven independent sources of scene-dependent path-kernel
growth without changing Blender graph values or Cycles closure semantics:

1. emission code is recorded only for surface tags whose scene-unioned closure
   plan may emit; and
2. repeated Principled dielectric physical setup is represented by two shared,
   strongly typed Luisa callables instead of being cloned into every material
   topology and every surface operation; and
3. the smaller but more frequent Principled diffuse allocation/setup is
   represented by one additional typed callable; and
4. Principled metallic Fresnel, GGX energy, allocation, and layer attenuation
   are represented by two energy-policy-specialized typed callables; and
5. Cycles image interpolation is represented by the reachable members of a
   finite set of typed interpolation/extension callables instead of being
   expanded once per Image Texture node and surface operation; and
6. RGB/HSV/HSL conversion is represented by the reachable members of a finite
   typed transform family instead of being expanded once per color node and
   surface operation; and
7. Mapping vector transforms are represented by the reachable members of the
   finite POINT/TEXTURE/VECTOR/NORMAL family instead of being expanded once per
   Mapping node and surface operation.

On the unchanged Lone Monk export, the complete cold Vulkan path module falls
from 203,652 to 127,838 pre-restructure XIR instructions (37.2%). The main
kernel falls 35.7%, raw SPIR-V falls 38.8%, and native AST-to-SPIR-V time falls
46.9%. The rendered PPM and Combined PFM hashes remain bit-identical.

## Formal reachability rule

For a scene surface tag `t` and operation `o`, code generation now obeys

```text
record(t, emission) iff capabilities(t).may_emit
```

For a Principled closure, `may_emit` may be disproved only by a finite direct
socket value: alpha `<= 0`, an exactly zero emission color, or an exactly zero
emission strength. Linked and non-finite values remain unknown and therefore
retain emission. The proof is computed per material instance and unioned over
all instances sharing one deduplicated topology before code is recorded. This
prevents one material's direct value from specializing code used by another.

`SurfaceDispatch::emission` dispatches over the proven emissive tag set with a
zero default. Capability calculation, the operation dispatcher, and the
emission evaluator consume the same scene-unioned feature mask, so metadata and
generated code cannot disagree.

Regressions cover direct-zero emission, linked-zero conservatism, topology
union, capability refinement, and a recording surface that proves a
non-emissive branch is absent from the generated AST.

## Typed populate/setup boundary

Lone Monk contains 35 source materials, 37 compiled material instances, and 24
deduplicated shader-graph topologies. Those topologies contain 19 reachable
Principled dielectric occurrences. Previously, the physical Fresnel, GGX
energy, table lookup, allocation, sample-weight, albedo, and lower-layer
attenuation algebra was expanded once per occurrence in each surface operation.

The new boundary follows Luisa's multistage model:

```text
authored graph sockets + shading-point state
                 |
                 v
        typed dielectric populate
                 |
                 v
       typed physical setup callable
                 |
                 v
       original Cycles closure record
```

The host-side `ShaderServices` abstraction optionally supplies a
`SurfaceClosureSetupProvider`. Production path-tracer services supply shared
callables; standalone `GraphSurface` users retain the canonical inline path.
The ABI names every semantic field and returns a typed result. No `float4`
parameter protocol, closure array, material-name switch, host evaluation,
Blender/Cycles pre-bake, or weakened closure is involved.

GGX energy preservation is immutable topology metadata, so it produces exactly
two callables: ordinary GGX and preserve-energy GGX. All authored numeric values
remain Luisa expressions. The physical helpers also received narrow overloads:
adjusted IOR depends only on IOR and Specular IOR Level, while GGX energy depends
only on roughness, the host preservation flag, incident cosine, and `Fss`. This
avoids constructing a mostly-unused `TracedClosure`, whose default DSL fields
would themselves record dead AST.

The populate/setup split was selected by measurement, not convention. Moving
the derived glossy normal and incident cosine back to each material call site
made the module larger because those computations became live in every
topology:

| Boundary | Total XIR | Main kernel | `surface_evaluate_light` |
| --- | ---: | ---: | ---: |
| raw semantic state; derive inside shared callable | 169,378 | 120,200 | 33,251 |
| derived normal/cosine at every call site | 173,492 | 123,050 | 34,699 |

The rejected boundary was 2.4% larger overall and 4.3% larger in light
evaluation. The retained architecture lets standalone graph evaluation reuse
values already populated for sibling Principled lobes, while the path tracer
fuses populate and setup inside the cross-topology callable.

## Shared diffuse setup follow-up

The next family was selected by total scene cost rather than by the size of one
closure body. The focused single-topology marginal costs are approximately 544
XIR instructions for diffuse after dielectric and 1,221 for metallic after
dielectric. Lone Monk, however, contains 18 reachable diffuse occurrences and
only 3 metallic occurrences. Diffuse therefore has the larger repeated cost.

`PrincipledDiffuseComponent` now exposes one canonical transfer function. Its
input is the original typed `{lower_weight, color, subsurface_weight}` socket
state. The shared callable performs the color clamp, lower-layer product,
subsurface attenuation, Cycles allocation cutoff, and sample-weight setup. It
returns only the three fields that are not already present in the authored
closure record. No closure array, material value, or weakly typed register
block crosses this boundary.

The regression includes the fixed callable definition and ABI packing in its
module count. This deliberately records the negative boundary as well as the
amortized one:

| Focused fixture | Inline XIR | Shared XIR | Result |
| --- | ---: | ---: | --- |
| one dielectric+diffuse topology | 5,355 | 6,050 | shared is not amortized |
| four independent topology branches | 16,834 | 14,547 | shared is 13.6% smaller |

Thus the optimization is justified by cross-topology reuse, not by hiding the
callable body from the counter. On the full scene the new callable contains 24
instructions and produces the following post-optimization change relative to
the dielectric-only checkpoint:

| Metric | Shared dielectric | Shared dielectric + diffuse | Change |
| --- | ---: | ---: | ---: |
| XIR definitions | 23 | 24 | +1 shared definition |
| total XIR instructions | 169,378 | 168,706 | -672 (-0.40%) |
| main-kernel XIR instructions | 120,200 | 119,736 | -464 (-0.39%) |
| `surface_evaluate_light` XIR instructions | 33,251 | 33,019 | -232 (-0.70%) |
| raw SPIR-V words | 1,111,962 | 1,107,189 | -4,773 (-0.43%) |
| optimized SPIR-V words | 1,010,627 | 1,005,854 | -4,773 (-0.47%) |
| ordinary XIR inline | 274.17 ms | 280.65 ms | +2.4% |
| SPIR-V XIR legalization | 6,296.58 ms | 6,339.37 ms | +0.7% |
| native AST-to-SPIR-V | 16,077.31 ms | 15,952.92 ms | -0.8% |
| driver compute-pipeline creation | 77,884.27 ms | 77,476.73 ms | -0.5% |
| complete shader JIT | 94,077.8 ms | 93,552.1 ms | -0.6% |

The two small frontend-pass regressions are reported rather than rounded away;
the downstream module, SPIR-V, driver time, and end-to-end JIT all improve.
The one-pixel PPM and Combined PFM retain the exact fingerprints below.

## Shared metallic setup follow-up

`PrincipledMetallicComponent` applies the same typed populate/setup boundary to
the metallic lobe. Its input remains the authored base color, metallic,
roughness, tint, normal, incoming direction, and shading-point state. The
shared callable performs the canonical normal correction, clamps, Fresnel F82,
GGX energy lookup, allocation, sample weighting, albedo estimate, and lower
layer attenuation. It returns named physical fields; it does not serialize a
generic closure or evaluate graph values on the host.

The common normal/cosine/roughness/tint populate relation used by metallic and
dielectric is now one host-side C++ function that records the same Luisa AST.
The inline path passes its already-populated state to avoid recomputing sibling
Principled inputs, while the production callable starts from raw semantic
inputs so the work is shared across material topologies.

As with diffuse, the complete-module negative boundary is explicit:

| Focused fixture | Inline XIR | Shared XIR | Result |
| --- | ---: | ---: | --- |
| one metallic+dielectric topology | 6,071 | 7,408 | shared is not amortized |
| four independent topology branches | 19,698 | 16,328 | shared is 17.1% smaller |

On Lone Monk only the ordinary-energy metallic specialization is reachable; it
contains 208 instructions and replaces repeated setup in both the main path
kernel and light evaluation:

| Metric | Shared dielectric + diffuse | + shared metallic | Change |
| --- | ---: | ---: | ---: |
| XIR definitions | 24 | 25 | +1 reachable definition |
| total XIR instructions | 168,706 | 167,408 | -1,298 (-0.77%) |
| main-kernel XIR instructions | 119,736 | 118,754 | -982 (-0.82%) |
| `surface_evaluate_light` XIR instructions | 33,019 | 32,495 | -524 (-1.59%) |
| raw SPIR-V words | 1,107,189 | 1,096,841 | -10,348 (-0.93%) |
| optimized SPIR-V words | 1,005,854 | 996,375 | -9,479 (-0.94%) |
| ordinary XIR inline | 280.65 ms | 284.00 ms | +1.2% |
| SPIR-V XIR legalization | 6,339.37 ms | 6,502.61 ms | +2.6% |
| native AST-to-SPIR-V | 15,952.92 ms | 16,180.90 ms | +1.4% |
| driver compute-pipeline creation | 77,476.73 ms | 76,792.49 ms | -0.9% |
| complete shader JIT | 93,552.1 ms | 93,093.7 ms | -0.5% |

The frontend timing changes are within one cold-run noise sample and are not
claimed as wins. The structural counts, downstream SPIR-V size, driver time,
and end-to-end JIT all improve, while exact output fingerprints remain stable.

## Shared Cycles texture sampling follow-up

Lone Monk has 45 reachable Image Texture nodes. Before this checkpoint every
node expanded the complete software `TextureInterpolator` body into every
surface operation that evaluated its graph: coordinate splitting, four
extension policies, nearest/linear/cubic interpolation, and as many as sixteen
image reads. This was graph-node-level duplication, so topology deduplication
alone could not bound it.

Interpolation and extension are immutable shader-graph metadata. The new
formal bound is therefore

```text
texture_sampler_definitions(module)
    <= |reachable canonical (interpolation family, extension mode) pairs|
    <= 3 * 4
```

The three canonical families are nearest, linear, and cubic (including Smart);
the four extension modes are repeat, clip, extend, and mirror. Texture handles
and UVs remain typed Luisa expressions. A host-stage provider selects the typed
callable while recording the shader, so no device mode switch, generic
`float4` register protocol, graph VM, host texture evaluation, or pre-baking is
introduced. Consumers outside the production path tracer retain the same
canonical inline implementation.

The focused fixture repeats one linear/repeat sample eight times and counts the
complete module, including its reachable callable definition:

| Focused fixture | Inline XIR | Shared XIR | Change |
| --- | ---: | ---: | ---: |
| eight independent sample sites | 5,945 | 790 | -86.7% |

Another shape guard records all four source interpolation tags and all four
extension tags. It requires exactly twelve definitions, while two independently
constructed bundles that use the same specialization must collapse to one
definition by complete callable AST hash. Device regressions compare the old
direct and shared paths at seven interior and boundary coordinates for all
sixteen source-tag pairs.

Only linear/repeat and linear/clip are reachable in Lone Monk, so the production
module gains two small definitions and eliminates the repeated bodies:

| Metric | + shared metallic | + shared texture sampling | Change |
| --- | ---: | ---: | ---: |
| XIR definitions | 25 | 27 | +2 reachable definitions |
| total XIR instructions | 167,408 | 136,521 | -30,887 (-18.5%) |
| main-kernel XIR instructions | 118,754 | 98,253 | -20,501 (-17.3%) |
| `surface_evaluate_light` XIR instructions | 32,495 | 22,270 | -10,225 (-31.5%) |
| raw SPIR-V words | 1,096,841 | 933,506 | -163,335 (-14.9%) |
| optimized SPIR-V words | 996,375 | 850,086 | -146,289 (-14.7%) |
| ordinary XIR inline | 284.00 ms | 243.92 ms | -14.1% |
| SPIR-V XIR legalization | 6,502.61 ms | 5,107.07 ms | -21.5% |
| native AST-to-SPIR-V | 16,180.90 ms | 12,559.43 ms | -22.4% |
| driver compute-pipeline creation | 76,792.49 ms | 77,701.16 ms | +1.2% |
| complete shader JIT | 93,093.7 ms | 90,365.8 ms | -2.9% |

The single driver timing regression is retained as measured and is within the
noise of one cold pipeline build; it is not presented as an improvement. The
structural reductions and exact output hashes are deterministic.

## Shared color-transform follow-up

Lone Monk contains 32 reachable Hue/Saturation nodes, in addition to any HSV
blend and Separate/Combine Color operations selected by static graph metadata.
Previously every use expanded the full RGB-to-HSV and HSV-to-RGB algebra into
each surface operation that traversed the topology.

Color conversion is a closed family of four pure typed functions:

```text
{RGB -> HSV, HSV -> RGB, RGB -> HSL, HSL -> RGB}
```

`ShaderServices` now exposes an optional host-stage
`SurfaceColorTransformProvider`. Standalone GraphSurface clients evaluate the
canonical inline functions. Production buffer services expose typed callables
recorded from those same functions. The graph still evaluates its original
typed values in topological order; no generic value register, device opcode,
host color evaluation, or material pre-bake is introduced.

The formal module bound is

```text
color_transform_definitions(module)
    <= |reachable transform endpoints|
    <= 4
```

Constructing the provider does not attach all four definitions. The regression
requires a kernel using only RGB-to-HSV to contain exactly one definition, all
four endpoints to contain exactly four, and two independently constructed
providers using RGB-to-HSV to collapse to one definition by complete callable
AST hash. The focused Hue/Saturation-shaped roundtrip gives:

| Focused fixture | Inline XIR | Shared XIR | Change |
| --- | ---: | ---: | ---: |
| eight RGB-to-HSV-to-RGB sites | 3,169 | 524 | -83.5% |

Only RGB-to-HSV and HSV-to-RGB are reachable in Lone Monk. Their definitions
contain 38 and 43 instructions; the HSL endpoints are absent. Relative to the
shared-texture checkpoint:

| Metric | + shared texture sampling | + shared color transforms | Change |
| --- | ---: | ---: | ---: |
| XIR definitions | 27 | 29 | +2 reachable definitions |
| total XIR instructions | 136,521 | 130,971 | -5,550 (-4.07%) |
| main-kernel XIR instructions | 98,253 | 94,959 | -3,294 (-3.35%) |
| `surface_emission` XIR instructions | 507 | 437 | -70 (-13.8%) |
| `surface_evaluate_light` XIR instructions | 22,270 | 20,003 | -2,267 (-10.2%) |
| raw SPIR-V words | 933,506 | 897,508 | -35,998 (-3.86%) |
| optimized SPIR-V words | 850,086 | 816,866 | -33,220 (-3.91%) |
| ordinary XIR inline | 243.92 ms | 221.83 ms | -9.06% |
| SPIR-V XIR legalization | 5,107.07 ms | 4,962.88 ms | -2.82% |
| native AST-to-SPIR-V | 12,559.43 ms | 12,134.65 ms | -3.38% |
| driver compute-pipeline creation | 77,701.16 ms | 79,303.15 ms | +2.06% |
| complete shader JIT | 90,365.8 ms | 91,539.7 ms | +1.30% |

The structural counts and frontend reductions are deterministic. The single
driver timing sample regressed enough to make end-to-end JIT slower despite the
smaller optimized SPIR-V; it is reported as measured and is not claimed as a
runtime win. Driver pipeline creation remains the dominant, noisy tail.

## Shared vector-mapping follow-up

The source Lone Monk scene contains 38 Mapping nodes. After material-instance
normalization and topology deduplication, 22 Mapping occurrences remain
reachable in recorded surface operations. All 22 use Blender's POINT mode.
Previously, each occurrence expanded the Euler sine/cosine rotation and affine
transform into every operation that traversed its graph.

Mapping mode is immutable graph metadata and belongs to the closed family

```text
{POINT, TEXTURE, VECTOR, NORMAL}
```

`ShaderServices` now exposes an optional host-stage
`SurfaceVectorMappingProvider`. Production services select one strongly typed,
lazy callable while recording the graph. POINT and TEXTURE carry the semantically
required `{input, location, rotation, scale}` values; VECTOR and NORMAL use the
narrower `{input, rotation, scale}` ABI. Standalone GraphSurface clients retain
the same canonical inline definitions. Static axis remapping remains a small
host-recorded component selection and does not become a device mode switch.

The formal module bound is therefore independent of Mapping-node count:

```text
mapping_definitions(module)
    <= |reachable Mapping modes|
    <= 4
```

The regression requires a POINT-only kernel to contain exactly one definition,
all modes to contain exactly four, and independently constructed POINT providers
to deduplicate by complete callable AST hash. It compares the direct and shared
paths for every mode on fallback, HIP, and native Vulkan, including zero and
negative scale components. The focused shape result is:

| Focused fixture | Inline XIR | Shared XIR | Change |
| --- | ---: | ---: | ---: |
| eight POINT Mapping sites | 1,473 | 361 | -75.5% |

Only the 35-instruction POINT definition is reachable in Lone Monk. Relative
to the shared-color checkpoint:

| Metric | + shared color transforms | + shared vector mapping | Change |
| --- | ---: | ---: | ---: |
| XIR definitions | 29 | 30 | +1 reachable definition |
| total XIR instructions | 130,971 | 127,838 | -3,133 (-2.39%) |
| main-kernel XIR instructions | 94,959 | 92,649 | -2,310 (-2.43%) |
| `surface_emission` XIR instructions | 437 | 437 | unchanged |
| `surface_evaluate_light` XIR instructions | 20,003 | 19,145 | -858 (-4.29%) |
| raw SPIR-V words | 897,508 | 875,893 | -21,615 (-2.41%) |
| optimized SPIR-V words | 816,866 | 795,251 | -21,615 (-2.65%) |
| ordinary XIR inline | 221.83 ms | 229.64 ms | +3.52% |
| SPIR-V XIR legalization | 4,962.88 ms | 5,260.00 ms | +5.99% |
| native AST-to-SPIR-V | 12,134.65 ms | 12,481.60 ms | +2.86% |
| driver compute-pipeline creation | 79,303.15 ms | 79,793.11 ms | +0.62% |
| complete shader JIT | 91,539.7 ms | 92,379.1 ms | +0.92% |

The XIR and SPIR-V reductions are deterministic. This single cold timing sample
was slower at every noisy timing boundary despite the smaller module, so no
compile-time improvement is claimed for this checkpoint. The result is retained
for its bounded structural shape and exact semantic parity; repeated controlled
driver measurements remain separate from this compile-isolation check.

## Lone Monk cold Vulkan measurements

Each row is one shader-cache-disabled process compiling and executing the full
scene path kernel at 1x1 and 1 spp. The tiny launch isolates shader construction;
it is not a rendering-throughput benchmark. Vulkan selected the RX 9070 XT via
RADV and used native XIR-to-SPIR-V throughout; DXC was not loaded.

The retained vector-mapping run used:

```sh
PSYCLES_DISABLE_SHADER_CACHE=1 \
LUISA_LOG_LEVEL=verbose \
LUISA_VULKAN_PROFILE_COMPILATION=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_XIR_TRACE_PASSES=1 \
build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-transmission-dbdcb17/export \
  /var/tmp/psycles-kernel-shape-mapping-20260810/vk.ppm \
  vk 1 1 1 1
```

The complete trace and linear passes are under
`/var/tmp/psycles-kernel-shape-mapping-20260810/` on the measurement host.

| Metric | Initial baseline | Emission pruning | Shared dielectric | + diffuse | + metallic | + texture sampling | + color transforms | + vector mapping | Initial to final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XIR definitions | 20 | 21 | 23 | 24 | 25 | 27 | 29 | 30 | +10 definitions |
| total XIR instructions | 203,652 | 179,622 | 169,378 | 168,706 | 167,408 | 136,521 | 130,971 | 127,838 | -37.2% |
| main-kernel XIR instructions | 144,135 | 127,331 | 120,200 | 119,736 | 118,754 | 98,253 | 94,959 | 92,649 | -35.7% |
| `surface_emission` XIR instructions | 8,302 | 1,074 | 1,074 | 1,074 | 1,074 | 507 | 437 | 437 | -94.7% |
| `surface_evaluate_light` XIR instructions | 36,930 | 36,930 | 33,251 | 33,019 | 32,495 | 22,270 | 20,003 | 19,145 | -48.2% |
| raw SPIR-V words | 1,431,985 | 1,192,798 | 1,111,962 | 1,107,189 | 1,096,841 | 933,506 | 897,508 | 875,893 | -38.8% |
| optimized SPIR-V words | 1,116,158 | 1,087,675 | 1,010,627 | 1,005,854 | 996,375 | 850,086 | 816,866 | 795,251 | -28.8% |
| ordinary XIR inline | 348.30 ms | 301.09 ms | 274.17 ms | 280.65 ms | 284.00 ms | 243.92 ms | 221.83 ms | 229.64 ms | -34.1% |
| SPIR-V XIR legalization | 10,045 ms | 6,607 ms | 6,296.58 ms | 6,339.37 ms | 6,502.61 ms | 5,107.07 ms | 4,962.88 ms | 5,260.00 ms | -47.6% |
| native AST-to-SPIR-V | 23,492 ms | 17,184 ms | 16,077.31 ms | 15,952.92 ms | 16,180.90 ms | 12,559.43 ms | 12,134.65 ms | 12,481.60 ms | -46.9% |
| driver compute-pipeline creation | 86,333 ms | 77,958 ms | 77,884.27 ms | 77,476.73 ms | 76,792.49 ms | 77,701.16 ms | 79,303.15 ms | 79,793.11 ms | -7.58% |
| complete shader JIT | 109,979 ms | 95,269 ms | 94,077.8 ms | 93,552.1 ms | 93,093.7 ms | 90,365.8 ms | 91,539.7 ms | 92,379.1 ms | -16.0% |

The final reachable setup definitions contain 247 instructions for ordinary
dielectric GGX, 319 for preserve-energy dielectric GGX, 24 for diffuse, and 208
for ordinary metallic GGX. The reachable linear/clip and linear/repeat texture
samplers contain 195 and 211 instructions. Their small fixed cost replaces
repeated physical setup in the 19 reachable dielectric, 18 reachable diffuse,
and 3 reachable metallic topology occurrences, plus texture sampling at 45
reachable Image Texture nodes. The reachable RGB-to-HSV and HSV-to-RGB
definitions contain 38 and 43 instructions, and the reachable POINT Mapping
definition contains 35. Driver pipeline creation remains 86.4% of the final
JIT wall time and is now the dominant Vulkan tail; further
IR work should still reduce its input, but XIR passes are no longer the
majority of the measured wall time.

The exact output fingerprints for all retained checkpoints are:

```text
PPM:      3ff6c5463ace13c0f26a735ac1af2bb96ab8a9ba1cb4398359cf2466f63a4d1b
Combined: 4f93bceff43a46a086454e9a50745a497b39cd27e8a694f617a5c934fe3ed3eb
```

Because this is a 1x1 compile-isolation run, it is not a visual parity gate and
no triptych is claimed here. Full-resolution Cycles/Psycles triptychs remain a
separate scene-quality requirement.

## Regression matrix

`psycles_luisa_principled_setup_callable_tests` records both the canonical
direct implementation and the shared-callable implementation. Six device
cases cover ordinary allocation, below-cutoff weight, unit IOR, adjusted IOR
below one, disabled reflective caustics, bump-normal correction, diffuse color
clamping, full subsurface attenuation, metallic Fresnel/tint, and metallic
layer attenuation. Both GGX energy modes for dielectric and metallic, plus
diffuse, compare every returned field. The AST guard requires exactly five
custom callable definitions: one diffuse and two energy-policy specializations
for each specular family.

| Check | fallback | HIP | Vulkan native XIR/SPIR-V |
| --- | ---: | ---: | ---: |
| closure direct/shared typed-ABI numeric parity | pass | pass | pass |
| five-definition closure-callable shape guard | pass | pass | pass |
| texture direct/shared numeric parity | pass | pass | pass |
| texture finite-bound/hash-dedup shape guards | pass | pass | pass |
| color-transform direct/shared numeric parity | pass | pass | pass |
| color-transform used-only/bound/hash-dedup guards | pass | pass | pass |
| vector-mapping direct/shared numeric parity | pass | pass | pass |
| vector-mapping used-only/bound/hash-dedup guards | pass | pass | pass |
| surface metadata/reachability regressions | pass | shared host test | shared host test |

A complete `cmake --build build -j$(nproc)` passed after the lazy used-only
provider was finalized. The nine texture/color/mapping backend tests pass, as
does the standalone `psycles_luisa_compile_tests` graph-construction suite.

The focused closure-plan fixture records 25,015 conservative instructions and
5,345 scene-specialized instructions, a 78.6% reduction. The final Lone Monk
run and all focused regressions used the same production implementation.

## Remaining hotspot

The next structural target is a fresh audit of the remaining 92,649-instruction
main kernel and repeated surface-operation bodies. Graph-value operators and
normal/bump evaluation now warrant measurement because the already-deduplicated
24 topologies still contain 1,438 unique values. Any extraction must preserve a
typed semantic contract and pass a complete-module negative-boundary A/B like
the ones above. Separately, Vulkan driver pipeline creation warrants
driver-level profiling against optimized SPIR-V shape; blindly adding more XIR
passes would not address its measured share.
