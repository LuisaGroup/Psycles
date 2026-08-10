# Surface-operation reachability and shared physical setup

Date: 2026-08-10

## Outcome

This checkpoint removes four independent sources of scene-dependent path-kernel
growth without changing Blender graph values or Cycles closure semantics:

1. emission code is recorded only for surface tags whose scene-unioned closure
   plan may emit; and
2. repeated Principled dielectric physical setup is represented by two shared,
   strongly typed Luisa callables instead of being cloned into every material
   topology and every surface operation; and
3. the smaller but more frequent Principled diffuse allocation/setup is
   represented by one additional typed callable; and
4. Principled metallic Fresnel, GGX energy, allocation, and layer attenuation
   are represented by two energy-policy-specialized typed callables.

On the unchanged Lone Monk export, the complete cold Vulkan path module falls
from 203,652 to 167,408 pre-restructure XIR instructions (17.8%). The main
kernel falls 17.6%, raw SPIR-V falls 23.4%, and native AST-to-SPIR-V time falls
31.1%. The rendered PPM and Combined PFM hashes remain bit-identical.

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

## Lone Monk cold Vulkan measurements

Each row is one shader-cache-disabled process compiling and executing the full
scene path kernel at 1x1 and 1 spp. The tiny launch isolates shader construction;
it is not a rendering-throughput benchmark. Vulkan selected the RX 9070 XT via
RADV and used native XIR-to-SPIR-V throughout; DXC was not loaded.

| Metric | Initial baseline | Emission pruning | Shared dielectric | + diffuse | + metallic | Initial to final |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XIR definitions | 20 | 21 | 23 | 24 | 25 | +5 definitions |
| total XIR instructions | 203,652 | 179,622 | 169,378 | 168,706 | 167,408 | -17.8% |
| main-kernel XIR instructions | 144,135 | 127,331 | 120,200 | 119,736 | 118,754 | -17.6% |
| `surface_emission` XIR instructions | 8,302 | 1,074 | 1,074 | 1,074 | 1,074 | -87.1% |
| `surface_evaluate_light` XIR instructions | 36,930 | 36,930 | 33,251 | 33,019 | 32,495 | -12.0% |
| raw SPIR-V words | 1,431,985 | 1,192,798 | 1,111,962 | 1,107,189 | 1,096,841 | -23.4% |
| optimized SPIR-V words | 1,116,158 | 1,087,675 | 1,010,627 | 1,005,854 | 996,375 | -10.7% |
| ordinary XIR inline | 348.30 ms | 301.09 ms | 274.17 ms | 280.65 ms | 284.00 ms | -18.5% |
| SPIR-V XIR legalization | 10,045 ms | 6,607 ms | 6,296.58 ms | 6,339.37 ms | 6,502.61 ms | -35.3% |
| native AST-to-SPIR-V | 23,492 ms | 17,184 ms | 16,077.31 ms | 15,952.92 ms | 16,180.90 ms | -31.1% |
| driver compute-pipeline creation | 86,333 ms | 77,958 ms | 77,884.27 ms | 77,476.73 ms | 76,792.49 ms | -11.0% |
| complete shader JIT | 109,979 ms | 95,269 ms | 94,077.8 ms | 93,552.1 ms | 93,093.7 ms | -15.4% |

The final reachable setup definitions contain 247 instructions for ordinary
dielectric GGX, 319 for preserve-energy dielectric GGX, 24 for diffuse, and 208
for ordinary metallic GGX. Their small fixed cost replaces repeated physical
setup in the 19 reachable dielectric, 18 reachable diffuse, and 3 reachable
metallic topology occurrences. Driver pipeline creation remains 82.5% of the
final JIT wall time and is now the
dominant Vulkan tail; further IR work should still reduce its input, but XIR
passes are no longer the majority of the measured wall time.

The exact output fingerprints for all three retained checkpoints are:

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
| direct/shared typed-ABI numeric parity | pass | pass | pass |
| five-definition callable-shape guard | pass | pass | pass |
| surface metadata/reachability regressions | pass | shared host test | shared host test |

The focused closure-plan fixture records 25,015 conservative instructions and
5,345 scene-specialized instructions, a 78.6% reduction. The final Lone Monk
run and all focused regressions used the same production implementation.

## Remaining hotspot

The next structural target is a fresh audit of scene-dependent host unrolling
and repeated surface-operation bodies. Any extraction must preserve a typed
semantic contract and pass a complete-module negative-boundary A/B like the
ones above. Separately, Vulkan driver pipeline creation
now warrants driver-level
profiling against optimized SPIR-V shape; blindly adding more XIR passes would
not address its measured share.
