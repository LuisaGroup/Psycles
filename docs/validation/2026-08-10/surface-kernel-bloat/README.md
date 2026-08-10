# Surface-operation reachability and shared physical setup

Date: 2026-08-10

## Outcome

This checkpoint removes two independent sources of scene-dependent path-kernel
growth without changing Blender graph values or Cycles closure semantics:

1. emission code is recorded only for surface tags whose scene-unioned closure
   plan may emit; and
2. repeated Principled dielectric physical setup is represented by two shared,
   strongly typed Luisa callables instead of being cloned into every material
   topology and every surface operation.

On the unchanged Lone Monk export, the complete cold Vulkan path module falls
from 203,652 to 169,378 pre-restructure XIR instructions (16.8%). The main
kernel falls 16.6%, raw SPIR-V falls 22.3%, and native AST-to-SPIR-V time falls
31.6%. The rendered PPM and Combined PFM hashes remain bit-identical.

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

## Lone Monk cold Vulkan measurements

Each row is one shader-cache-disabled process compiling and executing the full
scene path kernel at 1x1 and 1 spp. The tiny launch isolates shader construction;
it is not a rendering-throughput benchmark. Vulkan selected the RX 9070 XT via
RADV and used native XIR-to-SPIR-V throughout; DXC was not loaded.

| Metric | Initial reachable-closure baseline | Emission pruning | Shared dielectric setup | Initial to final |
| --- | ---: | ---: | ---: | ---: |
| XIR definitions | 20 | 21 | 23 | +3 shared definitions |
| total XIR instructions | 203,652 | 179,622 | 169,378 | -16.8% |
| main-kernel XIR instructions | 144,135 | 127,331 | 120,200 | -16.6% |
| `surface_emission` XIR instructions | 8,302 | 1,074 | 1,074 | -87.1% |
| `surface_evaluate_light` XIR instructions | 36,930 | 36,930 | 33,251 | -10.0% |
| raw SPIR-V words | 1,431,985 | 1,192,798 | 1,111,962 | -22.3% |
| optimized SPIR-V words | 1,116,158 | 1,087,675 | 1,010,627 | -9.5% |
| ordinary XIR inline | 348.30 ms | 301.09 ms | 274.17 ms | -21.3% |
| SPIR-V XIR legalization | 10,045 ms | 6,607 ms | 6,296.58 ms | -37.3% |
| native AST-to-SPIR-V | 23,492 ms | 17,184 ms | 16,077.31 ms | -31.6% |
| driver compute-pipeline creation | 86,333 ms | 77,958 ms | 77,884.27 ms | -9.8% |
| complete shader JIT | 109,979 ms | 95,269 ms | 94,077.8 ms | -14.5% |

The final two named setup definitions contain 247 instructions for ordinary
GGX and 319 for preserve-energy GGX. Their small fixed cost replaces repeated
physical setup in the 19 reachable dielectric topology occurrences. Driver
pipeline creation remains 82.8% of the final JIT wall time and is now the
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
below one, disabled reflective caustics, and bump-normal correction. Both GGX
energy modes compare every returned field. The AST guard also requires exactly
two custom callable definitions.

| Check | fallback | HIP | Vulkan native XIR/SPIR-V |
| --- | ---: | ---: | ---: |
| direct/shared typed-ABI numeric parity | pass | pass | pass |
| two-specialization callable-shape guard | pass | pass | pass |
| surface metadata/reachability regressions | pass | shared host test | shared host test |

The focused closure-plan fixture records 24,994 conservative instructions and
5,324 scene-specialized instructions, a 78.7% reduction. The final Lone Monk
run and all focused regressions used the same production implementation.

## Remaining hotspot

The next structural target is the remaining repeated physical closure setup in
the large surface-operation definitions, selected by occurrence count and XIR
cost rather than closure name alone. Any extraction must preserve the same
typed populate/setup contract and pass a negative boundary A/B like the one
above. Separately, Vulkan driver pipeline creation now warrants driver-level
profiling against optimized SPIR-V shape; blindly adding more XIR passes would
not address its measured share.
