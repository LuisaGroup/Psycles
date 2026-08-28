# Surface SVM last-use forwarding bound

## Outcome

The compact surface histogram now computes an exact definition/use relation
over the serialized typed-slot program and reports which adjacent handler
transitions can eliminate both a producer bank write and the target's bank
read. This is host-side observation only: it does not enter the shader AST,
change the bytecode image, or alter shader/cache identity.

On the Blender 5.2 Barbershop export, 213,556,758 hit-weighted transitions
satisfy the proof obligation. That is 91.79% of immediate direct dependencies
and 73.43% of all ordinary handler transitions. The corresponding optimistic
semantic bound is 213,556,758 producer writes plus 213,657,598 target reads,
or 427,214,356 typed-bank operations. The read count is slightly larger than
the transition count because 100,840 executions consume the source in two
target operand positions.

This makes result forwarding a materially stronger candidate than the rejected
superinstruction overlay. The overlay removed only fetch/dispatch structure
and regressed HIP; last-use forwarding targets the data movement itself. These
are semantic operation counts, not hardware transactions or a speedup claim.

## Formal model

For one program segment, let ordinary instruction `i` first read its operand
addresses `R_i` and then define local typed address `d_i`. Linear-scan coloring
may assign the same encoded address to several definitions. The analysis state
therefore maps each address to its current *definition epoch*, rather than
treating equal slot numbers as one value:

```text
A : encoded typed local address -> defining instruction offset
L : defining instruction offset -> final semantic use offset

for instruction i:
    for r in R_i, if r is local: L[A[r]] = i
    A[d_i] = i
```

Reads precede the definition update, which is required for in-place slot reuse.
An explicit surface-normal commit reads its selected address, records that use,
and clears `A`, because the normal prefix and endpoint root are independently
colored. After the final value instruction, every local closure Mix factor and
leaf operand records the distinguished terminal offset `instruction_count`.
Parameters never enter `A`; optional invalid closure operands are ignored.

Induction over the stream proves that immediately before each read, `A[a]` is
the unique most recent definition of `a` in the current transaction segment.
Therefore `L` names the exact last serialized observer of every definition,
including slot reuse, the normal boundary, and closure consumers.

For adjacent ordinary instructions `(i, i+1)`, the histogram records an exact
target operand bit mask

```text
M = { q | address(target operand q) = d_i }.
```

The write-and-read elimination condition is

```text
M != empty and L[i] = i + 1.
```

Under that condition, evaluate the producer to value `v`, omit its bank write,
substitute `v` for every target operand named by `M`, and execute the target.
All earlier observations precede the producer definition; all target reads see
the same `v`; and `L[i] = i+1` proves that no later instruction, normal commit,
or closure record observes the removed slot definition. Since value handlers
are pure apart from their explicit result-bank write, the suffix state is
unchanged. This proves the local rewrite independently of node names or scene
heuristics.

## Barbershop evidence

The run used compact populate-once surface execution, HIP on the RX 9070 XT,
320x240, 16 fixed samples, and staged wavefront:

```sh
PSYCLES_COMPACT_SURFACE_VALUES=1 \
PSYCLES_POPULATE_SURFACE_ONCE=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  /var/tmp/psycles-last-use-forwarding-final.TC3hUd/barbershop.exr \
  hip 320 240 16 16 - 0 0 0 0 16 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 0 0 0 1 1048576 \
  - /var/tmp/psycles-last-use-forwarding-final.TC3hUd/barbershop-histogram.json
```

The render completed in 0.408510 s. Histogram schema v5 reports `exact=true`,
3,364,440 surface populations, 298,373,202 value-instruction executions,
2,218,429 normal commits, 290,850,117 ordinary transitions, and 468 exact
transition classes.

| Classification | Executions | Relevant share |
| --- | ---: | ---: |
| All ordinary transitions | 290,850,117 | 100.00% |
| Immediate direct dependency | 232,655,813 | 79.99% of all |
| Direct and source last-used by target | 213,556,758 | 73.43% of all / 91.79% of direct |
| Forwardable target has a dynamic-route source operand | 113,998,097 | 53.38% of forwardable |
| Optimistic eliminated target reads | 213,657,598 | semantic upper bound |
| Optimistic eliminated source writes plus target reads | 427,214,356 | semantic upper bound |

Forwardable source-result banks are 87,380,483 scalar executions (40.92%) and
126,176,275 float3 executions (59.08%). No uint64 source occurs in this scene.

| Exact target operand mask | Executions | Exact classes |
| ---: | ---: | ---: |
| `0x01` | 175,337,880 | 133 |
| `0x02` | 28,076,803 | 24 |
| `0x03` | 100,840 | 2 |
| `0x04` | 5,734,255 | 8 |
| `0x08` | 4,173,118 | 1 |
| `0x10` | 1,799 | 1 |
| `0x20` | 132,063 | 2 |

The hottest exact last-use families span the material graph rather than one
special node:

| Source -> target | Executions | Target mask | Dynamic subset |
| --- | ---: | ---: | ---: |
| Mix -> Color to Scalar | 23,478,329 | `0x01` | `0x00` |
| Color to Scalar -> Color Ramp | 11,645,433 | `0x01` | `0x01` |
| Color to Scalar -> Math | 10,479,697 | `0x01` | `0x01` |
| Math -> Scalar to Color | 9,780,243 | `0x01` | `0x01` |
| Mapping -> Image Color | 7,378,232 | `0x01` | `0x00` |
| Color Ramp -> Mix | 7,304,513 | `0x02` | `0x02` |
| Scalar to Color -> Mix | 7,214,422 | `0x02` | `0x02` |
| Math -> Clamp | 6,540,964 | `0x01` | `0x00` |
| Subtract -> Multiply | 6,391,547 | `0x01` | `0x01` |
| Vector Math Vector -> Vector Math Value | 6,391,547 | `0x01` | `0x00` |

The broad operation distribution again rejects a Bump-only patch. An eventual
implementation must preserve one generic typed forwarding contract and prove
that its callable ABI, dispatch structure, and live payload do not recreate the
code-size/private-storage regression seen in pair fusion.

## Regression and backend validation

The compiler analysis has permanent host regressions for:

- an in-place scalar chain, proving operand reads use the old definition epoch;
- an ordinary read before any definition, which fails closed;
- a normal commit consuming the prefix definition and clearing the namespace;
- a post-commit read of a prefix slot, which fails closed; and
- a closure leaf retaining the final definition at the terminal offset.

The existing all-pass integration fixture additionally requires nonzero
last-use forwarding candidates, validates every operand mask and bank, and
requires exact equality between one sample batch and split dispatches.
Combined, Normal, Albedo, all light passes, volume passes, and every scheduler
mode in the fixture remain covered.

```sh
cmake --build build --parallel 32 --target \
  psycles_surface_program_metadata_tests \
  psycles_luisa_sample_dispatch_film_tests \
  psycles_render_blender_scene

build/psycles_surface_program_metadata_tests
build/bin/psycles_luisa_sample_dispatch_film_tests fallback
build/bin/psycles_luisa_sample_dispatch_film_tests hip

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
build/bin/psycles_luisa_sample_dispatch_film_tests vk
```

All four commands passed. The Vulkan run used the strict native
XIR-to-SPIR-V route with DXC disabled.
