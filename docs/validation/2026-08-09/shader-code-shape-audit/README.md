# Shader code-shape and callable-hash audit

This checkpoint removes material data from shader specialization without
changing the authored Blender graph. It is a compiler/JIT validation, not an
image-quality comparison; no Cycles oracle or triptych is needed because the
rendering equations and closure topology are unchanged.

## Compiler invariant

Every node property is classified by `PropertyRole`:

- `code_shape` participates in the graph structure signature and may select
  host-stage node/component implementations;
- `runtime_parameter` participates in the parameter signature, is excluded
  from the structure signature, and must lower to a typed `ParameterDesc` with
  `ParameterSource::property`.

`SurfaceProgramBuilder` rejects a graph if any runtime property was not lowered
into the parameter block. This makes the classification a checked contract:
excluding a value from the structure signature without providing its runtime
operand cannot silently reuse the wrong shader.

The first migrated properties are image/environment handles, image box blend,
UV and vertex-color attribute IDs, Brick numeric layout values, normal-map UV
ID, and Magic depth. Unsigned material data remains an exact 64-bit Luisa
`ulong` expression; regression data uses non-zero high bits so a float or
32-bit truncation cannot pass accidentally.

Magic depth previously selected one host-built callable per authored value.
It now drives one structured device `$for` containing one `$switch`, clamped to
the Cycles maximum depth of ten. XIR inspection asserts exactly one loop and one
switch in the complete kernel module.

## Luisa callable hash invariant

Luisa `next` commit `cb54780b7` makes completed equivalent callable definitions
canonical by their 64-bit structural hash. The hash audit covered every
code-generation or ABI-bearing `FunctionBuilder` field:

- function tag, body and statement boundary, return type;
- ordered arguments and bound/unbound resource-binding kinds;
- builtin, local, and shared declarations and ordered variable usages;
- captured constant type and exact payload hash;
- curve bases and direct/propagated builtin operation sets;
- block size, allowed warp size, atomic-float, printing, and cooperative flags.

Expression and statement member fields were audited as the recursive base of
the function hash. The audit fixed omitted debug-break wrapper identity,
CPU/GPU custom-operation semantics, buffer-binding size, and exact half payload
bits. Field tags and counts delimit the function-level groups, so different
partitions cannot alias merely because their flattened hash sequences match.

The deliberate exclusions are resource handles/offsets/payload values, which
are call-site operands after AST construction, and names/variable names/current
function attributes, which are non-semantic metadata with no codegen consumer.
Canonicalization happens only after each call has materialized its captures;
two equal-hash callables therefore share a definition while retaining distinct
resource operands. Kernels and coroutines are never merged because they are
independently addressable entry points. As agreed for this project, a true
64-bit collision is treated as negligible and does not trigger a deep-equality
fallback.

Regressions cover independently constructed equal callables, unequal literal
callables, repeated call sites, distinct kernel entries, different captured
buffers, declaration/binding differences, and each repaired leaf hash.
`test_xir_translators` passes 102 assertions in 23 tests.

## Complex-scene code-shape result

The official Classroom bundle produces 97 compiled surface records (materials,
light shaders, and world). Before runtime-property classification it produced
71 unique structure signatures. The same export now produces 59, a reduction
of 12 unique programs (16.9%), while all 97 records retain independently bound
material values.

The current result was reproduced with Blender 5.2.0 LTS and:

```bash
blender assets/official-blender-scenes/classroom/classroom.blend \
  --background --python-exit-code 1 \
  --python tools/export_psycles_scene.py -- <export-directory>

build/bin/psycles_inspect_blender_material <export-directory> '*' \
  | awk '/ signature / {print $NF}' | sort -u | wc -l
```

The final command reports `59`.

## Validation matrix

| Check | fallback | HIP (RX 9070 XT) | Vulkan native XIR/SPIR-V |
|---|---:|---:|---:|
| Magic numeric oracle and one-loop/one-switch XIR shape | pass | pass | pass |
| Exact 64-bit attribute ID through typed surface evaluation | pass | pass | pass |
| Psycles contract tests | pass | shared host contract | shared host contract |

The native Vulkan runs emitted SPIR-V directly: the Magic test optimized to
2,987 words and the attribute test to 1,590 words. HIP generated 5,172-byte
and 5,516-byte AMDGPU programs respectively before final linking.

## Remaining code-shape work

`SurfaceDispatch` still records the material-topology switch in several
operation-specific callables, and unsampled ColorRamp/RGB-Curve nodes still
expand host loops with authored element count. These are the next structural
targets; neither is hidden by the current signature reduction.
