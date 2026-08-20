# HIP surface translation working-set validation

## Scope and result

This experiment separates three quantities that must not be reported as one
"compression ratio":

1. the immutable material-program representation;
2. the final HIP code object, including the shared evaluator and the rest of
   the path tracer;
3. the private working set and spills of one shader invocation.

Only the first quantity has changed so far. The 189 unique Barbershop material
topologies lower to 216,272 bytes of typed value bytecode. The preceding
topology-expanded surface code object is 22,554,128 bytes. Dividing those two
numbers gives 104.3, but it is deliberately **not** claimed as the final code
object compression ratio: the bytecode does not yet include the future shared
device evaluator, closure population/evaluation, or non-material path-tracing
code.

The result nevertheless proves that the current 22.55 MB module and 37,216-byte
HIP private segment are not forced by the material graphs. The largest exact
typed local working set over all Barbershop topologies is 144 bytes. This is a
semantic storage bound for the selected value program, not an estimate derived
from LLVM spills.

## Failure mechanism

The failing 640x480 HIP megakernel uses the following final kernel metadata:

| Metric | Value |
| --- | ---: |
| Private segment per invocation | 37,216 B |
| VGPR allocation | 256 |
| VGPR spills | 693 |
| SGPR allocation | 108 |
| SGPR spills | 157 |
| Dynamic stack | enabled |

The same cached kernel completes sample zero and faults on sample one in a
normal process. It completes under `rocprofv3`, which records a
2,438,987,776-byte scratch allocation before the giant kernel. A full launch
and a 57,600-pixel row-band launch fault, whereas every row band succeeds when
capped at 32,768 invocations, including coverage of the complete image.

This rules out a fixed scene-buffer or pixel-coordinate out-of-bounds error.
The supported causal chain is:

```text
per-topology, per-stage shader expansion
  -> extreme LLVM control-flow/register pressure
  -> 37,216 B private state and heavy spills per invocation
  -> very large use-once HSA scratch demand
  -> page fault in the ordinary launch configuration
```

The experiment does not assign the failure to a particular signed integer
boundary: HSA scratch allocation is rounded and quantized by implementation
details. The structural dependency on dispatch population and private bytes is
the useful result.

## Formal typed storage model

The host graph is already in strict topological order. The storage planner
first proves that the selected instruction set is transitively closed over its
operands. It then computes exact last uses, with closure terminals modeled as
uses at the stream boundary. Parameter values remain immutable references into
the material parameter block and occupy no local slot.

For computed values, each typed value has one live interval in this fixed
order. Greedy expiration therefore colors the interval graph with its exact
peak number of simultaneously live values. An output may reuse a dying
same-bank operand slot only after all operands of the instruction have been
read. Forward references, inactive operands, unsupported result types,
read-before-write violations, and failure to converge to the declared boundary
outputs are rejected.

Barbershop produces:

| Metric | Value |
| --- | ---: |
| Unique material topologies | 189 |
| Unique value instructions | 13,335 |
| Reachable closure instructions | 559 / 559 |
| Value opcode kinds | 55 |
| Preparation-active values | 10,791 |
| Direct parameter references | 5,421 |
| Runtime instructions | 5,370 |
| Maximum scalar slots | 6 |
| Maximum vector slots | 10 |
| Maximum uint64 slots | 0 |
| Maximum typed local payload | 144 B |
| Sum of per-topology local payloads | 10,360 B |

The 37,216 / 144 = 258.4 ratio is also not a promised runtime reduction. The
LLVM private segment includes ABI state, compiler temporaries, spills, and
unrelated path state. It demonstrates that almost all of the present surface
pressure is introduced by translation/code generation rather than required by
the graph's data dependencies.

## Compact value-program image

The lowering preserves strong types and original closure inputs. It does not
evaluate or bake material values on the host. Parameter instructions are
elided and late-bound; computed values address one of three local banks. The
hot instruction is 16 bytes, variable-arity operands live in a compact address
stream, and uncommon immutable fields use a sparse 40-byte metadata side table.
Static tables are retained verbatim. Address and record layouts have compile-
time size and trivial-copyability checks.

| Bytecode component | Bytes |
| --- | ---: |
| Instructions | 85,920 |
| Operand addresses | 41,256 |
| Sparse metadata | 88,840 |
| Static table data | 256 |
| **Total** | **216,272** |

The intended structural change is one scene-pruned shared evaluator plus these
program records, so generated code scales mainly with used opcode/closure
families instead of with `(topology, stage)` pairs. The next acceptance point
must measure the final code object, private segment/spills, image equivalence,
and HIP render-only time. An interpreter can trade code size for indirect
loads and opcode dispatch; the current byte count alone is not a performance
claim.

## Regression and reproduction

The focused regression suite checks exact closure-output masks, transitive
closure, topological validity, read-before-write reuse, parameter elision,
typed-address round trips, operand ranges, metadata/static-table preservation,
and signed-zero preservation. The real-scene inspector lowers all 189
Barbershop topology programs.

```bash
cmake --build build \
  --target psycles_surface_program_metadata_tests \
           psycles_inspect_blender_material \
  --parallel "$(nproc)"

./build/psycles_surface_program_metadata_tests

./build/bin/psycles_inspect_blender_material \
  /var/tmp/psycles-official-redownload-20260814/exports/barbershop-5.2 \
  '*' \
  | rg '^(unique_|reachable_|value_opcode_kinds|preparation_|maximum_|topology_|value_bytecode_)'
```
