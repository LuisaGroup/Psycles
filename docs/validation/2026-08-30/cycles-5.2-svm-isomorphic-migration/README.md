# Cycles 5.2.1 SVM isomorphic migration contract

This document is the implementation contract for replacing Psycles' previous
surface execution plan. The sole oracle is Blender/Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. Unless the user explicitly changes
the requirement, an implementation is acceptable only when it is an
isomorphic Luisa DSL realization of that SVM. Similar output is not sufficient.

## Authoritative source map

| Concern | Cycles 5.2.1 source |
|---|---|
| Interpreter and dispatch order | `intern/cycles/kernel/svm/svm.h` |
| Node opcode order | `intern/cycles/kernel/svm/node_types_template.h` |
| Typed bytecode payloads | `intern/cycles/kernel/svm/node_types.h` |
| Stack/input encodings and semantic enums | `intern/cycles/kernel/svm/types.h` |
| Stack loads, stores, and typed fetch | `intern/cycles/kernel/svm/util.h` |
| Graph scheduling, stack allocation, closure traversal | `intern/cycles/scene/svm.cpp` |
| Compiler state and typed emission API | `intern/cycles/scene/svm.h` |
| Per-node compiler emission | `intern/cycles/scene/shader_nodes.cpp` |
| Closure execution | `intern/cycles/kernel/svm/closure.h` |

The checked-in ABI projection is
`include/psycles/compiler/cycles_svm_types.h`,
`cycles_svm_node_types_template.h`, and `cycles_svm_node_types.h`. The original
Apache-2.0 notices are retained. Platform-only Cycles utility spellings
`packed_float3`, `packed_float4`, and `PackedTransform` are represented as
standard-layout C++ structs with the same 12, 16, and 48-byte ABI. No SVM field,
enum value, or payload layout is changed.

## Formal machine state

For one shader invocation define the state

```
Q = (W, p, S, cw, sd, C, R, v, f, tau)
```

where:

- `W : uint32[N]` is the one global Cycles SVM word stream;
- `p` is the word program counter (`offset` in Cycles);
- `S : float[255]` is the Cycles SVM stack;
- `cw` is the current closure weight spectrum;
- `sd` is the mutable Cycles `ShaderData` projection;
- `C` is the closure allocation/population state owned by `sd`;
- `R` is the render/pass buffer state used by AOV nodes;
- `v` is `PathRayVisibility`;
- `f` is the path flag word;
- `tau` is exactly one of surface, volume, displacement, or bump.

`node_feature_mask` is a JIT specialization parameter, not mutable machine
state. It may erase the same cases as Cycles' `IF_KERNEL_NODES_FEATURE` macros;
it may not select a different runtime representation.

One interpreter step is:

1. Read `op = W[p]`, then set `p = p + 1`.
2. Enter one switch whose selector is exactly `op` in Cycles opcode order.
3. For a fixed payload type `T`, `svm_node_get<T>` denotes the bit-preserving
   view of `W[p .. p + sizeof(T)/4)`, followed by
   `p = p + sizeof(T)/4`.
4. Execute the corresponding Cycles node transition on `(S, cw, sd, C, R)`.
5. Repeat until `NODE_END`, an invalid node, or an AOV early return.

For every payload type, the preconditions are

```
alignof(T) <= alignof(uint32)
sizeof(T) mod sizeof(uint32) = 0
p + sizeof(T)/4 <= N
```

These are checked on the host image and represented directly in the Luisa AST.
There is no execution-family opcode, semantic-subtype opcode, value-program
side stream, closure-leaf side stream, or second closure decoder in this model.

## Exact control-flow transitions

- `NODE_END`: terminate the invocation.
- `NODE_SHADER_JUMP`: fetch `SVMNodeShaderJump`; replace `p` by the field
  selected solely by `tau` (`offset_surface`, `offset_volume`, or
  `offset_displacement`). An unsupported `tau` terminates.
- `NODE_JUMP_IF_ZERO`: after fetching the payload, add `jump_offset` to `p` iff
  `S[stack_offset] <= 0`.
- `NODE_JUMP_IF_ONE`: after fetching the payload, add `jump_offset` to `p` iff
  `S[stack_offset] >= 1`.
- Variable-length nodes (`RGB_RAMP`, curves, sky data, raycast attributes, and
  BSDF-specific records) advance `p` exactly where the corresponding Cycles
  handler advances its `offset`; they are not represented by side streams.
- A bump routine falls through into its surface routine exactly as Cycles does.
  Surface, volume, and displacement routines end with `NODE_END`.

The global stream starts with one four-word jump record per shader:

```
[NODE_SHADER_JUMP, surface_offset, volume_offset, displacement_offset]
```

The three offsets are absolute word indices in the aggregated scene stream.
The initial interpreter offset is `(sd.shader & SHADER_MASK) * 4`.

## Exact stack and input model

The stack domain is 255 32-bit float lanes. Offset 255 is invalid and is never
a legal lane.

- Scalar, integer, vector, color, normal, and derivative values use the same
  lane widths as `SVMCompiler::stack_size`.
- `SVMInputFloat` embeds an immediate float bit pattern, or embeds a stack
  offset as `0x7fc00000 | offset`. The runtime stack test compares the upper
  24 bits exactly as Cycles does.
- `SVMInputFloat3` embeds three immediate float bit patterns, or places the
  encoded offset in `x.bits` and zeroes `y.bits` and `z.bits`.
- `SVMInputInt` uses its immediate `value` when `offset == 255`; otherwise the
  integer is recovered by bit-casting the addressed float lane.
- Non-finite unlinked float defaults are canonicalized to zero at compile time,
  matching `SVMCompiler::input_float` and `input_float3`.

Stack allocation and scheduling are also part of the required implementation,
not an optimization freedom:

1. allocate the first contiguous free lane range;
2. track Cycles-style active users for each lane;
3. release producer outputs after their last unscheduled consumer;
4. release temporary unlinked-input loads after the consuming node;
5. schedule value DAGs with Cycles' Sethi-Ullman heuristic and node-id tie
   breaker;
6. fail the shader when no legal range exists instead of changing the machine.

## Exact closure flow

Closures are nodes in the same word stream and mutate the same `cw` and `sd`
state as Cycles.

- `NODE_CLOSURE_SET_WEIGHT`, `NODE_CLOSURE_WEIGHT`, and
  `NODE_EMISSION_WEIGHT` update `cw` directly.
- `NODE_MIX_CLOSURE` writes its two child weight stack values.
- The closure tree is emitted with Cycles' `generate_multi_closure` traversal,
  shared-dependency treatment, and zero/one jump placement.
- `NODE_CLOSURE_BSDF` is followed immediately by the exact closure-specific
  typed payload selected by `SVMNodeClosureBsdf::closure_type`.
- Emission, background, holdout, volume, volume-coefficient, and Principled
  volume nodes remain their native Cycles node types.
- Closure allocation, cutoff tests, labels, flags, normal handling, and
  Principled component construction follow `kernel/svm/closure.h` field for
  field. Psycles may project the resulting Cycles closure state into Luisa
  classes only at host/JIT construction time; the device transition cannot be
  replaced by a custom leaf format.

## Compilation equivalence obligation

Let `G` be a validated raw Blender shader graph and let `C52(G, tau)` be the
Cycles 5.2.1 compiler output for shader type `tau`. Let `P(G, tau)` be Psycles'
new compiler output after remapping only external resource identifiers such as
image and attribute table indices. The required relation is:

```
normalize_resources(P(G, tau)) = normalize_resources(C52(G, tau))
```

word for word, including node order, typed payload boundaries, jumps, embedded
immediates, and stack offsets. Runtime equivalence is then proved by induction
over interpreter steps:

- base case: both machines start at the corresponding four-word shader jump,
  with equal stack domain, zero closure weight, and equal projected ShaderData;
- induction step: equal `p` selects the same opcode and typed payload; the Luisa
  handler implements the same transition, therefore the next observable state
  remains equal;
- termination: both reach the same `NODE_END` or Cycles-defined early exit with
  equal closure/ShaderData/pass outputs.

Any node whose compiler emission or runtime transition has not been checked
against the cited Cycles function remains unsupported. It must fail validation;
it must not silently fall back to the previous Psycles execution plan.

## Regression gates

`tests/test_cycles_svm_abi.cpp` is frozen from an executable compiled directly
against the cited Cycles headers. It currently checks every SVM payload:

- all 104 `SVMNode*` types plus the six common packed/input types;
- every `sizeof`, `alignof`, standard-layout and trivially-copyable property;
- all 663 field offsets;
- all 422 SVM semantic and opcode enum constants;
- the 255-lane stack, invalid offset, and NaN input tag.

Later migration commits must add, in this order:

1. Cycles-generated golden word streams for small raw Blender graphs;
2. exact host compiler equality for each migrated node family;
3. Luisa XIR structural checks for one PC loop and one primary opcode switch;
4. HIP value/closure state comparisons against Cycles probes;
5. whole-scene visual and numerical parity before performance claims.

No old-path fallback is allowed to make these tests pass.

## First word-stream oracle checkpoint

The local Cycles 5.2.1 diagnostic build was given an environment-gated dump at
`SVMShaderManager::device_update_specific`, after per-shader streams had been
aggregated and before device upload. The dump records the final global words
and the exact `(shader id, shader name)` table; it does not alter compilation.

The canonical `diffuse_surface` Blender probe produced shader id 5,
`Diffuse Probe`, with global jump `(95, 111, 112)`. Removing only the global
jump-table relocation gives this exact local stream:

```
00000001 00000004 00000014 00000015
0000000b 00000001 00000000
00000005 3f2e147b 3e75c28f 3db851ec
00000002 00000002 000000ff
3f2e147b 3e75c28f 3db851ec 3edc28f6
00000000 00000000
00000000
00000000
```

This is now a permanent word-for-word test in
`tests/test_cycles_svm_compiler.cpp`. It proves the first complete compiler
path: local shader jump, Cycles default `Geometry::Normal` insertion, first-fit
three-lane stack assignment, closure set-weight, Diffuse BSDF header and typed
data, surface end, empty volume end, and empty displacement end. Unsupported
families reject the new image explicitly; the test also proves that they do not
select the old Psycles bytecode.
