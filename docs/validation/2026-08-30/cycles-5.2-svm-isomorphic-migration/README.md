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

## General graph/compiler checkpoint

The one-off Diffuse lowering has now been removed. The replacement host path is
an adaptation of the following Cycles 5.2.1 implementation, preserving the
same mutable graph/compiler state and the same algorithmic order:

| Psycles projection | Cycles 5.2.1 source operation |
|---|---|
| synthetic graph output at node id 0 | `ShaderGraph::ShaderGraph()` creates `OutputNode` first |
| default Geometry/Texture Coordinate producers | `ShaderGraph::default_inputs()` |
| closure-weight graph rewrite | `ShaderGraph::transform_multi_closure()` |
| scalar/vector/closure stack widths | `SVMCompiler::stack_size()` |
| first contiguous free range | `SVMCompiler::stack_find_offset()` |
| producer last-user release | `stack_clear_users()` / `is_sole_user()` |
| temporary literal release | `stack_clear_temporary()` |
| dependency collection | `SVMCompiler::find_dependencies()` |
| DAG order and node-id tie break | `SVMCompiler::generate_svm_nodes()` |
| closure feature filtering and weight offset | `generate_closure_node()` |
| shared dependency intersection and jumps | `generate_multi_closure()` |
| surface/volume/displacement routines | `compile_type()` and `compile()` |

The graph has one stack offset on every input/output, Cycles-style output user
lists, `nodes_done`, `closure_done`, and the exact 255-lane active-user array.
The scheduler is the Cycles Sethi-Ullman implementation: producer order uses
`SU(node) - output_size(node)`, multi-consumer producers contribute only their
output size, and node id is the sole tie breaker. Unsupported node emitters
still fail the new compiler; no legacy program is consulted.

The audit also found that Psycles' generic Math schema initialized its third
input to `0.5`, while Cycles `MathNode` declares `Value3 = 0.0`. The schema now
uses exact positive zero and a bit-level regression locks it. Generic authored
Authored Math is accepted only through the subsequently ported Cycles
graph-cleaning and constant-folding stages described below; it no longer
bypasses those stages.

### Constant closure-mix oracle

The existing `transparent_mix` Blender probe (`Transparent Probe`) produced a
global jump `(95,114,115)`. Normalized to a local stream, Cycles emitted 25
words:

```
00000001 00000004 00000017 00000018
00000008 3f1eb852 000100ff
00000005 3f400000 3f666666 3f19999a
00000002 0000001e 00000000 00000000 00000000
00000005 3f828f5d 3dc49ba6 3d1374bd
00000003 00000001
00000000 00000000 00000000
```

The matching permanent regression proves that the new graph transform emits
one `NODE_MIX_CLOSURE`, assigns its two outputs to lanes 0 and 1, visits the
Transparent and Emission leaves in Cycles order, and passes the two distinct
mix offsets into those closures. Peak stack usage is two lanes.

### Linked closure-mix/jump oracle

`dynamic_mix_shader` is a new canonical Blender probe whose Geometry
`Backfacing` output drives Mix Shader's factor. It prevents constant folding
from hiding the runtime closure-tree control flow. The exact oracle command was:

```bash
blender --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/dynamic_mix.blend dynamic_mix_shader
PSYCLES_CYCLES_SVM_DUMP=/tmp/dynamic_mix.svm52 \
  blender /tmp/dynamic_mix.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/dynamic_mix.exr 1 1 1 0 --cycles-device CPU
```

Cycles produced global jump `(89,117,118)`. The normalized 34-word stream is
frozen in `test_linked_mix_closure_jumps_match_cycles_5_2_1` and contains, in
order:

1. `NODE_LIGHT_PATH(NODE_LP_backfacing, lane 0)`;
2. `NODE_MIX_CLOSURE(fac=lane 0, parent=invalid, children=lanes 1/2)`;
3. `NODE_JUMP_IF_ONE(jump=9, lane 0)` and the Transparent leaf using lane 1;
4. `NODE_JUMP_IF_ZERO(jump=6, lane 0)` and the Emission leaf using lane 2;
5. the three routine terminators.

Peak usage is exactly three lanes. Together with the constant-mix oracle this
checks both branches of `generate_multi_closure`, including forward-word patch
distances rather than merely decoded semantics.

### Validation

```text
cmake --build build --parallel $(nproc)                 PASS
ctest --test-dir build -R \
  '^psycles\.cycles_svm_(abi|bytecode|compiler)$'      3/3 PASS
```

This checkpoint is host compiler equivalence only. It does not claim the
production renderer has switched to the new stream: the remaining required
work is the full family-by-family compiler port and the Luisa device
interpreter with one Cycles PC loop and one opcode dispatch.

## Host node compilation boundary

The temporary centralized node-type chain has been removed. The projected
nodes now expose the same host compilation boundary as Cycles 5.2.1:
`ShaderNode::compile(SVMCompiler&)`. Diffuse, Translucent, Transparent,
Emission, Geometry, Mix Closure Weight, the closure-transform Multiply node,
and Output each own their bytecode emission. Combine-closure nodes remain
empty because `SVMCompiler::generate_multi_closure` handles them, exactly as in
Cycles. Unsupported node subclasses fail compilation and never select a
different bytecode path. The constant and linked closure-mix word oracles above
are unchanged across this refactor.

## Cycles graph finalization and Math checkpoint

The pre-SVM graph now copies this Cycles 5.2.1 sequence from
`scene/shader_graph.cpp`:

```text
ShaderGraph::expand
ShaderGraph::default_inputs
ShaderGraph::clean
  constant_fold
  simplify_settings
  deduplicate_nodes
  optimize_volume_output
  break_cycles and remove unreachable nodes
ShaderGraph::transform_multi_closure
```

The mutable link lists, bottom-up ready queue, node-id ordering, multi-output
fold boundary, dedup equality relation, relink behavior, and volume linearity
walk follow the corresponding Cycles functions. `ConstantFolder` and
`MathNode::{constant_fold,is_linear_operation,compile}` are direct adaptations
of `scene/constant_fold.cpp`, `scene/shader_nodes.cpp`, and
`kernel/svm/math_util.h`. In particular, clamp uses Cycles' ordered
`min(max(x, lo), hi)` semantics; this preserves its NaN and signed-zero behavior
instead of substituting `std::clamp`.

Three new Blender probes freeze final SVM streams from the exact diagnostic
Cycles build:

- `svm_math_dedup`: two independently authored but equal Geometry nodes feed
  one dynamic Add. Cycles emits one `NODE_LIGHT_PATH`, then one `NODE_MATH`
  whose Value1 and Value2 both address lane 0. Its local image is 23 words and
  peak stack use is 2 lanes.
- `svm_math_constant_fold`: `0.12 + 0.23` feeds Diffuse Roughness. Cycles emits
  no `NODE_MATH`; the Diffuse payload contains `0x3eb33333` directly. Its local
  image is 22 words.
- `svm_mix_closure_fold`: a zero-factor Mix selects Transparent and discards
  its Emission branch before closure transformation. The local image is 16
  words and contains neither `NODE_MIX_CLOSURE` nor
  `NODE_CLOSURE_EMISSION`.

The permanent word-for-word regressions are in
`tests/test_cycles_svm_compiler.cpp`. The oracle was generated with:

```bash
blender --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/<probe>.blend <probe>
PSYCLES_CYCLES_SVM_DUMP=/tmp/<probe>.svm52 \
  blender /tmp/<probe>.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/<probe>.exr 1 1 1 0 --cycles-device CPU
```

Cycles restores a folded-away displacement root with a `ColorNode` followed
by its automatic Color-to-Vector `ConvertNode`. Those two exact node compilers
are not yet migrated, so that boundary currently rejects explicitly instead
of inventing a replacement or silently dropping displacement.

## First Luisa device interpreter checkpoint

The device path now contains the first executable projection of Cycles 5.2.1
`svm_eval_nodes`. Its implementation is split by the same source families:

| Psycles implementation | Cycles 5.2.1 source |
|---|---|
| `src/luisa/cycles_svm.cpp` | `kernel/svm/svm.h` main PC loop and opcode dispatch |
| `src/luisa/cycles_svm_stack.cpp` | `kernel/svm/util.h` stack and `SVMInput*` loads |
| `src/luisa/cycles_svm_math.cpp` | `kernel/svm/math.h` and `math_util.h` |
| `src/luisa/cycles_svm_value.cpp` | `kernel/svm/value.h`, `geometry.h`, and `light_path.h` |
| `src/luisa/cycles_svm_closure.cpp` | `kernel/svm/closure.h` and `kernel/closure/emissive.h` |

The Luisa AST has one dynamic word `offset`, one 255-float local stack, one
loop, and one primary switch on the fetched Cycles opcode. Typed payload reads
advance that same offset in source field order. The only representational
projection is spelling a typed payload as sequential typed word loads because
Luisa DSL cannot dynamically reinterpret a `Buffer<uint>` reference as a C++
payload struct; there is no second stream or decoded instruction array.

The copied executable cases in this checkpoint are `NODE_END`,
`NODE_SHADER_JUMP`, `NODE_CLOSURE_BSDF` payload skipping for feature-erased
paths, `NODE_CLOSURE_EMISSION`, `NODE_CLOSURE_SET_WEIGHT`,
`NODE_CLOSURE_WEIGHT`, `NODE_EMISSION_WEIGHT`, `NODE_MIX_CLOSURE`, both closure
jumps, `NODE_GEOMETRY`, `NODE_VALUE_F`, `NODE_VALUE_V`, `NODE_MATH`, and
`NODE_LIGHT_PATH`. A BSDF that would execute rather than take Cycles' exact
feature-erased skip transition reports unsupported; it is not evaluated by the
old surface path. Geometry tangent and volume-density services likewise remain
explicitly unsupported until their corresponding Cycles families are copied.

`tests/test_luisa_cycles_svm.cpp` executes the two previously frozen Cycles
streams for dynamic Math/dedup and linked Mix Closure directly on HIP. It also
walks the generated Luisa AST and requires exactly one PC loop and one primary
opcode switch. The runtime oracle locks front/back-facing branch selection,
emission accumulation, closure weight, ShaderData flags, termination status,
and the exact final word offset (21 and 32 respectively).

Validation on AMD Radeon RX 9070 XT (`gfx1201`):

```text
cmake --build build --target psycles_luisa_cycles_svm_tests \
  --parallel 32                                           PASS
build/bin/psycles_luisa_cycles_svm_tests hip              PASS
ctest --test-dir build --output-on-failure -R \
  '^psycles\.luisa_cycles_svm_hip$'                       1/1 PASS
ctest --test-dir build --output-on-failure -R \
  '^psycles\.cycles_svm_(abi|bytecode|compiler)$'         3/3 PASS
cmake --build build --parallel 32                         PASS
```

With cache disabled, the first HIP compilation produced a 14,160-byte AMDGPU
object before linking and an 11,456-byte loaded code object. These sizes are a
checkpoint, not a performance claim. The production renderer is not yet wired
to this partial interpreter; doing so before every reachable Cycles opcode and
closure transition has been copied would violate the no-fallback contract.

## Color transformation opcode checkpoint

The next copied device and host families are exactly the Cycles 5.2.1
`InvertNode`, `GammaNode`, `BrightContrastNode`, `HSVNode`, and `ClampNode`,
mapped to `NODE_INVERT`, `NODE_GAMMA`, `NODE_BRIGHTCONTRAST`, `NODE_HSV`, and
`NODE_CLAMP`. Their compiler emission and constant-fold methods come from
`scene/shader_nodes.cpp`; their device transitions come from `kernel/svm`'s
`invert.h`, `gamma.h`, `brightness.h`, `hsv.h`, `clamp.h`, `color_util.h`, and
`util/color.h`. The component branch order, HSV hue wrapping, oversaturation
clamp, gamma-zero case, RANGE min/max reversal, and output-validity checks are
kept as written in those sources.

Two new Blender probes were executed with the exact diagnostic Cycles build:

- `svm_color_pipeline` uses Geometry Backfacing to keep all five nodes live.
  Shader 5 had global jump `(89,134,135)` and normalized to a 51-word stream.
  The permanent host regression locks every word, lane reuse, opcode order,
  and peak stack usage of seven lanes.
- `svm_color_constant_fold` chains constant Invert, Gamma, Bright/Contrast,
  and RANGE Clamp into Emission. Cycles removed all four opcodes and embedded
  closure weight bits `(3db1030e,3dc5492b,3dddd033)` in a 13-word stream. The
  permanent regression locks that graph-cleaning result.

The dynamic probe renders a two-cell 4x4 matrix whose second cell has reversed
polygon winding. Cycles CPU's linear Emission pass was constant within each
cell:

```text
front: (0.035884645, 0.071987897, 0.159804747)
back:  (0.665142953, 0.000000000, 0.707822502)
```

The HIP interpreter consumes the same frozen 51-word stream and matches both
triples within `2e-6`, along with closure weight, ShaderData flags, ended
status, and final PC 49. The oracle generation and validation commands were:

```text
blender --background --factory-startup --python \
  tools/create_cycles_shader_probe.py -- \
  /tmp/svm_color_pipeline.blend svm_color_pipeline             PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_color_pipeline.svm52 \
  blender /tmp/svm_color_pipeline.blend --background --python \
  tools/render_cycles_golden.py -- \
  /tmp/svm_color_pipeline.exr 4 4 1 0 --cycles-device CPU      PASS
cmake --build build --target psycles_cycles_svm_compiler_tests \
  --parallel 32                                                PASS
build/psycles_cycles_svm_compiler_tests                        PASS
cmake --build build --target psycles_luisa_cycles_svm_tests \
  --parallel 32                                                PASS
build/bin/psycles_luisa_cycles_svm_tests hip                   PASS
```
