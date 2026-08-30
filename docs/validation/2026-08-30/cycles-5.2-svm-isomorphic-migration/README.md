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

## Combine/Separate Color opcode checkpoint

`SeparateColorNode` and `CombineColorNode` are copied from Cycles 5.2.1
`scene/shader_nodes.cpp`. Their SVM payload layout and runtime transitions map
directly to `kernel/svm/sepcomb_color.h`; RGB/HSV/HSL conversion semantics come
from `kernel/svm/color_util.h` and `util/color.h`. The host compiler and Luisa
interpreter cover every Cycles `NodeCombSepColorType` direction: RGB to
components, HSV to components, HSL to components, components to RGB,
components to HSV, and components to HSL. Socket names are projected to
Cycles' Red/Green/Blue inputs and outputs before compilation; mode values,
invalid-stack-lane guards, payload words, and switch defaults remain those of
Cycles.

The dynamic `svm_combsep_color_pipeline` probe deliberately keeps all six
paths live by connecting Geometry Backfacing through an HSL/HSV/RGB chain.
Cycles shader 5 had global jump `(89,142,143)`. Removing the two global padding
words gives an exact 59-word local stream with peak stack usage of six lanes;
the permanent compiler test freezes every word and opcode position. Cycles CPU
rendered the linear Emission pass as:

```text
front: (0.000000000, 0.180000007, 0.000000000)
back:  (0.564999998, 0.383161038, 0.000000000)
```

The constant `svm_combsep_color_constant_fold` probe traverses the same mode
sequence without a dynamic input. Cycles produced global jump `(89,96,97)` and
a 13-word local stream containing no Separate/Combine opcode. Its final
closure weight is frozen bit-for-bit as
`(0x3ed6f51a,0x3eb020c6,0x3e051eb7)`, corresponding to
`(0.419838727,0.344000041,0.129999980)`.

The HIP interpreter consumes the dynamic 59-word stream directly. It matches
both Cycles CPU triples within `2e-6`, plus closure weight, ShaderData flags,
ended status, and final PC 57. Oracle generation and focused validation were:

```text
blender --background --factory-startup --python \
  tools/create_cycles_shader_probe.py -- \
  /tmp/svm_combsep_color_pipeline.blend \
  svm_combsep_color_pipeline                                  PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_combsep_color_pipeline.svm52 \
  blender /tmp/svm_combsep_color_pipeline.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_combsep_color_pipeline.exr 4 4 1 0 \
  --cycles-device CPU                                         PASS
cmake --build build --parallel 32                             PASS
ctest --test-dir build --output-on-failure -R \
  '^psycles\.(cycles_svm_(abi|bytecode|compiler)|luisa_cycles_svm_hip)$' \
                                                               4/4 PASS
```

## Legacy MixRGB / `NODE_MIX` checkpoint

Blender's legacy `ShaderNodeMixRGB` and modern `ShaderNodeMix` are no longer
conflated. The former now projects to a dedicated host graph node that copies
Cycles 5.2.1 `MixNode`; the latter remains reserved for the separate
`MixColorNode`, `MixFloatNode`, and vector families. This distinction is
observable in both compiler state and bytecode:

- legacy Mix has Cycles inputs `Fac`, `Color1`, and `Color2` and emits
  `NODE_MIX`;
- its factor is always saturated by `svm_mix_clamped_factor` and therefore it
  has no `ClampFactor` property;
- `use_clamp` emits a second `NODE_MIX` whose type is `NODE_MIX_CLAMP`;
- modern Mix uses its own typed node opcodes and separately stored factor and
  result clamp fields.

The host implementation is a direct projection of
`scene/shader_nodes.cpp::MixNode`,
`scene/constant_fold.cpp::ConstantFolder::fold_mix`, and
`kernel/svm/color_util.h`. The Luisa transition is copied from
`kernel/svm/mix.h::svm_node_mix`: it fetches one `SVMNodeMix`, loads the exact
`SVMInputFloat`/`SVMInputFloat3` payloads, clamps the factor, dispatches the
same 19-value `NodeMix` switch, and stores three consecutive stack lanes. The
blend, add, multiply, screen, overlay, subtract, divide, difference,
exclusion, darken, lighten, dodge, burn, hue, saturation, value, color,
soft-light, linear-light, and clamp branches retain Cycles' operation and
condition order.

The dynamic `svm_legacy_mix_matrix` oracle contains one material for every
blend type. Geometry Backfacing produces factors 0.23 and 0.64 in two rows.
The ordinary modes normalize to exact 33-word local streams; clamped ADD has
the second clamp node and normalizes to 43 words. Both use five stack lanes.
The HIP interpreter consumes each frozen stream with one shared compiled
kernel and matches all 38 Cycles CPU Emission triples within `2e-6`, including
negative and greater-than-one unclamped components. It also matches closure
weight, front/back-facing ShaderData flags, termination, and final PCs 31 or
41.

The constant oracle deliberately links Blender Value and RGB nodes into every
MixRGB input. GDB confirmed that this reaches Cycles'
`MixNode::constant_fold`; leaving all three values as unconnected
Blender socket defaults instead bypasses that function and is folded by
Blender before the Cycles graph exists. The two evaluators differ by several
ULPs on some modes, so the latter is not a valid SVM oracle. With the linked
inputs, all 19 independently folded 13-word streams and the aggregate
`mix_rgb_legacy_modes` stream match Psycles word for word. The aggregate
closure bits are
`(0x3e8cedbc,0x3f1a6314,0x3f5b393e)`.

The import regression in `tests/test_blender_legacy_mix_import.cpp` locks the
raw `MIX_RGB` mapping, absence of the nonexistent `ClampFactor` property,
preservation of blend/result-clamp settings, and the final Cycles SVM fold.
Oracle and focused validation commands are:

```text
blender --background --factory-startup --python \
  tools/create_cycles_shader_probe.py -- \
  /tmp/svm_legacy_mix_matrix.blend svm_legacy_mix_matrix       PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_legacy_mix_matrix.svm52 \
  blender /tmp/svm_legacy_mix_matrix.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_legacy_mix_matrix.exr 152 16 1 0 \
  --cycles-device CPU                                          PASS
blender --background --factory-startup --python \
  tools/create_cycles_shader_probe.py -- \
  /tmp/svm_legacy_mix_constant_matrix.blend \
  svm_legacy_mix_constant_matrix                               PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_legacy_mix_constant_matrix.svm52 \
  blender /tmp/svm_legacy_mix_constant_matrix.blend \
  --background --python tools/render_cycles_golden.py -- \
  /tmp/svm_legacy_mix_constant_matrix.exr 76 4 1 0 \
  --cycles-device CPU                                          PASS
cmake --build build --target \
  psycles_cycles_svm_compiler_tests \
  psycles_luisa_cycles_svm_tests \
  psycles_blender_import_tests --parallel 32                   PASS
build/psycles_cycles_svm_compiler_tests                        PASS
build/psycles_blender_import_tests                             PASS
build/bin/psycles_luisa_cycles_svm_tests hip                   PASS
cmake --build build --parallel 32                              PASS
ctest --test-dir build --output-on-failure -R \
  '^psycles\.(cycles_svm_(abi|bytecode|compiler)|blender_import|luisa_cycles_svm_hip)$' \
                                                               5/5 PASS
```

## Modern typed Mix checkpoint

Modern Blender `ShaderNodeMix` now follows the four Cycles 5.2.1 node types
without sharing the legacy `NODE_MIX` path:

- RGBA maps to `MixColorNode` and `NODE_MIX_COLOR`;
- FLOAT maps to `MixFloatNode` and `NODE_MIX_FLOAT`;
- VECTOR/UNIFORM maps to `MixVectorNode` and `NODE_MIX_VECTOR`;
- VECTOR/NON_UNIFORM maps to `MixVectorNonUniformNode` and
  `NODE_MIX_VECTOR_NON_UNIFORM`.

The host code is an isomorphic Luisa-side projection of
`intern/cycles/blender/shader.cpp`'s `ShaderNodeMix` selection and socket-name
mapping, the four compile/constant-fold/linear-operation implementations in
`intern/cycles/scene/shader_nodes.cpp`, and
`ConstantFolder::fold_mix_color`/`fold_mix_float` in
`intern/cycles/scene/constant_fold.cpp`. The device transitions preserve the
load, factor saturation, interpolation or `svm_mix`, result saturation, and
store order from `intern/cycles/kernel/svm/mix.h`.

### Dynamic, constant, and typed oracles

Three Blender probes force Cycles itself to retain or fold every modern form:

- `svm_modern_mix_color_matrix` contains all 19 `NodeMix` modes. Every
  material has an exact 33-word local image, five stack lanes, and a
  `NODE_MIX_COLOR` payload whose blend and two clamp fields vary independently.
- `svm_modern_mix_data_matrix` contains both clamp states of MixFloat,
  uniform MixVector, and non-uniform MixVector. Their local image sizes are
  respectively 28, 32, and 28 words, with peak stack use 2, 5, and 6.
- `svm_modern_mix_constant_matrix` feeds the nodes through linked Blender
  Value/RGB/CombineXYZ nodes, so the values reach Cycles' own constant folders
  rather than being pre-folded by Blender. All 25 shaders become exact
  13-word images containing no modern Mix opcode; all 75 closure-weight words
  are frozen bit for bit in `test_cycles_svm_modern_mix.cpp`.

The HIP regression evaluates all 19 color modes and all six typed data cases
with one node-mask-specialized interpreter. It compares the Cycles CPU
Emission pass within `2e-6`, as well as closure weight, front/back-facing
flags, termination, and final PCs.

### Import, conversion alias, and stack lifetime oracle

`svm_modern_mix_import_chain` connects FLOAT to uniform VECTOR, evaluates a
non-uniform VECTOR from Geometry Normal, converts both vector results to
color, and feeds them into an OVERLAY RGBA Mix. This is the same graph used by
the permanent Blender-import regression. Cycles shader 5 had global jump
`(89,138,139)` and normalizes to this 55-word shape:

```text
LIGHT_PATH -> GEOMETRY Normal -> MIX_VECTOR_NON_UNIFORM -> MIX_FLOAT
           -> MIX_VECTOR -> MIX_COLOR -> EMISSION_WEIGHT
           -> CLOSURE_EMISSION -> END
```

The image uses ten stack lanes. There is no conversion opcode between Cycles
float3 socket kinds: `ConvertNode::compile` aliases the producer stack offset
with `SVMCompiler::stack_link`. Psycles now mirrors that reference-counted
alias lifetime, rather than emitting a copy or freeing the producer lane at
the conversion node. The same oracle also locks Cycles' Geometry rule:
ordinary Normal uses `NODE_GEOMETRY`, while derivative-capable Position,
Tangent, True Normal, Incoming, and Parametric select the derivative opcode
when the graph requires derivatives.

Cycles CPU's linear Emission pass for the two front/back-facing cells was:

```text
front: (0.259999990, 0.540000021, 0.000000000)
back:  (0.635999918, 0.000000000, 0.088000007)
```

The HIP interpreter matches both values and final PC 53.

### Linked socket values and partial-fold edges

The partial-fold oracle exposed a graph-model mismatch rather than a Mix-only
bug. A Cycles `ShaderInput` retains its socket value while linked. That value
becomes observable if constant folding disconnects the link. Psycles formerly
represented a link and its fallback value as mutually exclusive, so a valid
Cycles rewrite could leave an input with no value.

The contract now preserves an authored value through `ShaderGraph::connect`,
normalization fills schema defaults even on linked inputs, and Blender import
copies each raw linked socket default before installing its link. Runtime
lowering still reads the source while it is linked. This models the two
independent Cycles fields directly and fixes the entire rewrite class rather
than special-casing Mix.

`svm_modern_mix_fold_edges` freezes three Cycles-produced cases:

- same linked dynamic color on A and B with result clamp: Cycles cannot bypass
  the clamp, disconnects the other input, sets Factor to zero, and retains a
  33-word `NODE_MIX_COLOR` image with seven stack lanes;
- zero-factor unclamped MixColor: Cycles bypasses A and emits a 23-word image
  with no MixColor opcode and four stack lanes;
- one-factor MixFloat: Cycles bypasses B and emits a 17-word image with no
  MixFloat opcode and one stack lane.

The first, second, and third Cycles CPU Emission values were respectively
`(0,0,1)`, `(0,-0.4,1.2)`, and `(0,0,0)` for the front-facing probe cells.
Permanent regressions lock the exact words, opcode absence or presence, stack
peaks, linked-default contract, and raw Blender socket defaults.

Oracle and focused validation commands:

```text
/home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/svm_modern_mix_import_chain.blend \
  svm_modern_mix_import_chain                                  PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_modern_mix_import_chain.svm52 \
  /home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  /tmp/svm_modern_mix_import_chain.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_modern_mix_import_chain.exr 4 2 1 0 \
  --cycles-device CPU                                          PASS
/home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/svm_modern_mix_fold_edges.blend \
  svm_modern_mix_fold_edges                                    PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_modern_mix_fold_edges.svm52 \
  /home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  /tmp/svm_modern_mix_fold_edges.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_modern_mix_fold_edges.exr 6 2 1 0 \
  --cycles-device CPU                                          PASS
cmake --build build --target \
  psycles_cycles_svm_modern_mix_tests \
  psycles_graph_material_scene_tests \
  psycles_blender_import_tests \
  psycles_luisa_cycles_svm_tests --parallel 32                 PASS
build/psycles_cycles_svm_modern_mix_tests                      PASS
build/psycles_graph_material_scene_tests                       PASS
build/psycles_blender_import_tests                             PASS
build/bin/psycles_luisa_cycles_svm_tests hip                   PASS
cmake --build build --parallel 32                              PASS
ctest --test-dir build --output-on-failure -R \
  '^psycles\.(cycles_svm_(abi|bytecode|compiler|modern_mix)|graph_material_scene|blender_import|luisa_cycles_svm_hip)$' \
                                                               7/7 PASS
ctest --test-dir build --output-on-failure -Q -E \
  '(_fallback|_vk)$'                                          160/160 PASS
```

The 160-test gate is the complete configured host/HIP suite with only the
explicit fallback and Vulkan test names excluded. Its test inventory was
checked with the identical exclusion expression before execution. The
canonical probe-list regression also covers all five modern Mix probes, and
the source-size gate covers the probe split as ordinary project code.

## Separate/Combine XYZ checkpoint

Blender `SEPXYZ` and `COMBXYZ` are not aliases of Separate/Combine Color in
Cycles 5.2.1. The previous Psycles importer conflated both families, which was
a structural graph and bytecode error even when RGB component arithmetic
happened to give the same result. This checkpoint follows these exact sources:

| Stage | Cycles 5.2.1 source |
|---|---|
| Blender node mapping | `intern/cycles/blender/shader.cpp` |
| Node schemas, folding, and emission | `intern/cycles/scene/shader_nodes.cpp::{SeparateXYZNode,CombineXYZNode}` |
| Typed payloads | `intern/cycles/kernel/svm/node_types.h::{SVMNodeSeparateVector,SVMNodeCombineVector}` |
| Plain and dual transitions | `intern/cycles/kernel/svm/sepcomb_vector.h` |
| Interpreter case order | `intern/cycles/kernel/svm/svm.h` |

The graph projection therefore keeps dedicated `separate_xyz` and
`combine_xyz` nodes. It also preserves the non-obvious Cycles schema detail
that `SeparateXYZNode::Vector` is declared with `SOCKET_IN_COLOR`, while
`CombineXYZNode::Vector` is a vector output. The necessary socket conversions
remain ordinary Cycles conversion aliases; they are not folded into the XYZ
nodes.

`CombineXYZNode::compile` emits three `NODE_COMBINE_VECTOR` records with one
shared output base and vector indices 0, 1, and 2. `SeparateXYZNode::compile`
emits three `NODE_SEPARATE_VECTOR` records with one shared input and the three
corresponding outputs. Their constant folders run only when all inputs are
constant and produce the same whole-vector or selected-component constant as
Cycles.

### Exact word-stream oracles

The dynamic probe rotates a two-cell surface, feeds Geometry Normal through
Separate XYZ, permutes `(Z, X, Y)` with Combine XYZ, and sends the result to
Emission. Cycles shader 5 had global jump `(89,124,125)`. After only global
jump relocation, its 41-word local stream is:

```text
00000001 00000004 00000027 00000028
0000000b 00000001 00000000
00000054 7fc00000 00000000 00000000 00000300
00000054 7fc00000 00000000 00000000 00000401
00000054 7fc00000 00000000 00000000 00000502
00000056 7fc00005 00000000
00000056 7fc00003 00000001
00000056 7fc00004 00000002
00000007 7fc00000 00000000 00000000 3f800000
00000003 000000ff 00000000
00000000 00000000 00000000
```

It uses six stack lanes. In particular, the stream contains opcodes `0x54`
and `0x56`, not the Separate/Combine Color opcodes.

The constant probe links Value nodes into Combine XYZ, passes the vector
through Separate XYZ, permutes the components, and emits the result. Cycles
shader 5 had global jump `(89,96,97)` and folded to this exact 13-word image:

```text
00000001 00000004 0000000b 0000000c
00000005 3fa66666 bf333333 3e800000
00000003 000000ff 00000000 00000000 00000000
```

No vector split/pack opcode survives the fold. The raw Blender import
regression independently freezes the dynamic 41-word image and verifies that
`SEPXYZ`/`COMBXYZ` remain distinct graph-node types with all links intact.

### Plain and derivative transition relation

For a plain vector at base `b`, Cycles stores its components in lanes
`S[b+0..b+2]`. Separate writes component `i` to its scalar output, and Combine
writes its scalar input to `S[out+i]`.

For a dual vector, the exact lane relation is:

```text
value = S[b+0 .. b+2]
dx    = S[b+3 .. b+5]
dy    = S[b+6 .. b+8]
```

Separate maps component `i` to the adjacent dual-scalar lanes
`(out, out+1, out+2)`. Combine is the inverse mapping and writes value, `dx`,
and `dy` at offsets `i`, `i+3`, and `i+6` from the output base. Immediate
inputs have zero derivatives. The Luisa handlers preserve this mapping and
the Cycles validity test on the output offset directly.

The ordinary transition is covered end to end by the external word stream and
HIP oracle. The derivative opcodes are implemented from the same templated
Cycles handler, but no end-to-end derivative graph is claimed yet: Cycles'
bump graph derivative propagation has not yet been migrated. That later
checkpoint must produce the derivative bytecode through the copied Cycles
graph transform; this checkpoint does not synthesize a private test-only graph
mode.

### CPU and HIP results

The authoritative clean render binary is Blender 5.2.1 hash
`9e2066aef7ef`. The dump build's custom commit has that exact commit as its
parent and does not change the Blender mapping, shader-node compiler, SVM
payloads, or SVM interpreter; its only uncommitted SVM change copies the final
global word stream to an environment-selected file before upload. A clean
render and the dump build compared exactly across every EXR subimage with
OpenImageIO `--diff` for both probes.

Cycles CPU produced these linear emission values:

```text
dynamic:  ( 0.917831361, -0.191833034, -0.347542346)
constant: ( 1.299999952, -0.699999988,  0.250000000)
```

The HIP interpreter receives the same rotated normal
`(-0.191833034,-0.347542346,0.917831361)` and matches the dynamic value within
`2e-6`, along with closure weight, front/back-facing flags, termination, and
final PC 39.

Oracle and validation commands:

```text
/home/mike/Projects/blender-install-5.2-hiprt/blender \
  --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/svm_sepcomb_vector_pipeline.blend \
  svm_sepcomb_vector_pipeline                                  PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_sepcomb_vector_pipeline.svm52 \
  /home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  /tmp/svm_sepcomb_vector_pipeline.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_sepcomb_vector_pipeline-trace.exr 16 8 1 0 \
  --cycles-device CPU                                          PASS
/home/mike/Projects/blender-install-5.2-hiprt/blender \
  /tmp/svm_sepcomb_vector_pipeline.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_sepcomb_vector_pipeline.exr 16 8 1 0 \
  --cycles-device CPU                                          PASS
oiiotool -a /tmp/svm_sepcomb_vector_pipeline.exr \
  /tmp/svm_sepcomb_vector_pipeline-trace.exr --diff            PASS

/home/mike/Projects/blender-install-5.2-hiprt/blender \
  --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/svm_sepcomb_vector_constant_fold.blend \
  svm_sepcomb_vector_constant_fold                             PASS
PSYCLES_CYCLES_SVM_DUMP=/tmp/svm_sepcomb_vector_constant_fold.svm52 \
  /home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  /tmp/svm_sepcomb_vector_constant_fold.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_sepcomb_vector_constant_fold-trace.exr 4 4 1 0 \
  --cycles-device CPU                                          PASS
/home/mike/Projects/blender-install-5.2-hiprt/blender \
  /tmp/svm_sepcomb_vector_constant_fold.blend --background \
  --python tools/render_cycles_golden.py -- \
  /tmp/svm_sepcomb_vector_constant_fold.exr 4 4 1 0 \
  --cycles-device CPU                                          PASS
oiiotool -a /tmp/svm_sepcomb_vector_constant_fold.exr \
  /tmp/svm_sepcomb_vector_constant_fold-trace.exr --diff       PASS

cmake --build build --parallel 32                              PASS
ctest --test-dir build --output-on-failure -R \
  '^psycles\.(cycles_svm_(abi|bytecode|compiler|modern_mix|vector)|graph_material_scene|blender_import|luisa_cycles_svm_hip|source_size|shader_probe_runner_contract|blender_export_render_settings)$' \
                                                               11/11 PASS
ctest --test-dir build --output-on-failure -Q -E \
  '(_fallback|_vk)$'                                          161/161 PASS
```
