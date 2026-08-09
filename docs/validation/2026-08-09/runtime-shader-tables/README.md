# Runtime shader-table IR validation

## Scope

Color Ramp and RGB Curves tables used to be embedded in each
`ValueInstruction`. Luisa therefore recorded table cardinality and contents
into the callable AST: unsampled tables emitted one host-unrolled expression
sequence per control point, while sampled tables emitted one AST constant per
material table. Materials with the same graph topology but different ramp
contents could not share a callable.

The table is now material data. Each fixed parameter slot stores a descriptor
`{absolute scalar offset, element count, element width}` and the payload is
appended to the scalar parameter buffer. Sampled tables use direct dynamic
buffer indexing. The legacy control-point representation uses one device loop
whose trip count comes from the descriptor.

This follows the relevant Cycles execution model: exported sampled ramp/curve
data is indexed as table data rather than expanded into shader control flow.
It does not pre-bake a closure or material response; the original shader node
and its runtime inputs remain in Psycles' graph IR.

## Structural checks

`psycles_luisa_compile_tests` translates otherwise identical Color Ramp
programs to XIR and counts the complete module shape:

| Representation | Authored entries | XIR instructions | XIR loops |
| --- | ---: | ---: | ---: |
| control points | 2 | 913 | 1 |
| control points | 128 | 913 | 1 |
| sampled | 3 | 629 | 0 |

The 2-entry and 128-entry programs are exactly equal in instruction and loop
count. A separate compiler regression verifies that changing table length and
contents preserves the graph structure signature while changing only the
parameter signature.

The shared-subgraph fixture also verifies both stages of graph scheduling:
the source `ShaderNode` occurs once in `ShaderProgram::evaluation_order`, and
the lowered `SurfaceProgram` contains one value instruction whose ID is reused
by both consumers.

## Numeric/backend checks

The production staging/finalization code and device-side lookup were exercised
with sampled Color Ramp and RGB Curves fixtures. Expected results were
`(0.3, 0.4, 0.5, 1)` and `(0.25, 0.6, 0.75, 1)` respectively.

Commands run from the repository root:

```text
cmake --build build --target psycles_tests psycles_luisa_compile_tests psycles_luisa_attribute_tests psycles_luisa_volume_majorant_scene_tests --parallel 32
./build/psycles_tests
./build/psycles_luisa_compile_tests
./build/bin/psycles_luisa_attribute_tests fallback
./build/bin/psycles_luisa_attribute_tests hip
./build/bin/psycles_luisa_attribute_tests vk
./build/bin/psycles_luisa_volume_majorant_scene_tests fallback
```

All commands passed. HIP selected the AMD Radeon RX 9070 XT. Vulkan reported
native SPIR-V optimization and compilation (`3221 -> 2929` words) and did not
load or invoke DXC.

## Dependency-edge regression

Value instructions historically exposed fourteen operand slots, but two
reachability analyses enumerated only the first ten. Dependency enumeration
was centralized and a volume fixture places its only spatial dependency in
the final operand. Misclassifying that program as homogeneous now fails the
test. The follow-up IR cleanup replaces the fixed lettered slots with an
iterable operand container so this class of prefix omission is structurally
impossible.
