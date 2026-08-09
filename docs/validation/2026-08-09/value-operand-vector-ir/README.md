# Semantic value-operand vector IR validation

## Scope

`ValueInstruction` previously reserved fourteen lettered operand fields for
every value node, even though most nodes have zero to three inputs. The fixed
slots also encouraged consumers to enumerate an `a..j` prefix; that had
already caused the final Brick operands to be omitted from reachability and
volume-homogeneity analysis.

The instruction now stores its true arity in
`std::vector<ValueExpressionId> operands`. This is host-side shader graph IR;
it does not change device parameter layout or introduce a dynamically typed
shader value. Backends still record the graph once in topological order and
each node reads already-recorded typed Luisa expressions.

If profiling later shows that one allocation per non-leaf value instruction
is material, the storage can move to a small-vector or shared operand arena.
That decision is intentionally deferred until scene-level host/JIT profiles
show a bottleneck; the iterable, exact-arity IR contract will remain the same.

## Semantic layout contract

Every operation family has a shared `value_operand::<layout>` containing
named `static constexpr` indices and a `count`. Lowering uses
`make_value_operands<layout>` with explicit `{semantic_index, expression}`
assignments. The helper rejects missing, duplicate, and out-of-range
assignments. Luisa recorders use the same names through
`instruction.operand(layout::socket)`; no numeric operand indexing or
lettered slots remain.

`value_operation_operand_count()` is the exhaustive operation-to-arity
contract. Both `SurfaceProgramBuilder::append()` and the public
`SurfaceProgram` constructor enforce it, so tooling that bypasses normal graph
lowering cannot create a structurally malformed value instruction.

Generic dependency consumers now iterate the vector directly:

- value reachability and compaction;
- value-ID remapping;
- Luisa active-stage dependency masks;
- volume spatial-dependency/homogeneity analysis.

## Regression coverage

The contract test deliberately supplies RGB Curve operands out of storage
order and verifies that the named indices place every expression at its
semantic endpoint. Wave and Voronoi lowering check exact arity and validity of
the complete vector. The volume fixture places its only spatial dependency in
`value_operand::brick::squash_frequency`, the final Brick endpoint; an
implementation that scans only a prefix misclassifies the volume as
homogeneous and fails.

The existing runtime-table XIR regression also passed unchanged. Two and 128
Color Ramp control points still produce identical XIR shape (913 instructions
and one loop), while the sampled representation produces 629 instructions and
no loop. Thus the host IR storage change did not expand the recorded device
program.

## Commands and results

Run from the repository root with all build threads:

```text
cmake --build build --target psycles_tests psycles_luisa_compile_tests psycles_luisa_attribute_tests psycles_luisa_volume_majorant_scene_tests --parallel 32
./build/psycles_tests
./build/psycles_luisa_compile_tests
./build/bin/psycles_luisa_attribute_tests fallback
./build/bin/psycles_luisa_attribute_tests hip
env LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 ./build/bin/psycles_luisa_attribute_tests vk
./build/bin/psycles_luisa_volume_majorant_scene_tests fallback
./build/bin/psycles_luisa_volume_majorant_scene_tests hip
env LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 LUISA_LOG_LEVEL=verbose ./build/bin/psycles_luisa_volume_majorant_scene_tests vk
```

All commands passed. HIP selected the AMD Radeon RX 9070 XT. The Vulkan cold
run used native XIR-to-SPIR-V translation and reported XIR translation,
structured optimization, XIR legalization, and successful SPIR-V compilation;
no DXC stage was loaded.
