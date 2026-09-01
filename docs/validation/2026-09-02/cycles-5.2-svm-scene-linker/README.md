# Cycles 5.2 SVM scene linker validation

## Result

Psycles now has the scene-level SVM linking operation used by Blender Cycles
5.2.1. It consumes the local four-word `ShaderJump` images emitted by the
isomorphic per-material compiler and produces one dense global jump table
followed by all shader tails. No `SurfaceProgram`, graph interning, parameter
baking, or alternate opcode representation participates in this operation.

The implementation is in:

- `include/psycles/compiler/cycles_svm_scene.h`
- `src/compiler/cycles_svm_scene.cpp`
- `tests/test_cycles_svm_scene.cpp`

This is a host-scene construction milestone. The global image is not yet the
production `shade_surface` evaluator; device upload and production wiring are
the next migration boundary.

## Source correspondence

The copied source operation is Blender Cycles 5.2.1 commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`,
`intern/cycles/scene/svm.cpp`,
`SVMShaderManager::device_update_specific`.

For `n` shaders, let `J = 4` be the word size of `NODE_SHADER_JUMP`, let
`T_i` be local shader `i` without its first `J` words, and let

```text
G_i = n J + sum_{k < i} |T_k|
```

be the global base of that shader tail. For every local surface, volume, or
displacement entry `L`, the linker writes

```text
global_entry(i, L) = G_i + (L - J).
```

It copies every word in `T_i` verbatim. Therefore all shader-local forward
branch displacements remain unchanged. This is the same transformation as the
Cycles source, expressed with checked `size_t` arithmetic and an explicit
signed-`int32` validation boundary for the three ABI entries.

The linker rejects a local image if any of the following preconditions is
false:

1. the local compiler marked the image valid;
2. the stream begins with `NODE_SHADER_JUMP` and has a non-empty tail;
3. all three jump entries name words in that tail;
4. the node-use domain contains `NODE_SHADER_JUMP`;
5. peak stack use is at most Cycles `SVM_STACK_SIZE`;
6. the linked global offsets fit the signed 32-bit Cycles ABI.

These checks are representation invariants. They do not inspect or rewrite
individual shader node cases.

## External oracle

The regression freezes the complete 113-word final global `svm_nodes` image
from a diagnostic-only Blender Cycles 5.2.1 render containing these six shader
ids in order:

```text
default_surface
default_volume
default_light
default_background
default_empty
Diffuse Probe
```

Oracle file used locally:

```text
/tmp/psycles-cycles-svm-closure-oracle-20260901/diffuse_surface.svm52
SHA256 c8270f57dc3c1148d54c3787ce7d8f9beb760d35bc26b5ff347dcfdc7c9cfb6b
```

The test separates that external global image into its six mathematically
implied local images, invokes the Psycles linker, and requires exact equality
of every one of the 113 output words. This checks realistic default shader
tails as well as the custom diffuse surface/volume/displacement entries.
Malformed-prefix, missing-tail, negative-entry, escaping-entry,
incomplete-node-domain, and stack-overflow regressions are included.

No `.svm52` binary is checked in.

## Commands and results

```bash
cmake --build build --parallel 32 \
  --target psycles_cycles_svm_scene_tests \
           psycles_cycles_svm_compiler_tests \
           psycles_cycles_svm_bytecode_tests \
           psycles_cycles_svm_abi_tests

ctest --test-dir build --output-on-failure --parallel 4 \
  -R '^psycles\.cycles_svm_(scene|compiler|bytecode|abi)$'
```

Result:

```text
4/4 passed
```
