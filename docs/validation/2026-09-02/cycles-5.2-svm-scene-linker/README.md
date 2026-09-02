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

The same transaction now also constructs and uploads the dense native
`KernelShader` image. It is not yet the production `shade_surface` evaluator;
exposing the tables through `KernelGlobals` and constructing exact
`ShaderData` remain the next migration boundary.

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

## KernelShader metadata image

For every represented source shader index `i`, scene compilation now emits the
exact 32-byte, 16-byte-aligned Cycles `KernelShader` record `K[i]`. The record
is derived from the same finalized graph traversal that emits SVM bytecode:

```text
K[i].constant_emission = output_estimate_emission(surface_root)
K[i].cryptomatte_id    = hash_to_float(murmur3(shader_name))
K[i].flags             = Cycles Shader metadata and authored policies
K[i].pass_id           = Blender Material.pass_index
```

The metadata transfer follows `ShaderManager::device_update_common` in
`intern/cycles/scene/shader.cpp`: emission front/back sampling, transparent
shadow gating, ray-trace use, volume/only-volume/heterogeneous/attribute state,
BSSRDF and displacement bump state, volume sampling/interpolation, true
displacement, bump correction, constant emission, and Light Path presence.
Spatial and attribute facts are recorded only while the corresponding node is
actually generated in the surface or volume compiler state, matching
`SVMCompiler::generate_node`; closure-only facts are recorded in the matching
surface closure state. This avoids both a whole-graph false positive and an
opcode-based reconstruction that has already lost source semantics.

The table is total over the linked shader index domain. An unrepresented hole
has inert END-only SVM routines and an all-zero `KernelShader`; duplicate
indices are accepted only if both the complete local compiler image and the
complete `KernelShader` record agree. Thus no per-material policy can be lost
merely because two graphs emit equal bytecode.

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
incomplete-node-domain, and stack-overflow regressions are included. Additional
tests freeze constant-emission/hash/pass fields, transparent-shadow policy,
heterogeneous volume and cubic/equiangular flags, Light Path emission state,
surface BSSRDF bump, automatic/true displacement, zero holes, and rejection of
equal bytecode carrying unequal shader metadata.

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

The typed runtime canary also reads every `KernelShader` field through a Luisa
kernel using dynamic shader indexing. It passed on fallback and HIP, and on
Vulkan with `LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`. Blender 5.2 exporter/importer regressions passed
with CUBIC material and world volume interpolation, so the device flag is
sourced from the original scene policy rather than a renderer default.
