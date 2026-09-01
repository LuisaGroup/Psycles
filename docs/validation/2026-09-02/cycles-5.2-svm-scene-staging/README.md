# Cycles 5.2 SVM scene staging validation

## Result

The Luisa scene compiler now builds and uploads one native Cycles 5.2 SVM
image for the complete material table. This is a staging milestone, not a
claim that production `shade_surface` already executes the image: the current
renderer still constructs its legacy surface evaluator after the native image
has been validated and uploaded.

The transaction preserves Blender's source `cycles_shader_index` values.
Renderer-authored materials without a source index are assigned fresh values
strictly above the largest imported index, and the resulting material-to-SVM
mapping is retained explicitly. Sparse source tables are filled with inert
surface, volume, and displacement routines containing only `NODE_END`.

## Formal transaction boundary

Let every compile unit be

```text
(shader_index, ShaderProgram, ShaderCompileContext).
```

All units share exactly one `AttributeIDMap`, `ImageIDMap`, and `IESIDMap`.
The per-shader compiler therefore observes the same scene-global interning
domain as Cycles' `SVMShaderManager`. Linking is committed only if every local
image is valid and every repeated shader index denotes an identical tuple

```text
(words, node_types_used, peak_stack_usage).
```

A collision between distinct tuples rejects the whole transaction. No partial
word buffer or resource table is exposed. The linker then applies the already
validated Cycles equation

```text
global_entry = global_tail_base + local_entry - 4
```

to every occupied or inert slot. Named attributes are returned in their
assigned-id order; image and IES arrays retain the exact indices encoded in
the bytecode.

The Luisa runtime owns the host image plus optional device buffers for SVM
words and packed IES data. Empty buffers are not fabricated. Scene compilation
fails closed if this native table cannot be produced; it does not silently
substitute a differently encoded material table at this boundary.

## Closure and value-family parity added in this milestone

The scene transaction is accompanied by exact host encoders for the Cycles
5.2.1 node stream and typed payloads needed by the currently staged closure
set:

- Principled BSDF, including its 176-byte typed payload, feature state,
  three-domain jumps, and stack usage;
- Diffuse, Translucent, Transparent, Glass, Emission, and Background;
- Subsurface Scattering for Burley, Random Walk, Random Walk Skin, and the
  legacy Random Walk method;
- Absorption Volume and all five Cycles 5.2 Scatter Volume phases;
- `NODE_CONVERT_VF` and canonical Psycles binary-float helpers projected to
  Cycles Math nodes;
- canonical multiply-color projected to Cycles Mix Color with
  `MULTIPLY`, factor clamping, and no result clamping.

The relevant host classes now live in
`src/compiler/cycles_svm_closure_nodes.cpp`; the generic node factory owns no
concrete BSDF, BSSRDF, emission, or volume implementation. This is a real
translation-unit boundary rather than an included implementation fragment.
The resulting source sizes are 1835 lines for `cycles_svm_nodes.cpp` and 697
lines for `cycles_svm_closure_nodes.cpp`.

The external Blender/Cycles 5.2.1 dump identities frozen by the regressions
are:

| Oracle | SHA-256 |
| --- | --- |
| Principled Probe | `6e22050134bd344c2ada4fb3282755152f1697978178ccafc7451bc9797451f8` |
| Glass Transport 00 | `d84f339e9d25276cd8086105c47353e85f8187dea0535aac0bb4cbea7da33c5e` |
| Vector to Scalar | `b6ef1cfe605d43746ddfdedf01ba2f1b0d12b9364244c0d07d68afa77d0357c2` |
| Scatter Volume phases | `de1f3d1a21412283a882d4305a9d86da213c35ff0d8dde8d42cd247eeafc7f98` |
| BSSRDF methods | `1779888406dcb52efe14af75239ff52bde01f30ffad9c7ff2d298953ba8fef2f` |

`vector_to_scalar` and `volume_scatter_svm` are deliberately registered as
oracle-only probes. They verify exact Cycles bytecode while the production
renderer is still on the legacy evaluator; treating them as canonical image
probes would obscure that route boundary.

## Regression coverage

`tests/test_cycles_svm_scene.cpp` now covers:

- scene-wide compilation through a shared resource domain;
- preservation of sparse shader indices;
- inert three-domain holes;
- acceptance of genuinely identical duplicate indices;
- rejection of distinct local programs sharing one index;
- rejection of an absent source program;
- the existing complete 113-word external Cycles 5.2.1 global oracle.

Existing scene compilation tests exercise creation and upload on fallback,
HIP, and strict native Vulkan. The Vulkan environment requires native
XIR-to-SPIR-V and disables DXC.

## Commands and results

```bash
cmake --build build --parallel 32 --target \
  psycles_luisa psycles_luisa_scene_traversal_tests

ctest --test-dir build --output-on-failure --parallel 32 -R \
  '^psycles\.(cycles_svm_scene|luisa_scene_traversal_(fallback|hip|vk))$'
```

Result: 4/4 passed.

The complete configured suite was then rebuilt and run:

```bash
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure --parallel 32
```

Result: 464/464 passed in 15.10 seconds, including fallback, HIP, and strict
native Vulkan XIR-to-SPIR-V tests. The source-size and Blender probe-registry
contracts are part of this count.

No image triptych is reported for this staging-only milestone: production
`shade_surface` does not execute this word image yet, so a rendered comparison
would still measure the legacy evaluator. A Cycles/Psycles/difference triptych
becomes a required gate at the explicit production route switch below.

## Remaining boundary

The uploaded word image is intentionally not consumed by production
`shade_surface` yet. Doing so before the remaining Cycles closure families and
their exact evaluation/sample state exist would turn unsupported nodes into a
silent rendering fallback. The next migration work is to finish those typed
state transitions, construct the production `KernelGlobals` projection, and
then replace the legacy surface-program population as one explicit route
change.
