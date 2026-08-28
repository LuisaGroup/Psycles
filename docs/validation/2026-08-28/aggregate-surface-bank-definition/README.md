# Rejected aggregate surface-bank definition

## Result

Replacing the compact surface interpreter's per-leaf zero definitions with
one whole-aggregate constant assignment does not change the generated HIP
kernel. The experiment has therefore been rejected rather than retained as a
source-only rewrite.

The production Barbershop `shade_surface` kernel keeps the same 177-field,
864-byte coroutine frame and the same device resource counts:

| Metric | Per-leaf baseline | Aggregate candidate |
|---|---:|---:|
| HIP code object | 340,184 B | 340,184 B |
| Instructions | 60,719 | 60,719 |
| Scratch loads | 678 | 678 |
| Scratch stores | 543 | 543 |
| Global loads | 1,239 | 1,239 |
| Global stores | 277 | 277 |
| Private segment | 3,096 B | 3,096 B |
| VGPRs | 256 | 256 |
| SGPRs | 107 | 107 |

## Formal model

Let each typed value bank be an aggregate `B` with leaves `B[i]`. The existing
root definition emits

```text
for every i: B[i] := 0.
```

The candidate instead emitted the single source-level assignment

```text
B := zero(B).
```

Both establish exactly the same device-visible invariant

```text
Defined(B) := for every legal i, B[i] has a reaching definition.
```

That invariant is what allows coroutine alloca-scope contraction, so the
candidate was expected to preserve the frame proof while leaving store-width
selection to XIR and LLVM. It cannot remove any definition: the bytecode bank
index is dynamic, and a previous bounce's bank content would otherwise remain
observable unless every read were proved dominated by a current-lifetime
write.

## Why it did not work

The final HIP LLVM for both forms contains the same `[8 x float]` and
`[12 x [4 x float]]` private allocas. The aggregate constant is scalarized into
the same eight scalar-bank stores and 36 live vector-component stores at the
same program point. Padding lanes are not written in either form. Consequently
the final AMDGPU instruction topology and every relevant resource count are
identical.

The code objects are not byte-identical because generated callable numbering
changes and one scene-derived embedded floating constant differs between cold
processes. After inspecting the disassembly, neither difference belongs to the
bank-definition sequence. Since the proposed mechanism disappears before
code generation, a runtime A/B would measure only process and scene noise and
cannot justify keeping the rewrite.

The next bank optimization must alter the storage or lifetime model itself,
for example a conflict-aware shared-memory SoA abstraction or a formally
defined fresh-lifetime seed understood consistently by XIR, HIP, fallback, and
native SPIR-V. A naive per-lane shared AoS is not suitable: strides of eight
dwords for the scalar bank and 48 dwords for the vector bank create systematic
LDS bank conflicts.

## Validation

The candidate was built with the full project and passed compact surface
population/preparation on fallback, HIP, and strict native Vulkan XIR-to-SPIR-V
with DXC disabled before it was rejected.

```sh
cmake --build build --parallel 32

ctest --test-dir build --output-on-failure --parallel 1 \
  -R 'psycles\.luisa_(surface_population|compact_surface_preparation)_(fallback|hip)$'

LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
ctest --test-dir build --output-on-failure --parallel 1 \
  -R 'psycles\.luisa_(surface_population|compact_surface_preparation)_vk$'
```

The cold HIP evidence is in
`/var/tmp/psycles-aggregate-bank.t7WjFp`; the immediately preceding baseline is
in `/var/tmp/psycles-current-surface-ir.V5GwRp`.
