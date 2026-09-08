# Native HIP remainder and remaining Barbershop surface cost

## Result

Psycles `cc4d9974` / Luisa `d51a33d48` publish native OCML floating-remainder
lowering before JIT IPO. This removes an unnecessary late precise-division
sequence without replacing range reduction by an approximate quotient.
It is a small code-generation cleanup, **not a resolution of Barbershop's
roughly twofold surface cost**. No SVM semantics, frame layout, inline/noinline
policy, register limit or scene-specific backend setting changes.

The main interpreter and microfacet bodies are already inlined in final ISA.
The same four machine functions remain after this change: the surface entry,
one unnamed procedural helper, signed 4D Noise and OCML tangent. Their total
effective static instruction sites fall from 188,976 to 188,137 (-0.44%).
Register and private-storage metadata are unchanged. Static counts are not
dynamic work or a speedup estimate.

## Cause, reduced control and semantics

The old HIP lowering leaves LLVM `frem` until final target code generation.
For a dynamic f32 divisor, its normalized-divisor reciprocal then expands
into DIV_SCALE/refinement/DIV_FMAS/DIV_FIXUP. Original Cycles' HIP `fmodf`
uses the installed OCML body, with native RCP, rint, fused residual subtraction,
residual correction and exponent-stepped range reduction. Exposing that same
body before IPO lets the ordinary optimizer see and specialize it.

The existing exclusion of `frem` from blanket fast-math flags was not a
one-ULP workaround: `1e17 % 2pi` must retain approximately 3.187734, while a
rounded quotient/subtraction may lose the entire remainder. That exclusion
remains for other LLVM remainder instructions. The new XIR floating
`BINARY_MOD` uses exact native f16/f32/f64 call signatures, scalarizing vector
lanes at the library ABI; integer `%` is unchanged. Shader-cache codegen
revision 85 invalidates previously compiled HIP entries.

The original GPU probe calls the actual Cycles math/Noise headers under
`--offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 -ffast-math`; no shader formula
is implemented on the host. Across eight dynamic float4 inputs, all 96
direct scalar/vector remainder lanes match original Cycles exactly before
and after the change. Of 32 Noise lanes, one differs by one ULP in both
implementations. That difference is deliberately ignored.

The lowering-quality regression first tests the extracted old `CreateFRem`
expression unchanged: 24 checks fail across 12 scalar/vector ABI shapes.
The native implementation passes 396 IR/operand/ABI checks. Its independent
runtime regression passes 5,232 assertions for f16/f32/f64 scalar, per-lane
and broadcast operands in precise/fast modes, including large quotients and
precise NaN/infinity/signed-zero classes. Host `std::fmod` here checks a basic
arithmetic primitive; it is not a CPU shader or rendering oracle.

## Full original scene A/B/B/A

All runs use the complete Barbershop scene at 2048x858 / 64 spp / seed 0,
identical inputs and passes, with no overlapping heavy build, test or render.
A is the old lowering, B is native OCML, and restored A rebuilds the original
lowering. Stage symbols independently identify LLVM and ELF artifacts.

| Measurement | A | B, first | B, repeat | Restored A |
| --- | ---: | ---: | ---: | ---: |
| Render seconds | 10.6827 | 10.6817 | 10.4403 | 10.6712 |
| Surface GPU seconds | 6.056961 | 5.998918 | 5.887299 | 6.039241 |
| Surface invocations | 332,307,893 | 332,307,894 | 332,307,893 | 332,307,894 |
| Frame bytes | 416 | 416 | 416 | 416 |
| VGPRs / SGPRs | 256 / 107 | 256 / 107 | 256 / 107 | 256 / 107 |
| Fixed private bytes | 2,496 | 2,496 | 2,496 | 2,496 |
| Session initialization seconds | 28.1935 | 78.2774 | 26.2854 | 27.5524 |

The two native results vary more than the first A-to-B difference. These
controls do not establish a substantial or isolated end-to-end speedup.
The first native run incurs new downstream HIPRTC compilation; the repeated
run benefits from its normal cache. Session initialization includes
JIT/setup/baking, not compiler-only or matched cold-JIT time. Main shader
caching is disabled throughout.

Restored surface `.text` is byte-identical to A (SHA-256
`2c0374c4975eeabb289841e2f6f9dc3aabbc02546db84cf230ab6b780da44fbb`).
All 46 output channels are finite and all 15 before/after pass comparisons
complete. First B's Combined relative RMSE against A is about 6.56e-9;
its DiffInd control is about 6.15e-6. These are intervention controls, not
substitutes for original-Cycles image comparisons.

The instruction auditor now bounds each function by its ELF `STT_FUNC`
extent. Earlier unbounded textual counts included decoded alignment padding;
those counts are superseded by the bounded counts here. Padding caused a
false apparent tangent instruction change even though its 972-byte body was
unchanged. The new five-case parser suite covers independent dump counters,
ambiguous/missing symbols, symbol extent/address agreement and padding.

## Gates and qualifications

All builds use 32 hardware threads. Psycles passes host 168/168, HIP 182/182
and fallback 184/184. The complete Luisa unit selection in the HIP build
passes 131/131 (130 tests carry the literal `unit` label, one additional test
matches a `unit_*` label). HIP arithmetic and shader-cache checks also pass.
Gate elapsed times include cold cache misses and overlapping non-timed CPU
test builds; they are not renderer/JIT benchmarks.

The strict native Vulkan lamp/bump gate passes 2/2 with 31 native SPIR-V
modules and no DXC/DXIL loading. An additional isolated float32 remainder run
passes 1,744 assertions. **The additional all-type Vulkan remainder run fails
1,076 assertions in existing f16/f64 lowering**, which still emits direct
`OpFRem`; for example half `65504 % 6.28125` produces 28.25 instead of 3.125.
This is retained coverage, not waived as a rounding difference. No SPIR-V
implementation changes are in this HIP commit; the f16/f64 follow-up remains
open and the default new test continues to exercise all types.

The valid fallback all-type run passes 5,232 assertions. An initial diagnostic
mixed system-STL and non-system-STL Luisa build ABIs and was rejected; it is
not treated as a backend defect. Its log is retained separately from the
correctly rebuilt test.

## Complete four-scene 256-spp follow-up

Six runs use frozen binaries and the retained equal-pass original-Cycles
references, not fresh timing pairs. Barbershop runs three times; the other
scenes run once. All 46 channels are finite, all 15 pass comparisons complete,
and all four first-run Combined triptychs have been inspected.

| Scene / extent | Render seconds | Retained Cycles median | Session init seconds | Frame |
| --- | ---: | ---: | ---: | ---: |
| Lone Monk / 1440x1080 | 13.4562 | 13.2425 | 60.2862 | 220 B |
| Monster / 1080x1080 | 14.8523 | 14.2865 | 69.2298 | 280 B |
| Classroom / 1920x1080 | 18.3115 | 18.0319 | 51.1574 | 260 B |
| Barbershop / 2048x858, median | 40.2139 | 25.3775 | 24.8495 / 24.5166 / 24.6063 | 416 B |

Barbershop's individual renders are 40.2139 / 40.2494 / 40.1534 seconds. Its
median is 0.20% above the preceding 40.1352 seconds, so the full-resolution
follow-up does not establish a render improvement; the retained-Cycles gap
is still 58.5%. Unlike Barbershop, the other three runs encounter their new
native code's first downstream compilation. Their initialization times must
not be compared with earlier warm-cache values as an intrinsic JIT slowdown.
Initialization is excluded from render time in every row.

First-run Combined relative RMSE, in table order, is approximately
1.2404% / 0.5479% / 0.3533% / 1.0521%; DiffInd remains
12.8818% / 2.5525% / 17.8203% / 7.1262%. These are unresolved differences,
not declared sampling noise. Original Classroom's 25 invalid DiffDir and 27
invalid GlossDir pixels retain the explicit union-of-invalid exclusion.

## Reproduction and remaining work

Evidence root: `/var/tmp/luisa-hip-frem-audit-SAGCwq`. Original and Luisa probe
sources are `tools/cycles_fmod_noise_oracle.hip` and
`tools/luisa_fmod_noise_probe.cpp`; the latter's optional `ocml` argument is
diagnostic only. Original source is the pinned Blender Cycles 5.2.1 tree;
the task-local CMake file records all original compiler flags and 32-thread
build commands. The child contains permanent IR and runtime remainder tests.

`archive_results.py EVIDENCE results.json --parent cc4d9974 --child d51a33d48`
checks the complete profile controls, gate totals and six-run canary manifest
before freezing the report. Use the stage-identified ELF/ISA auditor, never
assume equal LLVM/code-object dump indices.

Barbershop's non-resource SVM layouts already match the original 279 used
shader images. Resource binding identity is still unresolved. The remaining
performance investigation must explain per-invocation surface cost, including
shared closure-case emission and final code generation, not assume extra
surface paths or missing inlining from source alone. Residual DiffInd/shadow
work, unsupported nodes and private legacy displacement removal also remain
open.
