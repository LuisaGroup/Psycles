# Maintainability audit

Updated 2026-09-08. This page describes current source debt and mandatory
boundaries. Historical GraphSurface decomposition, old custom-executor
performance, forced HIP noinline and coroutine-baseline claims have been
removed from current guidance; their dated validation reports and Git history
retain the original evidence.

## Four source-size violations remain

The first-party budget is 2000 lines per hand-written source file.
The current checker has no grandfathered debt entries or hand-written .inl
allowlist. Generated tables and third-party sources remain separate.

| File | Current lines | Required semantic decomposition |
| --- | ---: | --- |
| src/compiler/cycles_svm_nodes.cpp | 2026 | Native node compiler families with shared stack/payload contracts |
| tests/test_cycles_svm_compiler.cpp | 2088 | Independent compiler family regressions |
| tests/test_luisa_compact_surface_preparation.cpp | 2114 | Separate preparation/state GPU fixtures |
| tests/test_luisa_cycles_svm.cpp | 2038 | Independent runtime opcode/state families |

These are actual failures, not exemptions. Run
`python3 tools/check_source_size.py` after each change. Splitting must preserve
oracle inputs and coverage; moving code into textual includes or raising
the limit is not a fix.

## Native Cycles SVM owns material execution

Surface, volume, shadow, world and light consumers use the Cycles 5.2.1 SVM
word stream, typed payloads, stack addresses, PC loop, dispatch, closure state
and feature masks. Static scene/entry analysis omits unreachable node cases
and bounds local arrays. Profiles and scene names must never supply bounds.

The private displacement prepass still consumes the legacy evaluator.
Legacy GraphSurface/SurfaceProgram translation units and related helpers
remain removal debt, not a second endorsed architecture or evidence that
default-path cleanup is finished. See
[compatibility status](cycles-compatibility.md) for remaining native opcodes
and unsupported geometry/motion configurations.

The renderer is divided by Cycles responsibilities: immutable scene setup,
camera/path state, traversal, surface shading, direct/forward lighting,
volume transport, film and observational tracing. Compiler changes retain
normalization, graph compilation, serialization and runtime boundaries.
Preserve natural shared contracts when extracting a family.

## Coroutine facilities stay generic

Psycles policies belong in Coro Ext/Handler clients. Luisa lifetime analysis,
frame projection and scheduler facilities must remain reusable. Normal
scalar/vector initialization retains zero semantics. Lifetime-only scratch
declarations do not authorize arbitrary uninitialized reads.

Use stream insertion for scheduler dispatch commands. Frame storage belongs
to reusable scheduler/worker pools, without per-thread/per-resume malloc.
Leave inlining decisions to the compiler. Do not reintroduce forced noinline
or software floating-point paths to reproduce harmless last-bit differences.

## Verification and publication remain semantic gates

Every compiler/backend correction requires a formal root cause, minimal
failing example, permanent regression, generic fix and full original-module
verification. Build with all 32 hardware threads. Validate HIP first, then
fallback and strict native XIR -> SPIR-V Vulkan without DXC. Structural
refactors preserve Cycles state, sampler dimensions and observable control
flow; exercise the affected original-Cycles word/GPU fixtures and full scenes.

The [current validation index](../VALIDATION.md) distinguishes complete suites,
focused gates and known failures. The
[equal-pass benchmark](validation/2026-09-08/matched-pass-hip/README.md)
separates render time, session initialization, host compilation and frame
size. Reduced IR or storage is not independently a performance result.

Inspect both designated worktrees, preserve unrelated changes, and stage
exact files. Publish validated generic Luisa changes to origin/next before
advancing the parent gitlink; publish Psycles checkpoints to origin/main.
