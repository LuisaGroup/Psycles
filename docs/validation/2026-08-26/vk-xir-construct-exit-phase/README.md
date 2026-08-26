# Vulkan XIR construct-exit phase regression

Date: 2026-08-26

## Failure model

The strict native Vulkan curve-path canary exhausted the CFG restructuring
budget. Its largest definition started the post-restructure phase with 316
blocks and 9,902 instructions, then grew as follows:

```text
iteration 0: 316 -> 428 blocks,  9902 -> 10630 instructions
iteration 1: 428 -> 472 blocks, 10630 -> 10700 instructions
iteration 2: 472 -> 507 blocks, 10700 -> 10761 instructions
iteration 3: 507 -> 542 blocks, 10761 -> 10822 instructions
```

The repeated 35-block/61-instruction suffix was not intrinsic shader
complexity. Main-loop restructuring had recovered a new loop hierarchy, but
the post phase inferred construct-exit analysis validity from transient
generated-dispatch markers. With no surviving marker, selection-exit repair
observed the newly recovered hierarchy before its exits were closed and built
another state dispatch for the stale relation. Later canonicalization consumed
the marker, allowing the same rewrite shape to repeat.

## Formal correction

`fixup_construct_exits` computes construct-exit closure for one CFG version.
Any transformation that changes executable edges, construct ownership, or
entry/merge/continue roles invalidates that result. The pass now carries this
validity explicitly:

```text
construct_exits_dirty := preceding transformation changed the CFG

before header/selection repair:
    if dirty: recompute closure; dirty := false

after a phase that may mutate the relation:
    dirty := true
```

A pre-existing, unchanged structured input retains the original selection-exit
ordering and receives one closure analysis at the second phase boundary. A
hierarchy recovered by the main phase is closed before downstream
header/selection repair can observe it. Dispatch markers remain provenance and
role information; they are no longer used as a proxy for analysis validity.

The nested-loop regression has two raw loops sharing the outer continue path.
It now asserts that the new loop hierarchy is closed before selection repair,
so no selection-exit invalidation is needed. Under the old gate the case grew
from 16 to 30 blocks in the first post iteration and finished at 31 blocks; the
corrected phase order reaches 27 blocks without the redundant dispatch.

## Result

The strict source-dump canary now converges in two changing post iterations:

```text
iteration 0: 316 -> 356 blocks,  9902 -> 10007 instructions
iteration 1: 356 -> 358 blocks, 10007 -> 10009 instructions
iteration 2: 358 -> 358 blocks, 10009 -> 10009 instructions
```

The affected definition's `restructure_cfg_on_definition` time was 20.117 ms.
Two other large definitions converged at 644 and 796 blocks. Native SPIR-V
generation completed with optimized binaries of 22,389, 98,776, and 120,315
words; no incomplete-restructure fallback was used.

Validation commands:

```sh
cmake --build build/luisa-tests --parallel 32
cmake --build build --parallel 32 --target psycles_luisa_curve_path_tests

ctest --test-dir build/luisa-tests --output-on-failure -j 32 -L unit_xir

LUISA_DUMP_SOURCE=1 \
LUISA_XIR_TRACE_PASSES=1 \
LUISA_LOG_LEVEL=verbose \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  build/bin/psycles_luisa_curve_path_tests vk

build/bin/psycles_luisa_curve_path_tests fallback
build/bin/psycles_luisa_curve_path_tests hip
build/bin/psycles_luisa_curve_path_tests vk
```

All 59 XIR/coroutine unit tests passed. The curve-path runtime regression
passed on fallback, HIP, and Vulkan; Vulkan used the strict native
XIR-to-SPIR-V route.
