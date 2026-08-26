# Closure-order single-recording regression

Date: 2026-08-26

## Failure model

`cycles_closure_transparent_order` recorded the same material operation twice
on the host, once for requested closure index zero and once for index one. Luisa
DSL construction is multistage, so those calls emitted two complete copies of
the surface graph before any device compiler could optimize them. The resulting
kernel contained 11,529 unoptimized XIR instructions and exceeded its 11,000
instruction ceiling.

This was not material complexity and raising the ceiling would have hidden the
recording error. The XIR module contained one 11,529-instruction kernel and no
callable definition that could share the two expansions.

## Formal scheduling correction

Let the requested closure set be `I = {0, 1}`. The corrected launch uses one
thread for every `i` in `I` and records `closure_trace(i)` exactly once in the
kernel AST, with `i = dispatch_x()`. Each thread writes two disjoint records:

```text
metadata(i) -> output[i]
weight(i)   -> output[i + |I|]
```

For `i` in `I`, both maps are injective, their ranges are disjoint, and their
union is the original four-record output layout. Thus the launch is a bijection
over the two semantic cases, without omitted or duplicate closure requests.

## Result

The unoptimized XIR instruction count fell from 11,529 to 5,842, a 49.3%
reduction. The ceiling is tightened to 7,300, approximately 25% above the new
single-recording baseline. A future second full graph expansion therefore
cannot pass the gate.

Validation commands:

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_cycles_closure_tests

PSYCLES_REPORT_SHADER_SHAPES=1 \
  build/bin/psycles_luisa_cycles_closure_tests fallback

ctest --test-dir build --output-on-failure -j 1 \
  -R 'psycles\.luisa_cycles_closure_(fallback|hip|vk)$'
```

All three backend tests passed. The Vulkan test used the configured native
XIR-to-SPIR-V route.
