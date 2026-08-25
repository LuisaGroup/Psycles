# Orthogonal SVM record fields

## Outcome

The compact surface interpreter no longer treats instruction-owned evaluator
state as an exclusive choice. A record may now carry an SVM mode immediate, a
late-bound parameter identifier, and a static-table view at the same time. This
fixes a deterministic Barbershop fallback crash introduced by the finite-mode
SVM compaction while preserving one shared Color Ramp handler.

The production regression runs on fallback, HIP, and Vulkan. Vulkan is forced
through native XIR to SPIR-V code generation with DXC disabled.

## Root cause

The first failing commit is:

```text
ce54a69 psycles: data-drive finite SVM node modes
```

The exact source-level bisect was:

| Revision | Fresh Barbershop export, 640 x 480, 1 spp, fallback |
| --- | --- |
| `c5c8fa4` | pass |
| `ae3e1d0` | pass |
| `ce54a69` | SIGSEGV |
| `973199f` | SIGSEGV |
| `9b595cf` | SIGSEGV |

Color Ramp owns two independent pieces of record state:

1. its finite interpolation mode, encoded as the SVM immediate; and
2. the `ParameterId` locating its sampled table in the material parameter
   block.

The old evaluator returned immediately after installing the mode override, so
it never installed the parameter override. Once multiple Color Ramp records
shared a handler, the callable retained the host variant's original parameter
identifier. A record belonging to another material then read the wrong
parameter slot, and sufficiently complex scenes eventually interpreted the
invalid value as table or resource metadata.

## Formal model

For operation `o`, define the optional record projections

```text
I_o : Record -> Option<SvmImmediate>
P_o : Record -> Option<ParameterId>
T_o : Record -> Option<StaticTableRange>
```

The evaluator environment is their product:

```text
E_o(record) = BaseEnvironment
              x I_o(record)
              x P_o(record)
              x T_o(record)
```

It is not the disjoint union `I + P + T`. Correct handler sharing therefore
requires constructing all defined projections before calling `evaluate`
exactly once. The implementation loads only the fields selected by the
operation and record metadata, so operations that do not own a projection do
not pay for it in their shader AST.

This gives the following invariant:

```text
evaluate(shared_handler(o), record, operands)
  == evaluate(exact_host_variant(record), operands)
```

provided that every instruction-owned component of the exact variant is
represented by one of the record projections. Adding a new component must
extend this product explicitly; an early-return precedence chain is invalid.

## Regression construction

The compact preparation fixture now builds two live sampled Color Ramp graphs.
One graph uses a direct factor. The other inserts a live Add node before the
ramp, shifting the table `ParameterId`, and uses a different sampled table.

The test first inspects the production compact executable and proves:

- there is exactly one Color Ramp handler variant;
- its instruction stream contains at least two distinct, valid late-bound
  table parameter identifiers; and
- both mode and parameter metadata are therefore required on the same shared
  handler fiber.

It then compares compact execution against the existing expanded production
evaluator. Shader tables are staged and finalized with the same production
helpers used by full scenes; no CPU reference renderer or pre-baked material
result is involved.

## Validation

The former full-scene crash reproducer now completes:

```text
build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-primary-dispatch-matrix-20260825/barbershop-fresh/export \
  /var/tmp/barbershop-record-product-fix-640x480-1.ppm \
  fallback 640 480 1 1
```

Measured result:

```text
scene compile: 9.91823 s
shader JIT:    13.7475 s
render:         0.244527 s
result:         pass
```

The focused matrix passes 29/29:

```text
env \
  LUISA_VULKAN_USE_XIR=1 \
  LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 \
  LUISA_LOG_LEVEL=warning \
ctest --test-dir build --output-on-failure --parallel 1 \
  -R 'psycles\.(surface_program_metadata|surface_svm_(math_immediate|vector_math_immediate|record_immediates)|normal_map_semantics|luisa_(compact_surface_preparation|surface_mix_svm|surface_math_svm|surface_vector_math_svm|normal_map|normal_map_callable|bump_callable|noise_callable)_(fallback|hip|vk))$'
```

The complete build used all 32 host threads:

```text
cmake --build build --parallel 32
```

The complete 285-test suite reports 275 passes and 10 failures. To determine
whether any failure was introduced here, the four-file patch was stashed, the
parent `9b595cf` was rebuilt with 32 threads, and the exact ten failures were
rerun serially. Parent and patched trees produce the same ten test names and
the same diagnostics:

- four existing fallback numerical-oracle differences;
- two existing Vulkan numerical-oracle differences;
- three existing native-XIR `restructure_cfg` failures; and
- one existing `cycles_closure_transparent_order` instruction-ceiling failure.

No threshold was widened and no failing test was disabled. These issues remain
independent follow-up work rather than being hidden by this SVM fix.

## Scene provenance

The real-scene reproducer is the freshly exported Blender 5.2 Barbershop file
from:

```text
/var/tmp/psycles-official-redownload-20260814/barbershop_interior.blend
```

Both exact Blender/Cycles 5.2 and Psycles report two genuinely missing external
assets in that source file: `generic_scratches.png` and
`guilder_ornament.png`. The five-way benchmark records those source diagnostics
instead of silently substituting or baking textures.
