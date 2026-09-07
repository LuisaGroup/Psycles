# Native SVM static pruning audit

Unused SVM opcodes are omitted before device AST construction, including the
case label. This audit also repairs ten missing whole-body feature guards:
Background's emission guard and nine derivative nodes' non-volume guards.
It does not introduce a different interpreter or rely on backend dead-code
elimination to discard unused node handlers.

## Semantic boundary

The reference is Cycles 5.2.1 source
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`, specifically
`intern/cycles/kernel/svm/svm.h`, `kernel/features.h`, and the node catalog.

Two independent decisions must not be confused:

- `node_types_used[n] == false`: omit the entire `$case(n)`. The host compiler
  records usage when emitting an opcode, not when appending typed payload
  words; the scene linker unions those usage sets.
- A used opcode's Cycles node feature guard is false: record the same empty
  case/break as Cycles, with no node-body instructions or payload consumption.
  Moving a typed payload read outside this guard would change the PC.

The first rule is implemented with an ordinary host `if` outside `$case`:

```cpp
if (node_types_used[NODE_TEX_IMAGE]) {
  $case(static_cast<unsigned>(NODE_TEX_IMAGE)) {
    // Record the original handler.
  };
}
```

This is not `$if`, and an unused node does not leave an empty device label.
The same usage union reaches production surface, light, and shadow SVM
entrypoints. Their different Cycles node feature masks remain distinct.
The word image, typed payloads, stack addresses, PC loop, and node/closure
state transitions are unchanged.

Partial feature guards still follow their individual Cycles semantics:
BSDF can retain typed-payload skipping or Principled emission without BSDF
setup; Light Path keeps visibility queries when depth counters are disabled;
bump/displacement can consume a payload without updating the corresponding
state. A material parameter or a runtime branch is not a license to invent
additional per-value specialization.

## Findings and permanent regression

`tests/test_luisa_cycles_svm_static_pruning.cpp` inspects the recorded AST,
before XIR, LLVM, or SPIR-V optimization. It checks an empty usage set, every
one-hot usage bit, and the all-enabled set against the implementation inventory.
All 99 existing dispatch handlers obey the outer usage guard, including the
three handlers added by the [native volume port](../native-volume-svm/README.md).
Of the 110 enum entries, nine still lack runtime handlers and two are non-executable placeholders
(`NODE_NONE` and `NODE_PAD1`); these are explicit gaps, not pruning successes.

The test additionally checks all 25 currently implemented whole-body feature
guards, enabled and disabled in each of the three shader domains. Nine use
emission/bump/bump-state/volume flags; all 16 derivative variants use the inverse
volume flag. The derivative list comes from the Cycles node catalog, not from
the implementation under test.

Before the fix, the opcode-usage tests passed but the feature matrix failed
for Background and derivative Image, Image Box, Environment, Vector Math,
Mapping, Texture Mapping, Tangent, Separate Vector, and Combine Vector.
Each failed in all three domain recordings. After the fix, their disabled
bodies contain only the DSL's structural break, with zero payload reads,
stores, runtime conditionals, or helper calls. Enabled bodies remain present.

## Lone Monk and the old SSS-exit question

Read-only GDB observations of the active worktree's actual HIP JIT builders,
using the complete production scene bundles at 16x16 / 4 spp, recorded:

| Scene | Scene kernel features | Used opcode bits | BSSRDF setup recordings | SSS-exit setup recordings |
| --- | --- | ---: | ---: | ---: |
| Lone Monk | `0x8090b`, subsurface off | 40/110 | 0 | 0 |
| Monster | `0x2091b`, subsurface on | 44/110 | 6 | 1 |

Both processes exited successfully. Monster is a positive control for the
breakpoints; zero Monk observations are not merely missing instrumentation.
These are host code-generation counts, not per-ray execution counts, and use
the inherited worktree feature-mask work rather than only this small commit.
The original GDB script and LLVM dumps are retained with the logs at
`/var/tmp/psycles-feature-pruning-8a86Az` and
`/var/tmp/psycles-feature-pruning-monster-sVf1gs`.

Before `46a1d32f`, native surface consumers recorded an SSS-exit override
behind `$if(query.subsurface_exit)` without a host scene-feature guard. An
unused exit body could therefore enter IR. The subsequent native surface
state change removed that consumer override and guards exit setup on the
host. This explains the old pruning hole; it does not prove that Monk rays
executed SSS or uniquely explain a previous image discrepancy. See the
[surface-state checkpoint](../native-surface-state/README.md).

## Validation and scope

Evidence: `/var/tmp/psycles-svm-static-pruning-M6CfEm`.

- Active all-target build: 32 threads, passed.
- New AST regression and native closure-budget compiler regression: 2/2.
- Full HIP suite: 167/167; full fallback suite: 169/169.
- Strict native Vulkan focused suite: 12/12.
- Isolated Psycles snapshot: `4efe6df5` plus only this pruning patch and its
  regression; all-target build with 32 threads passed (923 steps).
- Isolated AST plus focused HIP: 10/10; fallback: 9/9; strict Vulkan: 9/9.
- Isolated complete renderer: Map Range 16x16 / 4 spp on HIP, then fallback,
  then strict native Vulkan, with shader-cache reuse disabled. All outputs
  are finite. Vulkan records 25 successful native SPIR-V compilations and no
  DXC/DXIL library load in its `LD_DEBUG=libs` log.

Isolation compiles the application against clean Luisa `8e2b0ac78` headers
and active SDK libraries. It is not an independent clean SDK rebuild and does
not include the inherited coroutine/closure/feature-mask application changes,
the pending closure-budget change, or a modified Luisa gitlink.

The broad host suite has two independent known failures: four pre-existing
oversized source files, and a stale policy-test expectation that volume
preprocessing still disables fast math. The render-settings exporter test was
not selected. Neither failure is hidden by claiming a completely green CTest.

This checkpoint proves code-generation boundaries, not a renderer speedup
or complete SVM family coverage. Full-resolution scene canaries and further
scene-sized local-storage work are tracked separately.
