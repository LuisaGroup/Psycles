# Generic recording-time shading-terminator pruning

The two native BSDF frequency-correction blocks are now omitted at Luisa DSL
recording time when the complete finalized object image proves their native
`frequency > 1` predicates false. This is an ordinary host C++ `if`, not a new
compiler pass, a profile-derived decision, or a scene-specific exception.

The original GPU fixture also exposed a separate pre-existing algorithm
mismatch: Psycles used ordinary `acos` where Cycles calls `fast_acosf`.
The bounded repair reuses Psycles' existing native fast helper at that one
call site. It does not add a precision-compensation or slower software path.

As of this checkpoint, the final candidate builds with all 32 threads, passes
**186/186 host tests**, all **six recording controls**, **4/4 focused HIP
tests**, the **193/193 full HIP suite**, **195/195 fallback tests**, and
**4/4 strict native Vulkan controls**. The original six-case GPU fixture
passes both the enabled projection and its three eligible disabled cases.
All four fresh 256-spp HIP canaries and sixty pass comparisons complete,
but the numerical gate is **3/4**: Classroom has 18 non-finite actual lanes
at eight pixels. [The Phase A report](PHASE_A.md) retains all results,
including residual DiffInd. Fresh exports and twelve new timing pairs remain
unrun. No rendering speedup or complete Cycles parity is claimed.

## The static proof and its lifetime

Cycles' finalized `KernelObject` field is a frequency, not the authored UI
offset: `frequency = 1 / (1 - 0.5 * offset)`. Both original
`scene/object.cpp` and the existing Psycles
[native object packer](../../../../src/compiler/cycles_svm_object_scene.cpp)
use that conversion. The new query reads the final packed field, after the
normal object/geometry transaction, rather than interpreting the UI value
again or looking at material names.

For a finalized image, the host certificate is false only when the object and
flag arrays have consistent extents and **every** frequency is finite and at
most one. It is true for an invalid/incomplete transaction, any non-finite
frequency, or any frequency above one. A valid empty table is vacuously off.
Finite values below one also satisfy the original no-op predicate; the proof
does not unnecessarily narrow it to exact equality with one.

The [image query](../../../../src/luisa/path_tracer_cycles_svm_object.cpp) is
evaluated once by the actual
[PathCyclesSvmKernelGlobals constructor](../../../../src/luisa/path_tracer_cycles_svm_kernel_globals.cpp).
The recorded kernel retains that immutable scene-image decision. A changed
object table goes through finalization and new KernelGlobals/JIT construction;
there is no process-global cached scene answer. Independently supplied or
dynamic services inherit a conservative **true** default from
[KernelGlobals](../../../../include/psycles/luisa/cycles_svm.h), unless the
provider can explicitly meet this complete, immutable-image proof contract.

[The host regression](../../../../tests/test_cycles_shading_terminator_scene.cpp)
uses the real geometry/object transaction with arbitrary object identities and
analytic-light rows. It tests all-off, mixed, all-on, and off-again regeneration;
the prior snapshot remains unchanged. It also tests sparse/default rows,
rejected geometry, inconsistent extents, NaN/infinities, the finite no-op
domain, and a valid empty image. The actual production KernelGlobals is
constructed without uploaded buffers and its host query must record no
`BUFFER_READ`. This verifies the production wiring, not just a test service.

## Exact native placement, without removing neighboring work

The two recording-time guards are in
[cycles_svm_bsdf.cpp](../../../../src/luisa/cycles_svm_bsdf.cpp):

```cpp
if (kernel_globals.has_shadow_terminator_shading_offset()) {
  const auto frequency =
      kernel_globals.object_shadow_terminator_shading_offset(shader_data.object);
  $if(frequency > 1.0f) { /* existing native frequency correction */ };
}
```

When enabled or unknown, the original per-object device predicate and body
remain. In sampling, this stays inside the non-transmission, non-`LABEL_NONE`
branch. In evaluation, the existing nonnegative closure-normal cosine check
remains. No closure dispatch, label, PDF, state, or SVM word/PC/stack contract
is redesigned.

Both original shader-side reads of `shadow_terminator_shading_offset` are in
these native BSDF consumers. The separate bump-shadowing multiplication stays
outside each host guard. Geometry shadow-terminator offset handling and ray
origin logic are untouched; the host test even retains an active geometry
offset while proving the shading frequency off. Transmission handling and
subsurface-exit preparation are not folded into this capability. The focused
BSDF-dispatch and subsurface-exit regressions remain separate controls.

The actual service's existing `OBJECT_NONE` adapter is unchanged. The original
GPU fixture deliberately uses real object indices: it does not dereference an
invalid original object and does not certify the Psycles sentinel adapter.

## Authentic recording-time red and green

The standalone pre-API witness is retained at
`/var/tmp/psycles-holdout-CXlJR7/shading-frequency-recording-5sfsqG`.
It links the frozen published runtime and uses the actual old public header,
which had no host capability query. Its future derived method was declared
without `override`, allowing the same behavioral expectation to fail against
the old API. No backend or optimizer executes in this test.

The [old log](recording-before.log) has exactly two failures: disabled eval
and disabled sample each still call the dynamic object-frequency service and
record one ACOS. All four enabled/default controls pass. The service returns
a dynamic buffer read, so this is not constant folding of a known frequency.
All six complete pre-optimization AST modules and their hashes are retained.

After the host guard, [the intermediate log](recording-guard-only.log) records
zero service calls and zero ACOS in disabled modes; enabled/default modes
remain one each. This proves recording-time omission independently of the
later fast-angle repair.

The [permanent recording regression](../../../../tests/test_cycles_shading_terminator_recording.cpp)
now follows the native fast algorithm: no ACOS appears in any diffuse-only
mode, and the correction contributes exactly one COS. The sampler's genuine
disk mapping already has one COS and is not removed. It traverses complete
entry and reachable custom callable bodies, without double-counting shared
AST expressions. [Final observed controls](recording-native-fast.log):

| Operation | Capability | Frequency service calls | COS calls | ACOS calls |
| --- | --- | ---: | ---: | ---: |
| Eval | false | 0 | 0 | 0 |
| Eval | true / default | 1 | 1 | 0 |
| Sample | false | 0 | 1 | 0 |
| Sample | true / default | 1 | 2 | 0 |

These are pre-backend recording facts, not an inferred final instruction
count, a coroutine-frame measurement, or an end-to-end speed result.

## Original Cycles GPU oracle and the separate angle mismatch

The [external HIP probe](../../../../tools/cycles_shading_terminator_oracle.hip)
calls original Cycles 5.2.1 setup/sample/eval functions and its bump helper.
The host only packs/uploads inputs and prints results; there is no CPU BSDF
reference or expected-value generator. Original revision is
`cb168525138fecc792cc393f94afc39582b0103c`; the probe builds with
`hipcc -parallel-jobs=32 -O3 -ffast-math --offload-arch=gfx1201`.
Exact commands, source/toolchain hashes, definedness, layout, and proof limits
are in [fixture provenance](../../../../tests/data/cycles_shading_terminator.json).

The [six captured rows](../../../../tests/data/cycles_shading_terminator.txt)
contain 216 finite float lanes and 72 integer lanes: reflective and
translucent frequency-one/two pairs, a defined `LABEL_NONE` control, and an
active bump correction at frequency one. Reflective evaluation changes from
0.254647911 to 0.0891074389; its PDF is unchanged. Translucent outputs remain
unchanged. The original bump factors 0.998325288 and 0.970562756 ensure the
disabled projection cannot silently delete bump correction as well.

The [runtime test](../../../../tests/test_luisa_cycles_shading_terminator.cpp)
compares actual BSDF value/PDF/direction/roughness and state with those rows,
using normal finite tolerances and fast math. It parses the diagnostic bump
row but does not clone private bump arithmetic on the host. Enabled mode
admits all six rows; disabled mode admits only the three frequency-one rows,
with the production scene-wide proof tested independently.

The guard-only HIP run [failed in exactly three RGB lanes](guard-only-hip.log)
of positive-frequency evaluation: about **0.0891268 instead of 0.0891074**.
All disabled-projection rows, sample/transmission controls, and integer state
passed. This exposed the pre-existing `shift_cos_in` angle implementation,
not a regression from suppressing the proven-off branch.

Original `kernel/closure/bsdf.h::shift_cos_in` calls
`util/math_fast.h::fast_acosf`: absolute value, a clamp/denormal-crush
partition, square root times a cubic, then sign reflection. The already
existing [cycles_fast_math::arc_cosine](../../../../include/psycles/luisa/cycles_fast_math.h)
implements that same short formula and is already used by hair. The repair
changes only the angle call at `shift_cos_in`; its following COS/division and
the other ACOS consumers are unchanged. There are no new tables, strict-FP
barriers, noinline policy, ULP compensation, or scene heuristics. Normal fast
math/FMA freedom stays enabled. The repair restores the source algorithm;
it does not seek bitwise alignment with the mathematical inverse cosine.

[The final HIP capture](native-fast-hip.log) has zero mismatches for all six
enabled rows and the three disabled rows. The independently retained
guard-only source/runtime and full LLVM before/after/final dumps remain at
`/var/tmp/psycles-shading-pruning-build-xZQ1nk`; this diagnostic capture is
not a benchmark. One earlier observer invocation failed before compilation
because of a plugin/context path, and one recording invocation used the
wrong executable path. Corrected invocations are identified explicitly;
neither startup error is presented as a renderer/compiler defect.

The [complete-kernel LLVM audit](native-fast-acos-llvm-comparison.md) and
[opcode inventory](llvm-kernel-counts.awk) distinguish the visible short
formula from final generated work. Enabled entry instructions grow
**1,287 to 1,319 before optimization**, then shrink **750 to 738 in final
LLVM**. Final blocks, loads/stores, branch/switch counts, and total sqrt/COS
counts are unchanged. Ordinary ACOS was already fully inlined: there is no
missing-inline or indirect-call explanation. The removed fmuladd calls are
LLVM intrinsics, not runtime call overhead. The disabled complete modules
remain identical before/after/final optimization after replacing only their
kernel hash name; no arithmetic or control flow is normalized. These exact
counts were independently replayed, but they are not renderer timings or
evidence that the full SVM/frame was reduced by this amount.

## Checkpoint identity and remaining gates

[evidence.json](evidence.json) pins the archived logs, complete ASTs, fixture,
and implementation identities. Root base is `d1a12b8e` plus this uncommitted
bounded change; SDK is `5c7de2bb9`, with unchanged 98f production libraries.
The final runtime SHA-256 is:

```text
86fe7274ae712134bcba6b6262f6fea7e69263fc73a2998856858b4ee3a950fd
```

| Gate | Final-candidate status |
| --- | --- |
| Full all-thread build | Exit 0, `full-build-4.log` |
| Full host | 186/186, 1.58 s, `full-host-3.log` |
| Complete recording controls | 6/6, `recording-native-fast.log` |
| HIP oracle and adjacent controls | [4/4, 2.23 s](focused-native-fast-hip.log) |
| Full HIP | [193/193, exit 0, 1001.33 s](full-hip.log) |
| Fallback | [195/195, exit 0, 748.61 s](full-fallback.log) |
| Strict native Vulkan | 4/4, exit 0, 3.29 s; six native SPIR-V compilations |
| Fresh 256-spp HIP canaries | 4/4 renders and 60/60 pass comparisons completed |
| All-46-channel finite gate | **3/4; failed** on Classroom, actual 18 lanes / 8 pixels |
| Fresh exports / twelve equal-pass timing pairs | Not run |

The full HIP detail log contains exactly 193 `Test Passed.` markers and is
hashed in `evidence.json`. Its 1001.33 s is a correctness-suite elapsed time:
isolated host compiler builds/replays were allowed to overlap part of this
run. It is not fair renderer/JIT performance timing and is not compared with
older suite durations to claim a speed change. The final runtime remains
`86fe7274…`; the frozen execution package/source lock is unchanged.

Fallback detail independently contains 195 passed-test markers. The strict
Vulkan log records all three native-XIR guards for each of four tests, six
successful SPIR-V compilations, and no DXC/DXIL loader entries. Its full and
detailed logs are pinned in `evidence.json`. Optional libudev/Mesa symbol probes
in loader diagnostics did not prevent test completion. These are backend
correctness gates, not scene rendering or performance measurements.

The earlier full-host 186/186 guard-only result and its runtime
`ad884114…` are historical intermediate evidence, not the final binary.
The existing prospective benchmark plan under
`/var/tmp/psycles-shading-terminator-7owG2a/benchmark-plan.{json,md}` is frozen
at that old identity. [The capture plan](CAPTURE.md) requires a separately
pinned final-candidate execution manifest and new output directories.

The [four-scene capture](PHASE_A.md) records single render observations of
37.1113 / 12.7550 / 13.6912 / 17.5279 s for Barbershop / Monk / Monster /
Classroom, against retained original reference images. These are not new
paired medians or an isolated guard speedup. Classroom's known invalid values
and localized/diffuse-indirect differences remain explicit open correctness
issues. A finite-domain metric, smaller recorded body, or focused oracle pass
cannot certify those remaining integrator-level gaps.
