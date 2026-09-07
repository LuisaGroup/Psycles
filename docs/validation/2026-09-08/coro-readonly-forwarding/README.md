# Read-only uniform forwarding across ordinary coroutine calls

Base: Psycles `9bfb8835`, Luisa `fb315d27d`. Worktree:
`/home/mike/Projects/Psycles-surface-svm`. Evidence directory:
`/var/tmp/psycles-native-volume-svm-06XnDX`.

The generic fix and both permanent regressions are published in Luisa
`9ea3b720f` on `origin/next`; the parent checkpoint advances its gitlink.

## Formal cause

Two independent predicates govern reference-to-value promotion:

1. No use of the candidate reference, including uses in defined callees,
   modifies or escapes that reference's storage.
2. At every call site of the function being promoted, the actual is
   thread-local storage, disjoint from every reference the function may write.

The old pass established (1) only for loads/GEPs, relying on leaf-first
signature promotion to remove calls from pointer use lists. But an inner
call can fail (2), because its actuals are caller reference arguments with
unknown aliases, without invalidating (1). Conflating those proofs loses
transitive read-only information even when the outer call has distinct
local-allocation roots and is safely promotable.

The original Barbershop coroutine demonstrates that exact shape. In the
captured original XIR, entry argument `%405` initializes local `%444` of
type T0 (`RenderKernelParameters`). `path_volume_segment` accepts it as
reference `%173138` and forwards it to `svm_eval_nodes` reference `%444546`.
The latter only reads the camera projection through GEP/load operations,
but remains a reference because other writable actuals have unknown roots
inside the enclosing function. The outer reference therefore used to look
potentially writable too. Coroutine liveness retained the uniform aggregate
as per-path state, including all four 4x4 matrices.

The fix propagates read-only effects through defined direct calls using a
memoized pointer-use traversal. Unknown calls, writes/escapes and unresolved
recursive effects remain conservative. It does not weaken either the
thread-local or cross-argument alias proof. Effect caches are discarded
before the next function is transformed, avoiding stale signature/use data.
No renderer-specific type, field name, lifetime marker or noinline policy
enters the Luisa pass. Normal scalar/vector zero initialization is unchanged.

Temporary XIR export instrumentation was removed after capturing the
original 120 MB module in `barbershop-coro-after-ref-promotion.xir`. Existing
`LUISA_CORO_DUMP_FRAME_LAYOUT` and `LUISA_CORO_SHADER_MAP` supplied live-range
and kernel-name evidence; these do not profile or size local arrays.

## Permanent counterexamples

`test_xir_pass_promote_ref_forwarding` covers one/three wrapper layers,
root aliases, the same pointer forwarded to read/write formals, transitive
writes, shared storage, one unsafe call among safe calls, unknown callees
and recursive forwarding. Each well-formed module is verified before and
after transformation; a second pass must be stable. Before the fix, the
two positive cases fail while the seven conservative controls pass. After
the fix all nine cases pass (51 assertions).

`test_coro_readonly_forwarding` records a real nested ordinary call and
two coroutine boundaries around a mutable accumulator and a 256-byte
immutable uniform aggregate. It checks statically that only scalar/loop
state occupies the frame (at most 16 B of user payload), then runs 257
logical invocations in a 64-frame pool, both AoS/SoA layouts, and repeated
dispatch with changed uniforms. The frame assertion fails before the fix;
afterwards all 1031 assertions pass on both HIP and fallback. No expected shading interpreter
or renderer is implemented by this compiler regression.

## Baseline kernel evidence, not a timing promotion

Original Barbershop at 2048x858 / 64 spp / seed 0, native fast math, separate
surface NEE queue, same production Cycles HIP build `9e2066aef7ef`, RX 9070 XT.
These runs use rocprofv3 kernel tracing and must not replace unprofiled
256 spp wall-time baselines. Shader hashes are mapped using the unchanged
native scheduler's structural-hash map, not inferred from duration/order.

| Stage | Psycles GPU sum s | Cycles GPU sum s | Psycles / Cycles launched lanes |
| --- | ---: | ---: | --- |
| Intersect closest | 1.63828 | 1.42519 | 431532256 / 444428032 |
| Shade surface | 6.70560 | 2.93558 | 332756480 / 332883968 |
| Shade volume | 1.47976 | 0.35962 | 82964160 / 85329920 |

Grid sizes include padded/inactive lanes, so these are not exact active-path
counts. They do not support a factor-of-two excess in surface invocations,
but also do not prove full path equivalence. Surface and volume throughput
are concrete optimization targets. Resource metadata before the fix:
surface VGPR 256 / scratch 2656 B in Psycles versus 192 / 6976 B in Cycles;
volume 256 / 3216 B versus 192 / 10560 B. Scratch allocation alone does not
establish actual spill traffic or explain the speed difference.

Artifacts: `barbershop-kernel-profile-{shared,cycles}/`,
`barbershop-shader-map.log`, `barbershop-frame-static-audit.log`,
`coro-readonly-*.log`.

## Original full-kernel canary after the correction

The full 32-thread Psycles build succeeds. The original Barbershop is
rerendered at 2048x858 / 256 spp with shader caching disabled, fast math and
the same staged scheduler/queue options as before. The main coroutine still
has six stages. Frame size falls from **896 B / 153 fields to 416 B / 93
fields**; all four matrix fields disappear. Ordinary reference promotion
count rises from 115 to 116: one additional outer uniform-aggregate snapshot
is sufficient. No scene-specific local/frame extent is supplied.

Scene compile / main JIT / render-only times are
15.8432 / 76.5101 / **40.8750 s**. This single render takes 17.5% less time
than the preceding 49.5599 s canary, but is still 42.4% slower than the fresh
original Cycles main-loop observation of 28.7062 s. Main JIT includes
compilation-phase logging in this diagnostic run and remains variable; no
JIT improvement or paired performance promotion is claimed.

All 46 channels remain finite. Combined / DiffCol / DiffInd relative RMSE
are 0.01079811 / 0.00159771 / 0.07450762, essentially unchanged from the
pre-correction image. The aggregate before/after RMS difference over all
output channels is 0.00001019; this is not a bit-identical rendering claim.
Artifacts: `barbershop-readonly-forwarding-{hip.log,hip.exr,metrics.json}`.

The other three original scenes were rerendered with the same 256 spp,
native fast math and cache policy. All 46 channels are finite in every
scene, and their frame sizes remain unchanged. Original bundles/goldens
are the same as the [shared-closure checkpoint](../shared-closure-weights/README.md).

| Scene / extent | Scene compile s | Main JIT s | Render s | Frame / stages | Combined / DiffCol / DiffInd rel. RMSE |
| --- | ---: | ---: | ---: | --- | --- |
| Monk / 1440x1080 | 5.01381 | 19.1313 | 13.9315 | 220 B / 4 | 0.01241366 / 0.00054593 / 0.12881629 |
| Monster / 1080x1080 | 1.52980 | 22.6170 | 15.2468 | 284 B / 6 | 0.00547924 / 0.00011006 / 0.02552799 |
| Classroom / 1920x1080 | 2.18309 | 18.8041 | 18.7757 | 264 B / 5 | 0.00353334 / 0.00010232 / 0.17820336 |

Artifacts use `{monk,monster,classroom}-readonly-forwarding-*`. The reduced
JIT observations are not controlled speedup measurements. No GPU profiler,
concurrent GPU task or CPU build ran during any of these rendering canaries.

Full HIP passes 177/177 (209.18 s). Full fallback passes 178/179 (200.33 s),
retaining only the previously recorded sample-dispatch film `light_ng.z`
difference: expected `0xbf1f8bfd`, actual `0xbf1f8a50`. Host passes 152/153
(17.65 s), retaining only the four old source-size violations. The full Luisa
build also succeeds with 32 threads. Its entire registered non-device-specialized
CTest selection passes 140/140 (26.78 s), including the separately checked
69/69 XIR/coroutine selection. The HIP regression run overlapped the CPU-only
Luisa build; its suite duration is validation evidence, not a performance
comparison.

Strict native Vulkan passes the five Psycles focused tests (16.75 s) and
the new full coroutine regression (1031 assertions). Shader caching is
disabled in the latter; the loader log shows native SPIR-V compilation and
no DXC/DXIL library load. All three fail-closed flags are set:
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`.

The standalone Vulkan build initially retained CUDA interoperability and
failed to load `libcuda.so.1` before any shader ran. Setting only that local
CMake build's `LUISA_COMPUTE_ENABLE_VK_CUDA_INTEROP=OFF` resolves the unrelated
configuration issue. No system/toolchain change or compiler fallback was
made. The successful loader audit is
`coro-readonly-runtime-vulkan-native-loader.log`; the initial load failure
is preserved separately in `coro-readonly-runtime-green-vulkan-loader.log`.

Finally, the existing `test_coro_frame_runtime` passes 22 tests / 306
assertions on both HIP and fallback, explicitly retaining ordinary
scalar/vector/array default-zero semantics. The final registered new HIP
and fallback coroutine tests pass 2/2 (0.96 s).
