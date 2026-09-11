# Surface instruction distribution and register-pressure probes

The strongest new code-generation witness is long-lived frame addresses:
the Barbershop surface continuation computes and spills 45 output pointers
before material evaluation, then reloads them at writeback. This reserves
360 bytes of private scratch per work-item for addresses alone. It is a
specific optimization candidate, not yet a measured explanation of the
rendering gap.

This report follows the [inclusive ISA audit](../noise-shape-pruning/isa-audit.md).
Static instruction counts are code instances, not dynamically issued counts.
The attempted gfx1201 instruction/wait counters returned unusable zeros;
no dynamic overissue ratio is asserted.

## Identity and method

Root: `670de2e4c3a2b97121200115cb861c699240e652`.
Luisa: `3bc7ee7897c5de25f77ab142dd77dc37b747f0be`.
Evidence: `/var/tmp/psycles-register-pressure-25tyjqu3`.
The frozen baseline executable and project libraries, their hashes, exact
command and environment, per-run source diffs, LLVM, bitcode and code objects
are retained there. The full original Barbershop scene uses HIP, 2048x858,
256 spp, 64 spp per dispatch, wavefront staged, surface block size 128 and
the existing sorting/direct-light queues. The main shader cache is disabled;
the HIPRTC/driver cache is not independently reset. Builds and heavy analysis
do not overlap timed renders.

Match artifacts by symbol and hash, not dump counter. In `baseline-capture`,
the surface is LLVM/bitcode **16** but ISA object **12**; bitcode 12 is a
different traversal kernel. Surface symbol: `kernel_5779fbe6d0fbd100`.
Its baseline 414,056-byte code object has SHA-256
`496f997cc087063af469d64e62dd197d50a6e89d895f95aef66d0aaa4bf1fdb7`,
identical to the published Noise-pruned candidate.

## Instruction distribution

These percentages use the same ELF-size-bounded, dual-lane-aware parser as
the preceding audit. The Cycles entry is `integrate_surface<1979>`;
its inclusive scope contains 137 unique functions, versus two for Psycles.
Those scopes contain different retained alternatives and call boundaries,
so they are not a matched count of work executed on this scene.

| Instruction family | Psycles entry | Cycles entry | Psycles inclusive | Cycles inclusive |
|---|---:|---:|---:|---:|
| Total static lanes | 78,004 | 86,303 | 78,175 | 299,398 |
| Conditional masks/selects | 3.99% | 1.54% | 4.01% | 3.18% |
| Scalar/vector compares | 5.88% | 3.81% | 5.87% | 5.28% |
| Vector integer/hash family | 11.67% | 9.57% | 11.69% | 10.99% |
| ALU dependency waits/delays | 22.99% | 24.50% | 23.00% | 24.72% |
| Memory wait instructions | 2.50% | 3.35% | 2.50% | 3.14% |
| Scratch accesses | 2.07% | 0.85% | 2.06% | 0.30% |
| Generic flat accesses | 0.00% | 0.39% | 0.00% | 1.06% |
| Global accesses | 1.52% | 2.26% | 1.52% | 1.68% |

The largest apparent select/compare differences shrink when Cycles' helpers
are included. The current Psycles code is no longer statically larger than
the Cycles entry. Earlier pruning removed many unused Noise/hash paths but
barely changed render time; their static size was not proportional to their
runtime cost. The broad scalar/ALU-wait distribution is not unusually high
in Psycles. Conditional masks also implement value selection, min/max and
other expressions; their presence alone does not identify converted `if`s.

Raw and normalized opcode counts, including the category regular
expressions, remain in the linked audit's `isa-report.json` and
`isa_audit.py`. The integer/hash class includes address arithmetic and is
not exclusively procedural noise. Scratch and flat counts need the memory
scope qualification below.

## Rejected policy probes

| Configuration | Full render seconds | Surface VGPRs | VGPR spills | Private bytes |
|---|---:|---:|---:|---:|
| Frozen published baseline | 36.5768 | 256 | 449 | 2,368 |
| Maximum 192 registers, run 1 | 39.2840 | 192 | 698 | 2,768 |
| Maximum 192 registers, run 2 | 39.2149 | 192 | 698 | 2,768 |
| Module inliner default threshold 0 | 36.5674 | 256 | 455 | 2,368 |

The register cap changes the shared path-shader option, so several other
shaders change too. Its approximately 7.2–7.4% regression is not a clean
measurement of a surface-only cap. Both full runs failed the performance
criterion and the temporary change was reverted.

The lower default inline threshold reduces the surface code object to
395,664 bytes but does not reduce register allocation or private size.
Including retained helpers, its code falls from 78,175 to 75,544 static
lanes. The main entry alone falls from 78,004 to 74,064, but this excludes
the additional `callable.16` body retained by the experiment.
One effectively equal render is insufficient evidence of a speedup. The
temporary policy change was also reverted. Other inline thresholds and
target-specific cost adjustments remain active; threshold zero does not
mean every positive-growth inline is prohibited.

The coroutine compile profile does not identify the large locals as
coroutine-frame candidates. It scans 27,570 allocas, classifies 26,948 as
nonreplayable, rejects 25,713 as scope-local projections, and promotes 528
(only 9 nonreplayable, 3,900 bytes). The `[60 x [4 x i32]]` alloca is
`ClosurePool::_storage`, sized from the scene closure high-water mark; the
`[33 x float]` alloca is the dynamically indexed SVM stack. The former escapes
through `svm_eval_nodes`, and the latter cannot be field-scalarized while
preserving arbitrary SVM offsets. They are therefore not safe generic targets.
The unresolved spill pressure is in compiler-generated continuation SSA and
aggregate state after these ordinary locals are accounted for.

Both diagnostic patches and build logs are retained in the evidence
directory. These probes completed the original full render but were not
promoted to validated rendering changes: their output images were not
compared against the Cycles oracle, and no full backend suite is claimed.
The restored source was rebuilt with `cmake --build build --parallel 32`.

## Current SDK backend gate

After restoring the diagnostic policy, the exact active root/SDK pair was
validated with a freshly configured driver at
`/var/tmp/psycles-sdk3bc7-gates-20260911` (the old handoff driver was pinned
to a different SDK and was rejected). The all-thread build completed 46/46
tasks. Host controls passed 6/6, HIP passed 17/17, fallback passed 17/17,
and strict native Vulkan passed 17/17. Vulkan logs contain 1,023 successful
SPIR-V compilations and no DXC/DXIL matches. The machine-local
`results.json` records the exact library hashes and complete logs.

## Offline raw-buffer probe

`/var/tmp/frame-raw-lowering-repro` contains an exact LLVM transformation and
its reproduction notes. It replaces 68 scalar frame loads in the captured
surface module with `llvm.amdgcn.raw.buffer.load.{i32,f32}` calls using the
descriptor format already emitted for HIP raw atomics. `llc -mcpu=gfx1201
-O3 -verify-machineinstrs` succeeds. Static code length falls from 313,304 to
312,312 bytes and private allocation from 2,368 to 2,176 bytes; VGPR, SGPR
and occupancy metadata remain 256, 107 and 5. The probe omits stores,
aggregates, booleans and full semantic validation, so it is evidence for the
direction only and is not a production compiler change.

The current XIR resource operation does not carry frame-buffer provenance.
The proposed implementation therefore needs an internal marker from the
coroutine frame helpers through AST→XIR metadata, with raw lowering gated to
marked scalar operations. Ordinary byte buffers must retain the existing
64-bit pointer path. The required regression covers capacities 1, 3 and 37,
conditional dormant-field preservation, packed-word read/modify/write and
all non-HIP backends before any `next` publication.

## Frame-address witness

The unoptimized surface LLVM constructs frame addresses at the individual
load/store operations. Optimized LLVM merges equivalent expressions across
the inlined material body. Of 58 GEPs from the selected frame-buffer base,
46 are reused by final writeback and 12 are load-only. Definitions near
lines 859–1129 of `baseline-capture/hip_kernel_final_16.ll` are reused by
stores around lines 50903–50950. Source-line distance is only a locating aid,
not an executed-instruction count.

The ISA contains a repeated address-spill sequence. One example, at
`0x377C–0x3790`, computes the pointer for an existing frame field:

```text
v_add_co_u32        v26, s2, s80, v83
s_wait_alu         0xf1ff
v_add_co_ci_u32    v27, null, s81, 0, s2
scratch_store_b64  off, v[26:27], off offset:1540
```

Writeback at `0x65010` reloads the field value from scratch offset 1396,
then at `0x6501C` reloads this address from offset 1540, waits for loads,
and stores the value through that pointer. Adjacent fields repeat the same
pattern with address slots 1548 and 1556. There are 45 address-spill pairs
in the prologue range `0x355C–0x3B18`.

The entry loads already use scalar buffer base plus 32-bit vector offset.
The late stores instead consume full 64-bit pointer pairs kept across the
material body. A possible fix would keep address arithmetic local to memory
operations or preserve the scalar-base/offset form through allocation.
It must preserve offset arithmetic, field liveness, conditional old values,
packed-word read/modify/write and exact memory semantics. Simply dropping
writeback fields or adding opaque/volatile barriers is not justified.

Cycles' `kernel_gpu_integrator_shade_surface` metadata reports 192 VGPRs,
2 VGPR spills and 6,976 private bytes. Its larger private segment despite
fewer reported spills demonstrates why private bytes and spill counts must
be kept separate. Cycles also uses generic `flat_*` instructions, which can
access private storage; comparing only `scratch_*` opcode counts would omit
part of its private-memory traffic.

## Inlining and if-conversion scope

Luisa optimization absorbs the top-level SVM evaluator into the continuation.
Several generated callable helpers and `cycles_signed_noise_3d` remain in
the final LLVM capture and disappear later inside HIPRTC. An unchanged dump
around one HIPRTC legacy always-inline pass does not locate or rule out
those earlier HIPRTC transformations.

The active custom coroutine pre-distill pipeline does not run Luisa's
if-conversion pass. A fresh replay of the current exact surface bitcode
also produces byte-identical output with all eight tested LLVM machine
if-conversion variants disabled. Both production and disabled outputs are
414,056 bytes with the baseline SHA-256 above. This rules out those
switches for this input; selects can be introduced by other transformations.

Reproduce with the retained helper, which always includes the production
`-amdgpu-inline-max-bb=0` option and passes a separate `-mllvm` for each
additional option:

```sh
cd /var/tmp/psycles-register-pressure-25tyjqu3
inline-audit/link_ctx_current baseline-capture/bitcode/hip_kernel_16.bc \
  inline-audit/current-no-ifcvt.co \
  -disable-ifcvt-simple -disable-ifcvt-diamond \
  -disable-ifcvt-forked-diamond -disable-ifcvt-triangle \
  -disable-ifcvt-simple-false -disable-ifcvt-triangle-false \
  -disable-ifcvt-triangle-rev -disable-early-ifcvt
```
