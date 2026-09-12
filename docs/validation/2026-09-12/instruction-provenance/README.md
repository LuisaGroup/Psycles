# Corrected HIP ISA provenance, 2026-09-12

This is a read-only audit of retained artifacts. No new GPU workload or compiler
replay was run for this audit. All counts below are static code instances;
they do not measure dynamically issued instructions.

## Wrong attribution to retract

`/var/tmp/psycles-register-pressure-25tyjqu3/baseline-capture/bitcode/hip_kernel_16.s`
is Psycles, not Cycles. Its entry at line 12105 is
`kernel_5779fbe6d0fbd100`. Its `codeLenInByte = 313304` is from an offline
LLVM assembly output, not the production HIPRTC capture. Likewise,
`/var/tmp/psycles-switch-table-20260912-a/hip_kernel_12.s` has Psycles entry
`kernel_2427035293d48dc3` and `codeLenInByte = 311056` (line 78195).
Comparing these two files does not compare Psycles against Cycles. The claimed
matching 40 row multiplies, 1069/1072 scratch loads, 389/391 scratch stores,
and 1884/1888 cndmasks must not support a Cycles parity conclusion.

Those `.s` files also retain several helper bodies and use a different compiler
path from production. The table `.s` reports 382 VGPR spills; the actual raw
production code object reports 407. Code-length and spill differences between
these paths cannot be attributed to the switch-table transform alone.

## Verified production artifacts and counts

The real Cycles artifact is:

* `/var/tmp/psycles-lamp-routing-sCLxKs/barber-isa-47Dvhd/cycles-compute.co`
* SHA-256 `58e11f859e0855e2ab13a06a9593084a544c125f8752de6934f3b40976b2327e`
* Disassembly: same directory, `cycles-compute-isa.txt`
* Surface implementation: `_Z17integrate_surfaceILj1979EEiPK16KernelGlobalsGPUiPf`
  at ELF address 0x3f370c, size 0x6bcec (441580 bytes).
* Wrapper: `kernel_gpu_integrator_shade_surface` at 0x45f400, size 2552 bytes.
  Wrapper metadata: 192 VGPRs, 102 SGPRs, 6976 private bytes, 2 VGPR spills.

Raw-frame Psycles artifact:

* `/var/tmp/psycles-register-pressure-25tyjqu3/raw-current-3/isa/hip_isa_12.co`
* SHA-256 `7d0e5f4717404acb6e440f9485f4b498c5d97642197beaae437ec2b48738030f`
* Disassembly: `/var/tmp/raw-current-3.dis`
* Entry: `kernel_2427035293d48dc3`, ELF address 0x1a00, 408716 bytes.
* Metadata: 256 VGPRs, 107 SGPRs, 2192 private bytes, 407 VGPR spills,
  60 SGPR spills.

Frozen pre-raw Psycles surface object:

* `/var/tmp/psycles-register-pressure-25tyjqu3/baseline-capture/isa/hip_isa_12.co`
* SHA-256 `496f997cc087063af469d64e62dd197d50a6e89d895f95aef66d0aaa4bf1fdb7`
* Entry `kernel_5779fbe6d0fbd100`, 409104 code bytes.

Counts were recomputed using the `Module` class from
`docs/validation/2026-09-11/noise-shape-pruning/isa_audit.py`, without executing
the script's output-writing main section. ELF symbol-size bounds exclude
padding; dual-issue lanes are counted separately and opcode suffixes normalized.

| Scope | Symbols | Code bytes | Static lanes | cndmask | cmp_eq_u32 | scratch load | scratch store | flat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cycles1979 entry | 1 | 441580 | 86303 | 1333 | 157 | 570 | 164 | 333 |
| Cycles1979 unique callable closure | 137 | 1545124 | 299398 | 9520 | 1113 | 655 | 232 | 3179 |
| Raw-frame Psycles entry | 1 | 408716 | 77927 | 3116 | 794 | 1037 | 577 | 0 |
| Raw-frame Psycles unique callable closure | 2 | 409688 | 78098 | 3138 | 796 | 1037 | 577 | 0 |

Cycles has 229 statically resolved call sites across the closure; Psycles has 7.
No calls were unresolved by the parser. Cycles includes generic alternatives
and separate helper bodies, so entry counts and inclusive counts are different
scopes. Neither scope gives a matched dynamic workload. Cycles flat accesses
can also address private memory, so scratch opcodes alone do not measure its
whole private-memory traffic.

## Corrected source of the mask chain at 0x4DAC

The previous claim that raw ISA 0x4DAC corresponds to
`node_closure_bsdf_skip` is wrong. That SVM switch has sparse closure opcodes
and merges constants into one scalar result. The actual ISA does something
structurally different:

1. 0x4D84 computes `v1 = v183 << 2`.
2. Four sequences compare `4*i + {0,1,2,3}` against lanes 0 through 11.
3. Each sequence conditionally replaces twelve different destination VGPRs
   with one of four nonconstant input VGPRs: v187, v25, v26, v27.
4. 0x51F8 increments v183, followed by a second row update with sentinel values.

Range [0x4D84, 0x51F0) has **48 cndmask**, **50 vector eq compares**,
46 `s_wait_alu` and 24 `s_delay_alu` instructions. Two of the eq compares
are separate scalar-array entry updates. The remaining 48 compares/selects
implement four dynamic inserts into a twelve-lane vector.

The source chain is `VolumeStack::_identity`, an array of `uint4` rows:

* `include/psycles/luisa/volume_stack.h:53`: `Local<uint4> _identity`.
* `src/luisa/volume_stack.cpp:51`: `_write` stores object, shader, surface_tag,
  and parameter_block as one uint4 at a runtime index.
* `src/luisa/volume_stack.cpp:172`: `_append` writes at `_count`, increments
  `_count`, then writes the mandatory sentinel. This matches the increment
  between the two ISA update sequences.
* `src/luisa/volume_stack.cpp:198`: `_exit` can swap the last row into a removed
  row and clears the last slot, producing other occurrences of the same shape.

The retained XIR has precise correspondence:

* `kernel.36141c5c2087763b.opt.rq.xir:1842`: `_reg_184` is
  `array<vector<u32, 4>, 3>` (`%1357`).
* XIR lines 8180-8182: whole-row GEP and store at `%1536` (the count).
* XIR line 8187: count increment.
* XIR lines 8196-8198: whole-row GEP and sentinel store at the incremented count.
* XIR lines 2024-2026 and 126192-126194 show whole-array state load/store.

Exact baseline LLVM:

* `/var/tmp/psycles-register-pressure-25tyjqu3/baseline-capture/hip_kernel_before_opt_16.ll:25236`
  allocates `_reg_184 = alloca [3 x [4 x i32]]`.
* `hip_kernel_after_opt_16.ll:1132` and `hip_kernel_final_16.ll:1132` already
  have the flattened `<12 x i32>` SSA value. Thus promotion occurs in Luisa's
  LLVM optimization pipeline before HIPRTC, not in the coroutine XIR pass.
* `hip_kernel_final_16.ll:3241` computes count*4; lines 3242,3244,3246,3248
  insert the four entry words at count*4+0..3.
* Lines 3251-3259 increment count and insert -1,-1,-1,0 as the sentinel.
* Further dynamic inserts at lines 3185,3204,50153,50172,50210,50221 correspond
  to exit/append updates in the two surface paths.

Luisa uses system LLVM 22.1.8 (`LLVM_DIR=/usr/lib/cmake/llvm`). Its HIP LLVM
pipeline registers target callbacks and builds the per-module default pipeline
in `third_party/LuisaCompute/src/backends/hip/llvm_codegen/hip_codegen_llvm_impl.cpp:1166`
and line 1188. The upstream LLVM 22.1.8 implementation of
`AMDGPUPromoteAlloca.cpp` recursively multiplies nested array sizes to create
one flat vector (lines 909-933), and emits per-element dynamic insertelements
for partial vector stores (lines 748-755). This is the algorithm seen here.
Source: https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/Target/AMDGPU/AMDGPUPromoteAlloca.cpp

The exact pass ordering within the captured run was not instrumented, but the
before/after IR and upstream promotion implementation identify this conversion;
no sparse-switch source-to-address inference is needed.

## Previous dynamic-array-vector probe did not test this issue

`/var/tmp/psycles-register-pressure-25tyjqu3/dynamic-array-vector-1789143460/run.json`
contains a temporary XIR SROA patch requiring every alloca use to be a GEP with
exactly two indices and the inner lane constant. It rejects whole-row GEPs,
whole-array loads/stores, and escaping uses. `_identity` has precisely those
whole-row and whole-array accesses, so it is excluded. Its surface code object
has SHA-256 `7d0e5f...38030f`, **byte-identical to the raw baseline**. The 33.3632 s
render from this probe therefore says nothing about the proposed volume-stack
storage or vector-insert fix.

## Concrete next experiment

A source-only diagnostic can replace the four-field `Local<uint4> _identity`
with four `Local<uint>` columns while preserving all VolumeStack transitions,
capacity, count, sentinel, and entry fields. Each dynamic component insert then
has only three candidate lanes; a row write should require twelve selects,
and the same three row predicates can be shared across components. This avoids
the current 48-select expansion and does not require profiling-based semantics.
It changes coroutine frame layout, so full output/device checks remain required.

A generic compiler fix could lower dynamic vector inserts using known index
bits, preserving only reachable lanes (here `(index & 3)` is fixed), or preserve
row grouping when promoting nested arrays. A narrowly targeted LLVM pass after
the default optimizer could explicitly scalarize an insertelement into constant
lane extracts/inserts plus only compatible-lane predicates. It must preserve
poison/out-of-range behavior and be validated on the full original module.
Globally disabling alloca promotion is a much broader pressure tradeoff; passing
the disable flag only to HIPRTC is too late because the supplied final bitcode
already contains the promoted vector.
