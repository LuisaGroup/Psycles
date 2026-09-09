# Whole-kernel LLVM comparison: native fast-acos restoration

Capture root for the following relative paths and replay commands:
`/var/tmp/psycles-shading-pruning-build-xZQ1nk`.

Inputs are the immutable complete test modules in `baseline-ir/` and
`candidate-ir/`. Kernel 0 runs the six original-Cycles fixture cases with
frequency correction available. Kernel 1 runs the three identity-frequency
cases with correction omitted by the host guard. These are the original
complete test kernels, not extracted angle-function siblings. Both captures
use HIP/gfx1201 with fast math. This is a code-generation comparison, not a
rendering performance measurement.

`llvm-kernel-counts.awk` inventories actual LLVM instruction opcodes in defined
functions, excludes declarations/metadata/multiline operands, and prints the
complete kernel separately from module-wide imported OCML functions. Example:

```sh
awk -f llvm-kernel-counts.awk baseline-ir/hip_kernel_final_0.ll
awk -f llvm-kernel-counts.awk candidate-ir/hip_kernel_final_0.ll
```

## Complete entry counts

| Kernel measure | Ordinary ACOS baseline | Original fast-acos candidate |
|---|---:|---:|
| Enabled before-optimization instructions | 1,287 | 1,319 |
| Enabled after-optimization / final instructions | 750 | 738 |
| Enabled final blocks | 29 | 29 |
| Enabled final loads / stores | 20 / 44 | 20 / 44 |
| Enabled final conditional/unconditional branch instructions | 26 | 26 |
| Enabled final switch instructions | 2 | 2 |
| Enabled final COS / SIN intrinsics | 4 / 2 | 4 / 2 |
| Enabled final sqrt intrinsics, all forms | 12 | 12 |
| Enabled final non-intrinsic / indirect calls | 0 / 0 | 0 / 0 |
| Disabled before-optimization instructions | 1,206 | 1,206 |
| Disabled after-optimization / final instructions | 675 | 675 |
| Disabled final blocks / loads / stores | 25 / 18 / 44 | 25 / 18 / 44 |
| Generated enabled AMDGPU object bytes (compiler log) | 9,788 | 9,416 |
| Linked enabled code-object bytes (compiler log) | 7,744 | 7,616 |
| Generated / linked disabled object bytes | 8,980 / 7,232 | 8,980 / 7,232 |

The enabled final reduction is 12 LLVM instructions (1.6%). Before optimization,
explicitly recording the short native formula replaces two calls with visible
arithmetic, so the entry grows by 32 instructions. Do not characterize this as
an unoptimized-IR reduction. Both initial modules contain 582 definitions due
to linked OCML; after optimization only one kernel definition survives on each
side. Those dead imported library functions are not executed work.

## Exact changed angle structure

Baseline final LLVM already inlines both ordinary ACOS calls into OCML's
range-split polynomial. There is no missing-inlining or indirect-call defect
here. Each expansion contains eight `llvm.fmuladd.f32` calls, an AMDGCN sqrt,
and extra range/sign selections. Candidate final LLVM contains the original
clamped sqrt-times-cubic in each location; the sampled-angle region shrinks
from 20 to 14 instructions, the evaluated-angle region from 17 to 11.

The whole final opcode delta is: call -14, fadd +4, fcmp -2, fmul +2,
fsub +2, select -4; all other opcode counts are unchanged. The 16 removed
`llvm.fmuladd.f32` calls are intrinsics, not runtime function-call overhead.
The native cubic remains ordinary `fast` multiplies/adds and may contract
normally during backend lowering. Both variants still have the same total
sqrt and COS counts. No strict-FP barriers, FP64 compensation, new memory
accesses, or alternate dispatch structure were added.

## Disabled control is unchanged

The complete disabled before/after/final modules compare identically after
normalizing only `kernel_[hex-hash]` to `kernel_NORMALIZED`. No instructions,
SSA IDs, constants, control flow, declarations or attributes are normalized.
Pairwise normalized SHA-256 hashes:

```text
701afaa816ff51b47c9bf1cc306446b074ed155eabb6447f87d7b676fdbfd211  before_opt_1 (both)
ba751c5ae18f2b67be4910dda2008dcc9f7dd0aaa0e80910fca83494785ac18b  after_opt_1 (both)
6c52f63ec703652cbb8f2117902f2fd2c8cc30fca5528d465f2f306733359c01  final_1 (both)
```

The baseline run has exactly three expected RGB mismatches in the enabled
positive-frequency evaluation, with zero mismatches in the disabled cases.
The candidate run reports zero mismatches for all six enabled and all three
disabled original-Cycles cases. Compilation/link duration is not compared:
these diagnostic captures have different cache/link conditions.

```text
dbcebfe9f7c167f5f0033402a0ea1c0088b73003b5366be55d3d7e4a3ce5e2ae  baseline-ir/hip_kernel_final_0.ll
496424ad4f34d775e550715dfdeb8f3acb32a113ee386be48d58c12d5ddd4ee2  candidate-ir/hip_kernel_final_0.ll
2a3f3847d955b930b454957655fb0af5dd2a383f75d6d147563767497c01cd6e  candidate-ir/run.log
```
