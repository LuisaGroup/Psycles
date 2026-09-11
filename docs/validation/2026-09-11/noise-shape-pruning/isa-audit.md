# Inclusive HIP ISA audit

This is a static code-layout audit, not a measurement of executed or issued instructions. Counts split AMD dual-issue `::` lanes. ELF symbol-size bounds exclude section padding. The resolver uses `s_getpc PC + 4 + signed s_add_co_u32 immediate` and requires an exact symbol start; it never rounds to the nearest symbol.

Reproduce on the evidence host with `python3 docs/validation/2026-09-11/noise-shape-pruning/isa_audit.py`. The input code objects and disassemblies remain at the recorded machine-local paths. Input paths and complete normalized/raw opcode histograms, edges, symbol sizes, and unresolved calls are in [isa-report.json](isa-report.json). `summary.txt` is a short readout.

| Scope | Symbols | Static lanes | Code bytes | Static call sites | Unresolved calls |
|---|---:|---:|---:|---:|---:|
| Cycles integrate_surface1979 entry | 1 | 86,303 | 441,580 | 90 | 0 |
| Cycles1979 unique callable closure | 137 | 299,398 | 1,545,124 | 229 | 0 |
| Cycles Noise entry | 1 | 4,524 | 21,752 | 64 | 0 |
| Cycles Noise unique callable closure | 22 | 19,323 | 94,412 | 75 | 0 |
| Cycles 3D FBM helper | 1 | 1,331 | 6,704 | 0 | 0 |
| Psycles generic packed-pruned entry | 1 | 109,597 | See JSON | 47 | 0 |
| Psycles generic packed-pruned closure | 4 | 113,077 | 576,528 | 47 | 0 |
| Psycles specialized candidate entry | 1 | 78,004 | 409,104 | 7 | 0 |
| Psycles specialized candidate closure | 2 | 78,175 | 410,076 | 7 | 0 |

The previously quoted approximately 4K Cycles Noise body excluded its helper bodies. Noise directly calls the 20 dimension/type variants plus shared `snoise_4d`; the 4D wrappers add 11 calls to `snoise_4d`. The inclusive body is 19,323 lanes, so comparing approximately 35K inlined Psycles Noise against approximately 4K Cycles was not a like-for-like comparison. The full Cycles callable closure is also substantially larger than its flat entry. Neither flat entry nor unique inclusive code size proves runtime overissue: runtime node usage, branch paths, loop iterations, repeated calls, and wave occupancy are not measured here.

The older text-only symbol scope counted padding after the last function: the candidate entry's 78,128 became 78,004 after excluding 124 trailing rows; generic 109,695 became 109,597 after excluding 98; Cycles1979 86,305 became 86,303 after excluding two rows.

## The FP64-looking range reduction is shared with Cycles

The suspicious sequence uses FP64 conversion and exponent extraction, not FP64 arithmetic. Cycles has the same precise fmod/range-reduction expansion:

| Scope | cvt_f64_f32 | frexp_exp_i32_f64 | frexp_mant_f32 |
|---|---:|---:|---:|
| Cycles 3D FBM helper | 6 | 6 | 6 |
| Cycles Noise closure | 84 | 84 | 84 |
| Cycles1979 closure | 108 | 108 | 264 |
| Psycles generic closure | 212 | 212 | 212 |
| Psycles specialized candidate closure | 45 | 45 | 45 |

The full Cycles closure also has 156 `frexp_exp_i32_f32`, accounting for its additional FP32 mantissa-extraction sites. Counts across whole candidate vs one shared FBM helper are not matched dynamic counts.

A matched local sequence starts at Cycles `noise_fbm<float3>` address 0xC9AB4 and Psycles candidate address 0xDD68. Both convert absolute input to FP64, extract exponent, extract the FP32 mantissa, scale by 2^12, subtract exponent 17, and branch into the same reduction loop. Both then multiply by 0x3F27C5AC, round, fused multiply-add 0xBFC35000, add 0x3FC35000, select by sign, and repeat with exponent decremented by 12. Cycles uses `v_ldexp_f32 ..., 12`; Psycles uses `v_mul_f32 ..., 0x45800000` (4096). There are small scheduling/dual-issue differences, but no gross extra algorithm in this sampled path. This evidence does not justify changing FRem precision or claiming that the FP64 conversion pair explains the rendering gap.

The Cycles 3D FBM helper has 427 vector integer/hash lanes, 105 conditional-mask lanes, 126 compares, 323 wait/delay lanes (318 ALU dependency waits/delays), and no global/flat/scratch memory accesses. Its body includes two 3D noise evaluation sites (six coordinate-reduction sequences). The Psycles candidate has all relevant code in its main symbol; an exact isolated FBM3 body cannot be delimited from symbol names alone. No matched whole-core overissue ratio is asserted.

## Scope limits

The graph traverses every `s_swappc_b64` in each reachable symbol, and all 229 Cycles sites resolved exactly. Saved-return/computed-jump `s_setpc_b64` instructions are not traversed as additional callable edges; this script is not a full CFG or dynamic call-graph reconstruction. Counts include runtime-dead alternatives retained in Cycles' generic kernels. The comparison uses the supplied Cycles1979 artifact and the supplied candidate-1 code object, with no new compilation or GPU measurement.

Cycles also has generic flat-address memory instructions; these can address private storage. Comparing only opcodes named `scratch_*` between engines is therefore not a complete private-memory comparison. Neither static scratch counts nor compiler spill counts measure dynamic traffic.
