# Surface ownership: post-continuation IR check

The ShaderData-to-SurfacePoint-to-ShaderData bridge remains a source-level
ownership difference. It is **not evidence of three executed shader-flag
loads**: the current Barbershop optimized surface kernel shares one load
between geometric setup, volume-only classification and population.
This read-only follow-up narrows W1 of the
[whole-surface audit](../surface-semantic-audit/README.md); it neither repairs
the remaining boundaries nor explains the performance gap.

## Captured scope

Psycles `9e180833`, Luisa `6e58928d8`, RX 9070 XT, native HIP fast math.
All six renderer/library hashes remain identical to the
[native-continuation campaign](../native-surface-continuation/results.json).
The same Barbershop command runs at 2048 x 858, 256 spp, staged capacity
1,048,576, with `PSYCLES_DISABLE_SHADER_CACHE=1` and
`LUISA_DUMP_LLVM_IR=1`. No build or other device job overlaps it.

Evidence is retained under `/var/tmp/psycles-surface-ownership-FeArPk`.
The surface entry is `kernel_11eae98514a796c2`, in `hip_kernel_final_12.ll`;
its 33-float SVM stack and twelve 80-byte closure slots identify the native
surface body, rather than assuming a particular dump index is always surface.
The main coroutine is still six stages, 92 fields, 416 bytes. The diagnostic
render completes in 37.6806 s. It is not included in the earlier three-run
median, nor treated as an isolated timing experiment or a new image-parity gate.

## Exact load use-def witness

The captured kernel masks the native shader identity with `541065215`, uses
the 32-byte KernelShader stride, and loads its flags at byte offset 16.
The resulting SSA value is `%.unpack12967.i`, defined at IR line 1324.

| Consumer | Captured final IR |
| --- | --- |
| Setup/backfacing projection | Line 2587, `%_reg_159.i` phi uses the loaded flags. |
| Volume-only routing | Line 3086, `%2102` tests the same value with SD_HAS_ONLY_VOLUME (524288). |
| Population flags | Line 3337, `%2185` ORs that value with the backfacing bit. |
| Population ABI invariant | Line 3344, `%2186` tests the same value for SD_BSDF/SD_BSSRDF (20), followed by llvm.assume. |

The source has explicit KernelShader reads in `setup_cycles_svm_ray_shader_data`,
the pipeline's volume-only branch, and `CyclesSvmPopulatedSurface` construction.
Their same-hit flags consumers share this final load. The repeated shader mask
later in the IR initializes SVM addressing; it is not another flag load.
This is a scoped use-def proof, not a count of all shader-table loads in the
whole module or a dynamic memory-traffic measurement. No compiler workaround
is justified by these three source reads.

## Remaining ownership and control obligations

Geometric setup already computes native P/N/Ng, compact differentials and
parametric derivatives. The bridge expands these into SurfacePoint, adds
eager triangle/normal/material data and world vertices, then reconstructs
ShaderData for the retained population. Neither a host C++ aggregate nor a
source-level read alone establishes surviving GPU storage or extra traffic.

The following have **not** been discharged by the flag-load witness:

- Whether all repeated triangle/normal/object accesses are eliminated, and
  which cached values remain live through the SVM or are cheaper to fetch at
  their native consumer. Normal reads are conditional in native setup, but
  the current triangle bridge requests them unconditionally.
- Whether volume-only, terminated or non-NEE paths execute unnecessary
  material-binding or geometric work in the final kernel.
- W2/S5 data-pass predicates and placement, W6 early forward-MIS rejection,
  W7 proposal finalization and W8 packed lamp-inverse ownership.
- S3 holdout before emission, transparent-glass controls, shadow catcher,
  remaining semantic opcodes, services and the legacy displacement prepass.

These remain separate correctness/performance obligations. In particular,
the three already-CSE'd flag reads must not be used to justify a claimed
Barbershop speedup, and missing holdout support must not be blamed for scenes
with no holdout geometry. No renderer, compiler, inlining or math policy is
changed by this check.

`verify_capture.py` checks the frozen artifact hashes and the function-scoped
load/use witness. The file/SSA identifiers deliberately describe this capture;
they are not proposed as stable compiler regression expectations.
