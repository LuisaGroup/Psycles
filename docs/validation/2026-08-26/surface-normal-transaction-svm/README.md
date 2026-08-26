# Single-stream surface-normal SVM transaction

This checkpoint replaces the duplicated automatic-normal and endpoint-root
interpreter invocations with one Cycles-style bytecode transaction. It is a
data-model change, not an inlining hint: the final optimized HIP LLVM contains
one top-level interpreter loop where the preceding production revision
contained two.

The Cycles oracle is Blender 5.2 commit `fbe6228777e7`. Its
`intern/cycles/kernel/svm/svm.h:102` has one `svm_eval_nodes` loop, and
`NODE_CLOSURE_SET_NORMAL` inside that stream commits a stack value through
`svm_node_set_normal` (`kernel/svm/closure.h:1519`). Psycles retains its typed
SoA slots and Luisa-generated handlers; it adopts the single ordered stream,
not Cycles kernel text.

## Formal model

Let `P` be the incoming surface point, `L` the typed local banks, `N` the
topologically closed automatic-normal prefix, `R` the independently closed
endpoint program, and `A(P)` the exact displaced/undisplaced point selected by
the material policy. The previous semantics were

```text
(L1, _) = eval(N, A(P), L0)
n       = read(normal_output, L1)
(L2, Q) = eval(R, commit(P, n), L1)
```

The new bytecode is `N ; commit_normal(normal_output) ; R`, interpreted by one
loop. Sequential equivalence follows by induction over the instructions of
`N` and `R`: ordinary instructions and their typed reads/writes are unchanged;
the boundary reads the same `normal_output`, changes only the shading normal,
restores the endpoint geometric fields, and precedes every instruction in
`R`.

Typed storage plans for `N` and `R` may share physical slots. This is sound
because the commit consumes the sole required `N` live-out before `R` starts;
there is no `N` value live across the boundary. Parameters remain immutable
entry values and require no local initialization.

The compiler and verifier enforce these proof obligations:

- `N` and `R` are independently compatible, topologically closed schedules.
- A transaction cannot contain a nested transaction and contains at most one
  normal commit.
- The commit has no operand or metadata range and reads a vector address.
- Definite initialization is a forward must-property. Parameters are defined
  at entry; a local becomes defined only after an ordinary instruction writes
  it. All operands and the commit output must be defined before their read.
  Reads are checked before result definition, preserving legal dying-operand
  slot reuse.
- Unknown program flags are rejected, and the undisplaced-evaluation flag
  requires a commit.
- The immutable-variant and Bump-call side streams carry an invalid entry at
  the commit. Bump height programs contain no commit.
- Emission receives the prefix only when dependency analysis proves that its
  root observes the automatic normal. Closure lowering continues to use the
  root's original typed addresses; no closure or texture value is baked.

Sparse definite-initialization sets make verifier cost proportional to emitted
bytecode rather than to untrusted declared bank capacity.

## Permanent regressions

`psycles_surface_program_metadata_tests` now covers:

- a consuming boundary where normal and root deliberately reuse vector slot
  zero;
- a parameter-only normal prefix (zero ordinary prefix instructions);
- rejection of incomplete execution inputs, a second commit, unknown flags,
  and an uninitialized local commit;
- correct Bump edge relocation across the inserted commit.

The compact runtime regression checks every root and height program, all
parallel semantic side streams, automatic-normal presence per topology, and
all Combined, Normal, Albedo, light-pass, BSSRDF, and Bump preparation results.

Commands, all passing:

```sh
cmake --build build \
  --target psycles_surface_program_metadata_tests \
           psycles_luisa_compact_surface_preparation_tests \
           psycles_render_blender_scene \
  --parallel 32

./build/psycles_surface_program_metadata_tests
./build/bin/psycles_luisa_compact_surface_preparation_tests fallback
./build/bin/psycles_luisa_compact_surface_preparation_tests hip
LUISA_VULKAN_USE_XIR=1 \
LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
LUISA_VULKAN_DISABLE_DXC=1 \
./build/bin/psycles_luisa_compact_surface_preparation_tests vk
```

The strict Vulkan canary produced five native optimized SPIR-V modules
(135506, 19193, 6029, 12706, and 185179 words) and did not load DXC.

## Final-IR and production evidence

LLVM LoopInfo was run on the final optimized main path kernel:

```sh
opt -passes='print<loops>' -disable-output MAIN.ll
```

The transaction artifact was produced with the same cold-cache production
command as the preceding revision (the positional renderer arguments are kept
verbatim so this observation can be repeated):

```sh
PSYCLES_DISABLE_SHADER_CACHE=1 \
AMD_COMGR_CACHE_DIR=/var/tmp/psycles-svm-transaction.eRuSeZ/comgr \
LUISA_DUMP_HIP_ISA=/var/tmp/psycles-svm-transaction.eRuSeZ/isa \
LUISA_DUMP_LLVM_IR=1 \
build/bin/psycles_render_blender_scene \
  /home/mike/Projects/psycles-benchmarks/barbershop-480p-64spp/export \
  /var/tmp/psycles-svm-transaction.eRuSeZ/out.exr hip \
  640 480 64 64 - 0 0 0 0 64 - 1 0 wavefront-staged \
  32 32768 32 1 1 0 4 2 4096 131072 0 0 1
```

The old `callable.6` has two depth-one loops; the new `callable.6` has one.
The result cannot be explained by source-level callable sharing or a test-only
kernel: both inputs are the dumped final LLVM from the 640x480, 64 spp
Barbershop production render.

| Metric | Before | Transaction | Change |
|---|---:|---:|---:|
| Root programs | 567 | 378 | -33.33% |
| Total programs | 671 | 482 | -28.17% |
| Final main LLVM | 4,443,821 B | 4,054,798 B | -8.75% |
| Final main LLVM lines | 71,806 | 65,354 | -8.99% |
| Main callable lines | 14,400 | 9,745 | -32.33% |
| Main HIP ISA | 472,912 B | 430,032 B | -9.07% |
| Main LLVM codegen | 1373.72 ms | 1303.04 ms | -5.14% |
| Main COMGR link | 3404.47 ms | 2916.21 ms | -14.34% |
| Render-only | 3.27005 s | 3.25630 s | -0.42% |

Codegen/link and render timings are one cold paired observation, so they are
reported as directional evidence, not a statistically established speedup.
The structural LLVM/ISA reductions and one-loop LoopInfo result are exact.
Full machine-readable values and artifact paths are in [metrics.json](metrics.json).

## Numerical and visual inspection

The before/after renders are independent 64 spp stochastic runs. They are not
expected to have an exact hash, but their pass means and spatial structure must
remain stable. The full 15-pass report is
[barbershop-before-after.json](barbershop-before-after.json).
That report uses the existing differential-tool schema, where fields named
`cycles` denote the reference image; `reference_label` records that the
reference in this checkpoint is the preceding Psycles revision, not Blender
Cycles.

- [Combined](triptychs/combined.png): RMSE 0.01077; luminance mean ratio
  0.99876. Geometry, textures, cabinet/floor boundaries, and lighting structure
  match by visual inspection; the amplified difference is sparse sampling
  noise and isolated high-energy paths.
- [Normal](triptychs/normal.png): RMSE 0.002085; channel-mean ratios are within
  0.012% of one. No normal island, orientation, or bump boundary moved.
- [Diffuse Color](triptychs/diffcol.png): RMSE 0.002525; luminance mean ratio
  0.99989. The texture/UV structure is unchanged.

All remaining pass triptychs are in [triptychs](triptychs). Environment and
both volume passes are bit-identical for this scene. The same ten unavailable
optional image bindings occur before and after, so this checkpoint establishes
an SVM structural no-regression against the prior Psycles render; it does not
use this asset snapshot to claim new absolute Cycles parity.

Triptych SHA-256 values for the three manually opened images are:

```text
b36a5f0292912cb6544d37caf2ab15d3830ca1471a62068f9aa4830d9343192e  combined.png
fb3fb3c5cc62cd1c6565d9da8d79585fd85829751e576e166c9478537b5866b9  normal.png
81cc86be111308ba5f0355ac77b7b080c85bb13bc2631ca622e0bc5a488015a3  diffcol.png
```
