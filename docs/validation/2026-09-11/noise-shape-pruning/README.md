# Native SVM Noise shape pruning

This experiment preserves the Cycles 5.2.1 SVM words and interpreter. It uses
compiler emission facts to omit Noise dimension/type switch arms that cannot
be selected by any shader in the relevant scene entry domain.

## Proof and implementation

`SVMNodeTexNoise` stores dimensions, noise type and normalization in the native
payload. The compiler collects the first two fields while emitting that exact
payload. Its 20-bit set represents the joint `(dimension, type)` domain;
independent dimension and type unions would admit nonexistent combinations.
Both arms of runtime graph branches contribute through ordinary emission.
Bump emission contributes to the surface ShaderJump entry. Volume and
standalone displacement maintain independent entry facts.

A compiler-generated entry starts with the empty set. A raw or malformed
Noise emission retains all shapes. Images without an entry proof, including
legacy hand-authored entry metadata, default to the full domain. The linker
unions every contributing shader's entry facts; entries with no Noise opcode
contribute nothing. Inert shader-index holes therefore cannot erase a valid
proof or disable a live shape. An advertised Noise opcode with an empty or
out-of-domain shape mask also retains the full domain. Rebuilding the scene
recomputes the set.

The Luisa handler records the same nested switches and runtime payload loads,
with ordinary C++ guards around their cases. Normalization, distortion, color
and factor output validity, inputs, clamp behavior, stack addressing and PC
movement remain unchanged. No new device resource, opcode or dispatch queue is
introduced. Recorded callable caches are keyed by the admitted type set for
their coordinate dimension; AST hashes naturally distinguish generated bodies.

## Measurement protocol

Evidence: `/var/tmp/psycles-noise-shapes-20260911`.
Baseline: root `5e23df7e`, SDK `3bc7ee789` (both published). The baseline
executable and its complete project library set were copied before rebuilding;
all five production hashes in the previous full packed-sampler capture match.
A read-only loader check confirmed all project libraries resolve into the
frozen directory. No old/new project libraries are mixed.

The retained original Barbershop command renders 2048x858, 256 spp, 64 spp per
dispatch, HIP wavefront staged, unchanged seed/frame/integrator and pass set.
The main shader cache is disabled. Builds, tests and image comparisons run
separately from timings. Static ISA counts include dual-issue lanes separately
and are not dynamic issued-instruction counts. Cycles outlines helper bodies,
so an entry-symbol-only size comparison is not an inclusive algorithm cost.

## Results

This is a code-size reduction with effectively neutral render performance,
not a resolution of the Barbershop performance gap.

| Full Barbershop render | Baseline seconds | Candidate seconds |
|---|---:|---:|
| Run 1 | 36.7721 | 36.6133 |
| Run 2 | 36.8019 | 36.6295 |
| Run 3 | 36.9143 | 36.7223 |
| Median | 36.8019 | 36.6295 |

Execution order was baseline-1, candidate-1, candidate-2, baseline-2,
candidate-3, baseline-3. The median difference is -0.47%; these few runs do
not establish a substantial renderer improvement. Main JIT times were
12.4878/12.2854/11.9103 seconds for the baseline and
30.6098/10.7667/11.6375 seconds for the candidate. The first candidate's JIT
was slower; repeated candidate JIT was lower. The main shader cache was
disabled, but the HIPRTC/driver cache was not independently reset, so this
is not an isolated cold-compilation comparison.

A final full-resolution render after the malformed-proof fallback repair
took 36.6173 seconds, with 10.8129 seconds main JIT. Its surface code object
is byte-identical to candidate-1: SHA-256
`496f997cc087063af469d64e62dd197d50a6e89d895f95aef66d0aaa4bf1fdb7`.

### Instruction distribution

The following counts use exact ELF function sizes, exclude padding, count
both lanes of dual-issued instructions, and include every statically reachable
ordinary helper once. These are static code instances, not hardware issue
counters. Full raw and normalized opcode histograms and resolved edges are in
[isa-report.json](isa-report.json); [isa-audit.md](isa-audit.md) explains the
Cycles comparison and the sampled range-reduction instructions.

| Instruction family / resource | Baseline | Candidate |
|---|---:|---:|
| Total static instruction lanes, including helpers | 113,077 | 78,175 |
| Vector XOR | 3,835 | 1,102 |
| Vector bit-align | 2,039 | 495 |
| Vector integer subtract, non-carry unsigned | 3,308 | 729 |
| Conditional mask/select | 5,581 | 3,138 |
| Scalar/vector compares | 7,584 | 4,590 |
| ALU dependency waits/delays | 25,936 | 17,979 |
| Memory wait instructions | 1,907 | 1,953 |
| Scratch accesses | 1,982 | 1,614 |
| Global accesses | 1,192 | 1,187 |
| Native image samples | 20 | 20 |
| Surface code object bytes | 580,672 | 414,056 |
| Allocated VGPRs | 256 | 256 |
| Reported VGPR spills | 337 | 449 |
| Private segment bytes per work item | 2,352 | 2,368 |

Noise shape pruning removes much of the static integer hash and select code,
including unused dimensions and fractal variants. It does not materially
reduce global memory instructions, lower register allocation, or lower the
private-memory allocation. The compiler's spill count increases even though
the total number of static scratch accesses decreases. These quantities
measure different things and cannot be interpreted as dynamic traffic.

Most wait/delay instructions are ALU dependency controls; treating their sum
as memory stalls would be incorrect. Repeated FP32-to-FP64 conversion and
exponent extraction initially looked suspicious, but the sampled Cycles 3D
FBM helper uses the same precise range-reduction algorithm. This does not
justify changing remainder semantics.

The exact entry alone shrinks from 109,597 to 78,004 lanes. Earlier text-only
counts of 109,695 and 78,128 included 98 and 124 padding rows. Including
helpers is also essential on the Cycles side: its `integrate_surface<1979>`
entry is 86,303 lanes, while its unique callable closure is 299,398 lanes.
That closure includes generic runtime alternatives which Psycles can omit
from complete emission facts. Neither total is a measure of the work actually
executed on this scene.

### If-conversion and inlining

The active coroutine path uses `create_coro_pre_distill_pipeline` in
`third_party/LuisaCompute/src/coro/coro_compile.cpp`; this custom pipeline
does not run Luisa's if-conversion pass. The default pipeline's if-conversion
entry is therefore not a relevant switch for this renderer path.

An earlier exact-binary replay used the production 576,224-byte HIP bitcode
capture with the production HIPRTC link options. Disabling generic LLVM
simple, diamond, forked-diamond, triangle, false/reverse variants and early
if-conversion produced the same 654,656-byte code object, SHA-256
`be9ee61b90a64f92c83e5980fadd0d62bec4985ccfc262b466217207396c6859`.
Inputs, helper and exact commands remain under
`/var/tmp/psycles-isa-cause-audit-20260911`, including
`replay_binary_commands.sh`. This rules out an effect from those tested
switches; it does not rule out select formation at other compiler stages.

Ordinary HIP callables are not universally marked always-inline. The generic
callable ABI pass strips inline hints, production LLVM optimization uses O3
with its automatic module inliner, and HIPRTC separately receives
`-amdgpu-inline-max-bb=0`. Coroutine lowering preserves ordinary callables.
Noise's factor, color, distortion and octave sites can nevertheless duplicate
helper bodies when automatic inlining removes those boundaries. No inlining
or if-conversion policy is changed here.

The remaining register pressure and scratch accesses are useful investigation
targets. Dynamic instruction overissue remains unmeasured: the attempted
gfx1201 SQ instruction/wait counters returned zero while SQ_WAVES worked,
and those zero counters were rejected as evidence.

## Validation

All targets built with `cmake --build build --parallel 32`. All 186 host tests
and ten selected device tests on each backend passed, sequentially HIP,
fallback, then native Vulkan with all three required guards. The final
malformed-proof repair was followed by another complete build, the same host
and backend checks, and the full original Barbershop render above.

Host regressions retain exact Cycles words across the 20 dimension/type
shapes and both normalization values, test independent surface/bump/volume
entry unions, and preserve conservative legacy, unknown and malformed input
handling. Device regressions cover all 20 singleton shapes, a mixed shape
set, both normalization values, cursor movement, clamp/output validity and
the original PC-loop evaluator. The recorded 3D FBM test module falls from
34,475 to 8,372 XIR instructions; every singleton and the mixed set have
explicit instruction/loop reduction checks. Generic oracle coverage remains.

All 15 linear image passes are finite. Baseline-to-candidate aggregate RMSE
is 2.64015e-6, compared with 2.43166e-6 between baseline repeats and 4.05281e-6
between candidate repeats. The maximum baseline-to-candidate absolute change
is 0.00780939; the corresponding repeat maxima are 0.00507677 and 0.0222305.
Per-pass signal scales and metrics are retained in
[image-reruns.json](image-reruns.json).

The official comparator verified the same Cycles build identity and all 46
channels for the 15 requested passes. Combined RMSE against Cycles is
0.0023104919 before and 0.0023104921 after; Normal RMSE is 0.00050519628 and
0.00050519622. Existing indirect-light differences remain: candidate relative
RMSE is about 7.12% for diffuse indirect and 11.40% for glossy indirect.
This patch does not claim to resolve those differences. Combined and Normal
triptychs were visually inspected; all 15 triptychs and full comparisons are
in the evidence directory's `image-validation` subdirectory.

This checkpoint preserves a smaller native SVM module and a reproducible
instruction-distribution diagnosis. Multi-scene correctness and HIP
performance parity with Cycles remain open.
