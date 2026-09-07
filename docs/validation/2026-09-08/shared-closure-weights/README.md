# Shared closure weights: sum incoming paths, do not multiply them

Base: Psycles `ddf1b92a`; Luisa remains `fb315d27d`. Cycles source:
`cb168525138fecc792cc393f94afc39582b0103c`. Evidence directory:
`/var/tmp/psycles-native-volume-svm-06XnDX`.

## Formal cause and minimal counterexample

For a shared closure C, let each incoming closure edge contribute weight
w_i after the Mix nodes along that path have attenuated it. Cycles
`ShaderGraph::transform_multi_closure` connects C to the **sum** of these
weights. Its newly created MathNode has the default `NODE_MATH_ADD`, as
declared in `scene/shader_nodes.cpp`.

Psycles instead constructed a synthetic Math with MULTIPLY. Its Math opcode
decoder also independently forced every synthetic Math to MULTIPLY, ignoring
the node's Operation property. Merely fixing the graph property still failed
all four new word regressions. The fix restores addition in the graph and
uses the ordinary property decoder for synthetic Math too; it does not add
an opcode-specific or scene-specific exception.

The smallest source counterexample is Add(C,C): Cycles gives C weight 2,
whereas Psycles gave weight 1. For the nested Mix in Barbershop, the shared
transparent closure should receive `(1-c) + c*t`. The incorrect product was
`(1-c)*c*t`, which is zero for either value of the camera predicate c.
This removed a real transparent continuation, not merely a display pass.
Closure sharing is retained; no graph/geometry deduplication rule is changed.

## Original oracle and permanent regressions

`tools/create_cycles_shared_closure_probe.py` creates four original Blender
graphs: surface Add, nested surface Mix, volume Add and nested volume Mix.
The local word images are extracted from the original Cycles dump without
changing anything except the three shader-entry addresses. Their lengths
are 22, 58, 23 and 54 words. Before the fix, each differs from Psycles at
exactly one word: Math opcode 2 (MULTIPLY) instead of 0 (ADD), at local words
5, 14, 6 and 15 respectively. After the fix all four complete images match.

Word fixture SHA-256:
`ef6c9cfabddb3d08b2e3dda914c8539c46e93edae72d016e6d3027d3c68ba070`.

`tools/cycles_svm_shared_closure_oracle.hip` runs those original words through
the original GPU interpreter. Forty-eight cases cross the four graphs with
camera/diffuse/glossy/shadow visibility and closure capacities 0/1/64. The
Luisa regression compiles the same high-level graphs, executes complete
surface/volume streams, and checks flags, allocator count/remaining space,
closure type/weight/sample weight, transparency/extinction, emission and PC
termination. It uses compiler-derived node masks and stack size. One ULP is
allowed in float comparisons; discrete state and PC remain exact. This is
not a CPU reference renderer or a fabricated expected stream.

State fixture SHA-256:
`0d6fb9d9e0e9de18e05192d9fec32d0f6db66393ac4e6495f805940f60fd70bc`.

Both the host word regression and HIP runtime regression fail before the
fix (`shared-closure-host-before.log`, `shared-closure-runtime-before-hip.log`).
They pass after the two-part generic correction. All builds use 32 threads.

| Validation | Result | Seconds |
| --- | --- | ---: |
| Full HIP | 177/177 | 123.27 |
| Full fallback | 178/179 | 190.55 |
| Host | 152/153 | 13.41 |
| Strict native Vulkan focused | 5/5 | 18.84 |

Fallback retains only the already recorded sample-dispatch film `light_ng.z`
trace difference; no arithmetic/tolerance changes were made to that test.
Host retains only the four existing source-size violations (the modified
native node file decreased from 2029 to 2026 lines). Vulkan uses XIR->SPIR-V
with DXC disabled and native compilation required. Tests cover shared
closures, failed images, volume nodes, closure pool and a full volume path.

## Original Barbershop, 2048x858 / 256 spp

Same original scene and bundle as the failed-image checkpoint, exact Blender
build metadata, fixed samples [0,256), seed/frame/effective seed unchanged,
native fast math, staged wavefront with separate surface NEE queue. No
profiler, CPU build or concurrent GPU job runs during the timing canary.

| Relative RMSE | Before | After |
| --- | ---: | ---: |
| Combined | 0.13641207 | 0.01079813 |
| DiffCol | 0.12181950 | 0.00159745 |
| Normal | 0.07101407 | 0.00091640 |
| Emit | 0.31933805 | 0.00021197 |
| DiffInd | 0.26861247 | 0.07450770 |
| TransDir | 0.99995996 | 0.02814885 |

All 46 output channels are finite. Combined mean luminance is 0.9999533 of
Cycles. Scene compile / main JIT / render-only times are
15.4390 / 27.5668 / 49.5599 s. The main shader cache is disabled but auxiliary
caches retain their normal policy. The shorter single JIT observation versus
95.2809 s is not a controlled compiler-speedup attribution. Rendering is not
faster than the incorrect 48.0181 s run; the missing continuation now exists.
There is no claim that the full performance target is achieved.

At film pixel (1260,616), sample 0/256, both renderers now traverse the same
four recorded object/primitive/shader events, with closure counts 1,2,1,1 and
selected types 30,30,2,30. All 45 recorded random fields compare exactly.
The full trace comparison still has 37 failures: small geometric float
differences and a later NEE choice of adjacent emitter triangles
(111140/9500700 in Cycles versus 111139/9500699 in Psycles), plus downstream
light/shadow fields. The sampled-emitter difference is not yet diagnosed;
neither RNG alignment at this pixel nor near-equal mean energy proves whole
scene path equivalence. No strict-FP or software intersection workaround is
introduced.

The frame remains 896 B / 153 fields / 6 stages. Its captured layout includes
four 4x4 matrices totaling 256 B; their provenance and cross-stage lifetime
need a separate static compiler/application audit before changing storage.

## Four-scene HIP canaries and Cycles baseline

All four original scene bundles were rendered again after the correction,
at 256 spp. Each output has 46 finite channels. Native fast math and the
staged scheduler remain enabled, and no profiler or concurrent GPU job was
used. These are single observations, not paired benchmark medians.

| Scene / extent | Scene compile s | Main JIT s | Render s | Frame / stages | Combined / DiffCol / DiffInd rel. RMSE |
| --- | ---: | ---: | ---: | --- | --- |
| Monk / 1440x1080 | 4.58993 | 64.5614 | 13.9186 | 220 B / 4 | 0.01240020 / 0.00054592 / 0.12881657 |
| Monster / 1080x1080 | 1.52133 | 75.4817 | 15.1713 | 284 B / 6 | 0.00547924 / 0.00010994 / 0.02552799 |
| Classroom / 1920x1080 | 2.1331 | 58.1559 | 18.9070 | 264 B / 5 | 0.00353348 / 0.00010236 / 0.17820336 |
| Barbershop / 2048x858 | 15.4390 | 27.5668 | 49.5599 | 896 B / 6 | 0.01079813 / 0.00159745 / 0.07450770 |

Production Blender `9e2066aef7ef`, RX 9070 XT HIP with hardware ray tracing,
fixed sampling and no denoise/adaptive sampling supplies the reference
images. Source bundles are `barbershop-export` and `monster-export` under the
evidence directory, `/var/tmp/psycles-fullres-hip-20260830/exports/lone-monk-current`,
and `/var/tmp/psycles-multiscene-52-Vw3BkE/classroom/export`. The command uses
`wavefront-staged`, 32-wide main blocks, frame capacity 1048576 and the
separate surface NEE queue; scene seed/frame settings are preserved.

A fresh, profiler-free original Barbershop render at the same 2048x858/256,
seed 0, takes **28.7062 s in the Cycles main loop**, not its 39.273 s Python
render call. The current Psycles canary is therefore about **72.6% slower**.
The log confirms HIPRT and hardware ray tracing. Fresh-versus-previous
original Cycles images have Combined relative RMSE 0.00023581 and DiffInd
0.00261939; they are not bit-identical. The table above retains comparison
against the previous original golden, with matching Blender build metadata.

Earlier fresh Cycles main-loop medians are 13.4429 s for Monk and 14.4242 s
for Monster (three runs each), and the fresh Classroom single is 20.7432 s.
Against those observations the current singles are about 3.5% and 5.2%
slower, and 8.9% faster, respectively. These are scope-matched indicative
comparisons, not a completed paired current-revision performance campaign.
JIT timings fluctuate substantially and do not establish an optimization.

Artifacts: `barbershop-shared-closure-*`, `barbershop-shared-1260-616-s0.*`,
`shared-closure/original.svm52`, `shared-closure/original-state.txt`, and
`shared-closure-*.log` in the evidence directory. Other scene outputs/logs
use `{monk,monster,classroom}-shared-closure-*`; the fresh original baseline
is `cycles-barbershop-shared-fresh.{exr,json,log}` and its original-to-original
control is `barbershop-cycles-fresh-control.{json,log}`.
