# Barbershop ordinary microfacet callable control

## Result

The current Barbershop microfacet BSDF bodies are already inlined in final
AMDGPU machine code. Adding ordinary callable boundaries to the two host
helpers does not improve compiler decisions here: surface time grows about
6.49 times, with essentially unchanged invocation counts. The experiment is
fully reverted. No inline/noinline marker, register limit or device-math
change is retained. This is a negative control, not a compiler repair or a
resolution of the remaining Cycles performance gap.

| Full Barbershop 2048x858 / 64 spp / seed 0 | A | B | Restored A |
| --- | ---: | ---: | ---: |
| Render wall seconds | 10.5028 | 52.1585 | 10.4971 |
| Surface GPU seconds | 5.904293758 | 38.297915033 | 5.895572170 |
| Closest GPU seconds | 1.680010410 | 1.774238361 | 1.681304056 |
| Surface invocations | 332,307,893 | 332,307,903 | 332,307,895 |
| Surface fixed private bytes | 2,496 | 94,896 | 2,496 |
| Surface VGPRs | 256 | 256 | 256 |
| Coroutine stages / fields / bytes | 6 / 93 / 416 | 6 / 93 / 416 | 6 / 93 / 416 |

These are sequential A/B/A profiler runs on identical scene inputs and pass
settings, without overlapping builds, tests or renders. They are not fresh
paired Cycles timings or the full-resolution 256-spp benchmark. The latest
pre-intervention production multi-scene results are in the
[socket-forwarding report](../svm-group-forwarding/README.md).

## Source boundary and final code

Original `kernel/closure/bsdf.h` groups GGX tags 12/21/25 and Beckmann
13/20/24 into one sample/eval call for each distribution. Psycles emits the
same host helper into the individual DSL cases. Distinct final LLVM branches
may permit useful type specialization; duplication alone is not a bug.

B wraps only the anonymous `microfacet_eval` and `microfacet_sample` bodies
in `$outline_with_name`, loading the selected closure inside that boundary.
This is an ordinary callable with no forced inline/noinline property. The
intervention preserves all tagged extra-payload branches, operations and
outputs. The complete patch and the pre-intervention hypothesis are retained
with source hashes in [results.json](results.json).

Final baseline ELF contains only 1D noise, 4D signed noise and OCML tangent
besides the main surface kernel. B also retains all four named microfacet
sample/eval functions. Its final pre-HIPRTC LLVM passes the 60 x uint4 closure
pool (960 bytes) by value in the demoted argument suffix, and the callees
materialize that aggregate. The generic XIR read-only-reference promotion
replaces a reference by a call-site snapshot and a callee-local store;
legality of that transformation is separate from its profitability.

The main function shrinks from 995,980 to 943,148 bytes, but B adds four
microfacet functions of 60,280 / 55,880 / 58,460 / 57,644 bytes.
Its main scratch-load/store sites grow from 1,469/610 to 4,532/7,731, with
thousands more stores in the outlined callees. These are static sites and
resource metadata, not measured dynamic scratch traffic or stall counters.
Do not attribute the entire measured slowdown to any one static counter.
Do not compare only main-function sizes when the callee set changes. Archived
textual instruction counts included alignment padding; the
[current bounded auditor](../../2026-09-09/hip-native-remainder/README.md)
uses ELF function extents. Timings, resource metadata and function byte sizes
are unaffected by this counting correction.

## Controls and reproduction

B passes 165/165 host tests and 10/10 focused original-HIP closure/microfacet
tests before the complete scene. Both B and restored A have 46 finite image
channels and all 15 pass comparisons. Combined relative RMSE against A is
0.000015926 for B and 0.000013362 for restored A; this is an intervention
control, not original-Cycles parity. Work changes by only ten surface visits
out of 332 million; extra path count cannot explain B's slowdown.

Both rebuilds use 32 threads. Restored source has no production diff, its
runtime library hash matches A, and its entire surface code-object SHA-256
matches A: `c32767d4981d0621072e85fe3c1d13ba464397c02805b5b34bac3e5d246dee76`.
No full HIP/fallback/Vulkan gate is claimed for the rejected implementation.
The unchanged published implementation retains its complete previous gates.

Evidence: `/var/tmp/psycles-microfacet-boundary-bWpsKZ`; directories
`baseline-XKWwaL`, `intervention-DuLKRk`, `restored-lYIxeE`. Each render log
contains the full command. Runtime stage hashes are checked against LLVM and
ELF symbols before inspecting code object 12. Run the existing
`lamp-routing-and-surface/audit_inline_experiment.py` on those three directories
to produce `audit-unscoped.json`, then this directory's `archive_results.py`
with the evidence root and output filename. The latter replaces the old
helper's intervention wording; the old broad inliner-policy experiment and
this source-boundary experiment must not be conflated.
