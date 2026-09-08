# Surface launch geometry: 1024 threads is slower; production stays at 512

Changing only Barbershop's surface continuation from 512 to 1024 threads
reduces VGPR metadata from 256 to 192, matching the original Cycles launch's
count, but makes median surface time **5.72% higher** and render time **3.06%
higher**. The experiment is fully reverted. This is not an inline-policy
change, a production optimization or evidence that fewer registers are faster.

Implementation: Psycles `e79644af` / Luisa `da8fff856`; the preceding
[four-scene validation](../closure-input-guards/README.md) remains current.
No production source or child gitlink change results from this control.

## Independent hypothesis and preconditions

Cycles `cb168525138fecc792cc393f94afc39582b0103c` uses
`GPU_KERNEL_BLOCK_NUM_THREADS=1024` and `__launch_bounds__(1024,1)` for
surface. The matched original trace records 797 launches at 1024 threads,
192 VGPRs and 6976 private bytes. Psycles' inherited continuation override
uses 512 threads, 256 VGPRs and 2480 private bytes. These are queue batches,
not per-path shading counts. Different launch constraints can change register
allocation, spill placement, occupancy and scheduling; metadata alone does
not identify which cost dominates.

The generic scheduler bounds-checks global dispatch index, obtains the true
frame index from the resume queue and operates on that frame. RNG identity
comes from path hash/offset/sample, not thread/block id. The captured surface
IR contains no workgroup-wide barriers or shared storage. Wave size remains
32, with the same SurfaceSortHandler, 490 native keys and one frame partition.
The temporary intervention changes only `execution_block_size` for
`path_transition::shade_surface`, from 512 to 1024. No register cap,
inline/noinline annotation, math, SVM words, feature masks, static allocation
extent or coroutine frame changes.

## Full-scene A/B/B/A

Four sequential HIP profiles use 2048x858, 64 spp, seed 0 and the same
15-pass/46-channel bundle. No agent-launched heavy task overlaps the timed
renders. Each dump-producing process has its own fresh evidence cwd; all
six implementation binary hashes are checked unchanged during each run.
These are intervention controls, not fresh paired Cycles timings.

| Run | Surface GPU seconds | Render seconds | Session initialization |
| --- | ---: | ---: | ---: |
| A, 512 | 5.868351 | 10.5276 | 18.6945 |
| B, 1024 | 6.213614 | 10.8629 | 45.7246 |
| B repeat | 6.212750 | 10.8723 | 18.7604 |
| A restored | 5.885352 | 10.5628 | 19.0069 |

Medians are 5.876851 / 6.213182 surface seconds and 10.5452 / 10.8676 render
seconds. Initialization includes JIT plus setup/baking, not compiler-only;
main shader caching is disabled but downstream caches are not cleared.
First/repeat B are not matched cold-JIT trials.

| Final main-kernel property | A, 512 | B, 1024 |
| --- | ---: | ---: |
| Function bytes | 852836 | 860096 |
| Static instructions, excluding padding | 159972 | 160873 |
| Static scratch load/store sites | 1274 / 622 | 1704 / 727 |
| VGPR / SGPR count | 256 / 107 | 192 / 107 |
| Fixed private bytes | 2480 | 2816 |
| VGPR / SGPR spill metadata | 504 / 65 | 672 / 65 |
| Call sites / image-sample sites | 49 / 20 | 49 / 20 |

Static instruction/spill sites are not executed counts or memory traffic.
All runs retain six stages, 93 frame fields and 416 frame bytes. Surface
visits are 332307892 / 332307892 / 332307898 / 332307895: no material added
shading workload explains the timing change. Other stage counts are archived,
not treated as exact original-Cycles parity.

All 46 channels are finite and all 15 comparisons complete. B-versus-A
Combined relative RMSE is 1.19e-5 / 1.50e-5 and DiffInd 4.70e-5 / 3.69e-5;
restored A has 2.01e-6 Combined and 2.89e-6 DiffInd. These intervention controls
do not waive the much larger original-Cycles residuals.

The original source file is byte-identical after restoration and both
32-thread renderer builds pass. Both A surface `.text` hashes are
`431b702c763b641fcc499690078472dc428fe62f7cc33297a4fab44c59e1e9fe`;
both B hashes are
`b6abe439670d704aadae181daa76d54439315760e28f95781653e1c9ff0d72c3`.
The full original application is recompiled and rerendered after restoration;
the prior full backend gates are not represented as newly rerun here.

## Evidence and next work

Raw evidence: `/var/tmp/psycles-surface-launch-vUy8LQ`. Its formal analysis
predates the intervention. The isolated-cwd runner, source backup, build
logs and exact commands are hashed in [results.json](results.json).
[archive_profiles.py](archive_profiles.py) takes the before, 1024, repeated
1024 and restored directories plus an output JSON; it verifies live block
sizes, restored `.text`, finite channels and all pass controls.

Barbershop's remaining per-surface cost, 164 resource binding identities,
residual DiffInd/shadow differences and general compiler audit remain open.
SVM and microfacet bodies are already inlined in final machine code; neither
missing inline nor this launch-policy difference explains the large gap.
