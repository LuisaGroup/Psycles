# SDK317 fresh HIP scene baseline

This capture uses the current Psycles root (`bdd783c0`) with LuisaCompute
`7df2fcc3a17f0c61ef36011e39b07e96766579b9` from `origin/next`, Blender 5.2.1
build `9e2066aef7ef`, and current exporter
`46254cd23f1b73bf7928be11c19f5726732e412a50975b088cae64bc6bbfe3b3`.
Each pair used the native scene extent, 256 samples, 64 samples per dispatch,
HIP on Radeon RX 9070 XT, fast math, `wavefront-staged`, staged surface sorting,
staged direct-light queue, 32-wide execution and persistent blocks, 32768
workers, and frame capacity 1048576. The main shader cache was disabled.

The fresh exporter bundles were regenerated because the older control bundles
have different scene and geometry hashes. Results below are render-phase
measurements from `benchmark.json`; process setup, compile, and JIT are shown
separately where useful.

| Scene | Cycles render (s) | Psycles render (s) | Psycles/Cycles | Finite passes |
| --- | ---: | ---: | ---: | --- |
| Barbershop (3 runs) | 25.28--25.32 | 36.975--37.030 | 1.456--1.464 | 15/15 each |
| Lone Monk (3 runs) | 13.19--13.20 | 12.329--12.390 | 0.934--0.939 | 15/15 each |
| Monster (3 runs) | 14.21--14.25 | 13.49--13.56 | 0.952--0.958 | 15/15 each |
| Classroom (1 native run) | 17.94 | 17.161 | 0.956 | Combined finite; DiffDir/DiffInd/GlossDir/GlossInd contain the known auxiliary nonfinite lanes |

The Barbershop result is the remaining performance outlier. Its completed
GPU profile recorded 59,762 HIP dispatches and 35.445 s accumulated GPU time
for a 38.602 s profiled render. This profile is diagnostic only and is not
used as a benchmark timing.

A generic scheduler A/B on the identical Barbershop bundle and render settings
rejected disabling the staged direct-light queue: render time increased to
48.464 s versus approximately 37.0 s with the queue enabled. The current
staged queue configuration remains the selected baseline.

The surface continuation's scheduler block size was also probed without
changing the path program. At 640x480/64 spp, 256 threads rendered in 1.80673 s
and 512 threads in 1.81682 s. One native-resolution 256-thread run rendered in
36.8091 s, within the existing 512-thread range of 36.975--37.030 s. The
low-resolution margin and single full-resolution observation are too small to
establish a general gain, so production retains the existing 512-thread
continuation until a repeated cross-scene sweep justifies changing it.

A bounded `rocprofv3 --kernel-trace --scratch-memory-trace --stats -f rocpd`
capture was also run on the current Barbershop export at 640x480/64 spp. The
trace is retained at
`/var/tmp/psycles-current-sdk317-20260910/rocprof-barber-640-1788986418/`.
SQLite reports 5,466 application kernel dispatches and 1,518.3 ms of
application kernel time (the full trace also includes HIPRT build work). The
largest application kernel, `kernel_d84793d13e03f5da`, ran 246 times for
955.8 ms (63.0% of application kernel time), with 256 VGPRs, 128 SGPRs, and
2,384 bytes of private scratch per thread. The next application kernel,
`kernel_378b2c84b694208e`, ran 292 times for 303.0 ms with 144 VGPRs,
128 SGPRs, 3,072 bytes of LDS, and 240 bytes of private scratch. The named
shadow stages were only 51.25 ms (`shade_light_nee`), 39.00 ms
(`intersect_shadow`), and 20.10 ms (`shade_shadow`). Scratch tracing observed
929 allocation/free pairs, a 149 MiB maximum allocation, and 33.76 GiB total
allocated during the run. The large 256-VGPR/private-scratch kernel is the
next material-heavy surface continuation target; the trace alone does not
justify a Luisa patch without same-bundle timing, LLVM/ISA, and Cycles-oracle
regression.

The staged path execution block size was checked on the same bundle at
640x480/64 spp after warm-up: 32 threads rendered in 1.81543 s, 64 in
1.82568 s, and 128 in 1.83668 s. The existing 32-wide setting remains the
measured choice for this workload.

The separate `shade_surface` continuation block was then tested at 128
threads against the production 512-thread setting. With the normal outlined
SVM evaluator, two native Barbershop runs at 2048x858/256 spp rendered in
36.6842 s and 36.7118 s, versus 36.975--37.030 s for the 512-thread baseline.
A native Monster run at 1080x1080/256 spp rendered in 13.4043 s, versus
13.560--13.613 s for its three 512-thread baseline runs. The Barbershop and
Monster outputs each passed all 15 finite-channel checks against their matching
Cycles HIP references. The production continuation block is therefore 128
threads. A separate noinline SVM evaluator experiment retained the same
92-field/416-byte frame but regressed reduced Barbershop renders to
1.92--1.95 s, so the evaluator remains outlined normally.

The surface geometry bridge now reads packed triangle normals only under the
native smooth-triangle predicate. Flat paths retain their zero-initialized
normal lanes, preventing the shadow-terminator consumer from introducing an
eager normal-buffer read. On the same fresh Barbershop bundle at 640x480/64,
the post-change render was 1.78016 s with all 15 channel finite checks passing.
That timing is within the existing continuation-128 spread, so this change is
recorded as a semantic guard rather than a material timing gain.

The forward-hit emission path now materializes world-space triangle vertices
only for sampled-emissive mesh surfaces, leaving non-emissive triangles in
object space until a consumer needs world coordinates. NodeDataView float3
inputs likewise follow Cycles' x-lane stack marker before reading literal y/z
words. On the same fresh Barbershop bundle at 640x480/64 spp, the combined
post-change render was 1.76904 s with all 46 channels finite; this is within the
prior 1.77308 s single-run spread, so no material timing gain is claimed.

The ten native-size pairs currently recorded for these four
current-exporter scenes show Psycles ahead on Monk and Monster, close on
Classroom, and still behind on Barbershop. This note does not claim global Cycles parity. The next
performance work requires a measured, semantics-preserving Barbershop shader
change with complete image, finite, LLVM/ISA, and backend validation before
publication.
