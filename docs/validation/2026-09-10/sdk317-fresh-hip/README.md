# SDK317 fresh HIP scene baseline

This capture uses the current Psycles root (`fff8c803`) with LuisaCompute
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
| Monster (1 run) | 14.24 | 13.560 | 0.952 | 15/15 |
| Classroom (1 run) | 17.94 | 17.161 | 0.956 | Combined finite; DiffDir/DiffInd/GlossDir/GlossInd contain the known auxiliary nonfinite lanes |

The Barbershop result is the remaining performance outlier. Its completed
GPU profile recorded 59,762 HIP dispatches and 35.445 s accumulated GPU time
for a 38.602 s profiled render. This profile is diagnostic only and is not
used as a benchmark timing.

A generic scheduler A/B on the identical Barbershop bundle and render settings
rejected disabling the staged direct-light queue: render time increased to
48.464 s versus approximately 37.0 s with the queue enabled. The current
staged queue configuration remains the selected baseline.

The full 12-pair campaign is still incomplete, and this note does not claim
Cycles parity for all scenes. The next performance work requires a measured,
semantics-preserving Barbershop shader or queue change with complete image,
finite, LLVM/ISA, and backend validation before publication.
