# Current v3 HIP paired benchmark

Captured 2026-09-11 from root commit `ccdde5786568c899b13d67b9e6c2f20b5003f838`
and LuisaCompute commit `5066ab8f23fd158ca50a967c6a4d4a61c2571f1e`. The
source manifests and rendered images remain in
`/var/tmp/psycles-v3-20260911`; this archive contains timing, identity and
comparison metadata only. No EXR or PFM files are copied here.

The runner was `tools/run_scene_benchmark.py` (schema
`psycles.scene-benchmark.v3`) with Cycles HIP and Psycles HIP
`wavefront-staged`, on an AMD Radeon RX 9070 XT. Each run used the scene's
native resolution, 256 samples in `[0,256)`, 64 samples per dispatch, fast
math, staged surface sorting and direct-light queue, 32-thread execution
blocks, 32,768 persistent workers, 1,048,576 frame capacity, and disabled
main shader cache. The run-specific benchmark manifests retain the exact
commands and output hashes.

| Scene | Cycles HIP render (s), median [min,max] | Psycles HIP render (s), median [min,max] | Psycles / Cycles | Result |
| --- | ---: | ---: | ---: | --- |
| Barbershop, 2048×858 | 25.3056 [25.2920, 25.3468] | 36.5278 [36.4613, 36.5402] | 1.4430× | finite; 44.31% slower |
| Monk, 1440×1080 | 13.2106 [13.1894, 13.2113] | 12.1597 [12.1545, 12.1709] | 0.9204× | finite; 8.65% faster |
| Monster, 1080×1080 | 14.2550 [14.2515, 14.2781] | 13.2501 [13.2248, 13.2532] | 0.9297× | finite; 7.03% faster |
| Classroom, 1920×1080 | 18.0049 [17.9813, 18.0453] | 16.7880 [16.7756, 16.7932] | 0.9327× | non-finite comparison; diagnostic timing |

Barbershop remains the performance outlier and the overall HIP parity target
is therefore not met. Classroom has recurring non-finite lanes in the paired
images: Cycles `DiffDir` 25 and `GlossDir` 27 pixels, Psycles `DiffDir` 7,
`DiffInd` 2, `GlossDir` 7 and `GlossInd` 2 pixels. Its timing is retained for
diagnosis and is not a validated performance win. All other scene/pass pairs
had zero invalid pixels.

For exact per-repeat timings, source and bundle hashes, binary identities,
output hashes, and per-pass comparison metrics, see
[`summary.json`](summary.json). The 12 source manifests are under
`/var/tmp/psycles-v3-20260911/{scene}/run-{1,2,3}/benchmark.json`.
