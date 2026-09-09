# Final-candidate four-scene HIP canaries

All four final-candidate renders and all **60 pass comparisons** completed.
The all-channel numerical gate **fails**: Classroom retains non-finite
separated-pass values. This is a completed capture, not complete correctness
or a fresh paired performance benchmark. Fresh exports and twelve new v3
timing pairs have **not run**.

## Same candidate, four native-resolution scenes

These are one observation per scene, at 256 fixed samples, with the original
extent, authored seed/frame, fixed sample range [0,256), and 64 samples per
dispatch. Both comparison sides contain exactly 15 linear passes / 46 channels,
including Combined alpha. Main shader caching is off and native fast math is
on; downstream/driver/OS caches retain their ordinary policy. No build,
replay, other render, profiler, or pixel comparison overlaps a timed render.
The finite scans and existing pass-comparison tool run serially afterward.

| Scene / extent | Scene compile s | Session/JIT s | Render s | Stages / fields / frame B | Combined relative RMSE | DiffInd relative RMSE |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Barbershop / 2048x858 | 15.7369 | 55.0679 | 37.1113 | 6 / 92 / 416 | 1.05211% | 7.12617% |
| Lone Monk / 1440x1080 | 4.93996 | 42.9222 | 12.7550 | 4 / 50 / 200 | 1.24043% | 12.88149% |
| Monster / 1080x1080 | 1.51741 | 55.2632 | 13.6912 | 6 / 68 / 272 | 0.54785% | 2.55253% |
| Classroom / 1920x1080 | 2.05483 | 40.2551 | 17.5279 | 5 / 63 / 252 | 0.35332% | 17.82034% |

`shader_jit_seconds` measures session creation, including compilation,
allocation and upload/setup; it is not compiler-only. `render_seconds` covers
sample rendering through synchronization/readback, before EXR serialization.
These are neither globally cold-JIT measurements nor medians. References are
the retained original Cycles HIP `run-1` images from
`/var/tmp/psycles-matched-pass-hip-2c98Q0`, not newly timed native runs.
No speedup, queue-only effect, or complete efficiency parity follows from
comparison with older observations. Frame sizes remain 416/200/272/252 B;
the host guard is not reported as a frame-size reduction.

## Numerical gate, including every channel

Barbershop, Monk and Monster are finite in all 46 channels on both sides.
Classroom has **18 actual non-finite lanes at eight pixels**, and **70
reference non-finite lanes at 27 pixels**. Combined is finite on both sides.

| Classroom side | Non-finite channels | Lane counts |
| --- | --- | --- |
| Actual | DiffDir.B / DiffInd.B / GlossDir.B / GlossInd.B | 7 / 2 / 7 / 2 |
| Original reference | Diffuse Direct.R / G / B | 4 / 4 / 25 |
| Original reference | Glossy Direct.R / G / B | 5 / 5 / 27 |

All four render exits and all four comparison exits are 0. Finite-check
exits are **0 / 0 / 0 / 2** in table order. The sticky failed numerical
verdict has policy code 2; this is a derived verdict, **not** an invented
combined-runner process exit. All fifteen Classroom pass comparisons still
run and disclose their valid domain. Its DiffInd result uses 2,073,598
valid pixels, excluding two invalid actual pixels and no invalid reference
pixels. A finite-domain RMSE is not an all-finite pass or a numeric waiver.

These invalid counts match the historical count-level outcome. This phase
does not compare old/new per-pixel invalid masks, so it does not establish
mask identity. The retained current mask hashes and all per-channel counts
are in [phase-a-results.json](phase-a-results.json). Residual DiffInd and
Monk's previously observed localized variation remain open; neither is
attributed to a cause by these comparisons or dismissed as noise or one ULP.

## Exact capture and immutable identity

The final package is
`/var/tmp/psycles-shading-terminator-7owG2a/final-candidate-tgZZND`.
Its `canaries/` directory retains four EXRs, render logs, command/terminal
records, 46-channel finite reports, fifteen-pass comparison reports and
triptychs. [The archived aggregate](phase-a-results.json) is byte-identical
to `canaries/phase-a-results.json`; it pins every output/log/finite/comparison
hash and records all sixty pass metrics. Relative `render_record` and
`checks_record` names in that JSON refer to the original `canaries/` directory.
The full command strings remain in those records and the execution manifest.

| Identity | SHA-256 |
| --- | --- |
| Final runtime | `86fe7274ae712134bcba6b6262f6fea7e69263fc73a2998856858b4ee3a950fd` |
| Execution manifest | `e4f2f082a00459cb28733c8ac2d6a6b54fe8675a847b99e3832130f19b3d77a2` |
| Source lock | `ec23a9df253af0208278c35da4bbf20a9ce3eca9acbd04f97c77abd80d489357` |
| Phase A aggregate | `b085aa8ca1ca1002bef8f66cb4ec58eb1b4a24a8f06d32db777daf692acad306` |

Root base is `d1a12b8e` plus the locked generic guard/fast-angle diff. SDK is
`5c7de2bb9`; the separate coroutine reachability/retirement candidate is not
included. [Preflight](phase-a-preflight.json) and
[postflight](phase-a-postflight.json) pass: all 32 pinned source files, eight
binaries, four input/reference sets and 310 loaded texture entries remain
unchanged. Actual renderer loader resolution is checked before execution.
The immutable final-runtime archive is
`/var/tmp/psycles-holdout-CXlJR7/frozen-shading-final-7bqEc8/bin`; ordinary
files, soname links and `.data` are retained, but `.cache` is excluded.

The older `ad884114…` prospective plan and the 6e/98f captures remain
untouched. Current exports must be fresh: old controls are valid for these
revision-pinned canaries but fail the current v3 exporter-identity check.
The [remaining capture plan](CAPTURE.md) retains the required geometry,
texture, metadata and native word-image audit before any new paired campaign.
