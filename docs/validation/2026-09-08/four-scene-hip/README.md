# Four-scene HIP benchmark, 256 spp

## Classroom is faster; the other three scenes remain slower

All 12 paired scene runs and 15-pass comparisons completed. Against original
Cycles 5.2.1 HIP, Psycles takes **4.1% more time on Lone Monk, 5.6% more on
Monster, 9.4% less on Classroom, and 42.8% more on Barbershop**, using the
ratio of three-run medians. This establishes a reproducible baseline, not
overall performance parity or complete path/indirect-light correctness.

The tested renderer is Psycles `eaa7c72e` (renderer implementation unchanged
from `2d89cf1d`), with Luisa `9ea3b720f`. This campaign precedes the subsequent
per-ShaderJump-entry specialization work. The renderer and loaded implementation
libraries were hash-checked unchanged across the entire campaign; see
[implementation hashes](implementation-sha256.txt).

## Matched scope and timing boundaries

Every pair uses the original, unmodified `.blend`, scene-owned frame and seed,
fixed 256 spp, no adaptive sampling or denoising, and the same RX 9070 XT
(gfx1201). The production Blender build is `9e2066aef7ef`, 5.2.1 LTS, not the
diagnostic trace build. HIP hardware ray tracing is enabled. The exported
bundle is regenerated once with the current exporter and then reused with
identity/hash checks. Scene order rotates between rounds; each pair runs
Cycles followed by Psycles, without a GPU profiler or concurrent build/render.

The primary metric is Cycles' **original main-loop wall interval**, parsed
from its debug log, versus Psycles' **render-only wall interval**. It is not
summed GPU kernel time or the whole Python `bpy.ops.render()` call. All
manifests use benchmark schema v2. Old v1 ratios used the wrong Cycles timing
scope and are not interchangeable with this campaign.

Psycles uses `wavefront-staged`, surface sorting, the separate direct-light
queue, fast math, a 1,048,576-frame pool, and at most 64 samples per dispatch.
`PSYCLES_DISABLE_SHADER_CACHE=1` disables the main shader cache; auxiliary
caches and the OS file cache retain their normal policy. “Main JIT” below
is therefore not a cold-machine or uncached-all-components measurement.

## Render times remain stable across repeats

Times are seconds. “Relative time” is Psycles median / Cycles median; a
number below one means less rendering time. Ranges are observed minima and
maxima, not confidence intervals. Exact values and sources are retained in
[summary.json](summary.json) and each scene's three manifests.

| Scene | Extent / seed | Cycles median [range] | Psycles median [range] | Relative time |
| --- | --- | ---: | ---: | ---: |
| Lone Monk | 1440x1080 / 0 | 13.4183 [13.4015, 13.4589] | 13.9693 [13.9594, 14.0269] | 1.0411 |
| Monster | 1080x1080 / 0 | 14.4289 [14.4179, 14.4603] | 15.2304 [15.1850, 15.2614] | 1.0555 |
| Classroom | 1920x1080 / 1 | 20.7624 [20.7338, 20.7869] | 18.8207 [18.5953, 18.8588] | 0.9065 |
| Barbershop | 2048x858 / 0 | 28.7312 [28.6916, 28.7474] | 41.0285 [41.0215, 41.2048] | 1.4280 |

The direction of every comparison is unchanged across the three repeats.
Classroom's lead must not hide Barbershop's substantially larger gap. The
Monk/Monster gaps are also larger than the observed repeat spread.

## Compilation remains a distinct first-render cost

These are medians in seconds, except frame size. Main JIT is not compared
with a fictitious Cycles JIT phase: this Cycles build loads precompiled GPU
kernels. Coroutine sizes are static compilation results, not profiled array
sizes. They remain identical across all three runs of each scene.

| Scene | Scene compile | Main JIT | Cycles whole render call | Frame / fields / stages |
| --- | ---: | ---: | ---: | --- |
| Lone Monk | 4.98274 | 18.8151 | 16.05099 | 220 B / 55 / 4 |
| Monster | 1.49867 | 22.3434 | 15.41285 | 284 B / 71 / 6 |
| Classroom | 2.05883 | 18.1598 | 22.18668 | 264 B / 66 / 5 |
| Barbershop | 15.1295 | 27.4798 | 39.35269 | 416 B / 93 / 6 |

The prior generic read-only-reference correction reduced Barbershop's frame
from 896 B to 416 B. Its larger compilation-phase-logging canary is not used
as a JIT baseline here. Smaller frames or IR do not, by themselves, establish
an end-to-end speedup.

## Pass residuals remain, without a broad energy shift

All **46 Psycles EXR channels are finite in every run**. The following are
the first pair's relative RMSE values; the complete 15-pass metrics for all
three pairs are retained in each scene's `run-*/report.json`. Original-sized
Combined triptychs were generated and visually reviewed for all four scenes.
The views show broadly matching scene appearance, but amplified residuals
are still visible; this does not establish structural path parity.

| Scene | Combined | DiffCol | DiffInd |
| --- | ---: | ---: | ---: |
| Lone Monk | 0.01240998 | 0.000545943 | 0.12881211 |
| Monster | 0.00547891 | 0.000110114 | 0.02552520 |
| Classroom | 0.00353330 | 0.000102385 | 0.17820333 |
| Barbershop | 0.01079831 | 0.001597650 | 0.07451034 |

Classroom's DiffInd mean luminance ratio is 0.99979792, while its relative
RMSE is 17.8%. This distinguishes average energy from same-sample agreement;
it does not explain or excuse the residual. Its TransDir relative RMSE is
53.6%, but the reference RMS is only 4.0648e-9 and absolute RMSE 2.1780e-9.
That nearly empty pass is not comparable in significance to DiffInd.

There is a reference limitation: in every Classroom pair, original Cycles'
DiffDir has 25 non-finite pixels and GlossDir has 27. Psycles has zero in
both. Those metrics exclude the union of non-finite reference/actual pixels,
and the counts are preserved explicitly; they must not be called whole-image
finite-domain proofs. Combined and the tabled passes have no such exclusion.

The previous same-path investigations remain relevant: Monk has a visibility
divergence at coincident leaves; the repaired Barbershop transparent closure
matches the first four surface events and all 45 sampled random fields at
the diagnosed pixel, but a later NEE event selects an adjacent emitter
triangle. Neither is permission to deduplicate geometry or introduce slow
floating-point/intersection emulation. No global RNG or full-path parity
claim follows from the average-energy agreement.

## Reproduction and remaining work

Each archived manifest records the exact commands, source/export hashes,
device choice, output hashes, logs and phase boundaries. `cycles.json`
retains original renderer settings/build identity; `report.json` retains
the full differential. Large EXRs, original-resolution triptychs and raw
logs remain under `/var/tmp/psycles-four-scene-hip-0PRq34`. A scheduling shell
received SIGTERM after seven complete pairs; the remaining five were run in
a second batch. No failed/incomplete render was counted or resumed silently.

Use `tools/run_scene_benchmark.py` with the saved arguments and a fresh
output directory to repeat a pair. Re-export before the first repeat if the
exporter identity changed. Source scene locations are recorded, not guessed.

The next implementation checks are entry-specific static SVM code/stack
specialization and alignment of NEE work with the Cycles BSDF-evaluation
predicate. Validate each change independently against original word/GPU
oracles and full scenes before attributing any performance change. Remaining
questions include the indirect/path residuals, surface/volume kernel cost,
unsupported opcodes and complete removal of the legacy displacement bridge.
Three repeats describe this machine and revision; they are not a broad
hardware claim or a statistical equivalence certification.
