# Native film state and destination routing

Barbershop's controlled 64-spp surface median improves **1.88%**, and render
wall time improves **0.98%**. This fixes real operation/state differences but
does not explain most of the remaining surface cost or DiffInd discrepancy.
The complete 256-spp follow-up has a 38.4822-second Barbershop median, still
51.6% above retained Cycles. Classroom's separated-pass finite-value gate is
**not green**, as detailed below; completing six renders is not six numerical
passes.

Implementation: Psycles `9e3ba165`, following `f690ead6` / runtime `4f487ba5`.
Measurements and full suites used Luisa `da8fff856`. Before publication the
upstream Metal/CI changes and their existing gitlink were fast-forwarded to
`8911828eb`: all six measured HIP binaries remain byte-identical after the
all-target build, and focused backend/complete host gates are rerun.
Original Cycles is `cb168525138fecc792cc393f94afc39582b0103c`.
[results.json](results.json) retains commands, hashes, full metrics and logs;
local evidence is `/var/tmp/psycles-film-routing-9piLUU`.

## Original state and operation contract

Original `kernel/film/light_passes.h` selects the destination before writing
each contribution. Surface and volume are mutually exclusive branches.
Surface writes glossy and transmission, then the common final diffuse
destination; volume uses only that final write. Directly visible forward
light writes Emit or Env. Eligible zero-valued writes are retained.

| Event | First-pass classification | Direct/indirect predicate |
| --- | --- | --- |
| Forward emission/background | Main path ANY_PASS, then surface/volume flags | Main bounce == 1 |
| Surface NEE | Copy existing weights if ANY_PASS; otherwise add SURFACE_PASS and calculate BSDF proportions | Captured shadow bounce == 0 |
| Volume NEE | Copy inherited classification; otherwise add VOLUME_PASS | Captured shadow bounce == 0 |

The old six-RGB splitter computed direct and indirect outputs, selected zero
for the unused half, and wrote both. Surface NEE ignored path flags at film
completion. Volume NEE reused the forward bounce==1 helper, misclassifying
one-bounce NEE. Surface-shadow preparation used bounce instead of ANY_PASS
and did not install the first SURFACE_PASS bit; repairing the sink alone
would therefore drop camera NEE. Deferred completion must consume the task's
captured flags, weights and bounce, never the advanced main path.

The repair uses shared ordinary host DSL recorders, with no device callable
boundary, extra local array or inline annotation. The six-RGB type, splitter
callable and all consumers are removed. Combined's independent volume-guiding
classification remains intact. Transmission uses native `(1 - d - g) * C`.
The common ratio helper restores original `safe_divide`'s zero-only guard:
the old `abs(sum) > 1e-20` discarded even normal 1e-25 BSDF proportions.
Main-surface first-bounce ratios still have a separate eager-placement lead;
this phase only restores their shared arithmetic and shadow preparation.

This is not a new SVM: words, typed payloads, stack/PC, closure/feature masks,
domains, RNG, array-bound analysis, launch/inlining policy and fast math are
unchanged. Current Barbershop has no world volume, zero volume bounces and
emission-only camera-dependent fog. The mixed volume-scattering route defects
cannot explain this scene's DiffInd. Their correctness and the profiled film
work reduction are separate findings.

## Permanent counterexamples and gates

`capture_cycles_film_routing.py` extracts original surface/volume shadow-pass
assignment blocks verbatim. The observer executes them and original GPU BSDF
weights, light clamp and film functions, wrapping original atomics only to
count destination writes. Its 240 boundary states cover four event kinds,
five flag combinations, three bounce depths and four contribution/BSDF kinds.
The harness supplies boundary state; it is not the entire original sampler
or traversal integrator, and there is no CPU film/shader oracle.

Against unchanged extracted production operations, film checks initially pass
351/540 and surface-shadow state checks 72/180. After repair they pass
540/540 and 180/180, including serial/atomic sinks and detached completion
after deliberate main-state mutation. All 52 film lanes are checked with
2e-6 absolute tolerance; discrete state is exact. The original 240-row fixture
is byte-identical before and after repair.

Five production-AST cases initially fail: 18 or 27 static pass atomics, none
with bounce-dependent addresses. All five now pass with nine sites, all
addresses dependent on bounce. The original maximum observed writer count
bounds these sites, including zero writes. This is a data-dependence/operation
structure check, not device traffic measurement or a CPU shader evaluator.

| Gate | Result |
| --- | --- |
| All-target build | Passed, 32 threads |
| Complete host | 176/176, 8.85 s; upstream integration rerun 176/176, 3.31 s |
| Complete HIP | 187/187, 484.37 s |
| Complete fallback | 189/189, 362.38 s, after HIP completion |
| Strict native Vulkan | 6/6, 32.17 s, 62 native compilations, no DXC/DXIL |
| Restored B focused | Film routing 1/1 on HIP, fallback, strict native Vulkan |
| Upstream integration focused | 6/6 each on HIP, fallback, strict native Vulkan; sequential |

The unrelated native Vulkan area-sample 36-lane discrepancy and f16/f64 FRem
1,076-assertion failures remain open. Passing this six-test selection does not
erase them. No original Cycles source was edited; its two inherited user
edits retain their recorded SHA-256 values.

## Full original Barbershop A/B/B/A

All runs use 2048x858, 64 spp, seed 0, 15 passes / 46 channels, 512-thread
surface groups, and isolated dump directories. No heavy agent-launched work
overlaps timed runs. Main shader caching is disabled; downstream/OS caches
retain normal policy. Initialization includes JIT, setup and baking, not
compiler-only time or a controlled cold-JIT experiment.

| Run | Surface GPU s | Render-only s | Session init s |
| --- | ---: | ---: | ---: |
| A before | 5.533560 | 10.1687 | 18.4574 |
| B routed | 5.436133 | 10.0974 | 55.0106 |
| B repeat | 5.428282 | 10.0626 | 18.5374 |
| A restored | 5.538502 | 10.1904 | 19.8340 |

Surface medians are 5.536031 -> 5.432208 s; render medians 10.17955 ->
10.08000 s. There are only two observations per treatment. Surface variation
is about 0.089% within A and 0.145% within B. All six A binaries and surface
text match after restoration; both B runs and final restored B match too.
This measures the full code-generation intervention, not isolated atomic cost.

Main instruction sites change 158,005 -> 157,717; function bytes change
842,816 -> 840,964. Both retain 256 VGPRs, 107 SGPRs, 2,464 private bytes,
47 calls, 20 image-sample sites and six stages / 93 fields / 416 frame bytes.
Raw VGPR spills change 503 -> 495, SGPR spills remain 64. Static scratch
load/store sites change 1242/617 -> 1252/612; these are not dynamic traffic.
The same 1D FBM, 4D Noise and OCML tan helpers remain outlined. Main SVM,
3D Noise and microfacet bodies were already inlined; this is not an inline fix.

Surface visits remain 332,307,891..332,307,897 and entry work is unchanged.
No reduced path/shading count or RNG repair is claimed. All 46 channels in
these four intervention images are finite. B-vs-A Combined relative RMSE is
1.339e-5 / 2.177e-6 and DiffInd is 1.556e-5 / 1.125e-5. Restored A itself has
1.855e-5 / 5.753e-5. Retained native surface GPU time is 2.962829 s, leaving
actual surface about 1.83 times native. These are not new Cycles timing pairs.

## Six 256-spp scene follow-ups and numerical limitation

All six final renders preserve earlier geometry/image bytes and refreshed
socket metadata; binaries are frozen. All 15 pass comparisons complete.
Four first-run Combined triptychs were viewed at resized viewer resolution;
their amplified differences remain visible. References are retained Cycles
runs, not fresh pairs. Five renders have all 46 channels finite; Classroom
does not. The runner's default finite gate remains strict. The explicit
keep-going option records failures, completes the campaign and exits **2**.

| Scene / repeat | Extent / seed | Render s | Session init s | Frame B |
| --- | --- | ---: | ---: | ---: |
| Barbershop 1 | 2048x858 / 0 | 40.4211 | 21.5272 | 416 |
| Lone Monk | 1440x1080 / 0 | 12.8729 | 12.0332 | 220 |
| Monster | 1080x1080 / 0 | 13.9104 | 15.3618 | 280 |
| Classroom | 1920x1080 / 1 | 17.4542 | 11.4340 | 260 |
| Barbershop 2 | 2048x858 / 0 | 38.4822 | 17.7848 | 416 |
| Barbershop 3 | 2048x858 / 0 | 38.4574 | 18.1808 | 416 |

All observations, including the slower first Barbershop, are retained. Its
median is 51.6% above native 25.3775 s. One-run and temporal follow-up changes
are not isolated treatment estimates. The initial interrupted campaign also
retains Barbershop 38.1864, Monk 13.3323 and Monster 14.4887 s; it stopped at
Classroom's finite assertion. No fast timing is selected in place of the
complete campaign. Warm downstream caches make session-init comparisons
with the preceding checkpoint especially unsuitable as JIT improvement claims.

Classroom has 18 invalid channel values at eight unique pixels, with exactly
the same channel coordinates in both attempts. Combined remains finite.

| Pass | Actual invalid pixels | Native reference invalid pixels | Excluded union |
| --- | ---: | ---: | ---: |
| DiffDir | 7 | 25 | 32 |
| DiffInd | 2 | 0 | 2 |
| GlossDir | 7 | 27 | 34 |
| GlossInd | 2 | 0 | 2 |

The comparator explicitly reports this union; no values are replaced with
zero. A separate nine-input original-GPU/production-DSL observer reproduces
the same NaN/Inf classifications for dynamic subnormal BSDF denominators
under fast math, with matching zero/normal controls. This demonstrates a
shared arithmetic boundary consistent with the Classroom failure; it does
not prove the input state of every failed full-scene path. The previous
epsilon cutoff masked this domain while incorrectly discarding normal 1e-25
weights. No cutoff, slower division or strict-math workaround is introduced
to force a finite result. The all-finite gate remains failed, not waived.

First-run Combined relative RMSE for Monk/Monster/Classroom/Barbershop is
0.01240070 / 0.00547855 / 0.00353320 / 0.01052120; DiffInd is
0.12881636 / 0.02552525 / 0.17820339 / 0.07126170, on reported finite domains.
Residual indirect/path differences remain unresolved. Mixed volume-scattering
film routing cannot account for Barbershop's emission-only fog case.

## Replay and next bounded investigations

`archive_profiles.py` takes the four isolated profile directories;
`archive_results.py` takes the evidence root, output, `--parent` and `--child`.
To rerun the independent arithmetic-domain observer without changing Cycles:

```bash
python docs/validation/2026-09-09/film-routing/observe_tiny_weights.py \
  /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  /var/tmp/psycles-tiny-weights-fresh
```

The next ownership discrepancy is per-ray lamp inverse reconstruction versus
original object-table reads; main-surface first-bounce weight placement is
another bounded lead. Neither is yet quantified as the main cost. Diagnostic
light/shadow SVM adapters also retain status handling, unlike release surface
evaluation. Surface already omits unused cases and diagnostic status lanes,
and shares one populated SVM closure state between NEE and continuation.
The remaining native opcodes, private displacement bridge, generic CFG proof
obligations and cross-scene structural/performance goals are still open.
