# Static SVM usage follows ShaderJump entries

Base: Psycles 14b3d8af / Luisa 9ea3b720f. Evidence is under
/var/tmp/psycles-native-volume-svm-06XnDX, with entry-usage prefixes.
The paired four-scene benchmark is the preceding baseline, not a measurement
of this change. Backend and scene validation are recorded below as completed.

## Formal cause and correction

Whole-image opcode usage and stack high-water marks mixed surface, volume
and displacement emission. Every native consumer inherited that union even
though its ShaderJump entry is known while recording Luisa DSL. For example,
a volume-only dispatch could retain surface closure cases and the surface's
stack bound; an END-only displacement entry could inherit both.

The compiler now records typed opcode emissions and allocation high-water
marks separately for the three original ShaderJump entries. It does not scan
payload words as opcodes, execute a host interpreter, profile a render or
rewrite the stream. All emitted arms of data-dependent jumps contribute.
The bump prefix has no independent runtime entry or END, so it is included
in surface usage. Since bump/surface allocation spaces are reused, the bound
is their maximum, not their sum. The original Cycles feature masks and
runtime predicates are unchanged.

Linking unions corresponding entries across admitted shaders. The union of
the three entry facts must equal the original whole-image mask and bound.
An image without emission metadata conservatively contributes its whole
image to every entry; missing metadata never means no usage. Inconsistent
metadata is rejected. Native surface, volume, background, shadow and light
consumers use the corresponding entry's mask and stack bound. A zero-use
entry records the one-lane storage sentinel, not a zero-sized Luisa array.

## Permanent regressions fail before the correction

The new independent compiler regression reuses four complete, unchanged
original Cycles shared-closure word images. It checks active/END-only entry
facts, both dynamic closure branches, linked domain separation, conservative
external-image fallback and invalid metadata rejection. The existing
238-word BOTH-displacement oracle additionally proves that the complete bump
prefix belongs to surface and is absent from displacement/empty volume.

The AST regression checks actual cases and array extents before backend
optimization. Its existing 99 implemented opcode checks and 25 feature-family
guards remain; the new checks detect another entry's closure cases and require
an empty entry to contain only ShaderJump/END.

Before the correction all three focused tests fail with these observations;
afterwards all three pass. The logs are entry-usage-red.log and
entry-usage-host-green.log. An earlier test-source typo caused a separate
compilation failure; that build failure is not the semantic red regression.

The runtime regression executes both conservative and entry-specialized
versions of the same linked stream. Each is compared directly with original
Cycles GPU state, not with the other Psycles variant: four graphs, four path
visibility states and closure capacities 0/1/64, repeated in both modes.
All 96 state comparisons retain exact count/type/flag/PC/status checks and
the existing native-arithmetic tolerance. No expected stream is regenerated
by Psycles and no shader arithmetic or inlining policy changes.

## Completed build and host/HIP checks

The full build succeeds with all 32 threads. Focused host checks pass 3/3.
The complete host selection passes 153/154, retaining only the four existing
source-size violations. Full HIP passes 177/177 in the completed serial run
(309.96 s). Suite durations are validation observations, not rendering
benchmarks.

Two earlier suite attempts were interrupted after GPU progress stalls,
first with 32 concurrent tests and then at the serial area-light fixture.
No OOM or kernel driver error was observed. The isolated fixture subsequently
passes with main caching disabled and enabled, and the complete serial suite
passes without an intervening source change. These observations do not prove
a cache or concurrency root cause and are not presented as a repaired
compiler defect. The unsuccessful attempt logs are retained separately from
entry-usage-hip-suite-serial-retry.log.

Full fallback passes 178/179 (427.20 s), retaining the existing film trace
light_ng.z difference: expected 0xbf1f8bfd, actual 0xbf1f8a50. The focused
Vulkan selection passes 5/5 (8.74 s): shared closure, volume, stack extent,
bump state and closure pool. All three native-XIR/require-SPIR-V/disable-DXC
guards are set. The verbose loader log records SPIR-V compilation without
loading DXC/DXIL libraries. No backend selection or numerical tolerance was
changed to obtain these results.

## Four-scene HIP canaries after entry specialization

Twelve Psycles renders completed, three per scene at the original extents
and 256 spp, against the preceding campaign's unchanged run-1 original
Cycles HIP images. These are repeated post-change canaries, **not** twelve
new interleaved Cycles/Psycles pairs. Original blend, exported scene/geometry,
reference EXR/metadata/log and implementation hashes were checked. The six
executable/library hashes stayed fixed across all twelve renders. Exact
commands, hashes, timings, frame layouts and all 15 pass metrics are archived
in [run 1](canaries-run-1.json), [run 2](canaries-run-2.json) and
[run 3](canaries-run-3.json). Original images and reviewed triptychs remain
at the paths recorded there.

Render times are wall seconds. The Cycles column is the preceding paired
campaign's three-run main-loop median, not a newly measured reference.
Psycles columns use three-run render-only medians. Native fast math and
wavefront-staged settings are unchanged; no profiler or concurrent build or
render overlapped these runs. Main shader caching is disabled; auxiliary,
downstream compiler and OS caches retain normal policy.

| Scene / extent / seed | Prior Cycles | Prior Psycles | Entry-specialized Psycles | Change vs prior Psycles | Relative to prior Cycles | Frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Monk / 1440x1080 / 0 | 13.4183 | 13.9693 | 13.9472 | -0.16% | +3.94% | 220 B |
| Monster / 1080x1080 / 0 | 14.4289 | 15.2304 | 15.2102 | -0.13% | +5.41% | 284 B |
| Classroom / 1920x1080 / 1 | 20.7624 | 18.8207 | 18.9568 | +0.72% | -8.70% | 264 B |
| Barbershop / 2048x858 / 0 | 28.7312 | 41.0285 | 39.5579 | -3.58% | +37.68% | 416 B |

The tiny Monk/Monster differences do not establish a speedup. Classroom's
small slowdown is retained. Barbershop's three observations are
39.4933 / 39.5579 / 39.6439 s, versus the prior median 41.0285 s; this is a
local improvement, not parity with Cycles. Coroutine frame sizes and stage
counts remain unchanged (4 / 6 / 5 / 6 stages respectively).

The existing CLI's `shader_jit_seconds` measures all of `create_session`,
including JIT and session initialization/baking. Its first observations
were much slower than repetitions without any source or binary change:

| Scene | Session initialization run 1 | Run 2 | Run 3 |
| --- | ---: | ---: | ---: |
| Monk | 36.9272 | 18.8283 | 18.7352 |
| Monster | 75.4606 | 22.0889 | 22.7432 |
| Classroom | 29.0150 | 18.7508 | 18.2957 |
| Barbershop | 83.4014 | 26.1290 | 25.8368 |

Logs locate large first-run delays in HIP final-bitcode linking: for example,
Monk's shadow-intersection link took 9631.44 ms and Barbershop's surface link
36084.65 ms. Later runs returned to the prior range. This is consistent with
downstream compilation/cache warming, but does not establish a specific cache
root cause. The first costs are not discarded or called cold-all-caches JIT.

All 46 actual channels are finite in all twelve images and all 15 comparison
passes complete. First-run Combined / DiffInd relative RMSE is
1.2399% / 12.8815% (Monk), 0.5479% / 2.5525% (Monster),
0.3533% / 17.8203% (Classroom), and 1.0798% / 7.4510% (Barbershop).
The original Classroom reference retains 25 non-finite DiffDir and 27
non-finite GlossDir pixels; the metrics explicitly exclude invalid unions.
All four first-run Combined triptychs were visually inspected at source
resolution. Existing indirect/visibility residuals remain; this change does
not establish full image parity or global path/RNG identity.

The result is static entry isolation with unchanged Cycles words and tested
state behavior, not a claim that every remaining unused operation or
structural performance difference is resolved. Legacy displacement removal,
missing semantic opcodes, indirect-path diagnosis and the cross-scene
performance goal remain open.
