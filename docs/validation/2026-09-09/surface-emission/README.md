# Native surface emission eligibility

Restoring the original operation guard removes a real source of unnecessary
surface work. Barbershop's 64-spp A/B/B/A medians improve by **4.70% for surface
GPU time and 2.69% for render wall time**. This is a modest gain, not closure
of the remaining gap or an inline-policy repair. The final 256-spp follow-up
has a 38.8536-second Barbershop median, still 53.1% above retained Cycles.

Baseline Psycles is `6cd48f1f` (runtime `ff8385c1`), repaired Psycles is
`4f487ba5`, and Luisa remains `da8fff856`. Original Cycles is
`cb168525138fecc792cc393f94afc39582b0103c`. Commands, original-source hashes,
implementation binaries, all pass metrics and logs are in
[results.json](results.json). The evidence root is
`/var/tmp/psycles-surface-emission-O0BGUX`.

## Zero contribution is not zero work

Original `integrator/shade_surface.h::integrate_surface` excludes BSSRDF exits
and requires `SD_EMISSION` before integrating surface emission. That operation
includes forward-hit MIS and film writes. Psycles instead selected a zero
emission RGB and still performed the operation. On an ordinary non-emissive
surface path, it could issue 24 float atomics, including additions of zero.

The repair encloses forward MIS and film effects in the original emission/
exit predicate. Diagnostic records retain their defined default values outside
the guard. Runtime flags must also be requested with NEE and tracing disabled;
otherwise a newly guarded forward-only render could incorrectly lose emission.
Pure emission preparation is not claimed to be entirely inside the new guard.
An ordinary host DSL-recording function exposes the production operation to
tests; it does not introduce a device callable or an inline annotation.

Flag-set zero RGB and grazing-angle emission remain eligible, as in Cycles.
There is no generic `value == 0` atomic suppression. SVM words, payloads,
stack addresses, PC, feature masks, closures, RNG, launch/inlining policies,
fast math and the existing direct/indirect split-film representation are
unchanged. Film address routing is a separate follow-up, not bundled here.

## Permanent red and original GPU oracle

Four production-AST configurations cover NEE and tracing on/off. All four
initially fail, identifying 30 static atomics across mutually exclusive film
branches and, with NEE, the forward-MIS call outside the required guard.
All four now pass. This is a side-effect dominance check, not a code-size cap.

The HIP observer calls original `surface_shader_emission` and
`film_write_surface_emission`, with original SOA path state and film tables.
It captures 80 boundary states plus a unity-throughput camera state, including
per-lane native atomic counts. The harness supplies the entry eligibility
predicate from the directly audited original integrator; it does not execute
the whole original integrator entry. It is not a CPU film/shader implementation.
The original checkout's two inherited user edits remain byte-identical.

The production runtime operation compares 40 film lanes for 320 configurations:
80 states, NEE on/off, and serial/atomic accumulation. States include flag
absence/presence, BSSRDF exits, camera/surface/volume routes, direct/indirect
depths, finite negative RGB, zero and grazing emission, and clamp boundaries.
The old code already passes these numerical tests: film values alone would
miss the extra writes. The AST regression is the required failing control.

A full emission-only SVM triangle additionally compares Combined and Emit to
the original camera capture with NEE on/off and megakernel/staged execution,
without diagnostic tracing. The fixture is unchanged by the repair. All
finite comparisons use a 2e-6 absolute component tolerance with fast math.
An initial camera-test draft confused authored average-channel clamp settings
with the kernel's RGB-sum limit; both original and actual upload multiply by
three. Only the test input mapping was corrected, not expected film or math.

## Full original-scene intervention

Four sequential runs use Barbershop 2048x858, 64 spp, seed 0, 15 passes /
46 channels and 512-thread surface groups. No agent-launched heavy work
overlaps a timed run. Surface time sums the identified kernel's GPU dispatch
durations; render time is render-only wall time. Initialization includes JIT,
setup and baking, separately. Main shader caching is disabled; downstream
caches retain ordinary policy. These are not fresh Cycles timing pairs.

| Run | Surface GPU seconds | Render seconds | Session init seconds |
| --- | ---: | ---: | ---: |
| A before | 5.819638 | 10.4723 | 18.4119 |
| B guarded | 5.530426 | 10.1758 | 44.4361 |
| B repeat | 5.536766 | 10.1745 | 18.7376 |
| A restored | 5.793219 | 10.4396 | 18.1106 |

Surface medians are 5.806429 -> 5.533596 seconds; render medians are
10.45595 -> 10.17515. There are only two observations per treatment; A's
surface variation is about 0.455%, B's about 0.115%. All six frozen A binaries
are identical after restoration; both B binaries and surface text match, and
the final restored B matches the measured B again. The intervention measures
the whole generated-code change, not the isolated time spent on atomics.

Main instruction sites change 158,088 -> 158,005 and function bytes
843,084 -> 842,816. Both retain 256 VGPRs, 107 SGPRs, 2,464 private bytes,
47 calls, 20 image-sample sites and six stages / 93 fields / 416 frame bytes.
Raw VGPR/SGPR spill metadata changes 500/66 -> 503/64; static scratch
load/store sites change 1264/618 -> 1242/617. These are not dynamic traffic
counts. The same three helpers remain outlined: 1D FBM, 4D Noise and OCML tan.
The main SVM/3D Noise/microfacet inline findings are unchanged.

Surface work remains 332,307,892..332,307,894 visits with 1,062 launches.
No reduced shading/path count or RNG repair is claimed. All channels are
finite; B Combined relative RMSE against A is 1.302e-5 / 1.362e-5 and DiffInd
is 1.098e-5 / 5.829e-5. Restored A itself differs by 1.472e-5 / 3.547e-5.
The retained original 64-spp surface total is 2.962829 seconds, leaving actual
surface cost about 1.87 times native. This remains a large unresolved gap.

## Six full-resolution 256-spp follow-ups

All runs use the frozen measured B implementation, exact earlier geometry/
texture bytes with refreshed socket metadata, and retained equal-pass Cycles
references. All 46 actual channels are finite and all 15 comparisons complete.
Four first-run Combined triptychs have been inspected. Difference images
retain visible residuals; visual similarity does not establish path parity.

| Scene / repeat | Extent / seed | Render seconds | Session init seconds | Frame bytes |
| --- | --- | ---: | ---: | ---: |
| Barbershop 1 | 2048x858 / 0 | 38.8536 | 17.3832 | 416 |
| Lone Monk | 1440x1080 / 0 | 12.8818 | 28.3660 | 220 |
| Monster | 1080x1080 / 0 | 14.4971 | 37.2553 | 280 |
| Classroom | 1920x1080 / 1 | 17.4903 | 26.8763 | 260 |
| Barbershop 2 | 2048x858 / 0 | 38.8298 | 17.6237 | 416 |
| Barbershop 3 | 2048x858 / 0 | 39.0386 | 17.4312 | 416 |

Barbershop's median is 2.93% below the preceding 40.0251 seconds; this temporal
follow-up is not a separately isolated causal estimate. It remains 53.1%
above retained native 25.3775 seconds. Initialization is neither compiler-only
nor a matched cold/warm comparison, and is excluded from rendering time.
First-run Combined relative RMSE for Monk/Monster/Classroom/Barbershop is
0.0124046 / 0.00547855 / 0.00353320 / 0.0105212; DiffInd is
0.128817 / 0.0255253 / 0.178203 / 0.0712617. This repair does not resolve it.
Original Classroom's 25 invalid DiffDir and 27 invalid GlossDir pixels remain
explicitly excluded by the invalid-union rule; no actual invalids are hidden.

## Backend gates and next structural boundary

All-target builds use 32 threads. Complete suites pass host 175/175,
HIP 186/186 (460.63 s) and fallback 188/188 (322.20 s). An accidentally early
fallback start was interrupted; the reported full run starts after HIP exits.
Device cases are sequential. These test durations are not performance trials.

The first native Vulkan test draft created two live Devices, violating Volk's
process-global dispatch-table restriction after all 320 film checks passed.
The full-render test now shares the existing native Device; no backend assert
or arithmetic tolerance changes. Focused HIP and fallback reruns each pass
1/1. The final strict native emission/film/volume-film/NEE canary passes 4/4,
with 49 native SPIR-V compilations and no DXC/DXIL load. Earlier native area
sampling's 36 lanes and f16/f64 remainder failures remain open separately.

Reproduce the observer with `hipcc -parallel-jobs=32 --offload-arch=gfx1201
-DHIPCC -std=c++20 -O3 -ffast-math`, the original Cycles include directory,
and `tools/cycles_surface_emission_oracle.hip`. The captured text is committed
under `tests/data`. [archive_profiles.py](archive_profiles.py) rebuilds the
four-run archive; [archive_results.py](archive_results.py) verifies fixtures,
gates, no-DXC loader evidence and campaign binaries before archiving results.
Keep LLVM dumps in isolated cwd directories, independently of ISA output paths.

Next, native film chooses one direct/indirect address and classifies by the
first scattering event; the actual six-RGB zero-selected intermediate writes
both slots. Source review also finds missing first-surface shadow pass flags,
surface NEE after volume scattering routed to surface passes, and volume NEE
using forward-emission bounce thresholds. These need original shadow-state
and film-address regressions before changing shared sinks. They are leads,
not yet a quantified explanation of the residual DiffInd. Per-ray inverse
light transforms, missing semantic opcodes, the private displacement bridge
and generic CFG proof obligations also remain open.
