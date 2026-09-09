# Native path lifetime: remove the aggregate truncation

Status: production lifetime correction validated. The permanent original-HIP
regression first failed on the unmodified implementation `9e3ba165`, SDK
`03a0f5158`. No oracle output, fastmath mode or tolerance is changed by the fix.
Local evidence: `/var/tmp/psycles-native-path-lifetime-v7burc`.
The broader renderer goal remains open. Ray Portal and object holdout from
the [whole-surface audit](../surface-semantic-audit/README.md) are not repaired
by this change. Neither are Barbershop's performance/DiffInd gap, Classroom's
nonfinite separated passes, or the newly isolated run-to-run variation below.

## Native contract and root cause

Cycles source is `cb168525138fecc792cc393f94afc39582b0103c`.
`kernel/integrator/path_state.h::path_state_next` owns ordinary and transparent
bounces. `path_state_volume_next` owns a separate lifetime-wide volume-boundary
counter and its `VOLUME_BOUNDS_MAX == 1024` check. In
`kernel/integrator/shade_surface.h`, a selected BSSRDF schedules entry transport
without an ordinary bounce; its synthetic diffuse exit advances that bounce.
Queue routing terminates when the native event returns no continuation, not
when an unrelated total-iteration counter expires.

| Main-loop continuation | Native progress / termination owner |
| --- | --- |
| Ordinary surface or volume scattering | Advances ordinary bounce and relevant type counter; native flags limit further scattering |
| Transparent or portal continuation | Independent transparent counter; portal has additional state |
| Pure volume boundary | Independent volume-boundary counter; changes tmin/RNG without an ordinary or transparent bounce |
| Successful BSSRDF entry | Sets a pending exit; the next event consumes that hit and its synthetic diffuse exit advances the ordinary counter |
| Failed sample/transport, native termination, background | Ends the path; final event contributions are retained |
| Analytic lamp traversal | Inner event loop advances the ray/endpoint state; it is not an ordinary surface iteration |

An SSS entry cannot repeatedly consume main iterations without progress: it
either fails or produces a pending same-object exit; that exit replaces the
closures with native diffuse transport. Empty geometry resolves to background.
Each admitted continuation is therefore owned by the existing native state
machine, including its safety limits. This is not permission to bypass those
limits or to introduce an unbounded volume-boundary counter.

The removed bound was `min(2 * synced_maximum + max(transparent_maximum, 1) +
1, 1024)`. It omitted volume boundaries and imposed another cap unrelated to
their independent limit. Two closed absorbing slabs consume four boundary
events; the next background event is valid, but authored ordinary/transparent
limits zero made the outer budget four. Raising only the transparent limit to
two made it five and restored the actual image while leaving original Cycles
unchanged. Adding another conservative constant would retain a second
termination mechanism with unnecessary proof obligations.

The correction uses the existing native termination branches as the main
loop exits. `path_step` is diagnostic only, increments at loop entry when
tracing is enabled, and does not bound execution. The host aggregate bound,
its kernel ABI member and tests asserting the incorrect formula are removed.
Scene-to-kernel socket mapping tests remain. No Cycles bounce limit, RNG
transition, volume-boundary cap, closure evaluation or scheduler policy changes.

## Permanent red regression

`tests/test_luisa_cycles_path_lifetime.cpp` imports the exact original exported
bundles and runs the full production renderer, both megakernel and staged
coroutines. All 256 RGB pixels are compared with original Cycles HIP build
`9e2066aef7ef`, at 16x16 / 1 spp. The staged frame pool has capacity 64, so it
also exercises refills. Fixtures include one slab, two slabs and the
transparent-budget control. Each image allows `2e-6 + 2e-6 * abs(original)`;
there is no locally calculated expected absorption or CPU reference renderer.

The checked-in fixture manifest pins every input, original render, metadata
and extracted text image. `tools/extract_cycles_path_lifetime_fixture.py`
performs byte/text extraction only. Before changing production:

| Case | Megakernel mismatched RGB lanes | Staged mismatched RGB lanes |
| --- | ---: | ---: |
| One slab | 0 / 768 | 0 / 768 |
| Two slabs | **768 / 768** | **768 / 768** |
| Two slabs, transparent limit two | 0 / 768 | 0 / 768 |

The new HIP CTest exited 8, with its test process failing as intended. The
dedicated existing volume-boundary transition oracle remains complementary:
it checks the native counter/next-float operations, including the boundary
cap, but did not exercise their composition with the old outer budget.

## Green implementation and original-module gates

All six full-render observations now have **0 / 768** mismatched RGB lanes.
The input/film fixture-integrity CTest checks exact hashes, original HIP build
identity, all image rows and the unchanged original transparent-budget control.
It does not implement a host transport formula.

| Gate | Result | Scope |
| --- | ---: | --- |
| Full build | Passed | All 32 hardware threads, including the changed kernel ABI |
| Host | 177 / 177 | Complete host selection, 32 workers |
| HIP | 188 / 188 | Complete suite, one device test at a time; 496.27 s |
| Fallback | 190 / 190 | Complete suite after HIP; 384.38 s |
| Strict native Vulkan | 7 / 7 | 129 successful SPIR-V compilations; no DXC/DXIL loader matches |
| Four-scene 256-spp follow-up | **5 / 6 finite** | All renders and all 15 pass comparisons complete; runner exits 2 |

The seven focused backend gates are lifetime, volume boundary, volume-emission
film, lamp routing, zero-BSDF, film routing and SSS exit. They include the full
production module used by the new scene regression. Native Vulkan is a focused
gate, **not** a claim that the unrelated area-sampling and f16/f64 remainder
failures have been repaired. This change includes no Luisa implementation edit;
the published SDK remains `03a0f5158b53768abefea555f480f87ee5bc5a1e`.

## Full-resolution follow-up, not a new paired benchmark

Exact retained geometry/image bytes and refreshed socket controls are reused.
Original Cycles references are the retained equal-pass campaign, not fresh
paired timings. All runs use 256 samples, native fast math and staged main /
separate shadow queues. No concurrent build, profiler or render overlaps them.

| Scene | Extent | Render seconds | Session init seconds | Main frame |
| --- | --- | ---: | ---: | ---: |
| Barbershop, run 1 | 2048x858 | 38.1444 | 59.2917 | 92 fields / 416 B |
| Lone Monk | 1440x1080 | 12.8658 | 46.7018 | 54 fields / 216 B |
| Monster | 1080x1080 | 13.8330 | 53.5706 | 69 fields / 276 B |
| Classroom | 1920x1080 | 17.5897 | 39.0266 | 64 fields / 256 B |
| Barbershop, run 2 | 2048x858 | 38.1387 | 18.0246 | 92 fields / 416 B |
| Barbershop, run 3 | 2048x858 | 38.2437 | 18.4055 | 92 fields / 416 B |

The Barbershop median is **38.1444 s**, versus retained Cycles 25.3775 s
(about 50.3% slower). This is not an isolated speedup estimate against the
previous 38.4822 s follow-up. Removing the diagnostic-unused loop counter
removes one persistent field. Frame sizes were 416 / 220 / 280 / 260 B in the
table's scene order; Barbershop remains 416 B because of layout padding.

Initialization includes JIT/setup/baking, not just compiler execution.
Main shader caching is disabled, while downstream caches keep normal policy.
The main HIPRTC bitcode link took 24,934.57 ms on first encounter and
89.12 / 89.47 ms on repeats, producing a same-sized 861,120-byte code object.
The separately logged 848,632-byte object is LLVM bitcode, not final ISA.
This localizes much of the first/repeat difference to the link interval; it
does not independently identify the downstream cache mechanism or establish
a controlled cold-JIT regression. No caching or compiler-policy change is made.

All 46 channels are finite except Classroom's separated passes: 18 invalid
lanes at eight pixels, with exactly the same NaN/+Inf/-Inf masks as the previous
film-routing capture. DiffDir/GlossDir have seven invalid pixels each;
DiffInd/GlossInd have two each. Combined being finite does not pass this gate.
Original Cycles has different invalid-pixel locations; the original/actual
comparisons retain their reported excluded unions. The runner records all six
observations and exits **2**. There is no numeric waiver or epsilon repair.

First-run DiffInd relative RMSE remains 7.126% / 12.882% / 2.553% / 17.820%
for Barbershop / Monk / Monster / Classroom. Combined relative RMSE is
1.052% / 1.240% / 0.548% / 0.353%. Four Combined triptychs were inspected for
gross topology/blackout changes; this is not a substitute for the split-pass
or path-state gates. Difference display scales are per scene, not comparable
error magnitudes across images.

### Source-quality control and remaining repeatability question

The previous film-routing images used a different XIR binary. Attributing all
before/after pixel changes to the lifetime repair would therefore be invalid.
An isolated control loads only the preserved old renderer runtime, while
retaining the current executable, core, XIR, coroutine and HIP libraries.
`LD_DEBUG=libs` confirms the actual provider paths; SHA-256 checks confirm all
six implementation identities. Neither build outputs nor library search paths
outside that child process are changed.

Lone Monk is rerun at the same 1440x1080 / 256 spp / seed 0. The same-SDK
old/new comparison still differs locally, so a second control repeats the
current input and binaries without any code change:

| Comparison | Normal RMSE | Combined RMSE | DiffDir RMSE |
| --- | ---: | ---: | ---: |
| Old runtime / current SDK vs repaired runtime / current SDK | 0.000114827 | 0.000498861 | 0.00313942 |
| Current input and binaries, run 1 vs repeated run | 0.000127948 | 0.000531415 | 0.00326515 |

The repeated-current Normal difference exceeds 0.001 at 177 pixels, and its
DiffDir difference at 162 pixels (the latter in x=599..840, y=723..729).
These are descriptive counts, not new acceptance thresholds. Run-to-run
variation is already of the same magnitude as the intervention. Its cause is
**unresolved**; neither RNG divergence, acceleration tie ordering nor a floating
point explanation is proved. In particular, this is not evidence that the
removed bound was exhausted in Monk or that SDK integration caused its pixel
changes. No bitwise requirement, software math path or tolerance relaxation is
introduced. This repeatability finding remains part of the next structural
diagnosis, separately from the deterministic two-slab regression.

## Reproduce and audit

Run from the explicit Psycles worktree:

```bash
cmake --build build --parallel 32
ctest --test-dir build --parallel 32 --output-on-failure -E '_(hip|fallback|vk|cuda|metal|dx)$'
ctest --test-dir build --parallel 1 --output-on-failure -R '_hip$'
ctest --test-dir build --parallel 1 --output-on-failure -R '_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 LUISA_VULKAN_DISABLE_DXC=1 \
  ctest --test-dir build --parallel 1 --output-on-failure \
  -R '^psycles\.luisa_cycles_(path_lifetime|volume_boundary|volume_emission_film|lamp_routing|zero_bsdf|film_routing|svm_subsurface_exit)_vk$'
python docs/validation/2026-09-09/native-path-lifetime/validate_results.py
```

[results.json](results.json) pins source/fixture/binary identities, gate logs,
the six full commands and 15-pass metrics, isolation controls and image checks.
`compare_followup.py` compares captured images only; it supports selected scenes
and explicit manifests for isolation controls. `capture_results.py` takes the
evidence, canary, same-SDK-control and current-repeat directories and emits the
snapshot to stdout. `validate_results.py --live` additionally requires those
retained files and the captured implementation bytes. After another source or
binary change, default report integrity and a new renderer gate are distinct
operations. The original audit's red captures remain immutable historical
evidence; this report supersedes only its S1 production status.
