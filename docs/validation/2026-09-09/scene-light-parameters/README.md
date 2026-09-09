# Scene-owned light parameters and Barbershop surface cost

Moving area/spot parameters back to Cycles' scene-preparation phase removes
a real ownership/structure mismatch, but explains little of the remaining
Barbershop slowdown. A full-scene A/B/B/A control changes median surface GPU
time by -0.73% and render time by -0.22%, with two observations per treatment.
The main code loses only 30 instruction sites and two calls. This is not a
large speedup or proof that missing inline boundaries caused the gap.

The original reference is Cycles `cb168525138fecc792cc393f94afc39582b0103c`;
the baseline is Psycles `124cee8c`, and the repair is `ff8385c1`, with Luisa
unchanged at `da8fff856`. The source/binary hashes, fixtures, commands and
full pass metrics are pinned in [results.json](results.json).

## Scene constants no longer become per-ray trigonometry

Original `scene/light.cpp::copy_to_kernel` constructs immutable derived light
fields during scene synchronization. Original GPU light sampling, forward
intersection, light-tree importance and volume intervals consume the table.
Psycles instead stored raw angles and regenerated these values per use.
In the actual Barbershop surface LLVM, a full-spread area's tangent was even
computed before an eager select chose `FLT_MAX`. This is a phase/state
ownership mismatch, not a missing inline annotation.

| Native light | Scene-owned fields now prepared once |
| --- | --- |
| Area | `tan_half_spread`, `normalize_spread` |
| Spot | `cos_half_spot_angle`, `half_cot_half_spot_angle`, reciprocal `spot_smooth`, `cos_half_larger_spread`, `ray_segment_dp` |

The field constructors mirror the original source, including the exact-pi
sentinel, Taylor threshold, raw transform-axis squared lengths and reciprocal
blend width. The existing host packer rebuilds them with light/transform data.
Device consumers use the original parameters and predicates; the approximate
`light_flag_full_spread` shortcut and raw spot-angle fields are removed.
Area attenuation uses the original cross/dot tangent, not a unit-vector-only
square-root approximation. Zero-blend spot attenuation and light-tree energy
use the original smoothstep comparisons, without a `safe_divide` substitute.

Authored angles needed by host light-tree construction now come from a
host-only source-ID sidecar, not duplicate raw fields on the device. Portals
remain an unattenuated, unclamped primitive proposal. No SVM word, payload,
stack address, PC, feature mask, RNG dimension, inline policy, register cap,
launch policy or fast-math setting changes.

## Original tables and GPU states bound the regression

Seven production-AST consumers initially fail the no-scene-trigonometry gate
(0/7), then all pass (7/7). The test follows custom callables and inspects
TAN/COS operations only where they are scene-owned; geometry/RNG-dependent
sampling cosine is allowed. It is not an arbitrary code-size ceiling.

The metadata-only original observer captures 25 boundary lamps plus all 12
Barbershop area/spot lamps. Barbershop also has three point lamps outside this
field family's scope. The replayable observer patch and packer retain exact
field bits; the packer computes no expected shader values. The original
checkout's inherited `light.cpp` and `svm.cpp` modifications were restored
byte-for-byte, and its observer build was restored afterwards.

The runtime test prepares an actual Psycles analytic-light scene and compares
all 37 parameter tables, then compares 296 device executions to direct
original Cycles GPU calls. Each state contains 40 lanes covering attenuation,
UV, surface and volume samples, PDFs, positions, normals, and valid/invalid
intervals. Rejected samples compare validity, not undefined payloads. Device
consumer checks upload identical observed parameters; the host construction
is a separate numerical gate, and full scenes exercise both together.

Area probes include zero, small, Taylor-boundary, ordinary, near-pi and exact
pi spreads, rectangle and ellipse. Spot probes include zero/finite radius,
zero/ordinary/full blend, wide cones, nonuniform and negative scale. Blender
clamps an authored zero spot angle to 0.017453292; this is the observed input,
not a claim of an unclamped zero-angle fixture. All captured lamps have
`normalize=true`; this does not establish parity for non-normalized lamps.

Host fields use 8e-6 relative tolerance and exact classifications. Device
predicates are exact; finite arithmetic uses 2e-4 absolute-relative tolerance.
Interval lengths are normalized to the authored ten-unit segment. No runtime
precision workaround or strict-math mode is added.

## Denormal modes and ill-conditioned roots are explicit

Full-pi area normalization is a positive subnormal in the original host
table. Original HIP preserves it, whereas fallback explicitly enables
FTZ/DAZ; the tested native Vulkan environment also flushes it. The same
original GPU consumer is captured in preserve mode and with
`-fgpu-flush-denormals-to-zero`. A separate seven-value device classification
control selects the matching original fixture; no light result selects its
own expectation. Both normal/signed controls and light predicates remain
exact. This resolves 81 initial cross-environment mismatch lanes without
changing application or backend math.

Initial transverse rays through a 0.001-radian area cone produce quadratic
coefficients by subtracting O(1e6) terms to obtain O(1) values. Merely disabling
FMA contraction in the original GPU diagnostic moves a root by up to 0.178
units. Earlier on-axis cone and nearly coplanar rectangle diagnostics also
show native sensitivity to adjacent float inputs. Those singular probes are
retained as diagnostics, not cross-backend exact-root gates. The permanent
area interval rays check emitting-halfspace clipping, segment limits and
outside-footprint rejection with nearly normal, slightly oblique directions.
Zero-blend exact-edge `0*inf` NaNs are outside finite-only fast math; finite
blend still tests the exact cone edge, and zero blend tests interior/exterior.

Strict native Vulkan still fails 36 numerical lanes in rectangle-based area
surface sampling: PDF, position, UV and attenuation. Parameters, validity and
interval predicates agree. A baseline-only probe using restored old raw-angle
APIs fails 183 lanes; all 36 remaining failures already occur there with the
same printed values. Thus they are not introduced by this repair. Fast
GLSL.std.450 versus HIP/OCML math is a candidate cause, not a proved diagnosis.
The regression remains failing and its tolerance is not relaxed. The other
five native canaries pass; no DXC/DXIL library is loaded.

## Full-scene control shows only a small gain

Four sequential Barbershop runs use 2048x858, 64 spp, seed 0, 15 passes /
46 channels and 512-thread surface groups. Surface time is the sum of the
identified kernel's GPU dispatch durations; render time is render-only wall
time. Initialization includes JIT, scene setup and baking, and is separate.
Main shader caching is disabled; downstream caches retain normal policy.
No agent-launched heavy task overlaps a timed run. These are not fresh Cycles
timing pairs, and two samples per treatment are not a significance study.

| Run | Surface GPU seconds | Render seconds | Session init seconds |
| --- | ---: | ---: | ---: |
| A before | 5.803689 | 10.4584 | 18.8122 |
| B prepared parameters | 5.783963 | 10.4395 | 55.6852 |
| B repeat | 5.782816 | 10.4310 | 18.6046 |
| A restored | 5.847735 | 10.4580 | 19.9791 |

Surface medians are 5.825712 / 5.783390 seconds; render medians are
10.45820 / 10.43525. The two A surface observations differ by 0.76%, so the
small treatment effect needs that variability caveat. Each A/B pair has
identical frozen implementation binaries and surface `.text` within the pair.
After the restored control, final B sources are restored byte-for-byte.

Main ELF-function instruction sites change 158,118 -> 158,088 and bytes
843,220 -> 843,084. Calls change 49 -> 47. Both retain 256 VGPRs, 107 SGPRs,
2,464 fixed private bytes, 500/66 raw VGPR/SGPR spill metadata and 1,264/618
static scratch load/store sites. These are static counts, not executed
instructions or memory traffic. Frames stay six stages / 93 fields / 416 B;
surface visits remain 332,307,891..332,307,896 across the four runs.

All 46 channels are finite. B Combined relative RMSE against A is
1.864e-5 / 1.366e-5, and DiffInd is 5.789e-5 / 5.870e-5. Restored A itself
differs by 1.394e-5 / 5.917e-5. These intervention checks do not waive the
remaining original-Cycles indirect-light or visibility differences.

## Four-scene 256-spp follow-up remains performance-limited

Six final renders use the same frozen implementation binaries as prepared B,
retained equal-pass Cycles references, and the exact earlier geometry/texture
bytes with refreshed socket metadata. All 46 actual channels are finite and
all 15 pass comparisons complete. The four first-run Combined triptychs have
been inspected. These are follow-ups, not newly paired Cycles timings.

| Scene / repeat | Extent / seed | Render seconds | Session init seconds | Stages / fields / frame bytes |
| --- | --- | ---: | ---: | --- |
| Barbershop 1 | 2048x858 / 0 | 39.9809 | 18.9804 | 6 / 93 / 416 |
| Lone Monk | 1440x1080 / 0 | 13.6267 | 12.0773 | 4 / 55 / 220 |
| Monster | 1080x1080 / 0 | 14.9215 | 50.5102 | 6 / 70 / 280 |
| Classroom | 1920x1080 / 1 | 18.1945 | 36.8537 | 5 / 65 / 260 |
| Barbershop 2 | 2048x858 / 0 | 40.0251 | 18.5425 | 6 / 93 / 416 |
| Barbershop 3 | 2048x858 / 0 | 40.0313 | 17.8490 | 6 / 93 / 416 |

Barbershop's median is 40.0251 seconds, 0.11% above the preceding 39.9800:
effectively unchanged at this measurement scale and not a causal regression
estimate. It is still 57.7% slower than the retained Cycles median of 25.3775.
The small 64-spp profiled improvement does not establish a material 256-spp
render improvement. Initialization is not compiler-only and includes normal
downstream cache variation; it is excluded from render timing.

First-run Combined relative RMSE for Monk/Monster/Classroom/Barbershop is
0.0124043 / 0.00547855 / 0.00353320 / 0.0105212; DiffInd is
0.128817 / 0.0255253 / 0.178203 / 0.0712617. The original Classroom reference
has 25 invalid DiffDir and 27 invalid GlossDir pixels, explicitly excluded
by the comparator's invalid-union rule. No actual invalid pixels are hidden.

## Verification and reproduction

Full builds use 32 threads. Host tests pass 174/174. Complete HIP and fallback
gates pass 185/185 and 187/187; the new arithmetic-environment control also
passes separately on both. Strict native Vulkan is 5/6, with eight observed
native compilations and the numerical failure above retained.
The initial mistakenly parallel HIP test run was interrupted after concurrent
JIT/device jobs stalled; only its owned processes were terminated. Subsequent
device suites run sequentially. No system configuration or GPU reset changed.

Evidence is retained at `/var/tmp/psycles-light-parameters-cAduo0`. Use
[archive_profiles.py](archive_profiles.py) for the four profile directories
and [archive_results.py](archive_results.py) for the validated campaign and
revision-pinned JSON. Original GPU fixture reproduction:

```bash
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_light_parameters_oracle.hip -o /tmp/light-oracle
/tmp/light-oracle tests/data/cycles_light_parameters.txt > /tmp/light-state.txt
# Repeat with -fgpu-flush-denormals-to-zero for the FTZ companion.
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure --parallel 1 \
  -R '^psycles\.luisa_cycles_light_parameters_hip$'
```

Keep LLVM dumps in isolated working directories: their path follows process
cwd, independently of the HIP ISA output path. The
[static function/texture audit](../surface-static-audit/README.md) does not
support missing main SVM/3D Noise/microfacet inline boundaries or float32
expansion of byte textures as the main explanation. The next concrete
ownership mismatch is per-ray light inverse reconstruction versus original
scene-owned object inverse transforms. Remaining light fields, indirect
path work, unimplemented native nodes, the private legacy displacement
bridge and generic CFG proof obligations remain open.
