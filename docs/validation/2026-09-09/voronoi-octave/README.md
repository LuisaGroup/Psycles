# Voronoi octave structure and original GPU state

This checkpoint restores two concrete Cycles SVM differences: a duplicated
runtime F1 fallback and a wrong homogeneous W coordinate in 3D Voronoi
position. The independent shared-F1 full-scene control improves median
Barbershop surface time by about 1.17%, and render time by 0.66%. This is a
small structural gain, not the solution to the large remaining surface gap.
No inline policy, register cap, launch policy or bit-matching math changes.

The sole shader reference is original Cycles
`cb168525138fecc792cc393f94afc39582b0103c`. The source repair is Psycles `ffb3f7f2`, in
`src/luisa/cycles_voronoi.cpp`; the Luisa child stays at `da8fff856`.
[Results](results.json) pin the final parent revision, all binary and source
hashes, original GPU states, complete profiles and backend/scene validation.

## Formal causes and permanent reds

Original `kernel/svm/voronoi.h::fractal_voronoi_x_fx` chooses each octave as:

```cpp
feature == F2 ? f2(...) :
(feature == SMOOTH_F1 && smoothness != 0) ? smooth_f1(...) : f1(...)
```

Ordinary F1 and zero-smoothness Smooth F1 share one fallback. Psycles had
recorded F1 in both its F1 switch case and the Smooth F1 zero-smoothing arm.
That creates four neighborhood-search bodies instead of three even when
their numerical results agree. This is an application projection defect,
not evidence of a Luisa CFG-restructure defect. The repair restores the
original branch selection; it does not add a generic deduplication rule.

`tests/test_cycles_svm_voronoi_octave.cpp` inspects the recorded production
AST, without a host shader evaluator or arbitrary code-size ceiling. Under
both feature masks and dimensions 1–4, each active X-FX path must contain
two dimension-sized radius-one loop families (F1/F2), one radius-two family
(Smooth F1), and one octave loop. Inactive paths must contain none. Exactly
five configurations fail before repair, giving 3/8; all eight pass after it.
The host-specialized `Configuration` path already records one selected
algorithm and is not rewritten by this runtime-branch repair.

The new direct original-HIP node observer then exposes an independent
defined-state mismatch: 24 of 192 states fail only the 3D W output.
Cycles `voronoi_position(float3)` calls `make_float4(float3)`, whose overload
in `util/types_float4.h` explicitly supplies W=1. Psycles supplied W=0.
The octave lerp and final scale division act on that defined fourth lane;
it is not uninitialized vector padding or a last-bit discrepancy. A single
literal is corrected in the position projection. The 3D search-coordinate
embedding remains W=0, and 1D/2D/4D behavior is unchanged.

`tools/cycles_svm_voronoi_octave_oracle.hip` directly includes and executes
the original GPU Voronoi node. Its shared header is only an input marshaller,
not an expected-value implementation. The frozen fixture is byte-identical
to the original GPU output. Both masks, all dimensions, F1/F2/Smooth F1,
zero/positive smoothing, zero/fractional detail, metric and normalization
variations cover 192 nine-lane states. This is not a full Cartesian product
of every input parameter. All outputs are finite and compared at 2e-5
absolute-relative tolerance; payload PC advance is exactly 13 words.

The runtime regression changes from 168/192 to 192/192 without changing
the fixture or tolerance. The existing independent full-interpreter Voronoi
stream also passes with exact END/PC. Typed words, stack addresses, feature
masks and octave arithmetic are unchanged. Ordinary authored 3D materials
do not expose this W output; its state correction is not offered as an
explanation for Barbershop's large slowdown.

## Independent shared-F1 full-scene control

Four sequential runs use the same Barbershop 2048x858 / 64 spp / seed 0
bundle, 15 passes / 46 channels and 512-thread surface groups. No overlapping
heavy task is launched by the agent. Both A and B still use the old W
projection, isolating the shared-F1 change. These are not fresh Cycles pairs.

| Run | Surface GPU seconds | Render seconds | Session initialization |
| --- | ---: | ---: | ---: |
| A before | 5.846424 | 10.4964 | 18.6192 |
| B shared F1 | 5.781100 | 10.4309 | 60.0805 |
| B repeat | 5.783488 | 10.4335 | 18.6223 |
| A restored | 5.854947 | 10.5059 | 18.9749 |

Surface medians are 5.850686 / 5.782294 seconds and render medians
10.50115 / 10.43220. Two observations per treatment are a small control,
not a broad significance claim. Main shader caching is disabled; downstream
caches retain their normal policy. Initialization includes setup/baking and
is not matched cold JIT timing. It is excluded from render time.

Both A `.text` images are identical, SHA-256
`431b702c763b641fcc499690078472dc428fe62f7cc33297a4fab44c59e1e9fe`.
Both B images are identical,
`bf14b84fd4d81e0f966d1c59b2ffbbd378c89a188a6272e29ec3779f6607a734`.
Frozen implementation hashes also match within each pair.

Main-function ISA changes from 159,972 instructions / 852,836 bytes to
158,118 / 843,220. Static scratch load/store sites change 1,274/622 to
1,264/618; raw VGPR/SGPR spill metadata changes 504/65 to 500/66.
Fixed private bytes fall 2,480 to 2,464. Both retain 256 VGPRs, 107 SGPRs,
49 call sites, 20 image-sample sites and six stages / 93 fields / 416 frame
bytes. Static sites are not dynamic instruction or scratch-traffic counts.
Surface visits are 332,307,891 / 332,307,894 / 332,307,893 / 332,307,894.

All 46 channels are finite. Shared-B Combined relative RMSE against A is
1.753e-5 / 1.737e-5, versus restored A's 1.891e-5; DiffInd is
4.787e-5 / 4.709e-5, versus restored A's 5.906e-5. All 15 passes are archived.
The small intervention differences do not waive the original-Cycles
indirect-light residuals. The separate final profile enables both corrections
and is excluded from the shared-F1 treatment medians.

## Final version and four-scene follow-up

With both corrections, the final 64-spp profile records 5.798762 surface GPU
seconds and 10.4441 render seconds, with 60.2268 seconds of initialization.
Its main-function ISA counts/resources are the same as shared B above, but
the changed W literal gives a distinct `.text` hash:
`b753a0c81ef0dd8da71aceaa062c4c35b4cfe8f3a8551a409d9f85ef278e9810`.
It executes 332,307,896 surface visits; all 46 channels remain finite.
Combined/DiffInd relative RMSE against A is 1.881e-5 / 5.842e-5.
This single final observation is not a separate causal estimate of the W fix.

The six final 256-spp renders use the same frozen binaries as this final
profile and retained original Cycles references. All 46 actual channels are
finite and all 15 pass comparisons complete.

| Scene / repeat | Extent / seed | Render seconds | Session initialization | Stages / fields / frame bytes |
| --- | --- | ---: | ---: | --- |
| Barbershop 1 | 2048x858 / 0 | 40.0252 | 17.9685 | 6 / 93 / 416 |
| Lone Monk | 1440x1080 / 0 | 13.6398 | 11.4797 | 4 / 55 / 220 |
| Monster | 1080x1080 / 0 | 14.9535 | 52.6430 | 6 / 70 / 280 |
| Classroom | 1920x1080 / 1 | 18.2805 | 11.0960 | 5 / 65 / 260 |
| Barbershop 2 | 2048x858 / 0 | 39.9800 | 17.5688 | 6 / 93 / 416 |
| Barbershop 3 | 2048x858 / 0 | 39.9197 | 17.5393 | 6 / 93 / 416 |

Barbershop's median is 39.9800 seconds: 0.58% below the preceding checkpoint's
40.2134, but still 57.5% slower than the retained Cycles median of 25.3775.
This temporal comparison is not a new paired benchmark or isolated treatment
effect. First-run Combined relative RMSE for Monk/Monster/Classroom/Barbershop
is 0.0124076 / 0.00547855 / 0.00353320 / 0.0105212; DiffInd remains
0.128815 / 0.0255253 / 0.178203 / 0.0712617. Original Classroom's 25 invalid
DiffDir and 27 invalid GlossDir pixels remain explicitly excluded by the
comparator's invalid-union rule; no actual invalid pixels are hidden.

## Validation and reproduction

The full 32-thread build passes. Host tests pass 172/172, HIP 184/184,
fallback 186/186, and strict native Vulkan canaries 5/5. The Vulkan loader
audit captures 51 successful native SPIR-V compilations and no DXC/DXIL load.
It covers lamp routing, bump state, BSDF dispatch, closure guards and the new
Voronoi states. Existing native Vulkan f16/f64 remainder failures remain a
separate open issue; this focused canary is not a claim that all Vulkan tests
pass. No child source changed, so its earlier 133-test `unit*` gate is not
presented as a fresh run here.

Evidence: `/var/tmp/psycles-voronoi-octave-aqoNR5`. The formal analysis,
structural red/green, original GPU observer output, 24-state runtime red,
final build/tests, five profiles and six 256-spp canaries remain there.
The four first-run Combined triptychs are inspected. Canary comparisons use
retained equal-pass Cycles references and the exact prior geometry/image
bytes with fresh socket metadata, not fresh paired Cycles timing trials.

```bash
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_svm_voronoi_octave_oracle.hip -o /tmp/voronoi-oracle
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure \
  -R '^psycles\.luisa_cycles_svm_voronoi(_octave)?_hip$'
```

Run profile dumps only from a fresh isolated directory: `LUISA_DUMP_LLVM_IR=1`
writes to process cwd, independently of the HIP ISA output path.
Use [archive_profiles.py](archive_profiles.py) for the four control directories
plus `--final`, then [archive_results.py](archive_results.py) with the evidence
directory and final `--parent` / `--child` revisions.

The [Noise audit](../noise-codegen/README.md) does not support missing 3D
inlining as the large-gap explanation. Per-surface efficiency, residual
DiffInd/shadow work, 164 resource identities, remaining native nodes/private
legacy displacement bridge, and general CFG proof obligations remain open.
