# Psycles / LuisaCompute handoff

Updated 2026-09-12 12:00 +08:00. The owner requested continued work toward
Cycles-or-better HIP performance. No background build, render, test or replay
remains running at handoff. The current raw-frame result and exact SDK gates
supersede the older pending-gates snapshot below.

This replaces the obsolete July/August handoff, whose older Cycles revisions,
legacy execution paths and timing claims are not current. Its full text remains
available with `git show 8bf668f0:HANDOFF.md` and in the dated historical reports.

## 1. Exact worktrees and publication

Only develop in `/home/mike/Projects/Psycles-surface-svm` and its nested
`third_party/LuisaCompute`. Never use `/home/mike/Projects/Psycles` or the
independent `/home/mike/Projects/LuisaCompute` checkout.

First inspect, rather than trust cwd or this snapshot:

```sh
git -C /home/mike/Projects/Psycles-surface-svm status --short --branch
git -C /home/mike/Projects/Psycles-surface-svm/third_party/LuisaCompute status --short --branch
```

- Root branch: `codex/surface-common-projection`, tracking `origin/main`.
- SDK branch: `next`, tracking `origin/next`.
- SDK `7d1f44cb6` is published directly on `origin/next`. It tags only
  coroutine frame scalar byte-buffer accesses and lowers eligible HIP accesses
  through AMD raw-buffer intrinsics. The root gitlink is `556f5ef7` on
  `origin/main`.
- A fresh paired Barbershop benchmark with `PSYCLES_DISABLE_SHADER_CACHE=1`
  measures Cycles HIP at 25.4221 s and current raw-frame Psycles at 32.8914 s
  (1.29381x, 29.38% slower). Three
  additional raw repeats are 32.97--32.99 s; the frozen stock capture was
  36.58 s. The exact surface object is 409,104 ->
  408,716 code bytes, 2,368 -> 2,192 private bytes, and 449 -> 407 VGPR
  spills, with 256 VGPR / 107 SGPR unchanged. See
  `docs/validation/2026-09-11/frame-raw-lowering/README.md`.
- Static ISA review finds the largest excess mask chain in SVM
  `node_closure_bsdf_skip`: 3,116 `v_cndmask_b32` and 794
  `v_cmp_eq_u32` versus Cycles' 1,333 and 157. This is closure-opcode
  dispatch lowered from a side-effect-free switch, not coroutine resume
  dispatch. Grouped-switch and constant-table source probes preserved image
  output but were within timing noise; an early-return probe failed XIR
  verification. See
  `docs/validation/2026-09-11/surface-register-pressure/README.md`.
- The exact current-SDK driver passed 6 host, 17 HIP, 17 fallback, and 17
  strict native Vulkan tests. Its evidence is in
  `/var/tmp/psycles-sdk3bc7-patched-20260911-1789137615`.
- Root `8bf668f0066072b3e2d4cb0117d0ed59d79a845d` is published: generic
  recording-time shading-frequency pruning, native fast-angle reuse, tests,
  original-GPU fixture and four-scene report.
- The earlier SDK317 publication (`31721e1f...`) and its parent commit are
  historical context only; the current nested SDK is `7d1f44cb6` as recorded
  above.
- `5c7de2bb9`'s actual frame-relocation/compatible-resume-queue regression is
  preserved. No isolated dependency symlink/gitlink adaptation was imported.

Untracked files deliberately **not committed or overwritten**:

- User's 60 `hip_kernel_{before_opt,after_opt,final}_*.ll` files and `.rocprofv3/`.
- Unfinished W1 authored-input/probe drafts:
  `tests/cycles_shadow_origin_fixture.h`, `tools/cycles_shadow_origin_oracle.hip`.
  They have not been compiled or captured. Do not present them as passing tests.

Preserve existing stash `26bee45c309ba0f6f6303bc99169a021a82658a2`,
`/var/tmp/luisa-publish-coro-bQIjua`, and the evidence/frozen runtimes below.
The isolated SDK at `/var/tmp/luisa-next-cfg-replay-BIaLdW` was a reduction
worktree, not the production SDK; do not copy its dependency adaptations.

## 2. Hard constraints

Read [DEVELOP.md](DEVELOP.md). Reproduce Blender Cycles **5.2.1** SVM word
stream, typed payloads, stack, PC, dispatch, closure state, feature masks and
surface/volume/displacement/bump control flow. Cycles itself is the only oracle;
no CPU shader, sampler or reference renderer. Host input packing, compiler
tests and image analysis are allowed.

Use ordinary C++ guards over complete immutable finalized metadata to omit
DSL when the native predicate is provably inactive. Unknown retains the path;
scene/session rebuild recomputes the proof. No scene names, measured hot paths,
prerenders, hardcoded scene array sizes or benchmark thresholds. Static stack
bounds remain compiler analysis, not profiling.

Keep fast math. Do not pay for one-ULP/bit alignment, force noinline, or add slow
software arithmetic/texture paths. Keep ordinary scalar/vector default zero
initialization. Generic coroutine facilities stay in Luisa; renderer policies
belong in Coro Ext/Handler clients. Use `stream << scheduler.dispatch(...)`.
No per-thread/per-resume malloc.

Compiler fixes require formal cause -> minimal authentic failure -> permanent
regression -> generic repair -> full original-module validation. Build with all
32 threads. Host CTest can use 32; device tests are serial. Validate HIP, then
fallback, then strict native XIR -> SPIR-V with DXC disabled. Do not overlap
timed renders with compilation, tests, replay, profiling or image comparison.

## 3. Historical SDK317 integration snapshot

Evidence directory `B=/var/tmp/psycles-coro-predicate-integration-Ba05LR`.
`B/STATUS.md` was an intermediate note; **this handoff supersedes its pending
remaining-HIP status**. That job finished before handoff:

| Gate against actual SDK317 | Result |
| --- | --- |
| Root full all-target build, `--parallel 32` | Exit 0; `B/full-build.log` |
| Root full host | 186/186; exit 0; `B/full-host.log` |
| Root HIP focused structural/render controls | 15/15; exit 0; `B/focused-hip.log` |
| Root remaining HIP | 178/178; exit 0; `B/remaining-hip.log` |
| Complete root HIP coverage | 193/193 across the disjoint 15 + 178 selections, not one fabricated run |
| Actual-root SDK driver host controls | 6/6; 322 assertions, 30 executed cases |
| Standalone actual-SDK host suite | **80/81, exit 8**, existing flaky byte-equality test below |
| Three new coroutine controls | 231 assertions pass (11/161 + 3/56 + 3/14) |
| Same full original Barbershop module, actual SDK libs | Pass; details below |
| Current SDK317 root fallback / native Vulkan | **Not run** |
| Current SDK317 SDK17 runtime HIP/fallback/native matrices | **Not run** |
| Current SDK317 four full-resolution scene canaries | **Not run** |

Both HIP detailed logs were copied into B. Host builds/replays overlapped parts
of correctness validation; durations are not renderer/JIT performance results.

Important ABI facts: Scope/Stats changed and certificate schema is 9. The root
renderer/libraries/tests were rebuilt together. Never preload an old observer
or mix old test executables with current libraries. Root app runtime hash alone
does **not** distinguish SDK317 from its predecessor:

```text
build/bin/psycles_render_blender_scene
  83ede06f45502b78df7ff6888d389306ea1ca763110232654eb799f1420a99e9
build/libpsycles_luisa_runtime.so
  86fe7274ae712134bcba6b6262f6fea7e69263fc73a2998856858b4ee3a950fd
build/bin/libluisa-xir.so
  1e23422de25579f4143259d91d897fac3723d5654f6f8f8f7dbb9e2c88fb3e1f
build/bin/libluisa-coro.so
  80c2ec120b8b1a9f5e909ece9094fd1dd7668a9c63fdf710a6d5affbea1fd90b
```

### Immediate host failure: test fix NOT implemented

`third_party/LuisaCompute/src/tests/unit/dsl/test_switch_case_group.cpp:134`
asserts `a.serialize() == b.serialize()` for legacy vs one-label group.
Same actual executable/cwd repeated exit 0, 255, 255. N repeated 0, 0, 0.
GDB inspection, without changing/rebuilding source, proved the two 451-byte
streams differ **only in uint3 block_size padding**: field starts at 254,
xyz are zero in both, padding offsets 266-269 differ, next statement starts270.
Function hashes are equal (`9753947726266053246`) and structural checks pass.

`CallableLibrary::ser_value` memcpy-copies sizeof(T); VectorStorage<T,3> has
tail padding. Hashing uses three semantic lanes. There is no documented
canonical-byte guarantee; unordered-map emission also prevents one globally.
The existing test already acknowledges non-byte-idempotent roundtrips.

The approved next repair was **test-only**, but stopped before any edit:
retain direct/group hash and single-label shape checks; deserialize both
libraries, duplicate/recompute their hashes (do not trust stored hashes), and
compare semantic AST/selector/label/block-size fields. Do not simply delete
the assertion, add serializer padding clearing, or claim a new CFG bug.
Rebuild with32, repeat the regression e.g.100 times, then all81 host tests.
Artifacts in B: `switch-serialize-typed-observer.log`,
`switch-serialize-observer.py`, `switch-original-failing-source.cpp`,
`switch-original-failing-test`, `sdk-host-{ctest,focused,build}.log`.

Root `build` has SDK tests OFF. The separate `build/luisa-tests` was freshly
reconfigured/rebuilt from actual SDK317 (81 targets), with HIP/fallback ON and
Vulkan OFF. Host selection **must be anchored**:

```sh
ctest --test-dir build/luisa-tests --parallel 32 --output-on-failure \
  -L '^(unit_xir|unit_coro)$'
```

Unanchored `unit_xir|unit_coro` selects85 including four runtime cases via
`unit_coro_runtime`. Correct the older SDK report's example at next doc update.
Full build/selection recipes: `B/sdk-host-plan.md` and
`/var/tmp/psycles-holdout-CXlJR7/coro-transition-predicate-reduction/integration-preflight.md`.

### Generic coroutine change and complete-module evidence

The analysis keys May states by latest resume owner + raw semantic block +
Boolean valuations. Exact private Boolean slots are distinct from SSA load
snapshots; aliases/unsupported producers/writable Extensions are conservative.
Selected successors are sealed and consumed by discovery, dataflow, splitting,
actual branch emission, materialized ABI and scheduler edges. No renderer or
scene rule exists. Sparse `DEAD=(USE union DEF)-LIVE_AFTER` retirement prevents
dead intra-block relations exhausting ROBDD budget; widening remains safe.

Read SDK `docs/validation/2026-09-10/coro-feasible-transitions/README.md` and:

- `H/coro-transition-predicate-reduction/predicate-retirement-design.md`
- `H/coro-feasible-consumer-review.md`

Here and below `H=/var/tmp/psycles-holdout-CXlJR7`.
Authentic pre-fix red and pre-retirement libraries are preserved; never replace
them with new expected outputs.

Actual ABI-matched replayer: `H/coro-replay-actual317-doTs4V`, with README,
`replay-results.json` and complete log. Three source files copied unchanged
from `H/coro-distill-current-BsxSjd`; actual generated includes are
`ROOT/build/include`, not a guessed nested include directory.
It consumed the exact original 1,111,095-instruction module (SHA
`3e62bd7b664d5deb3bb1afb122b20bf7c23c08d6762fc461eccfbe9de1131568`), not a sibling:

- read-only distillation, states2882, widened0, selected99;
- all six actual token-effect sets match the certificate;
- production post-materialize/source-detach/reg2mem verifier: zero errors;
- normalized instructions1,114,278; graph6 nodes/12 boundaries;
- surface incoming only closest/volume, down from six in the frozen baseline;
- **frame unchanged: 92 fields / 416 B**. Masks in the log are cardinalities,
  not proof of identical slot membership or measured bandwidth savings.

Raw intermediate split/materialized dominance diagnostics (156 each) precede
the required production reg2mem boundary; baseline has the same kind of issue.
Do not patch those individually or misreport final verification as failing.

### Ready actual-root SDK runtime driver

`D=H/sdk-predicate-integration-fyrDOE`: freshly configured/built22 executables,
57 tests =6host +17 per backend. Host six already passed; backend51 unrun.
README, original driver sources/diff, provenance, binary hashes and
`host-results.json` are retained. It forces actual ROOT/bin loader paths,
correct argv0 and empty LD_PRELOAD, with native guards for Vulkan.

```sh
ctest --test-dir /var/tmp/psycles-holdout-CXlJR7/sdk-predicate-integration-fyrDOE/build \
  --parallel 1 --output-on-failure -L '^integration_hip$'
# Then integration_fallback, then integration_vk (same driver supplies guards).
ctest --test-dir build --parallel 1 --output-on-failure -R '_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 LD_DEBUG=libs \
  ctest --test-dir build --parallel 1 --output-on-failure \
  -R '^psycles[.]luisa_cycles_(path_lifetime|volume_boundary|volume_emission_film|lamp_routing|zero_bsdf|film_routing|svm_subsurface_exit|ray_portal_state|ray_portal_render|holdout_state|holdout_render|surface_queue|shading_terminator|svm_bsdf_dispatch|svm_hair_scattering)_vk$'
```

Require actual native SPIR-V compilation and no DXC/DXIL loader entry, not only
exit0/cache hits. If source HEAD changes, rebuild/reconfigure driver identity;
never force its expected SHA to mask a header/library mismatch.

## 4. Last full-scene performance capture: PRE-SDK317 only

[Phase A report](docs/validation/2026-09-10/shading-terminator-pruning/PHASE_A.md)
and its archived JSON are authoritative. All are **Psycles HIP**, 256spp,
fastmath on, native extents, main shader cache off, one observation per scene.
References are retained original Cycles HIP images, not fresh timing pairs.

| Scene | Extent | Render s | Session/JIT s | Main frame B |
| --- | --- | ---: | ---: | ---: |
| Barbershop | 2048x858 | 37.1113 | 55.0679 | 416 |
| Lone Monk | 1440x1080 | 12.7550 | 42.9222 | 200 |
| Monster | 1080x1080 | 13.6912 | 55.2632 | 272 |
| Classroom | 1920x1080 | 17.5279 | 40.2551 | 252 |

Render excludes scene compile and session creation; session/JIT includes
allocation/upload/setup, not compiler-only time. There is no current demonstrated
speedup from the host guard or new coroutine analysis. Pre-SDK317 tests were
host186/186, HIP193/193, fallback195/195, native Vulkan4/4; they cannot substitute
for the pending new-SDK gates.

All four renders/60 pass comparisons complete; finite gate **3/4 failed**.
Classroom actual18 nonfinite lanes/8pixels vs reference70/27; Combined finite.
DiffInd relative RMSE: Barber7.12617%, Monk12.88149%, Monster2.55253%,
Class17.82034% (finite domain). Do not dismiss residuals as noise/one ULP.

`E=/var/tmp/psycles-shading-terminator-7owG2a/final-candidate-tgZZND` preserves
all four EXRs/logs/15-pass triptychs/finite masks and exact source/execution
manifests. Frozen matching PRE-SDK317 runtime:
`H/frozen-shading-final-7bqEc8/bin` (all ordinary dependencies/.data, no.cache).
Older published pre-guard runtime: `H/frozen-next-srB7dk`.
Do not run either archive with current libraries mixed in.

The four current-exporter bundles and **12 fresh v3 benchmark pairs have NOT
run**. Old controls under `/var/tmp/psycles-hidden-socket-U0KI5L/controls` fail
current exporter identity; do not relabel them. Use E's frozen command template
with new output directories and new full binary identity, then audit geometry,
loaded textures, metadata and raw current-compiler SVM words/bound identities
before paired runs. Read the report's [capture plan](docs/validation/2026-09-10/shading-terminator-pruning/CAPTURE.md).
Keep all 15 passes/46 channels, seeds/frames, original extents,256spp,64spp per
dispatch and scheduler options unchanged; no scene-specific performance tuning.

## 5. Classroom nonfinite: newly captured original-GPU boundary

Source/image audit: `H/classroom-split-nonfinite-audit.md` contains all bad
coordinates. Actual diffuse B has9 NaNs and glossy B9 +Infs, paired by pixel.
Final color divisors are normal/nonzero, so the nonfinite already entered raw
film before host output conversion. Combined is clamped before multiplying
split weights in both original and Psycles. Current ratio source spells the
same guarded a/b as Cycles; no authored explicit reciprocal difference found.

New independent original GPU probe:
`T=/var/tmp/psycles-native-pass-ratio-PyUr24`.
`inputs.h`, `probe.hip`, README, `build-provenance.json`, `capture-1.txt/.log`
and `capture-validation-1.json` are complete. First32-job fastmath compile0;
one GPU capture0 on RX9070XT/gfx1201;9 dynamic authored cases,16 RGB observations.
Original kernel/util source trees match cb168525 HEAD. No CPU expected math.

Ordinary/minimum-normal/exact-zero controls are finite. All six nonzero
subnormal sums retain their input bits and pass sum!=0, yet **original Cycles**
returns diffuse NaN/glossy+Inf. These persist through original PackedSpectrum
storage, film writes and film reads, while Combined stays finite. Maximum
subnormal's reciprocal is representable, so reciprocal overflow alone is NOT
an adequate explanation; denormal lowering/mode still needs inspection.

This proves a native function-boundary failure signature, **not** that the
actual Classroom paths supplied those inputs or that the image gate is fixed.
Next: matching production Luisa ratio/film GPU probe, exact LLVM/ISA for both,
then scoped real-pixel/sample sum/lobe/weight observation. No epsilon, software
division, global fastmath disable or invalid-pixel waiver has been implemented.

## 6. Other audited structural work, not implemented

### W1: lazy geometry and finite shadow-ray order

Read `H/w1-audit.md` and `H/w1-fixture-test-review.md`. The two root untracked
drafts author19 static-triangle cases and invoke original GPU functions, but
no native capture/build or runtime regression exists yet.

Native order is shadow_ray_offset -> shadow_ray_setup (finite D/t recomputed)
-> conditional integrate_surface_ray_offset using that new D. After the last
certificate changes P, native DOES NOT re-aim D/t again. Current
`path_kernel_direct_light_transport.cpp` calls a combined certificate/origin
helper before finite D/t recomputation: definite structural mismatch.
Triangle vertices/normals are also eagerly bridged before consumer predicates;
native heavy offset runs only inside triangle/smooth/cutoff/amount guards.
Reuse retained native ShaderData and existing native certificate; do not rerun
SVM, invent a second algorithm, or blanket-deduplicate legitimate native fetches.
Prove dynamic laziness with runtime resource counters and complete final IR,
not host recording counters. The draft has no motion coverage.

### S5: exact film metadata / late data-pass predicates

Read `H/s5-audit.md`, `H/host-guard-audit.md`, `H/s5-session-mask-review.md`.
Current AOV preparation is earlier/broader than native emission/termination
and film eligibility. Native pass policy includes volume/catcher/sample-count
auto additions; do not derive it from four booleans or only geometric volumes.
Use actual nonzero-reference shader.has_volume and original Film observer proof.
Scene/session reset is the specialization boundary; unknown/unsupported pass
inputs cannot justify pruning. No S5 production API or patch exists yet.

### Remaining broad goals

- Eight native semantic opcodes missing: RADIAL_TILING, BEVEL,
  AMBIENT_OCCLUSION, RAYCAST, AOV_START, AOV_COLOR, AOV_VALUE, SCENE_TIME.
- Private legacy displacement bridge and residual old execution code removal.
- Indirect/path/RNG parity remains unproved by aggregate images or event counts.
- Volume-stack exit order (swap-last vs native ordered shift) differs for three
  active volumes; Barbershop max2 is not evidence that this caused its slowdown.
- Volume NEE shadow and camera volume initialization cut placement still differ.
- Transparent roughness threshold integration and full native film predicates.
- Full multi-scene correctness/efficiency goal remains open.

## 7. Original authority and useful archives

Cycles source `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`; preserve inherited dirty
`intern/cycles/scene/light.cpp` and `scene/svm.cpp`. Original kernel/util files
were clean at capture. Production Blender:
`/home/mike/Projects/blender-install-5.2-hiprt/blender` (9e2066aef7ef).
Word observer Blender:
`/home/mike/Projects/blender-install-psycles-trace-5.2/blender` (cb168525).

Earlier bump-state work from the old conversation is no longer the immediate
uncommitted task: consult current Git/tests instead of restarting that handoff.
The historical SDK317 pending items are superseded by the current raw-frame
validation recorded at the top of this handoff. The remaining performance gap
to Cycles still requires paired multi-repeat scene benchmarks and further
generic HIP instruction-pressure work.
