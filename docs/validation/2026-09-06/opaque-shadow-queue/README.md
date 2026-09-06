# Terminate opaque shadow paths in INTERSECT_SHADOW

## Root cause and minimal change

Cycles 5.2.1 `kernel/integrator/intersect_shadow.h:150-184` has two disjoint
outcomes after traversal:

- Opaque hit: terminate the shadow path in INTERSECT_SHADOW.
- Otherwise, including zero recorded hits: queue SHADE_SHADOW for transparent
  evaluation and render-buffer output.

Psycles unconditionally suspended into SHADE_SHADOW and only then tested
`batch.blocked`. Film values were correct, but opaque paths unnecessarily
crossed another coroutine boundary and entered the shading queue.

The production staged driver now tests `blocked` before that suspension.
The loop is exposed as `DirectLightTaskEvaluator::trace_staged` so the
regression executes production control flow, not a test copy. It uses the
caller's existing `active` and `visible` variables: the extraction adds no
outer branch or duplicate loop state. This is a host-side DSL recording
method, not an additional device callable or an inlining directive.

The four-hit batch remains local to the INTERSECT_SHADOW -> SHADE_SHADOW
edge. Empty unblocked batches still enter shading. Exactly full transparent
batches still re-enter traversal. Opaque closure extinction discovered during
shading still terminates there, not speculatively in traversal. Inactive NEE
proposals perform no shadow traversal or shading.

No node stream, SVM dispatch, feature mask, closure semantics, default scalar
or vector initialization, compiler pass, `noinline`, or software floating-point
path is changed. The fused shadow consumer is unchanged.

## External oracle and permanent regression

Source: `/home/mike/Projects/blender-cycles-trace-5.2`, commit
`cb168525138fecc792cc393f94afc39582b0103c`.

`tools/cycles_shadow_queue_oracle.hip` includes the original Cycles HIP
`integrator_intersect_shadow`, state-flow functions and GPU SoA definitions.
Only the two BVH calls are replaced by prepared traversal results. This is a
queue-boundary probe, not a traversal/shading oracle or a reference renderer.
It records the real queued-kernel ID, both atomic queue counters, and the
termination predicate. The four input combinations cover blocked/unblocked
and empty/nonempty results, with transparent shadows both enabled and disabled.

The generated file is `tests/data/cycles_shadow_queue.txt`, SHA-256:

`c531c3f64ab719682e53a06cae75d2faebff4efa5282491695df2732c34739e8`.

`tests/test_luisa_cycles_shadow_queue.cpp` runs the actual production driver
through the wavefront coroutine scheduler. It checks executed continuation
counts, visibility, throughput, transparent depth, RNG offset, volume counter,
and logical path identity in 12 scenarios. Each scenario executes 67 paths
through a 19-slot pool, under both SoA and AoS storage, forcing multiple refills.
The attenuation callable is deliberately prepared test data; real native SVM
and geometry remain covered by the separate native-shadow oracle regression.

The confirmed pre-fix HIP failure is `red-hip-confirmed.log`: opaque scenarios
0 and 2 each executed 67 unwanted SHADE_SHADOW continuations; scenario 6
(a full transparent batch followed by opaque traversal) executed 134 instead
of 67. All state assertions passed. The initial fixture had omitted the
collector's ray-maximum sentinel for inactive hit slots; that fixture error
was corrected before recording this isolated red result.

Reproduce the external probe and focused device checks:

```sh
cd /home/mike/Projects/Psycles-surface-svm
probe_dir=$(mktemp -d /var/tmp/psycles-shadow-queue-oracle-XXXXXX)
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_shadow_queue_oracle.hip -o "$probe_dir/oracle"
"$probe_dir/oracle" > "$probe_dir/oracle.txt"
diff -u tests/data/cycles_shadow_queue.txt "$probe_dir/oracle.txt"
cmake --build build --parallel 32 --target psycles_luisa_cycles_shadow_queue_tests
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_shadow_queue_hip$'
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_shadow_queue_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 \
  ./build/bin/psycles_luisa_cycles_shadow_queue_tests vk --no-cache
```

## Validation

Evidence: `/var/tmp/psycles-shadow-queue-ulMBrB`.

- Full build, 32 threads: passed (`final-full-build.log`).
- 156/156 HIP tests: passed (`final-all-hip.log`).
- 158/158 fallback tests: passed (`final-all-fallback.log`).
- 100/100 core/adapter tests: passed (`final-core-adapter.log`), using
  `^psycles[.](cycles_|blender_)`, excluding only the previously failing
  `psycles.blender_export_render_settings`. This is not an all-CTest claim.
- 12 strict native-XIR Vulkan canaries: passed (`final-strict-vulkan.log`).
- The new coroutine test also passes with cache disabled and all three native
  Vulkan guards, with `LD_DEBUG=libs` recording native SPIR-V compilation and
  no DXC/DXIL library load (`final-native-vulkan-recompile.log`).
- The exact staged Psycles source image was exported into `source/` and built
  separately with 32 threads (`clean-build.log`, `clean-final-build.log`).
  Four focused HIP and four fallback tests pass there
  (`clean-final-hip.log`, `clean-final-fallback.log`). Its Psycles core/runtime
  libraries resolve to that isolated build. The wrapper deliberately imports
  the existing Luisa headers/libraries; this does not claim a clean Luisa build.

The three protected path-tracer changes and the unfinished Luisa gitlink are
not part of this change. Luisa itself was not edited.

## Whole Lone Monk HIP: normal-path control

Scene: `/var/tmp/psycles-lone-monk-f297ec53-20260904/export`, frame 4,
1440x1080, 256 fixed samples, seed 0, RX 9070 XT, native SVM, staged wavefront,
identical queue/launch settings. Each original module is freshly JIT-compiled
with cache disabled and LLVM/ISA dumps. No other build or GPU test runs during
profiling. No launch-size tuning is mixed into the comparison.

The accepted control uses the normal executable/library paths. Only the
production loop is temporarily restored from `12c3a680` for the baseline,
then the final source and build are restored and verified. These runs bracket
the baseline with two fixed runs:

| Order | Variant | Render time | Evidence directory |
| --- | --- | ---: | --- |
| 1 | Fixed | 16.9265 s | `/var/tmp/psycles-shadow-queue-final-9deifP` |
| 2 | Original | 16.4580 s | `/var/tmp/psycles-shadow-queue-normal-baseline-x2dauT` |
| 3 | Fixed | 16.9280 s | `/var/tmp/psycles-shadow-queue-normal-fixed-Yziyr4` |

The fixed mean is 16.92725 s: **2.9% slower overall** than this original-loop
control, not a renderer speedup. This is a narrow one-scene measurement, not a
general performance claim. The change is a structural-alignment checkpoint;
the whole-renderer performance objective remains unfinished.

| Kernel | Original control | Last fixed run | Fresh Cycles HIP |
| --- | ---: | ---: | ---: |
| Surface | 8.212292 s | 8.594714 s | 6.017388 s |
| Closest intersection | 3.242083 s | 3.322318 s | 3.473637 s |
| Shadow intersection | 0.848540 s | 0.970363 s | 0.970888 s |
| NEE | 0.780709 s | 0.794543 s | 0.540722 s |
| Shadow shading | 0.568620 s | 0.268898 s | 0.365029 s |

Opaque early termination reduces shadow shading from 1195 to 390 dispatches
and from approximately 412.49 million to 77.71 million launched lanes (81.2%
fewer). The fresh Cycles run has 381 shading dispatches / 77.98 million lanes.
These lane counts include launch padding and are not exact active-path counts.
The regression independently checks exact state/continuation counts.

Surface dispatches increase from 1272 to 1495 even though total surface lanes
remain approximately 819.5 million. Closest dispatches increase from 1865 to
2048. Shadow intersection VGPR usage rises from 144 to 160 (scratch stays
400 B). The changed queue traffic offsets the shadow-shading savings.

Importantly, surface final LLVM IR is **byte-identical across these runs**
and the preceding checkpoint, SHA-256
`7e9a624f771f6118e69a13cd740eb3a1df75b24b0c7acfc32622f538d73902b5`.
It retains 256 VGPRs / 3456 B private scratch and a 512-thread group. The
surface slowdown is therefore not evidence of a surface inlining or IR-growth
regression.

Main/shadow scheduling remains a structural difference: Cycles owns separate
path queues, while this Psycles coroutine waits for shadow completion before
continuing the main path. The existing optional auxiliary consumer fuses the
shadow stages, so enabling it is not equivalent to completing that stage-by-stage
migration. Its separate-stage scheduling/admission needs its own regression and
validation; this document does not claim to have quantified all causal factors.

Reproduce a normal-path render from a fresh directory:

```sh
run_dir=$(mktemp -d /var/tmp/psycles-shadow-queue-render-XXXXXX)
cd "$run_dir"
rocprofv3 --kernel-trace --output-format rocpd \
  --output-directory "$run_dir" --output-file render -- \
  env PSYCLES_NATIVE_CYCLES_SVM_SURFACE=1 PSYCLES_DISABLE_SHADER_CACHE=1 \
  PSYCLES_DUMP_COROUTINE_FRAME=1 LUISA_DUMP_LLVM_IR=1 \
  LUISA_DUMP_HIP_ISA="$run_dir" \
  /home/mike/Projects/Psycles-surface-svm/build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-f297ec53-20260904/export "$run_dir/render.exr" \
  hip 1440 1080 256 64 - 0 0 0 0 256 - 1 0 wavefront-staged 32 32768 32 1 1 0
```

## Cycles comparison, image and persistent state

Fresh Cycles evidence: `/var/tmp/psycles-shadow-queue-cycles-aBpy7U`, using the
same `.blend`, 1440x1080 / 256 / seed 0 and HIP-only RX 9070 XT selection.
Its sample-render interval is **13.831431 s**, versus the fixed normal-path mean
16.92725 s: Psycles remains **22.4% slower**. Cycles' 16.5828 s total including
synchronization/setup and Python's 16.907836 s outer measurement are not used
as the pure render baseline. Full performance parity is not achieved.

Against this one fresh reference, original/fixed Combined relative RMSE is
0.0111400953 / 0.0111222482. Both fixed normal-path runs contain zero nonfinite
pixels in all eight passes; the queue fix introduces no material image change.
Last fixed DiffInd and GlossInd relative RMSE remain 0.14106125 and 0.16198232.
Image parity is unfinished.

The full coroutine stays at 89 fields, **448 B actual SoA / 448 B AoS**,
capacity 1,048,576. Summing the fresh Cycles device-allocation log gives
**172 B main + 224 B shadow = 396 B** per equal-capacity slot pair. Psycles is
52 B (13.1%) larger in this comparison. Neither figure includes queues,
sorting storage, registers or private scratch. The separately printed
424 B payload type includes its own padding and is not the SoA field sum.

## Discarded experiments and measurement caveat

An earlier extraction introduced an unnecessary outer `if` and duplicate
state, changed surface IR and measured 17.6655 s. It was replaced by the final
caller-state-preserving extraction before the accepted control. That experiment
is `/var/tmp/psycles-shadow-queue-monk-yhE7yT`; its source patch is retained as
`preliminary.patch` in the main evidence directory.

A second diagnostic used saved runtime libraries through child-process
`LD_LIBRARY_PATH`. Its four runs, preserved by `run-monk.sh` / `ab-summary.log`,
were original 16.9341, fixed 17.2683, fixed 17.2328, original 16.9856 seconds.
Their directories are `/var/tmp/psycles-shadow-queue-` followed by
`baseline-XbxIxM`, `fixed-kR4gvG`, `fixed-eZMn2N`, `baseline-99BaBL`.
These relocated-library timings are **not used as the normal-deployment
performance conclusion**.

The relocated original/fixed images both have approximately 1.33% Combined
relative RMSE against the same fresh reference, unlike the normal-path 1.11%.
The normal and relocated fixed runtime libraries have the identical SHA-256
`aeec171e35887ed47fa53c33a0a689f883934fb78d2ebb696a34d8797a8a46e8`.
Normalized `ldd` output differs only in that library's location; generated
surface IR is also identical. The cause of this relocation-associated image
difference is not localized here. It must not be attributed to the queue fix.

The fresh and historical Cycles reference metadata agree apart from output path
and elapsed time. Their images differ by only 0.0001043904 Combined relative RMSE
(`repeat-compare.json` in the fresh Cycles directory), too little to explain
the relocated harness's discrepancy. This is why the final conclusion uses
normal-path controls and one reference, not mixed historical or relocated runs.

This checkpoint establishes opaque-shadow queue alignment, not complete SVM,
geometry-domain, main/shadow scheduling, image or performance parity.
