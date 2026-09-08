# Canonical scene benchmark

The default full-scene performance and quality matrix has five entries;
focused HIP campaigns explicitly select only the two HIP renderers:

| Implementation | Device path |
|---|---|
| Blender Cycles | CPU |
| Blender Cycles | HIP, with an explicit device-name filter |
| Psycles/Luisa | fallback |
| Psycles/Luisa | HIP |
| Psycles/Luisa | Vulkan |

The fallback entry executes the production Luisa DSL/JIT program through
LuisaCompute's LLVM/Embree backend. It is not a separate Psycles CPU reference
renderer. Cycles remains the only rendering oracle; its CPU result is retained
alongside HIP to expose device-dependent intersection and floating-point
behavior.

By default, `tools/run_scene_benchmark.py` renders all five entries
sequentially with the
same `.blend`, scene-owned seed, resolution, fixed sample count, disabled
adaptive sampling, and disabled denoising. The default extent is 640×480, so a
canonical scene run never substitutes a tiny probe for full-scene behavior.

Both renderers write exactly 15 common linear passes (46 channels including
Combined alpha). Cycles' headless setup clears inherited auxiliary outputs,
including AO, Depth, Mist, indices, debug/denoising passes, custom AOVs and
light groups, before enabling the common set. It does not save these changes
to the source .blend, change authored lighting, or enable disabled view layers.
The runner reads the actual EXR headers and rejects missing, duplicate or
extra channels and multiple output layers. Pass aliases such as VolumeDir
are normalized for validation; the original channel names remain recorded.

On the validated AMD workstation:

```bash
TMPDIR=/var/tmp/psycles-compiler-tmp \
python3 tools/run_scene_benchmark.py \
  --blender /home/mike/Projects/blender-install-5.2-hiprt/blender \
  --psycles-render build/bin/psycles_render_blender_scene \
  --blend /home/mike/Downloads/lone-monk_cycles_and_exposure-node_demo.blend \
  --output-dir /var/tmp/psycles-lone-monk-five-way \
  --cycles-hip-device-name "Radeon RX 9070 XT" \
  --compiler-tmp /var/tmp/psycles-compiler-tmp \
  --width 640 --height 480 --samples 64
```

Native platform matrices retain the same manifest, timing boundaries, pass
comparisons, and explicit device selection. For example, the Apple Silicon
matrix compares one named Cycles Metal GPU with Psycles fallback and Metal:

```bash
python3 tools/run_scene_benchmark.py \
  --blender /opt/homebrew/bin/blender \
  --psycles-render build-macos/bin/psycles_render_blender_scene \
  --blend assets/official-blender-scenes/classroom/classroom.blend \
  --output-dir build-macos/benchmarks/classroom-apple \
  --bundle build-macos/scene-exports/classroom \
  --reuse-export --skip-cycles-cpu \
  --cycles-gpu-device METAL \
  --cycles-gpu-device-name "Apple M1 Max" \
  --psycles-backends fallback,metal \
  --width 640 --height 360 --samples 64
```

`--cycles-hip-device-name` remains a compatibility alias for the default HIP
matrix. A non-HIP run uses `--cycles-gpu-device` and
`--cycles-gpu-device-name`; `--psycles-backends` is an ordered comma-separated
list of backend module names.

The runner exports the final-render Blender dependency graph once, then reuses
that immutable bundle for all selected Luisa backends. Pass `--reuse-export`
only when the requested bundle has already been produced from the same source
scene. It is never used to bypass material export: the bundle still contains
the original node graphs, socket values, closure topology, and scene data.

An interrupted or failed matrix can be continued with `--resume`. The runner
first requires the manifest schema, renderer matrix, render settings, source
scene path and hash, and bundle path to match. A completed render is reused
only when its exact command, successful return code, output path, SHA-256, and
required timing metadata and actual pass inventory remain valid. Cycles metadata and the original log
are hashed and parsed again: both the main-loop interval and the enclosing
render-call interval must match the record. Missing, modified, or incomplete
outputs are rerun; a changed `--reuse-export` bundle is rejected because its
relationship to the recorded renders can no longer be proven. Comparisons are
cheap relative to full rendering and are regenerated from the validated final
matrix. The manifest's `resume.reused_stages` array makes every reused result
explicit.

For example, to continue the AMD command above after a backend failure, repeat
the identical command and append:

```bash
  --resume
```

## Recorded outputs

`benchmark.json` is updated after every completed stage and has
`psycles.scene-benchmark.v3` schema. It records:

- the exact selected matrix and execution order;
- source and exported-scene SHA-256 hashes;
- resolution, samples, and maximum samples per Luisa dispatch;
- every command, log and its SHA-256, process wall time, EXR path, and EXR hash;
- the exact pass contract and actual EXR channel inventory for each renderer;
- Cycles' selected device inventory, main-loop wall time (`render_seconds`),
  and enclosing Python render-call time (`render_call_seconds`);
- Psycles scene compilation, session initialization (`shader_jit_seconds`),
  render-only, and process wall times;
- render-only speedup and slowdown ratios against the selected Cycles GPU and,
  when enabled, Cycles CPU;
- differential-report paths for every Psycles backend against each selected
  Cycles device variant.

Each comparison covers all 15 common linear passes and writes real
reference/actual/absolute-difference triptychs. The panel labels include the
device path, for example `Cycles HIP` and `Psycles fallback`, so a CPU result
cannot be mistaken for a same-GPU comparison.

Performance conclusions must state which timing boundary is used. The primary
throughput comparison is **Cycles' original main-loop wall interval versus
Psycles' render-only wall interval**. The runner enables `--debug-cycles` and
requires exactly one `Rendering in main loop is done in ... seconds.` record;
missing, ambiguous, or non-finite intervals fail the benchmark. This is not a
sum of GPU kernel durations. Process wall time, Cycles' enclosing render call,
and Psycles compilation/JIT phases remain visible separately.

The CLI's `Luisa shader JIT completed` label and `shader_jit_seconds` field
measure the entire `renderer.create_session` interval. This includes shader
compilation plus session resource initialization and setup/baking; it is not
an isolated kernel-compiler timer. Disabling the main shader cache does not
disable every auxiliary or downstream compiler/OS cache. Report first-run
and repeated initialization observations explicitly when they differ.

The v1 runner incorrectly used Cycles' whole Python render-call time as
`render_seconds`, including scene synchronization/setup. Its ratios are not
valid render-only comparisons. The v2 runner corrected that timing boundary
but inherited extra Cycles output passes; AO can launch additional shadow
paths, so its ratios are not equal-work speed comparisons either. Neither
v1 nor v2 manifests can be resumed as v3. Retain them as qualified historical
artifacts and run a new exact-pass campaign. Quality
conclusions use numeric pass metrics and original-resolution visual
inspection. A benchmark is incomplete if any selected renderer fails or
Cycles silently selects a different device than the explicit name filter.
