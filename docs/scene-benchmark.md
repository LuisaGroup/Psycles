# Canonical scene benchmark

Every full-scene performance and quality checkpoint uses one fixed
five-renderer matrix:

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

On the validated AMD workstation:

```bash
TMPDIR=/var/tmp/psycles-compiler-tmp \
python3 tools/run_scene_benchmark.py \
  --blender /home/mike/Projects/blender-install-4fe17ef6/blender \
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
required timing metadata remain valid. Cycles metadata is parsed again and,
when available, its own hash is checked. Missing, modified, or incomplete
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
`psycles.scene-benchmark.v1` schema. It records:

- the exact selected matrix and execution order;
- source and exported-scene SHA-256 hashes;
- resolution, samples, and maximum samples per Luisa dispatch;
- every command, log, process wall time, EXR path, and EXR hash;
- Cycles' selected device inventory and render-call time;
- Psycles scene compilation, shader JIT, render-only, and process wall times;
- render-only speedup and slowdown ratios against the selected Cycles GPU and,
  when enabled, Cycles CPU;
- differential-report paths for every Psycles backend against each selected
  Cycles device variant.

Each comparison covers all available linear passes and writes real
reference/actual/absolute-difference triptychs. The panel labels include the
device path, for example `Cycles HIP` and `Psycles fallback`, so a CPU result
cannot be mistaken for a same-GPU comparison.

Performance conclusions must state which timing boundary is used. The primary
throughput number is the renderer-reported render interval; process wall time
and Psycles compilation/JIT phases remain visible separately. Quality
conclusions use numeric pass metrics and original-resolution visual
inspection. A benchmark is incomplete if any selected renderer fails or
Cycles silently selects a different device than the explicit name filter.
