# Capture status and remaining plan

Phase A is complete and its [recorded result](PHASE_A.md) has a failed
Classroom numerical gate. Fresh exports and twelve new pairs have not run;
the remaining plan is not evidence of their execution. Parent owns the single
CPU/GPU validation window. Preserve the original `.blend` files, the pre-API red,
guard-only red/green captures, and the frozen prospective benchmark plan.
Do not edit old metadata, logs, ASTs, LLVM dumps, or timing results.

## Implementation and completed evidence

The latest bounded candidate combines the scene-wide host capability with
one-site reuse of the existing native fast-acos helper. Pin its final source
diff/new files, SDK revision, executable and all loaded render/core/SDK shared
libraries before running. Recheck hashes afterward and stop on drift. The
README and `evidence.json` identify the already completed build, host,
recording, focused HIP, full 193-test HIP, full 195-test fallback, and four-test
strict native Vulkan stages. All four Phase A scene renders and sixty pass
comparisons also complete, with three all-finite actual/reference pairs and
one failed pair. Completed backend stages do not waive that numerical failure.

Use existing registered tests, not a new validation harness. Focused replay
commands from `/home/mike/Projects/Psycles-surface-svm` are:

```bash
cmake --build build --parallel 32
ctest --test-dir build --parallel 32 --output-on-failure \
  -R '^psycles[.]cycles_shading_terminator_(recording|scene|fixture)$'
ctest --test-dir build --parallel 1 --output-on-failure \
  -R '^psycles[.]luisa_(cycles_shading_terminator|cycles_svm_(bsdf_dispatch|subsurface_exit|hair_scattering))_hip$'
ctest --test-dir build --parallel 1 --output-on-failure \
  -R '^psycles[.]luisa_(cycles_shading_terminator|cycles_svm_(bsdf_dispatch|subsurface_exit|hair_scattering))_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 LD_DEBUG=libs \
  ctest --test-dir build --parallel 1 --output-on-failure \
  -R '^psycles[.]luisa_(cycles_shading_terminator|cycles_svm_(bsdf_dispatch|subsurface_exit|hair_scattering))_vk$'
```

The final full backend suites are additional gates, not just these four-test
selections. Record their exact discovered CTest selection, denominator,
terminal exit, elapsed time, and copied detailed logs. HIP precedes fallback;
strict Vulkan comes last. The Vulkan proof must include all three native-XIR
guards, actual successful native SPIR-V compilation, and loader inspection
showing no DXC/DXIL library load. A shader cache hit or success code alone is
not that proof. Keep loader tracing out of performance runs.

To preserve final recording modules, the host executable is in `build/`, not
`build/bin/`:

```text
build/psycles_cycles_shading_terminator_recording_tests <new-ast-directory>
```

Archive the six complete modules, test source hash, and service/COS/ACOS
counts. Do not replace the old ACOS-based witness with the new marker logic.
For any future LLVM inspection, use an isolated working directory and the
matching runtime/plugin path. Preserve full original modules and before/
after/final output; never overwrite user IR files in the repository root.

## Four-scene correctness before performance claims

Use the current existing v3 equal-pass runner and the frozen prospective plan
only as a command/input template. Create a **new final-candidate execution
manifest** with the final `86fe7274…` runtime hash (or the explicitly recorded
later final identity), rather than modifying the old `ad884114…` plan.

The separately frozen execution package is at
`/var/tmp/psycles-shading-terminator-7owG2a/final-candidate-tgZZND`.
Its `execution-manifest.json` uses new canary/export/audit/paired output paths;
`source-lock.json`, the complete tracked diff, and nine new-file snapshots pin
the exact final source and eight binaries. The old prospective plan remains
byte-identical. Parent released Phase A only after the backend gates; all
four canaries and serial comparisons ran, and the window is now released.
Preflight and postflight pass with source/input/binary identities unchanged.
Further export/pair execution still requires a separate released window.

1. **Completed, numerical gate failed:** capture fresh candidate Barbershop, Monk, Monster, and Classroom HIP
   canaries at their original extents and 256 fixed samples, using the pinned
   revision-specific control bundles and retained original reference images.
   These are correctness canaries, not fresh timing pairs.
2. **Not run:** use current exporter code to create four full new bundles in new
   directories. The old controls fail the current exporter-identity check;
   never relabel them. Compare geometry bytes, image manifests/loaded bytes,
   scene/seed/frame/integrator/object/graph metadata, and raw current-compiler
   old/fresh SVM word images plus bound identities. Explain actual changes.
3. **Not run:** after correctness review, run all twelve new equal-pass pairs with fresh
   original Cycles HIP references: three repeats per scene in the existing
   rotating order. The current v3 runner, not the old hardcoded campaign
   output tree, owns command/metadata/pass/timing capture. Do not `--resume`
   fresh timing repeats, skip comparisons, or run a CPU reference renderer.

Retain the exact 15-pass / 46-channel contract, including Combined alpha;
native extents; authored seed/frame; fixed [0,256) range; 64-sample dispatch
limit; fast math; and main shader cache disabled. Do not change scene options
to make an invalid image pass. Auxiliary/driver/OS caches are not globally
purged, so JIT is not claimed globally cold. No build, test, oracle, observer,
profiler, or comparison CPU workload overlaps a timed renderer.

The renderer's `shader_jit_seconds` covers session creation, including
compilation, allocations and upload/setup work; it is not compiler-only.
`render_seconds` covers its sample/render/readback interval, not later EXR
serialization. Cycles time is its original main-loop wall duration, not
process time or summed kernels. Report all individual pairs and per-scene
median/min/max; never blend historical and fresh timing pairs.

Run the existing comparison tool and the plan's all-46-channel finite check
outside timed windows. Preserve reference and actual invalid counts/masks,
all pass metrics, each exit code, and image hashes. Runner `complete` does
not mean finite. If Classroom fails again, keep that failure visible, finish
the remaining scenes when safe, and retain an overall failed numerical gate.
Further timing capture for the invalid scene is diagnostic only after review;
do not omit it from the table or claim a validated all-scene speedup.

Once the remaining gates terminate, append a distinctly dated final results
section with those exact binaries and logs. Do not silently promote this
partial checkpoint or the earlier 98f/6e captures to the final result.
