# Cycles shadow append-storage admission

## Reference and change

The external Cycles 5.2.1 source is
`/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`. Its
`PathTraceWorkGPU::compact_shadow_paths` reclaims an empty append pool, but
otherwise postpones compaction while extent is below 32 or less than twice
the live population. Dead holes therefore need not be available append slots.

The prior auxiliary contract admitted against abstract free capacity `C-q`
and forced the client to materialize it. The generic SDK now separately
queries materializable storage `A`, constrained by `A <= C-q`, before proving
`invocations * emission_bound <= A`. The client can defer relocation and drain
its existing side stages instead. An empty pool must materialize full capacity
for progress. The SDK contains no Cycles thresholds, kernel IDs, or state.
Its formal proof and permanent red/green witness are in Luisa's
`docs/validation/2026-09-07/auxiliary-admission/README.md`.

Psycles supplies the original Cycles host policy, including its float comparison
at large extents. The gather/compact/traversal/shading kernel bodies, SVM word
stream, feature masks, and ordinary scalar/vector initialization are unchanged.
No application noinline marking is introduced.

## Independent and production regressions

`tools/cycles_wavefront_policy_oracle.py` extracts the external DeviceKernel
enum and complete host policy functions verbatim, compiles those definitions,
and instruments only counters and storage-operation requests. It is not a CPU
renderer or a rewritten policy oracle. Reproduce with:

```
python tools/cycles_wavefront_policy_oracle.py /path/to/blender-cycles-5.2 \
  --build-dir /var/tmp/cycles-wavefront-policy-oracle
```

The checked-in output identifies the extracted source SHA-256:
`2e8d1c32186cfb832792cafb69d16257f72990791edc3762084d73bb7a92a075`.
The independent host test checks all 145 compaction cases, including empty,
19/31/32/64-slot boundaries, MNEE population accounting, and representability
above 2^24. The retained 108 global selection rows are not claimed covered;
global main/auxiliary equal-count priority is a separate unfinished integration.

The actual shadow-pool regression distinguishes two previously conflated cases:
19 allocated / 13 live must drain before appending, whereas 32 allocated / 12
live must compact and admit 20 items. It verifies stage counts, NEE payloads,
shade-owned traversal batches, final film/pass values, and scheduler reuse.
The old client fails the new test; the new client passes HIP and fallback.

Evidence: `/var/tmp/psycles-wavefront-admission-6YMdzL`. All builds use 32
threads. Full original HIP selection: 157/157; fallback: 159/159; compiler and
adapter selection: 101/101 (excluding known `blender_export_render_settings`).
All 14 affected SDK executables pass both backends: 157 cases each,
13607/13606 assertions. The independent SDK witness passes 144 assertions.
Strict Vulkan canaries use all three native-XIR/no-DXC environment guards and
loader tracing: SDK witness 46 native SPIR-V compilations, production pool 36,
both passing without DXC/DXIL loads.

The SDK tests use clean candidate headers/tests with active compiler/backend
libraries. The application gitlink is deliberately not advanced over pending
Luisa compiler changes; this API addition on origin/next is required separately.
Protected path-tracer, exporter, Local/scope/pass changes are not in this commit.

An isolated source snapshot in `/var/tmp/psycles-handler-candidate-a2CxC6`
contains the published Psycles HEAD plus only this checkpoint's source/test
changes. Its 32-thread build passes, followed by the host policy and two HIP
pool/sort tests (3/3), fallback pool/sort/full-film tests (3/3), and the strict
native Vulkan shadow-pool test without DXC/DXIL loads. Logs are
`isolated-build.log`, `isolated-hip.log`, `isolated-fallback.log`, and
`isolated-native.log` in the evidence directory. This isolates the application
changes, while retaining the active SDK compiler/backend libraries.

## Lone Monk A/B/B

Same frame 4, 1440x1080, 256 fixed samples, seed 0, RX 9070 XT; fresh process
directories, shader cache disabled, typed surface-sort Handler and independent
shadow stages enabled throughout. Export:
`/var/tmp/psycles-lone-monk-f297ec53-20260904/export`.

| Policy | Render seconds | Evidence directory |
| --- | ---: | --- |
| Old client, new default SDK hooks | 14.3138 | `/var/tmp/psycles-admission-control-ACyrDI` |
| Cycles append policy | 14.2950 | `/var/tmp/psycles-admission-monk-okqGB3` |
| Cycles append policy repeat | 14.3686 | `/var/tmp/psycles-admission-repeat-7okwWr` |

New-policy mean is 14.3318 seconds, 0.13% longer than the control: neutral,
not a demonstrated speedup. Main frame remains 55 fields / 220 bytes; total
main plus equal-capacity shadow task/batch storage is still 520 bytes versus
the measured Cycles 172+224 bytes. Packing/phase reuse remain unfinished.

Surface final IR remains byte-identical. Shadow final IR differs only in its
entry symbol, not instructions. Scheduling counts change (NEE 1072 to 1045,
shadow intersection 1071 to 1044, shade 384 to 441, compaction 904 to 1077),
but resources and measured whole-render efficiency are essentially unchanged.
The timed production binary SHA-256 is
`4fff0bdb81e0cc2392280ecaa5717672106157b5e00808c1f661e47c8724cd92`.
These timings retain the active tree's protected block-size/fused-count and
path-tracer work; they are not isolated commit-only performance measurements.

All three multilayer EXRs were compared against the matched external Cycles
HIP reference `/var/tmp/psycles-shadow-queue-cycles-aBpy7U/cycles.exr` with
metadata checks. Eight passes have zero invalid pixels. Combined relative
RMSE stays about 0.0111; diffuse/glossy indirect remain about 0.1411/0.1620.
This policy does not resolve that existing transport parity gap.
