# Volume-work diagnosis: source and measurement contract

Question: does the homogeneous-volume implementation explain Barbershop's
remaining HIP slowdown? Separate an implementation defect from its measured
contribution to the full render; kernel time alone does not establish why a
kernel is slow or how many active paths it processes.

## Authoritative implementation and scene

- Cycles source: `/home/mike/Projects/blender-cycles-trace-5.2`,
  revision `cb168525138fecc792cc393f94afc39582b0103c`.
- `intern/cycles/kernel/integrator/shade_volume.h`:
  `volume_direct_sample_method`, `volume_integrate_state_init`,
  `volume_integrate_result_init`, `volume_integrate_homogeneous`,
  `volume_direct_scatter_mis`, `volume_integrate_event`, and
  `integrate_volume_direct_light` control work placement. In particular,
  light proposal and selected equiangular initialization precede coefficient
  shading. They must not be skipped merely because the material emits only.
- Original Barbershop: `/var/tmp/psycles-official-redownload-20260814/barbershop_interior.blend`.
  Verified export: `/var/tmp/psycles-four-scene-hip-0PRq34/barbershop/export`.
  The sole volume material, `fog`, connects Light Path.Is Camera Ray through
  MULTIPLY by 0.02 to Emission.Strength. It has no scattering or absorption
  node, and no world volume. Volume sampling is MULTIPLE_IMPORTANCE;
  use_light_tree is false and volume_bounces is zero. Both renderers retain
  animated-seed semantics: authored seed 0, frame 1, effective seed 1267069554.

## Before-change profiling

Psycles source `253767f5`, Luisa `9ea3b720f`, Radeon RX 9070 XT / gfx1201,
HIP 7.2.53211. Both saved profiles below use 2048x858 / 64 spp. These are
instrumented, single-run kernel-duration observations, not the 256-spp
unprofiled benchmark. Do not mix the two workloads.

Evidence root: `/var/tmp/psycles-native-volume-svm-06XnDX`.

- Latest Psycles: `barbershop-kernel-profile-nee.log` and
  `barbershop-kernel-profile-nee/psycles_results.db`.
- Earlier original Cycles: `barbershop-kernel-profile-cycles/cycles_kernel_stats.csv`.
- Older Psycles readonly-frame checkpoint:
  `barbershop-kernel-profile-readonly/psycles_kernel_stats.csv` and its sibling log.

The new rocprofv3 result is SQLite, not CSV. Its `kernels.duration` is
`end - start` in nanoseconds. The `top_kernels` view divides by 1000 and
therefore reports microseconds. Recover stage identities from the actual
`stage='...' structural_hash=...` log entries, never from row order or time.
The exact read-only query for the three latest stages is:

```sql
SELECT name, COUNT(*), SUM(duration),
       MIN(vgpr_count), MAX(vgpr_count), MIN(scratch_size), MAX(scratch_size)
FROM kernels
WHERE name IN ('kernel_7c5a18556bfbdef7',
               'kernel_b717de734733be83',
               'kernel_7fca871f3dc049b0')
GROUP BY name;
```

| Stage | Latest Psycles ns / calls | Earlier Cycles ns / calls | Psycles VGPR / scratch bytes |
| --- | ---: | ---: | ---: |
| shade_surface | 5684112276 / 1120 | 2935582003 / 1110 | 256 / 2432 |
| intersect_closest | 1626827636 / 992 | 1425190982 / 1180 | 144 / 240 |
| shade_volume | 427889843 / 429 | 359619786 / 418 | 256 / 452 |

The older Psycles volume observation was 766112926 ns, not the latest
427889843 ns. Surface remains the larger outstanding kernel difference.
Dispatch count and padded grid size are not active-path counts. Register or
scratch allocation is not measured occupancy or spill traffic, so neither
is asserted as a causal explanation here.

The after-change profile is `barbershop-kernel-profile-volume-work.log` and
`barbershop-kernel-profile-volume-work/psycles_results.db`. Apply the same
query with volume hash 859d7ba000078b22; surface and closest hashes are
unchanged. After durations are respectively 5688461864, 1554720358 and
430448010 ns, with the same call/resource counts. Volume does not show a
measurable gain. Both commands use rocprofv3 `--kernel-trace --stats`,
PSYCLES_DISABLE_SHADER_CACHE=1, LUISA_CORO_SHADER_MAP=1, the same export,
HIP 2048x858 / 64 samples / 64 max samples per dispatch, wavefront-staged,
32 block threads, 32768 global-memory batch, native fast math and a
1048576-entry frame pool. Full renderer argument vectors for the 256-spp
canaries are archived separately; do not substitute their sample count.

Original Cycles had additional output passes, including AO. The saved
numbers are historical observations, not matched-work speed ratios. Common
pass images remain valid numerical oracles. A new exact-pass campaign is
required before claiming efficiency parity or an end-to-end Cycles ratio.

## Regression provenance

`tests/test_luisa_volume_work.cpp` invokes the real homogeneous component,
transport and HG phase implementation. GPU input flags supply controlled
coefficients; device atomics count callbacks at the existing light-provider
boundary. This is not a CPU shader, estimator or renderer. Rejection
predicates are taken from the original functions above. Numerical transport
is checked separately by original-Cycles and complete-render fixtures.

The first 12-case distance-sampling test failed 24 of 72 checks on the
unchanged production implementation: 21 excess light-provider observations
and three phase continuations on rejected/transmitted paths. An earlier C++
test-setup compilation error is not the red regression. Raw logs are
`volume-work-red-hip.log` and `volume-work-red-build-retry.log`.

The final work regression additionally covers equiangular and MIS modes,
with empty, pure-emission, pure-absorption, terminated, scattering, transmit,
disabled-NEE, zero-sigma, zero-length, invalid-interval and invalid-light
cases. Positive scatter cases ensure the implementation cannot pass by
removing all work. No scene-name optimization, altered sampler dimensions,
software floating-point path, inlining policy, or coroutine layout policy
is introduced.
