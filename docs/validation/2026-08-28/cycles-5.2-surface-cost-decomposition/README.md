# Cycles 5.2 surface cost decomposition

## Outcome

Fresh same-device probes isolate the remaining Barbershop surface gap to the
complex closure program and its dynamic value graph. Psycles is 10.40% faster
than Cycles for a single constant Diffuse closure and only 4.08% slower for a
single constant Glossy closure. Keeping the original closure topology while
disconnecting every non-shader input makes Psycles 1.692x slower; the complete
authored graph is 1.991x slower.

This rejects the hypothesis that the current twofold surface gap is a uniform
HIP, scheduler, or elementary BSDF cost. The next work should change the
representation and traversal of complex closure/value programs, not force a
backend resource number or perturb already competitive simple closures.

## Controlled probes

All four rows use the official Blender 5.2 Barbershop scene, the Radeon RX
9070 XT, HIP, 640x480, 64 fixed samples, Tabulated Sobol, adaptive sampling
disabled, and denoising disabled. Psycles uses the staged wavefront path with a
64-thread block and compact value execution. Cycles uses Blender 5.2.0 LTS
build `fbe6228777e7`.

The probes are nested structural interventions:

- `full` retains the authored scene and material graphs;
- `constant closure inputs` retains each material's shader/closure topology,
  but disconnects every non-shader input so authored defaults replace the
  dynamic value DAG;
- `constant diffuse` replaces every material surface with one Diffuse BSDF;
- `constant glossy` replaces every material surface with one Glossy BSDF.

The source transformation is implemented by
`tools/create_blender_surface_cost_probe.py` and has permanent contract tests
in `tests/test_blender_diagnostic_probes.py`. Geometry, instances, lights,
camera, film, and render settings are unchanged.

## Normalized HIP results

`ns/item` is total duration of the renderer's surface-shading kernel divided by
its actual launched work. Cycles uses
`kernel_gpu_integrator_shade_surface`. Psycles surface kernels were identified
from the generated stage/resource metadata; JIT time is excluded.

| Probe | Psycles calls | Psycles work | Psycles ms | Psycles ns/item | Cycles calls | Cycles work | Cycles ms | Cycles ns/item | Psycles / Cycles |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 366 | 53,601,728 | 1,154.071 | 21.530472 | 296 | 54,061,056 | 584.528 | 10.812362 | **1.991x** |
| constant closure inputs | 128 | 56,091,840 | 651.310 | 11.611494 | 190 | 56,187,904 | 385.533 | 6.861498 | **1.692x** |
| constant diffuse | 142 | 67,572,736 | 420.871 | 6.228408 | 194 | 67,653,632 | 470.295 | 6.951518 | **0.896x** |
| constant glossy | 100 | 67,632,512 | 237.012 | 3.504410 | 182 | 67,715,072 | 227.991 | 3.366910 | **1.041x** |

The work counts within every renderer pair differ by less than one percent,
so the ratios are not dispatch-count artifacts. The Psycles surface resource
records for the three transformed probes are:

| Probe | Kernel hash | VGPR | Private B | Render-only |
| --- | --- | ---: | ---: | ---: |
| constant closure inputs | `424bd600e57c5bdf` | 256 | 2,272 | 1.31855 s |
| constant diffuse | `698e7995aef2c744` | 256 | 1,920 | 1.14492 s |
| constant glossy | `14a4e366e5542768` | 256 | 1,936 | 0.627226 s |

Cycles reports 192 VGPRs and 6,976 B private storage for all three transformed
surface kernels. The earlier register-cap audit demonstrated that copying the
192-VGPR number onto the different Psycles program only increases spills and
time; private-byte count alone also does not predict the measured ordering.

## Interpretation boundary

The probes establish causal regions, but their row differences are not an
additive timing decomposition. Replacing a material changes path survival,
closure probabilities, and therefore later surface work; even the launched
work counts are not identical across probe modes. The sound conclusions are:

1. Elementary Diffuse and Glossy surface paths are at or near Cycles parity.
2. The original closure topology remains expensive even when its value inputs
   become constants.
3. Restoring the dynamic value DAG widens the remaining gap further.
4. A structural optimization must preserve the simple-path result while
   reducing complex-program classification, storage traffic, and repeated
   value/closure work.

These observations do not prove a particular source-level fix. Candidate
changes still require a formal equivalence argument, backend differential
tests, generated-object/resource inspection, and a fresh HIP profile.

## Reproduction and provenance

The retained Psycles revision is `2205eefbcbdd34a781b7d7915e05c0b83903e7e6`
with LuisaCompute `bddf1d6a08ea91833548c3724b3e30125ac934f8`.
The Cycles source reference is Blender 5.2 release revision
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`.

Each Psycles probe was traced with the same command shape:

```sh
rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  build/bin/psycles_render_blender_scene EXPORT OUTPUT.exr \
  hip 640 480 64 64 - 320 240 0 0 64 - 1 0 \
  wavefront-staged 64 32768 32 1 1 0 4 2 auto 0 0 0 1 1048576
```

Each Cycles probe was freshly rendered and traced with:

```sh
rocprofv3 --kernel-trace -f rocpd -d PROFILE_DIR -o trace -- \
  /home/mike/Projects/blender-install-5.2/blender PROBE.blend \
  --background --python tools/render_cycles_golden.py -- \
  OUTPUT.exr 640 480 64 0 --cycles-device HIP \
  --device-name 'Radeon RX 9070 XT' \
  --sampling-pattern TABULATED_SOBOL --scrambling-distance 1.0
```

Raw fresh traces are under
`/var/tmp/psycles-current-cost-probes-20260828.omDlFS` and
`/var/tmp/psycles-cycles-cost-probes-20260828.Z6QnX3`. The transformed `.blend`
files and exported Psycles bundles are under
`/var/tmp/psycles-barbershop-surface-cost-20260827`.
