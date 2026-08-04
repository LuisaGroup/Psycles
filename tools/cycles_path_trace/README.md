# Diagnostic Cycles patch

`0001-Cycles-add-Psycles-per-path-trace-oracle.patch` applies the version-1
trace instrumentation to official Blender commit
`29ccd5e2e824128c86fc6174c9c502c02212434a`.

The kernel target is the official sample-subset offset. Camera filter and lens
randoms are observed from the same camera-sample evaluation that produced the
ray, so tracing an absolute sample never substitutes sample-zero state.

Apply it only to a dedicated diagnostic checkout:

```bash
git switch --detach 29ccd5e2e824128c86fc6174c9c502c02212434a
git am /path/to/Psycles/tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch
```

Do not use the diagnostic build as the normal Cycles benchmark binary. The
instrumentation is observational and guarded by the complete trace-film
layout, but keeping it separate makes the oracle modification explicit.
