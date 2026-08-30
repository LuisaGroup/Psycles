# Diagnostic Cycles patch

`0001-Cycles-add-Psycles-per-path-trace-oracle.patch` applies the version-3
trace instrumentation to official Blender 5.2.1 commit
`9e2066aef7ef`.

The kernel target is the official sample-subset offset. Camera filter and lens
randoms are observed from the same camera-sample evaluation that produced the
ray, so tracing an absolute sample never substitutes sample-zero state.

Apply it only to a dedicated diagnostic checkout:

```bash
git switch --detach 9e2066aef7ef
git am /path/to/Psycles/tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch
```

`nee_contribution` is total over every traced surface event. The diagnostic
kernel writes zero before direct-light eligibility, sampling, roulette, and
shadow traversal, then overwrites it only when the shadow state machine reaches
its visible terminal state. Consequently every early exit has the same explicit
zero value that it contributes to the film; absence never encodes occlusion.

Do not use the diagnostic build as the normal Cycles benchmark binary. The
instrumentation is observational and guarded by the complete trace-film
layout, but keeping it separate makes the oracle modification explicit.
