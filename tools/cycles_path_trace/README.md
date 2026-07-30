# Diagnostic Cycles patch

`0001-Cycles-add-Psycles-per-path-trace-oracle.patch` applies the version-1
trace instrumentation to official Blender commit
`ff404d072bb4bae52c578d2be3aeeea2a057ab63`.

Apply it only to a dedicated diagnostic checkout:

```bash
git switch --detach ff404d072bb4bae52c578d2be3aeeea2a057ab63
git am /path/to/Psycles/tools/cycles_path_trace/0001-Cycles-add-Psycles-per-path-trace-oracle.patch
```

Do not use the diagnostic build as the normal Cycles benchmark binary. The
instrumentation is observational and guarded by the complete trace-film
layout, but keeping it separate makes the oracle modification explicit.
