# Pin the actual Cycles scrambling property

Cycles 5.2.1 declares `auto_scrambling_distance` in
`intern/cycles/blender/addon/properties.py`, not
`use_auto_scrambling_distance`. The golden-image, path-trace and area-light
fixture scripts previously used the latter name through an optional-property
setter. That silently left automatic scrambling enabled in scenes which
authored it, despite requesting a fixed scrambling distance.

The fix uses the actual property. Golden and path-trace rendering share the
same sampler configuration function, and their metadata records the effective
automatic-scrambling setting. The area-light fixture's setter is corrected as
well. No renderer RNG, floating-point mode, or shading implementation changes.

## Regression and verification

The Python regression starts with automatic scrambling enabled; the old
golden helper leaves it enabled and fails. A dedicated Blender regression
checks all three scripts against real RNA properties, also starting enabled.
It only configures the area-light fixture: it does not render a CPU reference.
The existing export-settings check now starts enabled instead of inadvertently
relying on the default being disabled.

Evidence: `/var/tmp/psycles-path-structure-3ahR3G`.

- `sampler-red.log`: the golden helper fails the new non-default-state check.
- `sampler-blender.log`: all three configuration routes pass on Blender
  5.2.1, build `9e2066aef7ef`.
- `sampler-full-build.log`: full 32-thread build succeeds.
- `sampler-focused.log`: schema, comparator, decoder and new Blender sampler
  CTests pass, 4/4. The unrelated older export-settings probe-inventory failure
  is not counted as a passing test.

```sh
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure \
  -R '^psycles[.](blender_cycles_sampler|cycles_path_trace_decoder|cycles_path_trace_schema|cycles_path_trace_comparison)$'
```

Lone Monk's actual authored setting was already false, with TABULATED_SOBOL
and distance 1.0 (`monk-sampler-audit.log`). Thus this configuration bug does
not explain the current Lone Monk image residual. It prevents a different
scene's authored automatic-scrambling setting from invalidating future
same-seed comparisons. Exact discrete/RNG trace checks remain separate from
bounded continuous-value comparisons; this fix adds no bit-matching path.
