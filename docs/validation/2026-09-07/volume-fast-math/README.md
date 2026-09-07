# Restore fast math for the volume-density bake

## Audit finding

The production volume-majorant prepass was the only remaining Psycles shader
that unconditionally disabled fast math. Its comment claimed strict FP was
needed to preserve a conservative tracking bound. That justification was
incorrect: a finite set of volume samples is an estimate of the extrema, not
a proof of a conservative bound. Disabling reassociation does not establish
such a proof, nor does it guarantee an upper bound on runtime shader evaluation
that already uses fast math.

The authoritative Cycles 5.2.1 checkout is
`/home/mike/Projects/blender-cycles-trace-5.2`. Its HIP and HIPRT device compiler
options both contain `-ffast-math` (`intern/cycles/device/hip/device_impl.cpp:266`
and `intern/cycles/device/hiprt/device_impl.cpp:276`). The density bake is not
given a separate strict-FP compilation policy.

`kernel/bake/bake.h::kernel_volume_density_evaluate` samples the volume shader,
reduces extinction/emission extrema, and divides out object density.
`kernel/integrator/shade_volume.h::volume_null_event_coefficients` computes
`majorant = max(max_component(sigma_t), sampled_majorant)` and then
`sigma_n = majorant - sigma_t`. Psycles already projects that same runtime
rule in `HeterogeneousVolumeTracking::null_event_coefficients`.

For example, the existing original-Cycles collision fixture uses extinction
`(0.4,1.4,0.2)` with sampled majorant `0.8`. It retains the original raised
majorant `1.4`, nonnegative null coefficients, collision transition and
exceedance observation. This is a coarse structural case, not a one-ULP edge.
No claim is made that arbitrary underestimated bounds are mathematically safe;
the requirement here is to retain Cycles' existing estimator and handling.

## Change and regression

The prepass now enables fast math, removing its strict-FP exception and the
incorrect explanation. No bound padding, epsilon, software arithmetic, f64
device path, node, RNG operation, or tracking control flow is added or changed.
Default scalar/vector zero initialization remains unchanged.

The raw-volume prepass regression now compiles and runs both strict and fast
variants, with shader caching disabled. It keeps the existing original-Cycles
Sobol/hash fixtures and the same `2e-6` floating-extrema tolerance. No expected
values or tolerances were changed to admit the fast variant. The production
scene/resource fixture and the heterogeneous collision fixture also compile
their consumers with fast math, as production transport already does.

Before changing the production flag, all three updated HIP regressions pass.
This is removal of an unnecessary restriction, not a compiler wrong-code fix;
the previous implementation was not made to fail an invented numerical oracle.
HIP's shader-cache key includes the fast-math option, so an existing strict
artifact cannot satisfy the new production request.

Evidence is in `/var/tmp/psycles-volume-fastmath-1yBOeH/`:

- `regression-build.log`, `regression-hip-before.log`: 32-thread build and the
  three passing HIP fixtures before the production flag change.
- `full-build.log`: all-target build using all 32 threads after the change.
- `all-hip.log`: full HIP CTest, 166/166 passed.
- `all-fallback.log`: full fallback CTest, 168/168 passed.
- `volume-native-vk.log`: 18/18 volume tests passed, including production
  volume scene/resource construction, the complete volume/environment
  renderers, and native SVM volume emission. All three strict native gates
  are set: `LUISA_VULKAN_USE_XIR=1`,
  `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, `LUISA_VULKAN_DISABLE_DXC=1`.
  The dual-mode prepass and collision regressions disable shader caching.

This validation uses the active worktree and SDK libraries. Only the production
prepass, three targeted regression files, and this report belong to the patch;
the inherited application/SDK edits and Luisa gitlink are not staged.

The scene-level timing effect, if any, belongs to volume preprocessing, not
the steady-state surface-render timings in the separate shadow-relocation
report. No rendering speedup is claimed for this flag change.

## Policy-test follow-up

The broad host suite found that `test_luisa_shader_performance_policy.py`
still required the removed strict-FP exception and repeated its obsolete
conservative-bound claim. Its exact allowlist is now empty. The policy still
rejects any unreviewed production `enable_fast_math = false`, software
last-bit emulation, and scalar replacements for native normalization.
This follow-up changes the policy regression and documentation only, not
the renderer or its floating-point operations.
