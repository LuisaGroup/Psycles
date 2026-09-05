# Closure-free surface-shadow SVM

## Formal cause and required transition

Cycles 5.2.1 `integrate_transparent_surface_shadow` uses
`ShaderDataTinyStorage`. `surface_shader_eval` initializes `num_closure` and
`num_closure_left` to zero for shadow visibility, but dispatches the full
`KERNEL_FEATURE_NODE_MASK_SURFACE_SHADOW` interpreter. In `closure_alloc`,
zero available slots cause failure before addressing `sd->closure[]`.

The Luisa representation uses `ShaderData::closure == nullptr` for that
implicit zero-count, zero-capacity state. Rejecting `NODE_CLOSURE_BSDF` in
this state was incorrect. Removing only that rejection is also insufficient:
a DSL `$if(allocation.valid)` guards device execution, not the host/JIT-time
dereference of the absent pool while recording its body.

The repair must separate observable shader transitions from physical-pool
access. Setup operations whose effects require a successful allocation can
return when storage is absent, after consuming their complete typed payload.
Transparent extinction and flags, portal extinction, and Principled emission
and layer order must remain observable. In particular, thin-wall Principled
transmission becomes Transparent for sufficiently sharp non-camera rays;
it cannot return early as a whole. This is the same failed-allocation
transition, not a different node mask, alternate SVM, or dummy closure array.

## Minimal counterexample and permanent oracle

A single `NODE_CLOSURE_BSDF / CLOSURE_BSDF_TRANSPARENT_ID` with nonzero weight,
followed by an observable `NODE_CLOSURE_SET_WEIGHT` and `NODE_END`, suffices:
Cycles updates extinction and continues; the old Luisa diagnostic interpreter
returns unsupported. The 51-case regression includes that counterexample,
the complete standalone BSDF/BSSRDF/hair dispatch families, zero and tiny
weights, repeated signed transparency above one, portal plus transparency,
Principled alpha/sheen/coat/emission, and sharp versus rough thin walls.

`tools/cycles_svm_shadow_surface_oracle.hip` executes the external Cycles
full interpreter on HIP with genuine TinyStorage and Cycles' own albedo
tables. A second invocation of Cycles' same node handlers exposes PC and is
cross-checked against the full interpreter. Only typed input bytes are shared
with Psycles; `tests/data/cycles_svm_shadow_surface.txt` is its captured output.
The Luisa regression compares flags, emission, extinction, LCG, count/left,
diagnostic status, exact final PC and the observable successor. It exercises
both absent storage and a real four-slot pool with zero slots available,
through diagnostic and production entry points where available.

Evidence directory: `/var/tmp/psycles-shadow-surface-M9edH7`.
`hip-red.log` captures the failing baseline before the runtime repair.

Production shadow setup and its caller migration are separate remaining work;
this regression does not claim that the full renderer uses the new path yet.

## Reproduction and verification

External source: `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`. SHA-256:

| File, relative to `intern/cycles` | SHA-256 |
| --- | --- |
| `kernel/svm/closure.h` | `585a5e6a6a9507e76983a2cf2e5fdb757f70e9213c5f1b151e6b8ed47842c673` |
| `kernel/closure/alloc.h` | `0c4d641007cce198cf98133fbf50f796c1ccd2bdcaf25d99f38302f63732f346` |
| `kernel/closure/bsdf_ray_portal.h` | `cf127182faa727bea536233cf177ce67eb65c07458c52e60c1e3b8c9044239c1` |
| `kernel/integrator/surface_shader.h` | `1b36b3c2082b8992252b81869263a5e959e33713395b0b58de63d66c5052a69d` |

From the Psycles root:

```sh
probe_dir=$(mktemp -d /var/tmp/psycles-shadow-surface-XXXXXX)
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_shadow_surface_oracle.hip -o "$probe_dir/oracle"
"$probe_dir/oracle" > "$probe_dir/oracle.txt"
diff -u tests/data/cycles_svm_shadow_surface.txt "$probe_dir/oracle.txt"
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_svm_shadow_surface_hip$'
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_svm_shadow_surface_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 ctest --test-dir build -V \
  -R '^psycles[.]luisa_cycles_svm_shadow_surface_vk$'
```

The 2018-word input yields 51 oracle records. Their captured SHA-256 is
`cec0e87b5eab651bf6129a3947ed0f0d39a2f5ca5ad992e7879f08f0ce74dd4f`.
The original baseline fails 49 absent-storage cases (the two zero-mix cases
already bypass dispatch); both physical-storage variants already pass.
After repair all four variants pass all 51 records. In the thin-wall case,
extinction is approximately `(0.547194, 0.440399, 0.337642)`, not merely the
alpha contribution `(0.2, 0.2, 0.2)`. Signed repeated transparency reaches
`(-0.75, 1.5, 3)`: clamping this state to `[0,1]` would change Cycles semantics.

- Full active-tree build uses 32 threads and passes (`full-build.log`).
- All 154 HIP tests pass (`all-hip.log`), then all 156 fallback tests pass
  (`all-fallback.log`).
- 100 core/adapter tests pass (`core-adapter.log`); the previously documented
  unrelated Blender probe-inventory test remains excluded. This is not a
  claim that every registered CTest passes.
- Ten strict native Vulkan canaries pass (`strict-vulkan.log`). The new
  test is additionally recompiled without cache under all three native-XIR
  environment guards and `LD_DEBUG=libs`. Four SPIR-V compilations succeed,
  the regression passes, and no DXC/DXIL library loads are recorded
  (`native-vulkan-recompile.log`).
- The uncached absent-storage diagnostic HIP IR retains only the ordinary
  255-float SVM stack. The production entry has no `alloca`, `llvm.memset`,
  or `noinline` (`hip_kernel_final_0.ll`, `hip_kernel_final_1.ll`). No dummy
  pool was introduced and scalar/vector default initialization is unchanged.
- The selectively staged Psycles candidate was exported separately and its
  entire runtime dependency chain built with 32 threads. Its HIP and fallback
  regression pass (`clean-build-final.log`, `clean-{hip,fallback}.log`), using
  the existing Luisa headers/libraries. This excludes other Psycles dirt,
  but is not a clean-Luisa validation claim.

The isolation gate first caught incorrectly rebased zero-context staging
hunks (`clean-build.log`), then a test reference to a pending runtime feature
constant (`clean-build-fixed.log`). The patch's selected offsets were
corrected, its complete allocation/setup suffix was checked byte-for-byte
against the validated worktree, and the test now uses the already committed
compiler ABI constant. Neither issue required including unrelated changes.

## Whole-renderer HIP canary

Lone Monk frame 4, original exported scene, 1440x1080, 256 fixed samples,
seed 0, native surface SVM and staged wavefront. No other build, test, or
GPU workload from this task ran during profiling. Evidence:
`/var/tmp/psycles-shadow-surface-lone-monk-IFK3oU`.

| Measurement | Result |
| --- | ---: |
| Render elapsed | 15.998 s |
| Previous unchanged-scene baseline | 16.0064 s |
| Coro fields / SoA / AoS | 88 / 444 B / 448 B |
| Surface time / VGPR / scratch / block | 8.117835 s / 256 / 3424 B / 512 |
| Closest intersection | 3.216777 s |
| Shadow intersection | 0.841134 s |
| NEE | 0.747463 s |
| Shadow shading | 0.333118 s |

Surface remains `kernel_dfda7ac3b73c5abc`; NEE remains
`kernel_e4f35e0f4e9963ba`. Their final IR hashes are byte-identical to the
previous baseline (new dump indices 3/4, previously 4/5):

- Surface: `f09c834825072e7911b64aeeda8e562652915d4649ea5c751bc08d1e9192ff07`.
- NEE: `91bf6424b243b39cfa8d809de864597cdfd3cd1cf2420bab132cf946257040c3`.

Relative RMSE against the matched Cycles HIP reference:

| Pass | Relative RMSE |
| --- | ---: |
| Combined | 0.01113625797 |
| DiffCol | 0.00122990356 |
| DiffDir | 0.01098037734 |
| DiffInd | 0.14105862288 |
| Env | 0.00285286301 |
| GlossDir | 0.01914233384 |
| GlossInd | 0.16197772574 |
| Normal | 0.00237476458 |

All eight actual passes contain zero nonfinite pixels. These measurements
show no material canary regression, not image parity or a measured speedup.
The shadow renderer still needs native SVM setup/caller migration, and the
indirect-pass differences and remaining overall performance gap are open.
