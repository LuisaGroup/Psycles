# Closure-free Cycles SVM light emission

## Cause and semantic mapping

Cycles 5.2.1 sets `num_closure = num_closure_left = 0` for
`PATH_RAY_EMISSION` in `surface_shader_eval`. Principled emission still
evaluates alpha, sheen, and coat in layer order. Its
`bsdf_alloc_maybe_emission` uses local BSDF values for layer attenuation;
it does not require a physical closure array. Transparency updates flags
and extinction even when `closure_alloc` cannot allocate a slot.

Psycles rejected the Principled emission-only dispatch when
`ShaderData::closure == nullptr`. The allocation, sheen, coat, and
transparency helpers also dereferenced that host pointer while recording
DSL, including inside device-side branches. A device guard cannot protect
a host/JIT-time dereference.

The fix represents absent closure storage as the implicit state
`count == left == 0`. Allocation fails without touching storage; local
sheen/coat evaluation and transparent/emission state updates remain.
Only physical pool accesses receive host-side presence guards. There is
no dummy closure array, new SVM representation, changed typed payload or
PC rule, application `noinline`, altered scalar/vector initialization, or
software floating-point path.

## Permanent external-oracle regression

`tests/cycles_svm_light_emission_fixture.h` shares only serialized inputs.
Its 991-word image has 16 shader entries and uses the original shader jump,
scalar value, Principled BSDF typed payload, closure weight, and END
opcodes. The observable closure-weight successor verifies consumption of
the complete 176-byte Principled payload.

Cases cover plain emission, alpha, sheen, coat, combined layers, repeated
accumulation, zero emission, zero mix weight, a mix weight below the closure
cutoff, black coat tint, zero alpha, grazing incidence, negative sheen tint,
negative emission components, and zero coat/sheen roughness. Shader words
and incident cosines are device-buffer inputs, not per-case JIT constants.

`tools/cycles_svm_light_emission_oracle.hip` executes the external Cycles
`svm_eval_nodes<SURFACE_LIGHT, SURFACE>` on genuine
`ShaderDataTinyStorage` with no closure slots. Because that entry returns
void, a second focused dispatch over the same input invokes the original
Cycles closure function to expose the final PC. It checks agreement with
the full interpreter. All nine required BSDF lookup tables come from
Cycles' own `scene/shader.tables`. No CPU reference renderer or copied
reference shading formula is involved.

Golden rows in `tests/data/cycles_svm_light_emission.txt` contain:
case index, emission XYZ, flags, extinction XYZ, final PC, closure count,
and remaining slots. The final oracle run reproduces the checked-in text.
Numerical comparisons use relative tolerance 5e-5 with a 1e-8 scaling floor;
they do not require bitwise floating-point equivalence.

The Luisa regression exercises absent storage and a four-slot physical
array with zero remaining slots. It checks emission, extinction, flags,
count/remaining slots, unchanged LCG state, and diagnostic END status/PC
and successor weight. In the active worktree it additionally exercises the
pending production entry; the committed diagnostic API remains sufficient
to build and test this isolated change.

Before the runtime fix, HIP reports unsupported status for 15 of the 16
absent-storage cases; the zero-mix case correctly skips. For example,
case 0 returns emission (0,0,0), status 2, PC 114 instead of Cycles'
(2,1,0.5), ended status, PC 119. The zero-capacity physical-array variant
already passes. Both storage variants pass after the fix.

## Oracle identity and reproduction

External checkout: `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`.

SHA-256, paths relative to `intern/cycles`:

| Source | SHA-256 |
| --- | --- |
| kernel/svm/closure.h | `585a5e6a6a9507e76983a2cf2e5fdb757f70e9213c5f1b151e6b8ed47842c673` |
| kernel/closure/alloc.h | `0c4d641007cce198cf98133fbf50f796c1ccd2bdcaf25d99f38302f63732f346` |
| kernel/closure/bsdf_transparent.h | `0c962e9fc570bec559d77c1d28809fc2ac2327fbfa164f1b1da5c3ac60cf1b4c` |
| scene/shader.tables | `fbde5fea5750eb32e2a638b74b97ef6a39eed29283b2f46eb139893ca3af5175` |

From the Psycles root, with a fresh task-specific evidence directory:

```sh
probe_dir=$(mktemp -d /var/tmp/psycles-light-emission-XXXXXX)
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++17 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_light_emission_oracle.hip -o "$probe_dir/oracle"
"$probe_dir/oracle" > "$probe_dir/oracle.txt"
diff -u tests/data/cycles_svm_light_emission.txt "$probe_dir/oracle.txt"
cmake --build build --parallel 32 \
  --target psycles_luisa_cycles_svm_light_emission_tests
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_svm_light_emission_hip$'
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_svm_light_emission_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 ctest --test-dir build -V \
  -R '^psycles[.]luisa_cycles_svm_light_emission_vk$'
```

## Validation and limits

All builds use 32 threads. Evidence is in
`/var/tmp/psycles-light-emission-VBHAdP`.

- Full active-tree build passes.
- All 145 registered HIP tests pass, followed by all 147 fallback tests.
- All 40 compiler/SVM and host scheduler tests pass.
- The new Vulkan test passes with all three strict native-XIR environment
  settings above. Its uncached log records SPIR-V generation; DXC is disabled.
- An isolated candidate based on `b38a93ce` builds its complete
  `Psycles::luisa_runtime` dependency chain and passes the new HIP and
  fallback regression. It excludes other dirty Psycles changes, but uses
  the existing Luisa headers/libraries: this is not a clean-Luisa claim.
- After final formatting, the oracle and active/isolated regression targets
  rebuild, and focused active HIP/fallback/native-Vulkan plus isolated
  HIP/fallback tests pass again.

Logs include `regression-hip-before.log`, `full-build.log`,
`suite-{hip,fallback,core}.log`, `native-vulkan.log`,
`oracle-build-formatted.log`, `oracle-final.txt`,
`regression-build-formatted.log`, `clean-build-formatted.log`, and
`{final,clean-final}-{hip,fallback}.log`.

The uncached HIP IR probe uses the regression's `--no-cache` argument
with `LUISA_DUMP_LLVM_IR=1`. Its absent-storage diagnostic kernel retains
only the ordinary 255-float SVM stack allocation; the pending production
entry uses the fixture's one reachable stack lane and has no alloca
instruction after optimization. Neither emits closure-array storage,
`memset`, or `noinline`. This tiny fixture does not establish the full
renderer's register or scratch footprint.

## Whole-renderer canary: semantic prerequisite, not a speedup

Final HIP canary: Lone Monk frame 4, 1440x1080, 256 fixed samples, seed 0,
native SVM surface, staged wavefront. No other GPU test or build runs
during profiling. Evidence:
`/var/tmp/psycles-light-emission-canary-1wfXPq`.

| Measurement | Result |
| --- | ---: |
| Render elapsed | 19.7822 s |
| Surface kernel | 12.437055 s |
| Closest intersection | 3.202342 s |
| Shadow intersection | 0.831001 s |
| Shadow shading | 0.329377 s |
| Coro frame fields / AoS / actual SoA per slot | 90 / 456 B / 452 B |
| Surface VGPR / private scratch | 256 / 5252 B |
| Combined relative RMSE versus Cycles HIP | 1.115790% |
| Normal relative RMSE | 0.237835% |
| DiffCol relative RMSE | 0.122990% |
| Nonfinite pixels | 0 |

The matched Cycles HIP reference is
`/var/tmp/psycles-lone-monk-cycles-hip-profile-ulQYKD`;
its metadata confirms adaptive sampling and denoising disabled,
1440x1080, 256 samples, frame 4, seed 0, and elapsed 16.450413 s.
The surface kernel identity remains `kernel_e4d18539f77f6f15`.
No renderer speedup is attributed to this change.

Cycles' corresponding persistent SoA fields use 172 B main plus 224 B
shadow, or 396 B per equal-capacity slot pair for the compared feature
configuration. Psycles' 452 B is 56 B (14.1%) larger. Neither number includes
queues, sorting storage, registers, or private scratch, and the size
difference does not imply a proportional execution-time difference.

Native SVM nonconstant-emitter integration and Cycles-equivalent
`SHADE_LIGHT_NEE` stage placement remain unfinished. This correction
removes their closure-storage prerequisite; it does not silently replace
the existing renderer emission entry or claim the overall performance
goal has been achieved.
