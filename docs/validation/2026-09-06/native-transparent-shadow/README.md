# Native Cycles transparent-shadow state

The production native surface path now evaluates transparent shadow hits with
Cycles ShaderData and the full SURFACE_SHADOW SVM interpreter, instead of
projecting each hit through the legacy SurfacePoint/material evaluator.
This is a semantic migration, not evidence that the renderer has reached
Cycles image or performance parity.

## State and control-flow audit

| Cycles 5.2.1 operation | Native implementation |
| --- | --- |
| `shader_setup_from_ray` | Hit barycentrics, global object/primitive identity, native object matrices and packed triangle normals |
| Instance and backface setup | Shared triangle setup with NEE; no per-hit inverse or extra shading-normal hemisphere projection |
| Compact ray/UV differentials | `dP = ray.dP + t * ray.dD`, `dI = ray.dD`, original dominant-axis determinant branch |
| `surface_shader_eval<SURFACE_SHADOW>` | Shadow visibility, zero path flags, zero closure capacity, original word PC loop and node feature mask |
| `integrator_state_lcg_init` | Sample, RNG hash and the current per-hit RNG offset |
| `SD_HAS_ONLY_VOLUME` | Skip surface SVM, return unit transparency and advance volume-boundary count |
| `surface_shader_transparency` | Signed, unclamped spectrum; explicit ray-portal veto |
| `integrate_transparent_shadow` | Multiply the full shadow throughput, terminate on all-zero product before committing throughput/bounce/RNG |
| Surviving hit | Advance transparent bounce and RNG offset by `PRNG_BOUNCE_NUM = 16`; retain uint16 storage semantics |
| Volume-boundary guard | Terminate above `VOLUME_BOUNDS_MAX = 1024`, including the final empty segment |
| Four recorded hits | Re-enter shadow intersection even for an exactly-full final batch |

The previous positive-current-transparency predicate was not equivalent to
Cycles' nonzero-cumulative-throughput predicate. A negative spectrum is not
opaque, whereas two individually nonzero red/green filters can multiply to
zero. The latter terminates without committing the second hit's counters.
Keeping a separate unshadowed contribution and only a transparency product
also missed zeros originating in the receiving path/BSDF. Direct-light task
state now retains the complete Cycles shadow throughput. Film accumulation
does not multiply the unshadowed contribution a second time.

The optional fused tracing result still exposes a transparency factor for
legacy volume integration and debug records. Native direct lighting consumes
its complete throughput. Unused diagnostic fields are ordinary JIT dead code;
no `noinline` annotation or alternate SVM dispatch was introduced.

## External oracle

Source: `/home/mike/Projects/blender-cycles-trace-5.2`, commit
`cb168525138fecc792cc393f94afc39582b0103c`.

| Source file | SHA-256 |
| --- | --- |
| `kernel/integrator/shade_shadow.h` | `9754c74ff2f0e5525efe91ce96afb7aed00a94eec09e205ebadf4aab3c366d6d` |
| `kernel/geom/shader_data.h` | `d96bbb1cc712df9a5862943a9ee20fac726204e629f809610b8525bda19de666` |
| `kernel/geom/triangle.h` | `0c44e107f9df069697a73e127801eef4b9c66387893d4099014cee740752d4e8` |
| `kernel/integrator/surface_shader.h` | `1b36b3c2082b8992252b81869263a5e959e33713395b0b58de63d66c5052a69d` |

`tools/cycles_svm_native_shadow_oracle.hip` includes and executes those original
functions on HIP. It allocates genuine Cycles GPU shadow SoA and uses
ShaderDataTinyStorage. The shared fixture contains only inputs and serialized
typed node words; no reference shader or integrator formulas are translated
into the test. Expected data SHA-256:
`95991a6416897c9099a39f722cd1c3c96ef6783641dd3f1143293de6a6314eb6`.

The 25 setup/SVM inputs cover smooth/flat/corner normals, backfacing, applied
and unapplied negative scale, nonuniform/rotated transforms, a thin triangle,
opaque and portal closures, volume-only boundaries, signed transparency and
Light Path visibility/bounce queries. Instance IDs are permuted, and local
primitive zero maps through distinct native primitive offsets.

The 14 prepared intersection sequences exercise the production staged and
fused consumers, including cumulative color cancellation, empty batches,
exactly four hits, per-hit Light Path reads, volume limits and uint16 counter
wrap. Their collector supplies intersections rather than replacing a BVH;
the full renderer canary separately checks production traversal and routing.

Reproduce from the Psycles root:

```sh
probe_dir=$(mktemp -d /var/tmp/psycles-native-shadow-XXXXXX)
/opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201 \
  -DHIPCC -std=c++20 -O3 -ffast-math \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles -I include \
  tools/cycles_svm_native_shadow_oracle.hip -o "$probe_dir/oracle"
"$probe_dir/oracle" > "$probe_dir/oracle.txt"
diff -u tests/data/cycles_svm_native_shadow.txt "$probe_dir/oracle.txt"
cmake --build build --parallel 32
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_svm_native_shadow_hip$'
ctest --test-dir build --output-on-failure \
  -R '^psycles[.]luisa_cycles_svm_native_shadow_fallback$'
env LUISA_VULKAN_USE_XIR=1 LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  LUISA_VULKAN_DISABLE_DXC=1 \
  ./build/bin/psycles_luisa_cycles_svm_native_shadow_tests vk --no-cache
```

## Validation record

Evidence directory: `/var/tmp/psycles-native-shadow-0EWY8y`.

- The external oracle was built and executed successfully; its output is
  byte-identical to the checked-in 39 records.
- All 25 setup/SVM and 14 staged/fused cases pass on HIP
  (`native-bound-hip.log`). Film finalization additionally verifies that the
  pre-shadow diagnostic contribution is not multiplied in twice.
- The initial fixture launch crashed because its scene omitted the BSDF lookup
  buffer. The host launch backtrace identifies argument binding; uploading the
  same table as production fixes it without changing the shader cache identity
  or modifying Luisa (`native-gdb.log`).
- The shared NEE setup's existing HIP oracle passes (`nee-setup-hip.log`).
- Full 32-thread build passes (`full-build-layout.log`).
- All 155 HIP tests pass (`all-hip-green.log`), followed by all 157 fallback
  tests (`all-fallback.log`). The first HIP sweep caught a test's obsolete
  31-member layout assertion. The required volume-boundary counter occupies
  the old tail padding: AoS remains 224 bytes. The regression now also rejects
  array/structure members and verifies the new field at two SoA capacities.
- 100 core/adapter tests pass (`core-adapter.log`). The known unrelated
  `blender_export_render_settings` probe-inventory test is excluded; this is
  not a claim that every registered CTest passes.
- Eleven strict native Vulkan canaries pass (`strict-vulkan.log` plus
  `strict-vulkan-queue.log`). The new test additionally compiles both kernels
  without cache under all three native-XIR guards and `LD_DEBUG=libs`:
  25,728 and 27,735 SPIR-V words; no DXC/DXIL library loads are recorded
  (`native-vulkan-recompile.log`).
- The selectively staged Psycles candidate builds independently with 32
  threads (`clean-build-final.log`). Native shadow, NEE setup, NEE emission
  and direct-light layout tests pass on both HIP and fallback (4/4 each,
  `clean-hip.log`, `clean-fallback.log`). Its runtime/core libraries resolve
  to the isolated build. It uses existing Luisa headers/libraries; this is
  deliberately not a clean-Luisa validation claim.

## Whole-renderer HIP canary

Lone Monk frame 4, unchanged exported scene, 1440x1080, 256 fixed samples,
seed 0, native SVM and staged wavefront. Evidence:
`/var/tmp/psycles-native-shadow-monk-G2YXwy`. No other build, test, or GPU
workload from this task ran during profiling.

| Measurement | Previous legacy-shadow path | Native-shadow path |
| --- | ---: | ---: |
| Render elapsed | 15.998 s | 16.446 s |
| Coro fields | 88 | 89 |
| Actual SoA bytes per slot | 444 | 448 |
| AoS frame bytes | 448 | 448 |
| Surface kernel time | 8.117835 s | 8.160753 s |
| Closest intersection time | 3.216777 s | 3.210400 s |
| Shadow intersection time | 0.841134 s | 0.834924 s |
| NEE time | 0.747463 s | 0.783629 s |
| Shadow shading time | 0.333118 s | 0.561421 s |

This single-run comparison is approximately 2.8% slower, not a speedup or a
stable performance estimate. The native shadow kernel uses 256 VGPRs and
1904 scratch bytes, compared with 128 VGPRs / 16 bytes in the legacy shadow
projection. Surface uses 256 VGPRs / 3456 bytes; NEE 256 / 1456 bytes.
Their workgroups remain 512 (surface) and 32 (other path kernels).

Actual SoA size is the sum of the frame field sizes. The independently laid
out payload type includes its own alignment padding; its printed size must
not simply be added to the reserved fields to infer SoA allocation.

Relative RMSE against the matched Cycles HIP reference:

| Pass | Relative RMSE |
| --- | ---: |
| Combined | 0.01114060806 |
| DiffCol | 0.00122989714 |
| DiffDir | 0.01098686972 |
| DiffInd | 0.14106063976 |
| Env | 0.00285286301 |
| GlossDir | 0.01914543362 |
| GlossInd | 0.16198229484 |
| Normal | 0.00237878528 |

All eight actual passes contain zero nonfinite pixels. The image discrepancy
is essentially unchanged; indirect-pass parity remains unresolved.

The live per-kernel audit found a concrete remaining scheduling difference:
Cycles `integrator_intersect_shadow` terminates opaque paths in the
INTERSECT_SHADOW kernel, whereas our staged transport still suspends into
SHADE_SHADOW before testing `batch.blocked`. The matched Cycles control run
records 962 intersection and 381 shading dispatches; this run records 1195
of each. Dispatch counts also depend on batching, so this is not a normalized
work comparison, but the unconditional extra cut is directly visible in
source. Removing it requires the corresponding coroutine transition
regression and another complete renderer measurement; it must not be hidden
by a `noinline` or material-specific workaround.

Reproduce the timed render from an empty evidence directory:

```sh
run_dir=$(mktemp -d /var/tmp/psycles-native-shadow-monk-XXXXXX)
cd "$run_dir"
rocprofv3 --kernel-trace --output-format rocpd \
  --output-directory "$run_dir" --output-file render -- \
  env PSYCLES_NATIVE_CYCLES_SVM_SURFACE=1 PSYCLES_DISABLE_SHADER_CACHE=1 \
  PSYCLES_DUMP_COROUTINE_FRAME=1 LUISA_DUMP_LLVM_IR=1 \
  LUISA_DUMP_HIP_ISA="$run_dir" \
  /home/mike/Projects/Psycles-surface-svm/build/bin/psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-f297ec53-20260904/export "$run_dir/render.exr" \
  hip 1440 1080 256 64 - 0 0 0 0 256 - 1 0 wavefront-staged 32 32768 32 1 1 0
```

## Remaining scope

Native geometry setup currently targets the uploaded static-triangle image.
Motion, curves and points need their respective Cycles setup implementations;
they must not be reinterpreted as static triangles. Full native volume-stack
integration and texture-cache miss/resume handling remain separate unfinished
parts of the renderer. The existing legacy volume caller supplies static time
and still consumes a transparency factor. The main path does not yet carry
portal-bounce state, so that Light Path counter remains zero in the shadow
projection as well. This change does not establish
whole-renderer SVM, image, coroutine or performance completion.
