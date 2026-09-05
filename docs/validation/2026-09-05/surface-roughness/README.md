# Cycles roughness-pass classification

## Source and failure

The legacy thin-wall regression mixed two different fixtures in one assertion:
records 43/44 sample the thick BSSRDF (parameter block 5), whereas records
45/46 query the thin-wall rough-diffuse pair (block 4). Cycles preserves 0.4
for the latter; 1 is the fallback when no BSDF contributes, including a
BSSRDF-only prefix. Returning 0.4 was not a closure-allocation or JIT-pruning
failure.

The committed shared AOV operation used its glossy-pass classification as
the roughness-pass classification. Those predicates are not equivalent.
Cycles' `bsdf_get_roughness_pass_squared` includes Oren-Nayar and Rough
Translucent before skipping ordinary diffuse closures; singular closures
contribute zero, microfacets use their alpha product, and other BSDFs use
one. `surface_shader_average_roughness` filters to BSDFs and weights the
result by `fabsf(average(sc->weight))`.

The fix projects that classification onto the existing common post-setup
roughness value. It does not change normal/albedo reductions, allocation,
SVM streams, or initialization semantics. In particular it does not stage
the unrelated AOV rewrite already present in the main working directory.

`tools/cycles_surface_roughness_oracle.hip` invokes the actual Cycles device
function on HIP. The shared header contains input data only. The independent
test freezes 34 observed outputs and reads closure types/weights from runtime
buffers. Cases cover empty and BSSRDF-only prefixes, both rough-diffuse types,
thin-wall and mixed-roughness pairs, signed and zero-average weights,
singular closures, and the complete microfacet intervals. The latter include
the authoring-only multi-GGX identities, even though setup normally converts
them before producing retained closures.

On the isolated committed baseline, ten oracle cases fail. The pre-existing
dirty candidate fixed most of them but still omitted the multi-GGX-glass
endpoint. The permanent test catches both failure modes.

Oracle source: `/home/mike/Projects/blender-cycles-trace-5.2`, revision
`cb168525138fecc792cc393f94afc39582b0103c`.

- `kernel/closure/bsdf.h` SHA-256:
  `e2a897f1d85bb5c1f1eea000565e257883674baa9c816e08725e41bf33401383`
- `kernel/integrator/surface_shader.h` SHA-256:
  `1b36b3c2082b8992252b81869263a5e959e33713395b0b58de63d66c5052a69d`

## Validation

All builds use 32 threads. The active full build passed. All 39 compiler/SVM
tests and the host path-scheduler test passed. The registered HIP suite is
144/144 and fallback suite 146/146. The new roughness test and local shadow
traversal test also pass on Vulkan with all three strict settings:
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`; they were rerun after the final HIP render.

The isolated candidate `/var/tmp/psycles-roughness-clean-Zty1W2/source`
starts at `58f409b0` and contains only the new test/fixture/CMake registration,
the roughness classification fix, and the corrected thin-wall expectation.
It builds the complete `Psycles::luisa` dependency chain against the existing
Luisa libraries. Both roughness and thin-wall tests pass on HIP and fallback.
Before the production fix, the new test fails and the old thin-wall assertion
passes, confirming that the latter had frozen the incorrect behavior.

Logs are `/var/tmp/psycles-surface-roughness-*` and the isolated candidate's
`baseline-hip.log`, `fixed-hip.log`, and `fixed-fallback.log`. This does not
claim to resolve the previously recorded, normally unregistered HIP
all-scheduler film diagnostic in the unrelated dirty renderer work.

## Full scene and next structural work

The native-SVM Lone Monk HIP canary remains 1440x1080, 256 spp, frame 4,
seed 0, staged wavefront, with no other build/test running during profiling.
Elapsed time is 19.6886 s; surface is 12.329953 s for 819,494,400 rounded
lanes. The surface kernel identity remains `kernel_e4d18539f77f6f15`, with
256 VGPRs and 5,252 B private scratch. The coroutine remains 90 fields,
456 B AoS / 452 B SoA per slot. This legacy AOV fix does not establish a
native-renderer speedup.

Combined relative RMSE against the matched Cycles HIP oracle is 1.1159853%,
with zero nonfinite pixels. Evidence is
`/var/tmp/psycles-roughness-canary-MaPZtj`. Cycles remains 16.4504 s total
and 5.949119 s surface; the overall performance goal is not met.

Two live gfx1201 PMC probes under `/var/tmp/psycles-surface-pmc-vHs6h0`
count the expected waves but return zero for SALU, VALU, TEX_LOAD, wave32
instruction and VALU-cycle counters. These values cannot support a
compute-versus-memory bottleneck claim. No system/toolchain changes were
made to try to enable them.

The production route/IR audit confirms a remaining native-SVM integration
gap: `make_surface_callables` selects native SVM for surface population,
but nonconstant emitter evaluation still selects
`make_compact_surface_emission_callable`, emitting
`surface_emission_unified_svm`. Cycles instead evaluates its original stream
with `KERNEL_FEATURE_NODE_MASK_SURFACE_LIGHT` and `PATH_RAY_EMISSION`.
This is a concrete migration target, not yet a proven source of the timing
gap. The corresponding BSSRDF-normal consumer also still selects the legacy
route. Neither path is changed by this commit.
