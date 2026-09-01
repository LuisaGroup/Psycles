# Cycles 5.2 SVM closure-core validation

This checkpoint copies the first executable `NODE_CLOSURE_BSDF` state
transitions from Blender Cycles 5.2.1 into the single Luisa SVM interpreter.
It does not route production shade-surface through the exact interpreter yet,
and it makes no full-scene parity or performance claim.

## Copied semantic boundary

The implementation follows these pinned Cycles sources at
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`:

- `kernel/svm/closure.h::svm_node_closure_bsdf`
- `kernel/closure/alloc.h::{closure_alloc,closure_sample_weight,bsdf_alloc}`
- `kernel/closure/bsdf_diffuse.h`
- `kernel/closure/bsdf_oren_nayar.h`
- `kernel/closure/bsdf_transparent.h`
- `kernel/closure/bsdf_util.h::maybe_ensure_valid_specular_reflection`
- `kernel/closure/bsdf_microfacet.h::{bsdf_microfacet_*_glass_setup,`
  `bsdf_microfacet_setup_fresnel_generalized_schlick,`
  `microfacet_ggx_preserve_energy}`
- `kernel/closure/bsdf_util.h::{F0_from_ior,fresnel_dielectric_Fss}`
- `kernel/util/lookup_table.h`

The copied executable closure types are smooth Diffuse, improved Oren-Nayar,
Translucent, Transparent, Beckmann Glass, GGX Glass, and Multi-GGX Glass.
Other closure payloads still take Cycles' exact typed skip transition and
report unsupported only when the unported state transition is live.
Feature-erased BSDF nodes remain valid skips.

`ClosurePool` is the Luisa realization of the state tuple

```text
(closure[0..num_closure), num_closure, num_closure_left)
```

and not a second material representation. Its storage is SoA, while the
observable transition system is Cycles' prefix allocator:

```text
allocate(type, weight):
  if left == 0: invalid
  else:
    closure[count] = {type, weight}
    count' = count + 1
    left' = left - 1

allocate_extra(owner, slots):
  precondition: owner is the immediately preceding ordinary allocation
  if slots <= left:
    count' = count
    left' = left - slots
  else:
    count' = count - 1
    left' = left + 1
    result = invalid
```

Only the initialized prefix `[0, count)` is readable. `bsdf_alloc` first
clamps negative spectral weights, computes `abs(average(weight))`, applies the
`1e-5` cutoff, and then performs this allocation. Transparent setup uses its
distinct unclamped average, accumulates extinction, and either appends the
unique transparent closure or finds and merges the existing one. The
terminate-path `left = 1` exception is preserved.

Let `capacity` be fixed and let
`extra = capacity - count - left`. The combined ordinary/extra transition
system gives these inductive invariants:

```text
0 <= count
0 <= left
0 <= extra
count + left + extra = capacity
closure indices written by ordinary allocation are exactly [0, count)
SD_TRANSPARENT implies at most one transparent record
```

An ordinary allocation preserves `extra`; a successful extra allocation moves
exactly `slots` from `left` to `extra`. When extra allocation fails, Cycles
removes the ordinary closure allocated immediately before it. Therefore the
composite transition is an identity on `(count, left, extra)`. The Glass
regression proves both concrete cases for a one-slot Fresnel payload:

```text
capacity 8: (0, 8, 0) -> ordinary (1, 7, 0) -> extra success (1, 6, 1)
capacity 1: (0, 1, 0) -> ordinary (1, 0, 0) -> extra failure (0, 1, 0)
```

The terminate exception starts from Cycles' required surface-evaluation state
`count = 0, left = 0`; it temporarily makes one physical slot available and
therefore preserves the prefix invariant.

Glass deliberately allocates with `make_spectrum(mix_weight)`, not
`closure_weight * mix_weight`; this is Cycles' actual SVM semantics. It then
copies the normalized normal, bump-map correction, saturated-squared
roughness, front/back-face IOR transform, generalized-Schlick tints and
thin-film state, albedo-table sample-weight adjustment, optional Multi-GGX
energy preservation, and the exact shader flags. Cycles' BSDF tables have one
shared Luisa reader/interpolator used by both the native SVM and the legacy
surface services; the native SVM does not introduce a second table format.

The Oren-Nayar parameter copy uses Cycles' `M_2PI_F = 2*pi`. A review caught
and removed an initially mistaken `2/pi` interpretation before commit. The
regression freezes `a`, `b`, and the three multi-scatter components, so this
structural error cannot return while insignificant backend ULP differences
remain tolerated.

## External Cycles oracle

Four canonical raw-closure scenes were created with the repository probe
tool, rendered by the diagnostic Blender 5.2.1 build `cb168525138f`, and dumped
after Cycles linked its final global SVM buffer. The permanent device test
freezes the exact surface tails. Only each four-word shader-jump entry is
relocated from the original global offsets to its compact test buffer.

| Scene | Cycles closure state at event 0 |
| --- | --- |
| Diffuse Probe | type 3; weight `(0.68, 0.24, 0.09)`; sample weight `0.3366666734`; sphere shading normal |
| Translucent Probe | type 9; weight `(0.73, 0.28, 0.11)`; sample weight `0.3733333349`; sphere shading normal |
| Transparent Probe | type 30; weight/extinction `(0.285, 0.342, 0.228)`; sample weight `0.2849999964`; plane normal; emission `(0.6324, 0.05952, 0.02232)` |
| Glass Transport 02 | type 24 (Beckmann Glass); weight/sample weight `1`; normal `(0,0,1)`; sampled roughness `(0.03,0.03)` and eta `1.5` |

The Glass test additionally freezes the source-derived typed setup state:
alpha, IOR, energy scale, tangent, generalized-Schlick discriminator, film
IOR/thickness, exponent, both tints, F0, F90, flags, cursor position, and
closure-pool accounting. These fields come from the pinned Cycles setup
functions; only the common closure and later sampled roughness/eta are claimed
as externally observed trace fields.

The test also contains a source-derived legal Add Shader stream with two
Transparent nodes. It proves that the second transition merges into the first
record, adds both sample weights and extinction, keeps `count = 1`, and leaves
seven of eight slots. It is labeled separately from the external streams and
is not presented as an independent oracle.

Oracle generation:

```sh
/home/mike/Projects/blender-install-5.2-hiprt/blender \
  --background --factory-startup \
  --python tools/create_cycles_shader_probe.py -- \
  /tmp/closure.blend diffuse_surface

PSYCLES_CYCLES_SVM_DUMP=/tmp/closure.svm52 \
/home/mike/Projects/blender-install-psycles-trace-5.2/blender \
  /tmp/closure.blend --background --threads 0 \
  --python tools/render_cycles_path_trace.py -- \
  /tmp/closure-trace.exr --width 64 --height 64 \
  --pixel-x 32 --pixel-y 32 --seed 20903 \
  --total-samples 64 --sample 0 --cycles-device CPU

python3 tools/decode_cycles_path_trace.py \
  /tmp/closure-trace.exr /tmp/closure-trace.json
```

Artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| Diffuse `.blend` | `3162156fbfaa60190edfaffcb94096707342d567ab91fc8b48472699828a8220` |
| Diffuse final SVM buffer | `c8270f57dc3c1148d54c3787ce7d8f9beb760d35bc26b5ff347dcfdc7c9cfb6b` |
| Diffuse path trace | `7fe55855a6dd5b60d091736dfb6a8b099a5ce33f702bab6e05240cca43f351ae` |
| Translucent `.blend` | `75ecaa2e0784e2f699be3929ea7a402d70b532a113f6e4fa0f06418f7a6fb7d3` |
| Translucent final SVM buffer | `c6e2904ad9e714626976ee59bd06c84cff1db8162f368b9b192c3d3ff3ae9a68` |
| Translucent path trace | `0002199583b45faa06e1f55fb5ce6f3069bb1eb182e0e5dd6cbbcc940e549e91` |
| Transparent `.blend` | `4c9fadfee08796cc5a1abcc66f42c10cbe12bb6bf202092d7dc5c9d87fd6b6a8` |
| Transparent final SVM buffer | `6858564307bd306f97f12078527919da84e03c1e58799a513ae2505b5027e4ec` |
| Transparent path trace | `cc26986ddfb2a40a7a41ac4836a7280df52ba8082967df1471f9493f79f45eec` |
| Glass `.blend` | `05acbf14b94c9a7fd88b05779c0f5567062c5ef20c42c6fa2a2b5b2962af8a0f` |
| Glass final SVM buffer | `d84f339e9d25276cd8086105c47353e85f8187dea0535aac0bb4cbea7da33c5e` |
| Glass path trace | `c7a5f70cb0bf26b48435dda9e2da7a3b989404b778113c5fed260417067ef0fc` |
| Glass decoded trace | `01975118833f375b388160e2d3da6dc86f0b5de43a558eb0e7eee36d6fc0006c` |
| Glass render metadata | `f329bead2d022e493911fe70c95a519a3f2fa5b8f011eaf534522a049de71f0d` |

No `.svm52` binary is checked in.

## Backend validation

The exact frozen streams pass on fallback, HIP, and strict native Vulkan
XIR-to-SPIR-V:

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_cycles_svm_closure_tests \
           psycles_luisa_thin_film_fresnel_tests \
           psycles_luisa_thin_film_surface_tests

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_closure_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_thin_film_(fresnel|surface)_(fallback|hip|vk)$'
```

Result: closure 3/3 and thin-film 6/6 passed. A complete 32-way run after the
public KernelGlobals/table change passed 464/464 tests in 37.17 seconds. The
Vulkan test environment is
`LUISA_VULKAN_USE_XIR=1`, `LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`, and
`LUISA_VULKAN_DISABLE_DXC=1`.

The first Luisa transcription expressed the three Glass tags as adjacent DSL
`$case` blocks and exposed that DSL cases do not have C++ fallthrough. The
permanent three-backend regression failed at the first cursor divergence.
The final implementation computes one `is_glass` partition and records the
payload/setup body once; it neither depends on fallthrough nor triples the
Glass shader AST.

## Visual oracle check

The clean and diagnostic Cycles builds rendered the Diffuse Probe at 512x512,
16 spp, CPU, fixed seed, adaptive sampling disabled by the golden renderer.
Combined max absolute error, mean absolute error, and RMSE are all zero over
262,144 pixels. I inspected the retained image at original resolution: the
sphere boundary, shading gradient, and background agree visually, while the
difference panel is uniformly black.

![Clean Cycles, diagnostic Cycles, and absolute difference](diffuse-oracle-triptych.png)

The triptych hash is
`27016c6db32044601df0515518495ae4db721f9192497647d5555c960e519e87`.
This proves the diagnostic SVM dump and trace instrumentation are
observationally inert for the probe. It is deliberately not described as a
Psycles production-render comparison before exact SVM is wired into
shade-surface.

## Remaining boundary

This checkpoint establishes ordinary and extra closure allocation plus the
simple and Glass setup families inside the exact interpreter. Production still
calls the old custom `SurfaceProgram` path. The next required structural steps
are the remaining typed closure payloads (especially Principled), exact closure
evaluation/sampling, and only then replacement and deletion of the old
shade-surface route. The Cycles-style global shader linker is already staged,
but this document makes no production-render parity or performance claim.
