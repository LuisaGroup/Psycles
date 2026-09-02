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
- `kernel/closure/bsdf_sheen.h`
- `kernel/closure/bsdf_ashikhmin_velvet.h`
- `kernel/closure/bsdf_toon.h`
- `kernel/closure/bsdf_ray_portal.h`
- `kernel/closure/bsdf_hair.h`
- `kernel/closure/bsdf_principled_hair_chiang.h`
- `kernel/closure/bsdf_util.h::maybe_ensure_valid_specular_reflection`
- `kernel/closure/bsdf_microfacet.h::{bsdf_microfacet_*_glass_setup,`
  `bsdf_microfacet_ggx_setup,bsdf_microfacet_beckmann_setup,`
  `bsdf_microfacet_*_refraction_setup,`
  `bsdf_microfacet_setup_fresnel_conductor,`
  `bsdf_microfacet_setup_fresnel_f82_tint,`
  `bsdf_microfacet_setup_fresnel_generalized_schlick,`
  `microfacet_ggx_preserve_energy}`
- `kernel/closure/bsdf_ashikhmin_shirley.h`
- `kernel/closure/bsdf_util.h::{F0_from_ior,fresnel_dielectric_Fss}`
- `kernel/closure/bssrdf.h::{bssrdf_alloc,bssrdf_setup,`
  `bssrdf_setup_radius}`
- `kernel/util/lookup_table.h`

The copied executable closure types are smooth Diffuse, improved Oren-Nayar,
Translucent, Transparent, Beckmann/GGX/Multi-GGX Glass,
Beckmann/GGX/Ashikhmin-Shirley/Multi-GGX Glossy, and Beckmann/GGX Refraction.
The copied Metallic family includes F82-tint and physical-conductor Fresnel
models over Beckmann, GGX, and Multi-GGX distributions.
The standalone Sheen family includes Microfiber LTC Sheen and Ashikhmin
Velvet with the exact typed `SVMNodeSimpleBsdfData` payload. The standalone
Toon family includes Diffuse and Glossy Toon with the exact typed
`SVMNodeToonBsdfData` payload and reflective-caustics gate. The standalone
Ray Portal family includes the exact typed `SVMNodeRayPortalBsdfData`
payload, direct signed-weight allocation, transparent extinction, and portal
position/direction state. The standalone legacy Hair family includes
Reflection and Transmission with the exact typed `SVMNodeHairBsdfData`
payload, primitive-dependent tangent construction, and signed offset. The standalone
Principled Hair Chiang and Huang population/setup paths consume the exact
typed `SVMNodePrincipledHairBsdfData` payload and retain typed closures. Huang
also preserves the two-slot extra allocation, raw curve-key radius lookup,
elliptical local frame, and transparent fallback. Huang eval, sample, albedo,
and blur now execute the copied Cycles 5.2.1 scattering equations. Chiang
eval, sample, albedo, and blur likewise execute its copied Cycles 5.2.1
scattering equations. The standalone
Subsurface Scattering family includes Christensen-Burley,
Random Walk, Random Walk Legacy, and Random Walk Skin with the exact typed
`SVMNodeBssrdfData` payload.
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

The SoA payload is one tagged union, matching Cycles' `ShaderClosure` storage.
Oren-Nayar, Sheen, and Microfacet reuse the same seven `float4` payload rows;
the Microfacet plus discriminated Fresnel payload is currently the largest
member. Adding Sheen therefore adds no per-closure payload rows. This prevents
storage from growing as the sum of all closure-family structs while retaining
typed accessors at the Luisa DSL boundary.

Glass deliberately allocates with `make_spectrum(mix_weight)`, not
`closure_weight * mix_weight`; this is Cycles' actual SVM semantics. It then
copies the normalized normal, bump-map correction, saturated-squared
roughness, front/back-face IOR transform, generalized-Schlick tints and
thin-film state, albedo-table sample-weight adjustment, optional Multi-GGX
energy preservation, and the exact shader flags. Cycles' BSDF tables have one
shared Luisa reader/interpolator used by both the native SVM and the legacy
surface services; the native SVM does not introduce a second table format.

Standalone Glossy and Refraction deliberately retain their distinct Cycles
allocation rules. Glossy allocates `closure_weight * mix_weight`, squares the
saturated roughness, clamps anisotropy to `[-0.99, 0.99]`, and applies Cycles'
signed anisotropic alpha mapping and tangent rotation only outside the
`1e-4` isotropy threshold. Beckmann, GGX, and Ashikhmin-Shirley preserve their
own closure tags and setup flags. Multi-GGX is normalized to the ordinary GGX
runtime tag only after applying Cycles' constant-Fresnel table compensation;
its color payload is therefore emitted only for that distribution.

The graph stage also copies the prerequisite Cycles topology rule: an
unconnected non-isotropic Glossy Tangent is default-connected from
`Geometry.Tangent` through Cycles' NORMAL-to-VECTOR automatic conversion.
Only a literal, unlinked `abs(Anisotropy) <= 1e-4` permits
`GlossyBsdfNode::simplify_settings` to remove that edge. The old strict-type
default connector silently lost the non-isotropic edge; a dedicated external
word oracle now freezes both `NODE_GEOMETRY` records, tangent stack offset 3,
and peak stack use 6.

Refraction also allocates `closure_weight * mix_weight`, but squares the raw
roughness before the setup function saturates it, exactly preserving Cycles'
ordering for negative and out-of-range inputs. It selects the Beckmann or GGX
transmission tag and replaces eta with its reciprocal on a back face. These
are separate typed payloads in the word image; neither is translated through
the old Psycles surface representation.

Metallic preserves Cycles' distinct 13-word payload and white
`make_spectrum(mix_weight)` allocation. Static graph compilation prunes the
unused parameter family: F82 carries Base Color and Edge Tint, while physical
conductor carries IOR and Extinction. Roughness and anisotropy are saturated,
the standalone Metallic aspect ratio is `sqrt(1 - 0.9 * anisotropy)`, and its
tangent is rotated around the authored normal before bump-normal correction.
F82 and conductor use their own typed Fresnel payloads and sample-albedo
functions. Multi-GGX alone applies the table-derived energy preservation and
then uses the ordinary GGX runtime tag. The extra Fresnel record is represented
as one tagged SoA union: each variant reads and writes only its live fields.
The failure transition intentionally retains the setup flags written before
Cycles rolls the immediately preceding ordinary allocation back; the
one-slot regression freezes this source ordering rather than normalizing it.

The Oren-Nayar parameter copy uses Cycles' `M_2PI_F = 2*pi`. A review caught
and removed an initially mistaken `2/pi` interpretation before commit. The
regression freezes `a`, `b`, and the three multi-scatter components, so this
structural error cannot return while insignificant backend ULP differences
remain tolerated.

## Staged Principled word image

The compiler now has an external full-payload oracle for
`SVMNodePrincipledBsdfData`; this does not yet claim an executable Principled
device transition. The oracle finite product contains the untouched Blender
node plus two all-field nodes. It freezes all 44 payload words, both
distribution enums, Random Walk Skin and Burley subsurface tags, both Thin
Wall values, and Cycles' automatic Normal/Coat Normal/Tangent topology.

This audit found that the Psycles core schema defaulted Principled to GGX,
while the pinned Blender/Cycles 5.2.1 node defaults to Multi-GGX. The schema is
now corrected, and the untouched-node word image permanently locks the
default. Its peak stack use is three values because Normal and Coat Normal
share the default normal; the anisotropic all-field cases use six values after
Cycles inserts the default tangent conversion.

## External Cycles oracle

Focused closure families and their frozen transitions were created with the
repository probe tool, rendered by the diagnostic Blender 5.2.1 build
`cb168525138f`, and dumped
after Cycles linked its final global SVM buffer. The permanent device test
freezes the exact surface tails. Only each four-word shader-jump entry is
relocated from the original global offsets to its compact test buffer.

| Scene | Cycles closure state at event 0 |
| --- | --- |
| Diffuse Probe | type 3; weight `(0.68, 0.24, 0.09)`; sample weight `0.3366666734`; sphere shading normal |
| Translucent Probe | type 9; weight `(0.73, 0.28, 0.11)`; sample weight `0.3733333349`; sphere shading normal |
| Transparent Probe | type 30; weight/extinction `(0.285, 0.342, 0.228)`; sample weight `0.2849999964`; plane normal; emission `(0.6324, 0.05952, 0.02232)` |
| Glass Transport 02 | type 24 (Beckmann Glass); weight/sample weight `1`; normal `(0,0,1)`; sampled roughness `(0.03,0.03)` and eta `1.5` |
| Glossy BSDF Matrix 00 | type 12 (GGX); weight `(0.68,0.24,0.09)`; sample weight `0.3366666734`; alpha `(0.1600000113,0.1600000113)` |
| Glossy BSDF Matrix 08 | type 13 (Beckmann); weight `(0.5669999719,0.2430000156,0.0810000002)` after mix weight; sample weight `0.2970000207`; alpha `(0.0400000028,0.0400000028)` |
| Glossy BSDF Matrix 09 | input type 14 (Multi-GGX), output type 12; compensated weight `(0.5023012757,0.1866041273,0.0582876727)`; sample weight `0.2349678278`; energy scale `1.4335616827` |
| Glossy Ashikhmin-Shirley | type 15; roughness zero clamps to alpha `(0.0001,0.0001)` and still sets `SD_BSDF_HAS_EVAL` |
| Glossy anisotropic default Tangent | type 12 (GGX); unlinked source tangent becomes rotated `(0,1,0)`; alpha `(0.0800000057,0.3200000226)` for anisotropy `0.5`, rotation `0.25` |
| Refraction BSDF Matrix 00 | type 20 (Beckmann Refraction); weight `(0.3822740018,0.6512780190,0.8680070043)`; sample weight `0.6338530779`; alpha `0.0187969934`; eta `1.1599999666` |
| Refraction BSDF Matrix 01 | type 21 (GGX Refraction); otherwise the same source inputs and observed state as case 00 |
| Refraction BSDF Matrix 07 | type 20 on a back face; weight `(0.64,0.09,0.49)`; alpha `0.2024999857`; reciprocal eta `0.6666666865` |
| Metallic F82 GGX | type 12 (GGX); weight `1`; sample weight `0.2983346879`; alpha `(0.0324000008,0.0324000008)` |
| Metallic F82 Beckmann | type 13 (Beckmann); weight `1`; sample weight `0.5333424211`; alpha `(0.1224999949,0.1224999949)` |
| Metallic F82 Multi-GGX anisotropic film | output type 12; weight `(0.9768685699,0.9523739219,0.9431397319)`; sample weight `0.2315439284`; alpha `(0.2977624834,0.1503700614)`; energy scale `1.06322` |
| Metallic conductor GGX | type 12 (GGX); weight `1`; sample weight `0.6869250536`; alpha `(0.0324000008,0.0324000008)` |
| Metallic conductor Beckmann | type 13 (Beckmann); weight `1`; sample weight `0.6516747475`; alpha `(0.1224999949,0.1224999949)` |
| Metallic conductor Multi-GGX anisotropic film | output type 12; weight `(0.9931033254,0.9784969091,0.9566794634)`; sample weight `0.5281172991`; alpha `(0.2977624834,0.1503700614)`; energy scale `1.06322` |
| Principled Sheen | Transparent type 30, Sheen type 7, then Diffuse type 2; Sheen weight `(0.0003624241,0.0010147875,0.0006523633)` and sample weight `0.0006765249`; attenuated emission `(0.2077361,0.1038681,0.6232085)` |
| Principled Coat | Transparent type 30, dielectric GGX type 12, then Diffuse type 2; Coat weight `(0.5166048,0.5166048,0.5166048)` and sample weight `0.01746508`; tinted Diffuse `(0.1952816,0.4025556,0.4595241)`; attenuated emission `(0.1637846,0.09181092,0.4318419)` |
| Principled Metallic | Transparent type 30, F82 Metallic GGX type 12, dielectric GGX type 12, then Diffuse type 2; compensated Metallic weight `(0.4725275,0.4821480,0.4927247)` and sample weight `0.2328374`; lower Diffuse `(0.04782301,0.1461259,0.2178604)`; emission `(0.2145,0.06825,0.468)` precedes Metallic attenuation |
| Principled thick Transmission | Transparent type 30, generalized-Schlick GGX Glass type 25, dielectric GGX type 12, then Diffuse type 2; compensated Glass weight `(0.5829386,0.5830691,0.5831698)` and sample weight `0.4619407`; lower Diffuse `(0.07780470,0.1383195,0.1966730)`; emission `(0.19278,0.10206,0.57834)` precedes Transmission attenuation |
| Principled Thin Wall | Transparent type 30, GGX reflection type 12, Thin Glass Transmission type 22, dielectric GGX type 12, then Diffuse type 2; reflection weight `(0.02744345,0.01416836,0.05171960)` and sample weight `0.03107580`; transmission weight `(0.1949813,0.3594840,0.4788061)` and sample weight `0.3444238` |
| Principled Random Walk Skin | Transparent type 30, Random Walk Skin BSSRDF type 34, then Diffuse type 2; BSSRDF weight `(0.2291900,0.1119300,0.4050800)` and sample weight `0.7462000`; residual Diffuse `(0.1234100,0.0602700,0.2181200)` |
| Principled Burley | Transparent type 30, Christensen-Burley BSSRDF type 31, then Diffuse type 2; BSSRDF weight `(0.1057920,0.2997440,0.1719120)` and sample weight `0.5774480`; residual Diffuse `(0.0766080,0.2170560,0.1244880)` |
| Standalone Microfiber Sheen | type 7; post-LTC weight `(0.0028833640,0.0058426056,0.0012140479)`; sample weight `0.0033133393`; normal `(0,0,1)` |
| Standalone Ashikhmin Velvet | type 16; weight `(0.82,0.19,0.57)`; sample weight `0.5266666412`; normal `(0,0,1)` |
| Standalone Diffuse Toon | type 8; weight `(0.31,0.73,0.19)`; sample weight `0.4100000262`; normal `(0,0,1)` |
| Standalone Glossy Toon | type 18; weight `(0.84,0.22,0.56)`; sample weight `0.5399999619`; normal `(0,0,1)` |
| Standalone default Ray Portal | type 29; weight `(0.26,0.71,0.43)`; sample weight `0.4666666687`; normal `(0,0,1)` |
| Standalone authored Ray Portal | type 29; weight `(0.83,0.17,0.52)`; sample weight `0.5066666603`; normal `(0,0,1)` |
| Standalone Blender-default Hair Reflection | type 19; weight `(0.8,0.8,0.8)`; sample weight `0.8000000715`; normal `(0,0,1)` |
| Standalone authored Hair Transmission | type 23; weight `(0.83,0.17,0.52)`; sample weight `0.5066666603`; normal `(0,0,1)` |
| Standalone Random Walk | type 32; weight/albedo `(0.37,0.62,0.14)`; sample weight `1.1299999952`; normal `(0,0,1)` |
| Standalone Burley | type 31; weight/albedo `(0.21,0.74,0.48)`; sample weight `1.4299999475`; normal `(0,0,1)` |

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

For Metallic word images, the first command uses
`metallic_svm_oracle`. This oracle-only builder starts from the same finite
product as `metallic_bsdf_matrix`, literalizes its constant Value/Combine XYZ
links, and lets Cycles construct the default Normal/Tangent topology. It is
intentionally excluded from the canonical end-to-end probe runner until the
exact interpreter replaces production shade-surface.

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
| Glossy matrix `.blend` | `0cc6653d692a5f2df7c91956e064d245eb46dbc57083aea180fca0b5de628eec` |
| Glossy matrix final SVM buffer | `4c7f07490b6daa2ba9cb46d60b534176aa8f3465e564a6fe00a1a9d808efd6e0` |
| Glossy GGX path trace / decoded trace | `7cd5edbf861488200cacc1db4dcc59024272264bab8cb56bae3533b02872947f` / `c81e19700b547961cfa642bc0acbc076b59006484095acbf352e268f1b024105` |
| Glossy Beckmann path trace / decoded trace | `5c301aab4961aa78118739910057b431df034f29fb1ec3ae10d505cec0c6c93b` / `85f7d9c41b51945d71687d08554cee4a7d2b02cb304a3a6c0e00f2e39143ecc8` |
| Glossy Multi-GGX path trace / decoded trace | `8b09eec96fdb9a0707ea4d77925691f4312e37c1e8b5e104a5703ea329ca8b83` / `4e94e4a8eaca97b6a9e573928fe8e72bf83c11595bdfc0b838699c8a76da9da5` |
| Glossy Ashikhmin `.blend` / final SVM buffer | `5f1baff0351d595f72ef3ff2953fc412130f5930fd8f851261b20e2c9f6e96f2` / `e1a17a3b144229b677cb7fbaa758a8bc894906825157f5e1bd1f2d1012bec284` |
| Glossy Ashikhmin path trace / decoded trace | `2140808e3aff333ab03669e35e8b7aed8a6e929475c62bbc4569a9387638c700` / `b545626a5e535c8c21d1b1ea349a0d01aab91d184323c1ad78db056fa30e8780` |
| Glossy anisotropic `.blend` / final SVM buffer | `f9ea3e1f6699aa674655f85bc8c6f27cc554fd5187afac2b7c5b5f49fc64daa4` / `c9a6464c525e33efafee83ee5ef3cb8cdc3975c38c20219b38feb91c375e48f3` |
| Glossy anisotropic path trace / decoded trace | `a8f768cd84ea5bb910b2302f48aef2c0fe685f7cfc73bba6bf9096e53f4bd4b2` / `e95b16a529517ee99376073bc81e6ac3cf76c7a8f37a01c0be5d1618460631b8` |
| Refraction matrix `.blend` | `06478956fd3fcc24e318ffac76f818e23958bc9694451dc0970590f046c3a558` |
| Refraction matrix final SVM buffer | `57c302933acacc1b878993f2921542b12f31e40e42e413401047168e4509a63f` |
| Refraction Beckmann path trace / decoded trace | `fdaba6513cbc5ef60ee7509acd0a2f0a3d1a2f3346976b865c7d71e35d603a30` / `2dd3382831e4fccfebd81401f9896698fe0dadea7ad5706b1f47ab0911f3fdd4` |
| Refraction GGX path trace / decoded trace | `76a5eff91cda12cbac14aba4b4aa87a7307120e752d6f24a670155b3b85e132a` / `4da0653bcacb9aee9784d0a2102b0832b474fbeb32c5f41f84f6b831c9404f89` |
| Refraction back-face path trace / decoded trace | `dd67ccc2d0b0d34f0955981ac60b946ebc563bb2de936e3b6558b20bc83da8b5` / `c2cd1f894f0e24dc0bbd13abbee4b9cc5ac6c44652b9adb1a89b364272a22038` |
| Metallic linked matrix `.blend` / final SVM buffer | `c88d512e08e49a75d04d1f0732bc42e2e2d18ab826db7a758ff199ac0634c0b8` / `ca569e6e0fdf31b04109689ef8b3fb70a6655d464348f595de334cc66d637cfd` |
| Metallic literal oracle `.blend` / final SVM buffer | `e8895480b3f3aceaa94caacf512a184c1c143b8a948fa63140fdb5842dc6f675` / `b595c9b6b95702cde26185a2ff15c1572fd61923334df9a49d0e6cfe87f3eace` |
| Metallic F82 GGX path trace / decoded trace | `c91f6d42b67c09408426d10b5820a9b9a6b24c8179a4508052ae9b0cd13adb32` / `678a07a245ca29c414e14b6711b13e97ddf93203cdea506ea6c078e024bb6897` |
| Metallic F82 Beckmann decoded trace | `c0db0165550b2b98b1f51d9b8fa0df9ec8296f653cc34332ac3a530743584c78` |
| Metallic F82 Multi-GGX decoded trace | `a30e50987645645bc6b60cf4f67908ef01591e7064d532bbf154891d021d151f` |
| Metallic conductor GGX decoded trace | `89e4d69659388bf7a9bd1a9be5e4b52895236e39aeab211475a8ff0830cd2abd` |
| Metallic conductor Beckmann decoded trace | `27b34532f6e5ef01717579d0bf918a031cdc317fddde654813e4600f92065780` |
| Metallic conductor Multi-GGX decoded trace | `e7acdf099184f9323260819c40b36714293a87de224878d76e2da8b16cb004b5` |
| Principled oracle `.blend` | `9e747eeffac3fad668d67b6b78ed9cfcdd43549b0c174d2eae9c5f4e4a6ad7cd` |
| Principled final SVM buffer | `7a74aff57508b422f0a7c19e6eb9ab20fb9209a8ba217cf29571b91bb65cc4d1` |
| Principled diagnostic path trace | `752a6e150adca4cb128a76d59093d955125decfd5d46c24c0b6caeca7bcb9bd3` |
| Principled decoded diagnostic trace | `ca5ffcbce0488aa37b6ee9a96ea978161a29b921bcabf763f7a01ec8d4ed0bc5` |
| Principled Sheen oracle `.blend` | `05676f9210494dba18680025f26ac2e874bb762e0554f73a1ed7073885aca074` |
| Principled Sheen final SVM buffer | `1fa274d1c7b8a7610d9d8c5e5d5edbe0d1bd89ffcf6c9237534e3abc3297d777` |
| Principled Sheen diagnostic path trace | `7dfbbe58edf8c51f060df3fe6e3d0adeefad88e11b8f00cd23d2e28b42a262b9` |
| Principled Sheen decoded diagnostic trace | `2eaec51b67a38f6a9a09dc368f610b7ea32a7dde655525b0860cea9bdc7b9a33` |
| Principled Coat oracle `.blend` | `97bce9ee4a90886d51a9b0f937a5d5d95cf60f51e68bbef4b71452bac335e414` |
| Principled Coat final SVM buffer | `6865e185e8570a3921a9ebf845b6cfe4dff6e2f322997046ac06e0fbb320cfdd` |
| Principled Coat diagnostic path trace | `ebb438bef6583d5d7e4b45ed8aefac36f40aca831a232cbc38037b51cf5cbd4d` |
| Principled Coat decoded diagnostic trace | `6f64ca4347a22fc9adc24c0fcce52548f9897ce2559eed285b035842e465a319` |
| Principled Metallic oracle `.blend` | `171a02bd4a95fc09e1247a7950d0e71f016b6b053e672a82f5177b223d17fd86` |
| Principled Metallic final SVM buffer | `d4b02bc187d9e54099022f2b3fa8335224ede707a46cdf079d2eccd29d6dceaa` |
| Principled Metallic diagnostic path trace | `f68535b1f6d03500caf536f8dfbb92ee679a4b7c328cd05e0f8f17434a8aa204` |
| Principled Metallic decoded diagnostic trace | `e32b254552aa84462f7a1344ccc6025604e7a85d764302ad75d21cdbc0bfe0f4` |
| Principled thick Transmission oracle `.blend` | `b77342ce7398c25c171be6d1fd17d7665bc212dc3b6662dd6aa2b7d5f0bf2587` |
| Principled thick Transmission final SVM buffer | `3f5712896bde0a228a8782ac81993f86b119a2f85c2160e3c729e0b413d8d38f` |
| Principled thick Transmission diagnostic path trace | `792dac665596d76ce967441b568bf07a110c4902114a9c08ae77dfe0640807fa` |
| Principled thick Transmission decoded diagnostic trace | `4787279961ebab12324d2926f63db3bce2679f6a3c904403732417387fbf7ea8` |
| Principled Thin Wall oracle `.blend` | `069e3ebf3c1574e0b8d36de0322d181592673c977014a2f5f48f4ff8dc184d54` |
| Principled Thin Wall final SVM buffer | `b805579d5a8c24b8b9c9c1ad979433045fac3bbdd61a0ab74d235bd78f60172f` |
| Principled Thin Wall diagnostic path trace | `df299c20d4a2daa3e5745b816e6f52352e10e8a9b18923d2edafb6f7bf4ac5ca` |
| Principled Thin Wall decoded diagnostic trace | `ee03ef5ff6b160deb032e1ec84c9a8a038bacd79c02d3096fe6be28fbabc126e` |
| Principled Thin Wall render metadata | `76774bd3ff7caaf441bd6febd228a6f277d1eeebdb82c34d3ce9698f2fe43ee1` |
| Principled Random Walk Skin oracle `.blend` | `5c59c194dcde8448b7a101b7854a92501bc5ed6549b8b8cfe1dbaf8a53641dcc` |
| Principled Random Walk Skin final SVM buffer | `23a53385c0d7bcb3c4aff294b97a7bb937fa97af26c0e63dab50e6debec82837` |
| Principled Random Walk Skin diagnostic path trace | `ecc954ee2ed064b1fde7a35b3702caf67e5d766094e28da2c8a80b8180a6d7e7` |
| Principled Random Walk Skin decoded diagnostic trace | `44c3f499ec2a3f25b8810c136646dc6652dca289f071898230396104f34a1ef1` |
| Principled Random Walk Skin render metadata | `b6cacfb65817491062befc0eadbe2599649d2a31f0e9b162eee6f3bb75351ebf` |
| Principled Burley oracle `.blend` | `8a72b1bbd7491d55ff983d8b7ea24b9a90d4a9a420bd1277b7104b8e7630e934` |
| Principled Burley final SVM buffer | `ebdda594ae0fe8a2a6176e03bd9345921b6fd45b74e985c3ac22a3551b49ba0b` |
| Principled Burley diagnostic path trace | `394c8cd6972adc86f10ec7bc4cddf7e43cc38e462ba9e9d991f7185d6f3e1d0b` |
| Principled Burley decoded diagnostic trace | `7183e845f038fd23f481cc6d748276e4c9ab5cd3561874d2c8ebbad17c5cfc91` |
| Principled Burley render metadata | `40711f5aa1eb623685a7f6066761142db7404c13ca31830531953dd533206595` |
| Standalone Random Walk oracle `.blend` | `d01985cc46bc878c0b9e001a44cba4a3dab6e75b30de007d6ab84093437cfb91` |
| Standalone Random Walk final SVM buffer | `8b10065bebdce6182d72e4c7d797e6759f368d195ea5a6e60554f2619bee6a42` |
| Standalone Random Walk diagnostic path trace | `239b55acc9af1fb7759d5db727a7f98afc34d8f7c1c5f78387b4ad11ad7a0d3b` |
| Standalone Random Walk decoded diagnostic trace | `be3ffc1697fb0822584b134c39e4c3f9b74792c04e2aa595dae5a4349801f173` |
| Standalone Random Walk render metadata | `8a3fa778837e1d87b2db760c3deab59fe648121dbf3266546076020896dd2704` |
| Standalone Burley oracle `.blend` | `eb45a220276b6f019ad049a178670b90af46ad0679d7a7269af200e3c19cc997` |
| Standalone Burley final SVM buffer | `9a45ff56b808a715160b3ed6a6e870b3f7b06c553addb63f65112731fe6b6373` |
| Standalone Burley diagnostic path trace | `e5be5ef7958f24d287c957ced005d515c9329982b6c6a2c73e722ba40c591b48` |
| Standalone Burley decoded diagnostic trace | `f1358235826ce5555c09fa7872a1a3d7c9ba4b3f8eca6430374ce82ab937cdc7` |
| Standalone Burley render metadata | `22cfa882eb042b5f396d4181cd9441a2e11bfc39f8acf559dae7374a80548e3a` |
| Standalone Microfiber Sheen oracle `.blend` | `5211802094af3e3609d8f3517051ae1557a3eae64be1c9bbe9b85cb831d6cbd3` |
| Standalone Microfiber Sheen final SVM buffer | `bc20d02c75ead201d58035f3d4fc9f5f3ea15bd9c93cb42311fde41ae53dd863` |
| Standalone Microfiber Sheen diagnostic path trace | `836706080c2c76b3744868e39122361a6c4cb375f0c8582042e4f3b91cad33ef` |
| Standalone Microfiber Sheen decoded diagnostic trace | `6baae2943d1c853190bfb36521b108db8aca0b4c2e513037906eea45d3799010` |
| Standalone Microfiber Sheen render metadata | `97e8d13bf4afe6cd78eaed6e60cb18726b18f1a02a55b7f5edfa2bd31de3e7f9` |
| Standalone Ashikhmin Velvet oracle `.blend` | `e54799eafeb51b8c45a940afe2383701ddd66378690ef56f3e2b28cd56b38784` |
| Standalone Ashikhmin Velvet final SVM buffer | `25361469daf2c3f17db1f39a564f7c7b3361f3db989f84159b4e44a4452cd7a9` |
| Standalone Ashikhmin Velvet diagnostic path trace | `bb5f21ec48f666f61d83e9fd171476c2d9f5d28c71e0fa97dcfadcf466b0642d` |
| Standalone Ashikhmin Velvet decoded diagnostic trace | `df318c40fc9432170ff6b7e6cbe375ad658253fbb1a626c84160acc16525ccba` |
| Standalone Ashikhmin Velvet render metadata | `256959ff1929c98c9daa1236810e96b6308d72d43d6259c5c7a8dddfefb748b6` |
| Standalone Diffuse Toon oracle `.blend` | `991ae02942f735a51643498930d17992a980490e53a1e397678734840aeb2629` |
| Standalone Diffuse Toon final SVM buffer | `17b251397887d3ca8972453aac65cffb4470cef9a273b05d878dc7737085820b` |
| Standalone Diffuse Toon diagnostic path trace | `1bdb2ef66feb552ab0cbf8b962d8eb98ba48c184a81d1d5cf58ab71e929a9125` |
| Standalone Diffuse Toon decoded diagnostic trace | `49e078d2eebbdbfb6846cb61c48ed45175a0cec3293576ddd620f64eaabbd854` |
| Standalone Diffuse Toon render metadata | `0476e304af997dead8661b2a919e2397e44e3324b9e393bd1cd48f1ac304675b` |
| Standalone Glossy Toon oracle `.blend` | `9a5f9e7fbc0dbc7ab8bbe4b8dc69679a3644ecfd3a74d4e6950b1ae3f108cd49` |
| Standalone Glossy Toon final SVM buffer | `64d42154d482ebe54a6ab541b77f83942b4fc4cc05da9ed517bd37c93d325612` |
| Standalone Glossy Toon diagnostic path trace | `5847772f912bf9a3e90ae0d0f16054c4c453f33fb0594d9f499fce37bab829e4` |
| Standalone Glossy Toon decoded diagnostic trace | `706a4c8fe7e12523126591cb362de409874726b28c507a311de9b2cff9583e0a` |
| Standalone Glossy Toon render metadata | `e898f5a75c2e95fbc9cefe8ad8b0f38a7e3c9b5155daab7675df4c808afcc8cb` |
| Standalone default Ray Portal oracle `.blend` | `ba85f13c06ed878546dce0ff24a87ca167fc96bb9c866b3e42b8fa5e32246376` |
| Standalone default Ray Portal final SVM buffer | `9ae8c8903a8043a69194d8523d82a194c11c2eb69c8e813b7b38394a1debfbbf` |
| Standalone default Ray Portal diagnostic path trace | `1c5590b2a77866416a087963cf0568fa4cddc6ef425415694042838daa722e89` |
| Standalone default Ray Portal decoded diagnostic trace | `3d80a40ac1eda9b93082e4555dd1a8b16a1eb3cba8e2eb7c4501ec90a737f246` |
| Standalone default Ray Portal render metadata | `38e47ba9ded190ae6e4cbe4d1b70bc8788cdbd0ad84545ad1ea0b5edada33fc7` |
| Standalone authored Ray Portal oracle `.blend` | `c332ddb666f4796daabe5b2371dff822296cad3d37dce8ad760ae10a3f8b0646` |
| Standalone authored Ray Portal final SVM buffer | `67c35805f377c567a794bd272f00229818cc917568576267093f356b096a6f49` |
| Standalone authored Ray Portal diagnostic path trace | `0ef35b984b64d4cbbf385b969a40182f740ea51d05b57fbaad2cfd5e8c3be4b1` |
| Standalone authored Ray Portal decoded diagnostic trace | `654b44de2155581a74d593a2a33483e21fcfb5fe6e54e6db58a58f5bae95a2c7` |
| Standalone authored Ray Portal render metadata | `7ba9b1e869805347da32b807d3c72822f87092d5fb52cc6755c18e0994e44379` |
| Standalone Blender-default Hair Reflection oracle `.blend` | `9314cdf798fedf9139a3097112cc31780713d195f9580d003e23fa2d6e54782e` |
| Standalone Blender-default Hair Reflection final SVM buffer | `8c313ad247feaf4160c5bbf17dba5f70c943095ff2595c4e6b4f2c1baa52ceed` |
| Standalone Blender-default Hair Reflection diagnostic path trace / decoded trace | `8b94b5b41e2cdb5dc602f916ea23b846e7bc3a3b06c1d270c6ccac663317c274` / `a954bd7ac49afcc1bf16720092490a502cd3d07461022d13430f45abe130d247` |
| Standalone Blender-default Hair Reflection render metadata | `afb35d32840a4741e314c192236fdbbc0f31a7e5a88b85fab985dcdc896d7c97` |
| Standalone authored Hair Transmission oracle `.blend` | `db2fbe44dbb2ef57f5725fcb80bfeb86163c036874d10e6d8390d4744ae6ab5e` |
| Standalone authored Hair Transmission final SVM buffer | `d7316f111a07eed781d5f25ffae2e85892c339c8f6b331287a2e9f6bcb681f29` |
| Standalone authored Hair Transmission diagnostic path trace / decoded trace | `f99fcedf325a75c033a8ff951562b83f3c522d4bf184d1b8b952621eff2f7d65` / `1d0f9a541a8cf6a90bdf5145657ccbf42ad224d4fbb6fab7d30f534e293ccd55` |
| Standalone authored Hair Transmission render metadata | `45ff54e1e90a3491e6525b23273838c8d93d1fe46699595257e5a76db7e874e7` |
| Hair Reflection direct-eval oracle `.blend` | `0a804d03e5e5c5da6b4a44feb2633b15b0d0595a31830d996a4c59f6fbe73eaf` |
| Hair Reflection direct-eval path trace / decoded trace | `f4d15f73fbc455694b12d49abc156263f9d0a307ed9767619f39fa8af3a00491` / `8ebd5fe4224892727b110245e45154c8520790d1fc71b96941f9c467dd1f1c25` |
| Hair Reflection direct-eval render metadata | `cb92b0d4977bae3614ddd33e35a7ba121a007db7ae89c4909cbcb42456eecd73` |
| Hair Transmission direct-eval oracle `.blend` | `45a1391c63b29f6952158c8cf7c81d8eb354f3665cdb24d432388f1284cfd737` |
| Hair Transmission direct-eval path trace / decoded trace | `3bb87fc276ffb5e139bcb1082a08169c50f483ee1218e0ff5b8e877057d81639` / `f63cde7cdea31657f0d8d69b245a7c5cd33498cd479cf232f9638464f41836c8` |
| Hair Transmission direct-eval render metadata | `4224bcd7c36e78c437ece942cf8e30c600cb6a382c50f1d32dd12c71895b1c32` |
| Direct Cycles simple-BSDF HIP oracle source / executable | `37652166c999e30d4eb28fb34f6dfb4d5f16953859b4f72bc56e982d25aef22b` / `48a54c90bf8ba2769f17342d30b390e0214cd6bebe995a9fe6372b2fe848c16d` |

No `.svm52` binary is checked in.

### Standalone Sheen and Velvet runtime checkpoint

Both Cycles variants consume the same two-word `SVMNodeSimpleBsdfData`
record:

```text
param1, normal_offset
```

The handler always advances both words, loads the optional normal with
`stack_load_float3_default`, applies Cycles' exact-zero
`safe_normalize_fallback`, and calls `bsdf_alloc` with
`closure_weight * mix_weight`. It does not apply the specular bump-normal
correction used by Microfacet families.

Let `w = max(closure_weight * mix_weight, 0)` and
`s = abs(average(w))`. Allocation failure leaves the closure pool and shader
flags unchanged. After successful allocation, the two statically selected
transitions are the following disjoint sum:

```text
Microfiber:
  r0 = saturate(param1)
  r = clamp(r0, 1e-3, 1)
  (T, B) = make_orthonormals_safe_tangent(N, wi)
  (A, B_ltc, albedo) = three Cycles Sheen LTC table reads
  if abs(A) < 1e-5 or albedo < 1e-5:
    retain allocated input weight and typed payload
    type = NONE; sample_weight = 0; add no flags
  else:
    type = SHEEN
    weight = w * albedo; sample_weight = s * albedo
    flags |= SD_BSDF | SD_BSDF_HAS_EVAL

Ashikhmin Velvet:
  sigma = saturate(param1)
  invsigma2 = 1 / max(sigma, 0.01)^2
  type = ASHIKHMIN_VELVET
  weight = w; sample_weight = s
  flags |= SD_BSDF | SD_BSDF_HAS_EVAL
```

These branches share only cursor decoding, normal normalization, and prefix
allocation. The LTC invalid branch is not normalized into allocation failure:
it consumes one slot exactly as Cycles does. The permanent regression freezes
the two external 104-word global buffers as compact 19-word images, their
externally observed common closure states, the typed payloads, a real-table
invalid endpoint, an all-zero malformed table, a legal non-unit mix-weight
stream, and zero-capacity allocation. It passes on fallback, HIP, and strict
native Vulkan XIR-to-SPIR-V. The compiler regression also corrected and locks
Cycles 5.2.1's standalone Sheen roughness default of `1.0`; the previous
Psycles schema incorrectly used `0.5`.

### Standalone Toon runtime checkpoint

Diffuse and Glossy Toon share Cycles' three-word `SVMNodeToonBsdfData`
record:

```text
size, smooth, normal_offset
```

The handler consumes all three words before testing the optional caustics
gate. Let `G` mean Glossy Toon, `R` be the integrator's
`caustics_reflective`, and `D` mean that the current ray visibility contains
`PATH_RAY_VISIBILITY_DIFFUSE`. Cycles reaches normal loading and allocation
exactly when

```text
active = not G or R or not D
```

Thus only a Glossy Toon reached through a diffuse-visible ray with reflective
caustics disabled is skipped. The skip transition advances the program
counter over the entire typed record while leaving the closure pool and
shader flags unchanged. Diffuse Toon is not subject to this gate.

For an active transition, the optional normal is loaded and safe-normalized,
then `bsdf_alloc` receives `closure_weight * mix_weight`. Parameter loads and
setup occur only after successful prefix allocation. The two tags share the
following exact canonicalization:

```text
size'   = clamp(size, 1e-5, 1) * pi/2
smooth' = saturate(smooth) * pi/2
flags  |= SD_BSDF | SD_BSDF_HAS_EVAL
```

The state relation is therefore a disjoint product of component tag, caustics
gate, and allocation result; no case-specific recovery path exists. The
permanent regression covers both external word images and observed closure
states, both clamp endpoints, non-unit mix weight, an authored non-unit
normal, all three caustics partitions, zero capacity, and final cursor
positions. The compiler oracle separately freezes the exact compact 20-word
images and the default Diffuse component. Blender-side export checks prove
that the raw `ShaderNodeBsdfToon`, its Component property, and all four inputs
are preserved without pre-baking. The transition passes on fallback, HIP, and
strict native Vulkan XIR-to-SPIR-V.

This checkpoint copies setup state only. Exact Toon evaluation and sampling
remain part of the later closure-evaluation stage; no production-render
parity or performance claim is made here.

### Standalone Ray Portal runtime checkpoint

Ray Portal consumes the exact four-word `SVMNodeRayPortalBsdfData` record:

```text
direction[3], position_offset
```

Cursor advancement is unconditional. Let `w = closure_weight * mix_weight`,
`s = abs(average(w))`, `E` be transparent extinction, and `A` be the prefix
allocator result. Cycles' transition is exactly

```text
not (s >= 1e-5): identity
s >= 1e-5:        E' = E + w; A = closure_alloc(type 29, w)
A failed:         retain E'; do not set portal flags
A succeeded:      sample_weight = s; N = sd.N; P = input(Position, sd.P)
                  D = safe_normalize(input(Direction) == 0 ? -sd.wi
                                                            : input(Direction))
                  flags |= SD_BSDF | SD_RAY_PORTAL
```

The comparison form deliberately rejects NaN. Unlike ordinary BSDF setup,
Ray Portal calls `closure_alloc` directly: it neither clamps negative weight
components nor rolls back extinction when the pool is full. This makes the
four branches above a complete partition and prevents the implementation from
being expressed through the superficially similar `bsdf_alloc` helper.

The compiler regressions freeze the exact 21-word default and 23-word
authored compact images from Cycles 5.2.1. The default stream contains
Cycles' implicit Geometry Position node; the authored stream contains the
linked constant Position and immediate non-unit Direction. The runtime
regression additionally covers non-unit mix weight, signed spectra, cutoff,
NaN, zero capacity, feature-erased zero mix, pool accounting, flags, typed
`P/D/N`, extinction, and final cursor positions. Blender export checks prove
that the raw `ShaderNodeBsdfRayPortal` and its Color, Position, Direction, and
BSDF sockets survive without pre-baking. A separate bundle-import regression
takes the authored Blender JSON through raw lowering, core graph validation,
and SVM compilation, then compares the resulting 23 words to the same Cycles
oracle; this closes the adapter-composition gap instead of testing its layers
only in isolation. It passes on fallback, HIP, and strict native Vulkan
XIR-to-SPIR-V.

The external diagnostic trace exposes type, common weight, sample weight, and
normal. Its current record does not expose the internal portal `P/D` payload,
so those two fields are locked by the exact Cycles word stream, the pinned
source transition above, and the three-backend typed-state regression rather
than misrepresented as externally observed. Exact portal path integration is
a later stage; this checkpoint makes no production-render parity claim.

### Standalone Hair Reflection and Transmission runtime checkpoint

Both legacy Hair components consume the exact four-word
`SVMNodeHairBsdfData` record:

```text
roughness1, roughness2, offset, tangent_offset
```

The cursor advances over all four words before allocation. Let
`w = max(closure_weight * mix_weight, 0)`, `A = bsdf_alloc(w)`, and `curve`
mean `(sd.type & PRIMITIVE_CURVE) != 0`. The copied transition is

```text
A failed: no typed payload or flags
A succeeded:
  N = maybe_ensure_valid_specular_reflection(sd, sd.N)
  roughness1 = clamp(input(RoughnessU), 0.001, 1)
  roughness2 = clamp(input(RoughnessV), 0.001, 1)
  offset = -input(Offset)
  linked Tangent:    T = normalize(stack[Tangent])
  unlinked triangle: T = normalize(sd.dPdv); offset = 0
  unlinked curve:    T = normalize(sd.dPdu)
  Reflection:   type = 19; flags |= SD_BSDF | SD_BSDF_HAS_EVAL
  Transmission: type = 23; flags |= SD_BSDF | SD_BSDF_HAS_EVAL |
                                  SD_BSDF_HAS_TRANSMISSION
```

This is a total partition over allocation, Tangent linkage, primitive class,
and component tag. It uses Cycles' unchecked `normalize`, not a zero-safe
replacement, and ordinary `bsdf_alloc`, so negative spectral components are
clamped before cutoff and sample-weight calculation.

Two distinct defaults are intentionally preserved at their actual abstraction
boundaries. Cycles' internal `HairBsdfNode` schema declares
`RoughnessU = RoughnessV = 0.2`; Blender 5.2's UI
`ShaderNodeBsdfHair` declares `0.1/1.0`. The core registry follows the former,
while raw Blender export/import carries the latter unchanged. The compiler
regression freezes the internal-default compact image plus the external
18-word Blender-default Reflection and 23-word authored Transmission images.

That comparison exposed and fixed a structural graph bug: Psycles had marked
legacy Hair Tangent with Cycles' `LINK_TANGENT` flag. `default_inputs()` then
invented a `Geometry.Tangent` edge for both unlinked and explicitly linked
Hair nodes. Cycles 5.2 declares this socket as a plain `SOCKET_IN_VECTOR`; only
Principled, Glossy, and Metallic use the implicit Tangent flag. Removing Hair
from that flag restores `SVM_STACK_INVALID` for the unlinked stream and
preserves the authored `NODE_VALUE_V` producer for the linked stream. Both
forms are permanent exact-word regressions.

The device regression covers Reflection and Transmission, both roughness
clamps, linked non-unit Tangent, unlinked triangle and curve derivation,
signed input weight, non-unit and zero mix weight, cutoff, zero-capacity
allocation, feature-erased payload skipping, flags, prefix-pool accounting,
and final cursor positions. It passes on fallback, HIP, and strict native
Vulkan XIR-to-SPIR-V. The Blender export regression proves that the raw
`BSDF_HAIR` closure and sockets are not pre-baked, and the bundle test carries
the authored graph through raw lowering, validation, and exact SVM
compilation.

The external setup trace exposes Hair type, common weight, sample weight, and
normal; `T`, roughness, and signed offset are not fields in the diagnostic
trace. They are therefore claimed only from the pinned Cycles source
transition, exact word images, and typed three-backend regression.

The same checkpoint now includes direct Luisa projections of all four legacy
Hair scattering functions: Reflection/Transmission evaluation and sampling.
They use the exact Cycles 5.2 fast-acos, fast-atan2, fast-sincos, and safe-asin
arithmetic. The shared `cycles_fast_math` layer performs the polynomial and
range reduction in DSL rather than selecting unrelated backend-native
approximations; the paired sin/cos form performs one range reduction, as
Cycles does. Signed-zero atan2 quadrants, out-of-domain fast-acos clamping,
and the source predicate that maps a NaN fast-acos input to zero are permanent
regressions.

The formal support partition is preserved as written in Cycles:

```text
Reflection eval:
  dot(N, wo) < 0                                    -> zero
  pi/2 - abs(theta_i) < 0.001 or cos(phi_i) < 0    -> zero
  otherwise                                         -> density
Transmission eval:
  dot(N, wo) >= 0                                   -> zero
  pi/2 - abs(theta_i) < 0.001                       -> zero
  otherwise                                         -> density
Sampling:
  construct wo and both densities first
  pi/2 - abs(theta_i) < 0.001                       -> zero pdf
  otherwise                                         -> sampled pdf/value
```

The negated predicates are not rewritten as positive complements: that would
change Cycles' non-finite boundary. Reflection sampling also deliberately does
not reapply the evaluation hemisphere test. The external sample oracle in
fact produces `wo.z = -0.08913364` for a closure with `N = +Z`; the sample is
valid with PDF `0.03163304` and label 10, while independently evaluating that
direction returns zero. Transmission produces label 9, PDF `1.7582085`, and
weighted value `(1.4593130, 0.29889545, 0.91426837)`.

An additional Cycles CPU trace adds one zero-angle Sun to each unchanged
standalone material. Reflection evaluates direction `(0,0,1)` to weighted
value `(0.34622371,0.34622371,0.34622371)`. Transmission evaluates direction
`(0,8.7422777e-8,-1)` to `(0.0009033323,0.0001850199,0.0005659431)`. These
values independently cover the eval path; the test does not assume that a
sample PDF must equal re-evaluation of its generated direction. Cycles' fast
trigonometric maps are approximate and that false invariant differs by about
0.3% in the Transmission oracle. The three-backend regression passes on
fallback, HIP, and strict native Vulkan XIR-to-SPIR-V. Production
shade-surface replacement remains later work.

### Simple BSDF scattering checkpoint

The common scattering return types now project the output arguments of
Cycles' `bsdf_eval()` and `bsdf_sample()` directly: unweighted value, PDF,
outgoing direction, sampled roughness, eta, and the Cycles closure label.
Legacy Hair and the simple closure family share this one representation; no
second material or event model is introduced.

The copied simple family covers Diffuse, Translucent, Oren-Nayar, Rough
Translucent, and Transparent. Its support partition is preserved literally:

```text
Diffuse sample:          dot(Ng, wo) > 0  -> keep; otherwise zero
Translucent sample:      dot(Ng, wo) < 0  -> keep; otherwise zero
Oren-Nayar eval:         dot(N, wo) > 0   -> evaluate; otherwise zero
Oren-Nayar sample:       dot(Ng, wo) > 0  -> evaluate; otherwise zero
Rough Translucent:       Oren-Nayar with N=-N, Ng=-Ng, wi=reflect(wi,N)
Transparent sample:      wo=-wi, eval=pdf=1e6, roughness=0, eta=1
Transparent eval:        zero
```

The inequalities are not rewritten as complements, so zero and non-finite
boundaries retain Cycles' branch behavior. Oren-Nayar also keeps its
`b <= 0` diffuse limit and the exact `FLT_MIN` denominator guard in the
positive-correlation branch. Cosine-hemisphere sampling reuses the existing
Cycles concentric-disk map and Cycles-defined orthonormal-basis orientation.

An isolated HIP oracle included the pinned Cycles 5.2.1 headers and called
`bsdf_diffuse_*`, `bsdf_translucent_*`, `bsdf_oren_nayar_*`,
`bsdf_rough_translucent_*`, and `bsdf_transparent_*` directly. It did not
translate their formulas. The oracle used Cycles commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`, ROCm 7.2.53211, `gfx1201`,
`-O3 -ffast-math`, and produced the frozen 20-vector/6-label trace in
`test_luisa_cycles_svm_simple_scattering.cpp`. Representative outputs are:

```text
Diffuse sample:           wo=(0.765341878,-0.286503345,0.576339781), pdf=0.244151756, label=6
Oren-Nayar sample value:  (0.240954846,0.222976729,0.219377518), eta=1, label=6
Rough-translucent value:  (0.230091318,0.212113202,0.208513990), label=5
Transparent sample:       wo=(-0.349890679,0.149953157,-0.924711108), pdf=1e6, label=33
```

The production `enable_fast_math=true` test passes on fallback, HIP, and
strict native Vulkan XIR-to-SPIR-V. A deliberately reversed geometric normal
and a wrong-side Oren-Nayar direction permanently lock both rejection paths.

### Toon scattering checkpoint

Diffuse and Glossy Toon now project Cycles' four scattering functions over
the retained `ToonClosure` directly. The implementation reuses the shared
Cycles concentric-disk cone map; it does not substitute the more common
polar `(u, phi)` cone sampler. Let

```text
maximum = size
sample_angle = min(size + smooth, pi/2)
measure = one_minus_cos(sample_angle)
pdf = 1 / (2*pi*measure)
```

where `one_minus_cos(a)` is exactly Cycles' piecewise definition:

```text
a > 0.02: 1 - cos(a)
otherwise: 0.5 * a * a
```

The second-order branch is part of the sampling measure, not a numerical
oracle trick. Setup permits `size = 1e-5*pi/2`; evaluating `1-cos(size)` in
single precision can become zero and create an infinite PDF. The permanent
small-cone regression instead produces finite PDF `1.29006131e9`, value equal
to that PDF, and label 6.

The support partition is copied without complement rewrites:

```text
Diffuse eval:    dot(N,wo) >= 0 and angle < sample_angle -> density
Diffuse sample:  dot(Ng,wo) > 0                          -> label 6
Glossy eval:     dot(N,wi) > 0 and dot(N,wo) > 0 and
                 angle(reflect(N,wi),wo) < sample_angle  -> density
Glossy sample:   dot(N,wi) > 0, dot(Ng,wo) > 0, and
                 dot(N,wo) > 0                           -> label 10
otherwise:       value=pdf=0, label 0
```

The strict `< sample_angle` boundary makes a tangent direction evaluate to
zero when the cone is clamped to `pi/2`. The smooth shoulder retains Cycles'
piecewise linear intensity; the oracle case one quarter through the shoulder
has value `0.289572865` and PDF `0.386097133`.

The isolated oracle directly included pinned Cycles 5.2.1
`kernel/closure/bsdf_toon.h` and called its setup/eval/sample functions in a
HIP kernel. It used ROCm 7.2.53211, `gfx1201`, `-O3 -ffast-math`; the source
SHA-256 was
`46746ad177ec4cb234d2a90a1473d30321b3f6ab1480e0f64bd2ca2e528f2d8c`
and the executable SHA-256 was
`38f944cce7cd0f3a30e9cf836d78c7556cdee808f5a14b601d7908ad1a7b1b40`.
Representative outputs are:

```text
Diffuse sample: wo=(0.682372749,-0.296361089,0.668234646), pdf=0.411154330, label=6
Glossy sample:  wo=(0.677203298,-0.421548009,0.603069663), pdf=0.246169761, label=10
Wrong Ng:       value=pdf=0, label=0
Backside wi:    wo=value=pdf=0, label=0
```

`test_luisa_cycles_svm_toon_scattering.cpp` freezes the oracle values, labels,
eta/roughness outputs, smooth shoulder, tangent boundary, both rejection
families, and small-cone measure on fallback, HIP, and strict native Vulkan
XIR-to-SPIR-V. Ray Portal remains a disjoint integrator transition exactly as
in Cycles: generic BSDF evaluation is zero and generic BSDF sampling is
unreachable, so this checkpoint does not invent a portal sample function.

### Microfiber Sheen and Ashikhmin Velvet scattering checkpoint

The standalone Sheen setup had already copied Cycles' LTC-table lookup,
incoming-aligned tangent basis, invalid-table transition, and albedo fold. This
checkpoint adds the four scattering functions directly over the retained
`SheenClosure` and `VelvetClosure`; it does not route either closure through
the older graph-surface approximation.

For Sheen, let `o = (dot(wo,T), dot(wo,B), dot(wo,N))`, `a = transformA`, and
`b = transformB`. Evaluation now uses exactly

```text
L = (a*o.x + b*o.z)^2 + (a*o.y)^2 + o.z^2
value = pdf = (1/pi) * max(o.z, 0) * (a/L)^2
```

Sampling first applies Cycles' concentric square-to-disk map, then forms
`normalize(disk.x - diskZ*b, disk.y, diskZ*a)` and transforms it by the stored
`(T,B,N)` basis. The geometric-normal rejection is intentionally asymmetric:

```text
dot(Ng,wo) <= 0: value=pdf=0, label=REFLECT|DIFFUSE
otherwise:       value=pdf=LTC density, label=REFLECT|DIFFUSE
```

In particular, the implementation retains the negated `<= 0` predicate rather
than replacing it with `> 0`; this preserves Cycles' non-finite boundary
semantics. The label difference from Diffuse Toon is also permanent regression
coverage, not a normalization into a common Psycles event model.

Velvet evaluation preserves Cycles' two-stage domain proof. No division,
exponential, or fourth-power denominator is evaluated until the corresponding
strict predicate has succeeded:

```text
cosNI > 0 and cosNO > 0
abs(cosNH) < 1 - 1e-5 and cosHI > 1e-5
```

Inside that domain it copies the Ashikhmin distribution and masking terms
literally, with PDF `1/(2*pi)`. Sampling uses Cycles' uniform-hemisphere map,
which is not a polar `(z,phi)` map: it lifts the concentric disk using
`z=1-r^2` and rescales `xy` by `sqrt(z+1)`. A rejected geometric normal or a
failed Velvet half-vector domain returns label 0, while a valid sample returns
label 6. Both wrappers retain Cycles' sampled roughness `(1,1)` and eta 1.

The isolated oracle directly included the pinned Cycles 5.2.1
`bsdf_sheen.h` and `bsdf_ashikhmin_velvet.h` headers and called their
eval/sample functions inside one HIP kernel. It used Cycles commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`, ROCm 7.2.53211, `gfx1201`,
`-O3 -ffast-math`; the source SHA-256 was
`475564afae2652a370c55cc997fd09e29dcbfc8642c8a227fe9bbc6ef1778f9a`
and executable SHA-256 was
`2310b7192a311a287fb408daf6e8b09d325ca6b1d4c4f89423fac77473269bd1`.
Representative outputs are:

```text
Sheen sample:  wo=(0.552439630,0.507043540,0.661602020), pdf=0.287902415, label=6
Sheen direct:  value=pdf=0.209007919
Sheen bad Ng:  value=pdf=0, label=6
Velvet sample: wo=(0.893368244,-0.254301727,0.370437086), pdf=0.159154937,
               value=0.163679332, label=6
Velvet direct: value=0.153165922, pdf=0.159154937
Velvet bad Ng: value=pdf=0, label=0
```

`test_luisa_cycles_svm_sheen_scattering.cpp` freezes the complete 18-vector,
5-label trace, both geometric-normal rejection contracts, Sheen backface and
tangent evaluation, Velvet wrong-side and singular-half-vector domains, the
uniform-hemisphere direction, and wrapper roughness/eta. It passes on fallback,
HIP, and strict native Vulkan XIR-to-SPIR-V.

### Microfacet VNDF branch-structure checkpoint

The GGX and Beckmann visible-normal samplers now preserve the control-flow
partition in Cycles 5.2.1 instead of constructing both sides and selecting the
result afterwards. This matters structurally: the normal-incidence Beckmann
case must not build or execute the general Newton/root-solve path, and its
temporary values must not extend the shader AST or live ranges of that case.
The exact partitions are:

```text
GGX projected_length_squared > 1e-7: construct the projected basis
otherwise:                              basis_x=(1,0,0), basis_y=(0,1,0)

Beckmann stretched_incoming.z >= 0.99999: sample the analytic fast branch
otherwise:                                 run the bounded root solve
```

Both distributions expose a local-space helper so the Microfacet scattering
core can reuse its already-computed local incoming vector rather than repeat
world-to-local dot products. The existing world-space entry points remain thin
adapters and transform the sampled normal back exactly once.

An isolated HIP oracle directly included the pinned Cycles 5.2.1 sampling
code at commit `9e2066aef7ef7e20c142ad7bd3303138a4304c93`. It used ROCm
7.2.53211, `gfx1201`, and `-O3 -ffast-math`; the source SHA-256 was
`3bc336ab0246cfe869eed9c87c956287091b9cfd83d5c4ec7cdffdf6a297241d`
and executable SHA-256 was
`f0386584bf49f6277941b80d5d32d7d864ff614d48da1d27ff69e9a3edea6034`.
The branch-sensitive local-space samples were:

```text
GGX near-normal:        (-0.0842477456, 0.310166359, 0.946942031)
Beckmann normal branch: ( 0.0674229935, 0.533709824, 0.842975616)
```

`test_luisa_cycles_sample_mapping.cpp` permanently covers both fast branches
as well as the prior GGX and oblique Beckmann samples. The test passes on
fallback, HIP, and strict native Vulkan XIR-to-SPIR-V. This checkpoint makes no
runtime speedup claim before a production-kernel benchmark; it removes a
proved source of unnecessary AST construction and register pressure.

### Microfacet scattering checkpoint

The exact interpreter now has the Cycles 5.2.1 Microfacet eval/sample core,
not a projection through the old graph-surface approximation. GGX and
Beckmann are host-specialized instances of one Luisa DSL definition, matching
the source templates: a retained closure never pays for a runtime distribution
switch. The runtime state machine remains Cycles' own partition:

```text
type                     lobes
GGX/Beckmann              reflection
GGX/Beckmann Refraction   transmission
GGX/Beckmann Glass        reflection + transmission
Thin Glass Transmission   mirrored GGX reflection, relabeled transmission
```

Within that partition the implementation copies the same singular threshold,
half-vector construction, isotropic/anisotropic D and Lambda functions,
visible-normal sampling, Fresnel-energy lobe choice, geometric/shading-normal
rejection, transmission Jacobian, MIS scale, sampled roughness, eta, and label
bits. GGX alone applies `energy_scale` to eval while leaving PDF unchanged.
Thin Glass retains its separate `1e6` transparent limit and mirrored rough
path.

Fresnel is a separate typed-union projection rather than a weak float payload.
The common SVM record handles only `NONE`, `DIELECTRIC`, and
`GENERALIZED_SCHLICK`; conductor and F82 records use their typed overloads.
`DIELECTRIC_TINT` is not produced by Cycles SVM and is therefore not invented
here. A tag/payload mismatch fails closed instead of reinterpreting foreign
payload rows as `NONE`. The dielectric TIR domain is also a real negative
control-flow branch, so rejected lanes do not construct the regular Fresnel
divisions.

The direct oracle included the pinned Cycles 5.2.1
`bsdf_microfacet.h` in one HIP kernel at commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. It used ROCm 7.2.53211,
`gfx1201`, and `-O3 -ffast-math`; source SHA-256 was
`bd615054a2c547c38f96784f7bcfb9f66419c1bcaa5a7886d5f5ce48cc3f838d`
and executable SHA-256 was
`53fbf3253ac331f3c87a1a4519a7ee3fd764926c2c23147ac0012dac71499408`.
Representative branch outputs are:

```text
anisotropic GGX:       pdf=0.486668080, value=0.544672608, label=10
anisotropic Beckmann:  pdf=value=0.340102851, label=10
GGX Glass refraction:  pdf=25.3747864, value=27.6519051, eta=1.45, label=9
Conductor GGX:         pdf=0.322688937, value=(0.291490346,0.183629915,0.107035987)
Generalized refraction:pdf=19.1016998, value=(14.7117825,16.4667950,17.9401398)
Beckmann Refraction:   pdf=value=30.6961956, eta=1.33, label=9
singular GGX:          pdf=1e6, value=1.17e6, label=18; eval=(0,0)
singular Thin Glass:   pdf=value=1e6, label=17
```

`test_luisa_cycles_svm_microfacet_scattering.cpp` freezes all 42 numerical
slots and 13 labels, including reflection and transmission eval, pure
refraction masking, ordinary and singular events, all SVM Fresnel payload
families, wrong-side rejection, and Thin Glass. With production fast math it
passes on fallback, HIP, and strict native Vulkan XIR-to-SPIR-V.

### Ashikhmin-Shirley scattering checkpoint

The retained type-15 closure now uses a direct Luisa DSL transcription of
Cycles 5.2.1 `bsdf_ashikhmin_shirley_eval()` and
`bsdf_ashikhmin_shirley_sample()`. The transcription preserves the roughness
to exponent map, isotropic and anisotropic normalization, tangent basis, the
four mutually exclusive azimuth quadrants, original-paper pump, half-vector
reflection, shading/geometric-normal domains, `1e-4` singular boundary, MIS
scale, roughness output, and label bits. It also preserves two less-obvious
contracts: the sample rejects through `!(NdotI > 0)` so non-finite input does
not enter the positive domain, and the enclosing Cycles `bsdf_sample()` writes
`eta = 1` even when the closure sampler rejects the event.

The isolated HIP oracle directly included pinned Cycles 5.2.1
`bsdf_ashikhmin_shirley.h` at commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. It used ROCm 7.2.53211,
`gfx1201`, and `-O3 -ffast-math`; source SHA-256 was
`47312e59bc61be72f143e924f1e10ffee6ef8154709c4dba2871258143ce522e`
and executable SHA-256 was
`c971e7d10a923893d2c4c5b5b04cb5b040b23bad2fdbea304deef0b5367c031a`.
Representative results are:

```text
isotropic sample: pdf=0.414520204, value=0.393805861, label=10
isotropic eval:   pdf=0.281307250, value=0.259996474
anisotropic eval: pdf=0.0801215917, value=0.0740518868
quadrant PDFs:    0.535278857, 0.354017824, 0.621847391, 0.462558091
bad incoming:     pdf=value=0, label=0
bad geometric N:  pdf=value=0, label=0
singular alpha:   pdf=value=1e6, label=18; eval=(0,0)
```

`test_luisa_cycles_svm_ashikhmin_shirley_scattering.cpp` freezes all 27
numeric records and eight labels, including all four anisotropic branches and
both rejection paths. With production fast math it passes on fallback, HIP,
and strict native Vulkan XIR-to-SPIR-V.

### Principled Hair Chiang population/setup checkpoint

The interpreter now copies the type-27 branch of Cycles 5.2.1
`svm_node_closure_bsdf()` and `bsdf_hair_chiang_setup()`. The transition first
reads the complete 26-word `SVMNodePrincipledHairBsdfData` record, so cutoff,
zero-mix, and exhausted-pool paths cannot leave the program counter inside the
payload. Its live computation is the following source-shaped state map:

```text
random = curve-random attribute when found, otherwise payload random
roughness_factor = 1 + 2 * (random - 0.5) * random_roughness
roughness = input_roughness * roughness_factor
radial_roughness = input_radial_roughness * roughness_factor
sigma = DirectAbsorption | PigmentConcentration | Reflectance | brown fallback
allocation = bsdf_alloc(max(closure_weight * mix_weight, 0))
successful allocation -> Chiang setup -> SD_BSDF | HAS_EVAL | HAS_TRANSMISSION
```

Pigment concentration preserves Cycles' randomized melanin transform,
`-log(max(1-melanin, 0.0001))`, eumelanin/pheomelanin split, optional tint,
and concentration constants. Reflectance preserves the fifth-order albedo
roughness scale and squared logarithmic absorption. Setup preserves the
`[0.001,1]` clamps, the exact `pow20`/`pow22` multiplication trees, coat-to-M0
mapping, curve-tangent frame, ribbon `h=-v` branch, signed cuticle tilt, IOR,
and type-27 runtime tag. The shared closure pool stores this as a typed member
of the existing SoA `ShaderClosure` union; it does not introduce a second
material representation or additional per-closure rows.

The isolated HIP oracle directly included pinned Cycles 5.2.1
`bsdf_principled_hair_chiang.h` at commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. It used ROCm 7.2.53211,
`gfx1201`, and `-O3 -ffast-math`; source SHA-256 was
`18fe3da05bcf69f6ad6d3e5969372548d07573e49c3cc0cf8952d779e9249a96`
and executable SHA-256 was
`954a001845586011518bb6955483322f4c722d73dc817154cd2cb9258e0356a9`.
Representative post-setup records are:

```text
Reflectance sigma/v: (0.118423782, 0.0162182655, 0.00126328191, 0.224562809)
Reflectance s/alpha/eta/M0: (0.0875470489, -0.189999998, 1.54999995, 0.0959622115)
Pigment sigma/v: (0.129874706, 0.234955996, 0.526990354, 0.0659941882)
Pigment ribbon N/h: (0.762866139, -0.572149634, -0.301131397, -0.270000011)
Direct absorption sigma/v: (0.2, 0.7, 1.1, 5.28255782e-7)
Invalid-param brown sigma/v: (0.276265055, 0.590385675, 1.54966176, 0.107588485)
flags for every successful setup: 1036
```

`test_luisa_cycles_svm_principled_hair_chiang.cpp` freezes all four
parametrization partitions, attribute-found and fallback random selection,
thick and ribbon frames, roughness/coat clamps, allocation cutoff, an empty
pool, zero mix, feature-erased skip, runtime flags, pool accounting, and final
PC. It passes on fallback, HIP, and strict native Vulkan XIR-to-SPIR-V.

### Principled Hair Huang population/setup checkpoint

The same source-shaped handler now copies the type-28 branch of Cycles 5.2.1
`svm_node_closure_bsdf()` and `bsdf_hair_huang_setup()`. Chiang and Huang share
the complete 26-word read, curve-random selection, randomized longitudinal
roughness, and absorption parametrizations. The runtime type selects only the
model-specific radial roughness and setup after that shared prefix; no second
node decoder or alternative material representation exists.

Huang allocation is modeled as the following state transition, where `C` is
the ordinary closure count and `L` is `num_closure_left`:

```text
R <= 0 && TT <= 0 && TRT <= 0: (C, L) unchanged
ordinary allocation failure:   (C, L) unchanged
extra allocation failure:      (C + 1, L - 1) -> (C, L)
successful Huang allocation:   (C, L) -> (C + 1, L - 2)
outside projected radius:      (C + 1, L - 2) -> (C, L)
                                then Cycles transparent setup
```

The extra slot is associated with the ordinary closure through typed SoA rows;
rollback changes only prefix counters and never moves closure memory. The
stored state preserves non-negative R/TT/TRT modulation, pixel coverage,
`Y/Z/N` frame, local incoming direction, projected radius, eccentricity,
roughness, signed tilt, IOR, aspect ratio, absorption, and `h`.

For camera curves, pixel coverage uses the exact Cycles table expression

```text
k0 = curves[prim].first_key + PRIMITIVE_UNPACK_SEGMENT(type)
radius = mix(curve_keys[objects[object].position_offset + k0].w,
             curve_keys[objects[object].position_offset + k0 + 1].w, u)
pixel_coverage = 0.5 * dP / radius
```

`KernelGlobals` therefore exposes the raw `object.position_offset` and
`curve_keys[]` services. It deliberately does not substitute the existing
world-space `curve_thickness` Info service, which has different units and
semantics. Elliptical setup preserves the curve-normal attribute, major/minor
axis swap, exact `h=-v` ribbon partition, orthonormal fallback, local incident
direction, and projected-radius test. An outside hit restores both allocated
slots and calls the existing Cycles transparent transition, including
terminate-path capacity handling and transparent extinction.

The isolated HIP oracle directly included pinned Cycles 5.2.1
`bsdf_principled_hair_huang.h` and `closure/alloc.h` at commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. It used ROCm 7.2.53211,
`gfx1201`, and `-O3 -ffast-math`; source SHA-256 was
`202a708b9ca95f2ef7a51fcef2ff1ed4d045afebb9c3806d200995e9b416da04`
and executable SHA-256 was
`cb99e2b29a6f607a38f107e8db14712f727509a2820c1ab71664042629ae2584`.
Representative post-setup records are:

```text
circular N/h:       (0.794034362, -0.463883251, -0.392838061, 0.0675399899)
circular lobes/r:   (0, 0.699999988, 1.10000002, 1)
major-axis N/h:     (0.268715411, -0.329966754, 0.904938698, 0.0675399899)
major-axis a/e2:    (0.5, 0.75)
ribbon N/h:         (-0.70403403, 0.368139923, 0.607296586, -0.270000011)
ribbon radius:      0.976373792
transparent type/count/flags: (30, 1, 516)
successful Huang flags: 1036
```

`test_luisa_cycles_svm_principled_hair_huang.cpp` freezes circular, elliptical
major-axis, elliptical minor-axis/ribbon, camera and non-camera coverage,
source-shaped curve/object/key indexing, lobe-disable, cutoff, zero-mix,
feature erase, zero capacity, one-slot extra rollback, transparent fallback,
and terminate-path accounting. It passes on fallback, HIP, and strict native
Vulkan XIR-to-SPIR-V.

### Principled Hair Chiang scattering checkpoint

The type-27 closure now copies Cycles 5.2.1
`bsdf_hair_chiang_sample()`, `bsdf_hair_chiang_eval()`,
`bsdf_hair_chiang_albedo()`, and `bsdf_hair_chiang_blur()`. The projection
retains the trimmed logistic azimuthal distribution, finite Bessel
approximation, low/high-variance longitudinal branches, R/TT/TRT/TRRT+
attenuation and luminance-normalized sampling weights, cuticle tilt, Beer
attenuation, Fresnel term, local/global frame mapping, sampled label, and
filter-glossy state transition. The fixed three-lobe evaluation is a device
loop rather than three host-expanded copies.

The common Bessel and longitudinal implementation is shared with Huang in
`cycles_svm_principled_hair_math.{h,cpp}`. Its `argument > 12` branch does not
also record or execute the finite Bessel series, matching Cycles' control flow
and avoiding dead transcendental work. Sampling and evaluation deliberately
retain two different R variances: Cycles samples the R direction with `v`,
but evaluates its R density with `m0_roughness`. The first transcription
incorrectly shared the evaluation selector with sampling; the pure-R external
oracle scenario exposed the structural error immediately. The two selectors
are now separate functions, while TT, TRT, and residual scenarios lock their
own branches.

The isolated HIP oracle directly includes pinned Cycles 5.2.1
`bsdf_principled_hair_chiang.h` at commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`. It was compiled with ROCm
7.2.53211, `gfx1201`, and `-O3 -ffast-math`. The oracle source SHA-256 is
`8e0e89617ec5803845ed069b3ffb3cf6c2674cd4dae222af74534623378abe16` and
the executable SHA-256 is
`98c6e7b39e9dabef7ce788b853bb16618d3e19a08c88c2497dd5837a5fcd5ee3`.
Its normalized lobe weights are
`(0.0549212247, 0.928331614, 0.0164331775, 0.000313937053)`; the four random-z
cases therefore enter R, TT, TRT, and TRRT+ respectively. The regression
freezes 20 float4 records and four labels across sample, eval, albedo, blur,
sampled roughness, and eta.

`test_luisa_cycles_svm_principled_hair_chiang_scattering.cpp` passes on
fallback, HIP, and strict native Vulkan XIR-to-SPIR-V. HIP generated the
combined regression kernel in 39.47 ms and linked an 8,768-byte code object.

### Principled Hair Huang scattering checkpoint

The type-28 closure now copies Cycles 5.2.1
`bsdf_hair_huang_sample()`, `bsdf_hair_huang_eval()`,
`bsdf_hair_huang_albedo()`, and `bsdf_hair_huang_blur()`. The implementation
retains the R, TT, TRT, and TRRT+ construction, tilted elliptical
mesonormals, GGX VNDF samples, dielectric Fresnel terms, energy lookup-table
correction, Beer attenuation, geometric residual series, near-field visible
interval, and local/global frame conversion. Evaluation keeps the runtime
Composite Simpson and Monte-Carlo/Simpson loops. The number of integration
intervals is a device value; it is not expanded into the shader AST.

The RNG transition is part of the copied algorithm. Each two-dimensional
microfacet sample explicitly sequences its two LCG draws before constructing
the vector. This is required because C++ does not specify function-argument
evaluation order: the original shorthand
`make_float2(lcg_step(), lcg_step())` produced the same final state under both
host compilers while swapping the two sample dimensions between GCC's Luisa
AST construction and the HIP/Clang Cycles oracle. A final-state-only test
therefore could not detect the bug. The permanent regression isolates R, TT,
TRT, and TRRT+, and checks both the sampled directions/energies and every
control-dependent final RNG state.

The isolated oracle directly includes pinned Cycles 5.2.1
`bsdf_principled_hair_huang.h` at commit
`9e2066aef7ef7e20c142ad7bd3303138a4304c93`, with the GGX energy tables fixed
to one so the scattering algebra is independently visible. It was compiled
with ROCm 7.2.53211, `gfx1201`, and `-O3 -ffast-math`. The oracle source
SHA-256 is
`2dd4b4f417dc726001bce7b1691e43f165d5aaaecd5a41d6d5ea1fc111c63162` and
the executable SHA-256 is
`322daa1815a8c9bf480b8f937f8cbda66b1c94b1629f1c105cebb88a9d0bfa93`.
The eight scenarios freeze 40 float4 records and 24 integer records: mixed
near/far-field samples, pure-lobe samples, eval, albedo, blur, labels, sample
LCG state, and quadrature LCG state. Numerical checks use
`2e-6 + 4e-4 * abs(expected)`, so the small residual eval records cannot pass
by collapsing to zero.

`test_luisa_cycles_svm_principled_hair_huang_scattering.cpp` passes on
fallback, HIP, and strict native Vulkan XIR-to-SPIR-V. The HIP native LLVM
route generated a 29,248-byte linked code object for the combined regression
kernel.

### Standalone BSSRDF runtime checkpoint

`NODE_CLOSURE_BSDF` now consumes the exact eight-word
`SVMNodeBssrdfData` record:

```text
radius[3], scale, ior, anisotropy, roughness, normal_offset
```

The handler always advances all eight words. It safe-normalizes the optional
SVM normal, computes `max(radius * scale, 0)`, and passes two intentionally
different spectra into the shared setup transition: the common closure weight
is `closure_weight * mix_weight`, while BSSRDF albedo remains the original
`closure_weight`. This distinction is present in Cycles and is not folded into
a generic material parameter.

After the cutoff and prefix-allocation transition, let `C` be the set of
channels whose radius is at least `BSSRDF_MIN_RADIUS`. The copied state machine
has the following exhaustive partition:

```text
allocation failure: no closure, flag, or payload write
Burley with diffuse ancestor: C = empty; preserve NONE slot; append Diffuse
otherwise: move exactly channels outside C to Diffuse
C non-empty: set typed BSSRDF, sample_weight = abs(avg(weight_C)) * |C|
C empty: retain NONE slot with sample_weight = 0
```

Only the last two branches can change `SD_BSSRDF`, and only the non-empty one
sets it. Method-specific setup is also kept inside this typed transition:
Burley and Random Walk Legacy apply the compatibility radius scale, Random
Walk Skin applies Cycles' 12-step dipole inversion and forces roughness one,
ordinary Random Walk alone permits anisotropy through `0.99`, and all methods
clamp entry IOR to `[1.01, 3.8]`.

The permanent regression starts from the two external 110-word global Cycles
buffers and relocates only their shader jump to compact 25-word streams. It
then covers the two externally observed methods, source-derived Skin and
Legacy types, Burley's diffuse-ancestor branch, a negative-scale all-diffuse
branch, an authored non-unit normal, zero-capacity allocation, and a legal
non-unit mix input whose stack slot is disjoint from the external Geometry
node. The external cases end at PC 23 independently of allocation success;
the prepended mix stream ends at PC 26. The test uses numerical tolerance for
irrelevant backend ULP differences while freezing the word ABI, closure
types/order, typed payloads, flags, and pool accounting.

### Principled runtime checkpoint

The production SVM interpreter now consumes the complete typed 176-byte
`SVMNodePrincipledBsdfData` payload through named field offsets and advances
the PC by the exact Cycles struct size. Field reads remain lazy: the payload
view does not turn the typed record into 44 unconditional device loads.

The proved runtime subset preserves Cycles' layer order and implements alpha,
Sheen LTC, Coat dielectric GGX and tint absorption, emission, anisotropic F82
Metallic, thick- and thin-walled Principled Transmission, Random Walk Skin and
Christensen-Burley Subsurface, anisotropic dielectric specular, Multi-GGX
energy preservation, dielectric layer attenuation, and diffuse/Oren-Nayar.
None of these transitions consults or translates through the legacy
SurfaceProgram.

The Sheen transition copies `bsdf_alloc_maybe_emission` rather than treating
the layer as a color multiplier. On ordinary surface evaluation it appends a
typed `SheenBsdf`, stores Cycles' incoming-aligned LTC basis and table
coefficients, scales both closure and sample weights by LTC albedo, and then
attenuates lower layers. With `PATH_RAY_EMISSION` it computes the same setup in
temporary state and consumes no closure slot. An invalid LTC preserves the
already consumed `CLOSURE_NONE` slot, clears only sample weight, and leaves
lower-layer weight unchanged.

The Coat transition likewise copies `bsdf_alloc_maybe_emission`. It runs
Cycles' dielectric GGX albedo estimate and unconditional multiple-scattering
energy preservation before attenuating lower layers. Its colored medium is a
separate Beer-law transition: disabling reflective caustics skips allocation
and reflective layer attenuation, but deliberately keeps Coat tint absorption.
The regression freezes both branches. With `PATH_RAY_EMISSION`, the typed
Microfacet state is temporary and changes emission/flags without consuming a
closure slot.

The Metallic transition allocates Cycles' typed F82-tint Fresnel record after
the common Microfacet closure, uses the default Geometry Tangent through the
same anisotropic rotation, and keeps Multi-GGX as an energy-preservation
choice whose runtime closure tag is ordinary GGX. Its lower-layer attenuation
is outside the reflective-caustics branch. The regression therefore also
disables reflective caustics and proves that no Metallic or dielectric record
is allocated while Diffuse is still multiplied by `1 - metallic`; emission is
unchanged because Cycles evaluates it before Metallic.

The thick Transmission transition allocates one GGX Glass closure followed by
its generalized-Schlick extra record. It copies Cycles' independent reflective
and refractive caustics masks, reciprocal back-face bulk IOR and corresponding
thin-film IOR adjustment, square-root transmission tint, and Multi-GGX energy
preservation. The four-way caustics regression proves that either live lobe
keeps the shared Glass closure, that each mask zeros only its own tint, and
that disabling both lobes skips Glass allocation while still attenuating all
lower layers by `1 - transmission_weight`.

Thin Wall is a separate Cycles transition, not a different tag on the thick
Glass record. The copied implementation evaluates generalized-Schlick at the
front interface, evaluates the reversed film/IOR configuration at the back
interface when thin film is active, applies Beer-Lambert absorption, and sums
the infinite internal-reflection series component-wise. It then allocates an
ordinary GGX reflection and a type-22 mirrored-transmission closure. Both use
Cycles' unconditional constant-Fresnel GGX energy preservation; type 22 uses
the Kulla-Conty double-refraction roughness transform and normal `-N`.

The allocation partition is proved for all four caustics masks: both lobes
produce two closures, reflective-only produces only GGX reflection,
refractive-only produces only type 22, and neither produces neither while
lower layers remain attenuated. A separate source-derived zero-roughness word
image proves the non-camera singular branch: type 22 is not allocated and its
weight/sample weight/extinction merge into the pre-existing type-30 record.
The external word image differs from the paired thick oracle in exactly one
word, the literal Thin Wall field at compact word 52.

Thick Principled Subsurface now allocates the exact typed Cycles `Bssrdf`
state: common closure prefix followed by radius, albedo, anisotropy, entry IOR,
and alpha. Random Walk Skin copies the 12-step dipole inversion per spectrum
channel, uses the socket IOR, and forces entry roughness to one. Burley copies
the compatibility radius scale and the diffuse-ancestor policy. IOR and
anisotropy clamps, allocation cutoff, final sample-weight recomputation, flags,
and closure ordering are source-isomorphic.

The regression also proves the nontrivial setup partitions. A source-derived
one-small-channel stream moves only that channel's weight into a newly appended
Diffuse closure. Burley after a diffuse ancestor preserves the already
allocated `CLOSURE_NONE` record's original weight and unscaled radius, clears
only its sample weight/type, and appends the full BSSRDF weight as Diffuse.
Pool exhaustion cannot invent a BSSRDF flag. The `bssrdf_alloc` comparison is
kept as Cycles' negated `< cutoff` rejection, including its non-finite
allocation behavior; a structural regression prevents it from being rewritten
as `>= cutoff`.

Thin Wall remains a disjoint state machine. Live Subsurface is split by
anisotropy into reflection and transmission. Smooth inputs produce Diffuse and
Translucent; rough inputs share one copied Oren-Nayar parameter block and emit
Oren-Nayar plus type-4 Rough Translucent with normal `-N`. No BSSRDF tag or
`SD_BSSRDF` flag is present on that path.

For the untouched Blender 5.2.1 Principled default, the external diagnostic
trace records the following closure sequence at normal incidence. The new
device regression checks these values with numerical tolerance rather than
requiring irrelevant one-ULP identity:

| index | Cycles closure | weight | sample weight |
| --- | --- | --- | --- |
| 0 | Microfacet GGX | `(0.9220424294, 0.9220424294, 0.9220424294)` | `0.03738487884` |
| 1 | Diffuse | `(0.7700921297, 0.7700921297, 0.7700921297)` | `0.7700921893` |

The regression also locks the typed specular state (`alpha_x = alpha_y =
0.25`, `eta = 1.5`, generalized-Schlick `f0 = 0.04`, `f90 = 1`, exponent
`-1.5`), closure order/types, pool accounting, runtime flags, and final PC.
It passes on fallback, HIP, and strict native Vulkan XIR-to-SPIR-V.

## Unified Cycles BSDF consumer dispatch

The retained `ClosurePool` tagged union now has the direct Luisa projection of
Cycles 5.2.1 `kernel/closure/bsdf.h`. The copied consumer boundary is:

- `bsdf_get_specular_roughness_squared`
- `bsdf_sample`
- `bsdf_roughness_eta`
- `bsdf_label`
- `bsdf_eval`
- `bsdf_blur`
- `bsdf_albedo`
- `bump_shadowing_term` and the shading-normal shadow-terminator multiplier

This is one ClosureType dispatch over the same records populated by the SVM;
it does not translate closures into the legacy Psycles surface model. For a
Microfacet record, ClosureType selects GGX versus Beckmann and the independent
`MicrofacetFresnel` discriminator selects the exact tagged-union projection:

```text
dispatch(type, fresnel_tag):
  type in GGX family      -> microfacet<GGX>(payload(fresnel_tag))
  type in Beckmann family -> microfacet<Beckmann>(payload(fresnel_tag))
  type == Thin Glass      -> thin-glass transmission
  type == other closure   -> its exact Cycles family
```

The initial audit found a structural hole: the typed Conductor and F82-Tint
payloads had GGX entry points but no Beckmann entry points, even though Cycles'
`MicrofacetBsdf` union makes distribution and Fresnel model independent. The
fix adds both Beckmann typed projections; it does not special-case a material
or mutate the closure tag. The regression exercises both combinations through
the generic dispatch and compares them with their already oracle-frozen direct
family functions.

The albedo copy also found a domain-selection defect. Cycles selects
`ggx_gen_schlick_ior_s` only for negative generalized-Schlick exponents and
selects `ggx_gen_schlick_s` for non-negative exponents. A permanent sentinel
reader assigns distinct constants to the two table address ranges. With
`F0=(0.1,0.2,0.3)` and `F90=(0.9,0.8,0.7)`, it requires the positive-exponent
path to return `(0.3,0.35,0.4)` and the negative-exponent path to return
`(0.7,0.65,0.6)`. This test observes the selected Cycles table address and
cannot pass by comparing the dispatch with the same erroneous helper.

The outer modifier sequence is copied without normalization:

```text
sample transmit:
  add TRANSMIT_TRANSPARENT only below threshold and only when not diffuse
sample reflect:
  apply object shadow-terminator offset, then bump shadowing
eval:
  reject invalid smooth-normal hemisphere first
  evaluate closure, multiply bump term, then apply positive-cosine offset
label transmit:
  add TRANSMIT_TRANSPARENT below threshold, including diffuse labels
```

The test freezes the intentionally different `sample` and `label` transparent
rules, a nonzero raw Diffuse evaluation that bump correction rejects for
crossing the smooth surface, and the fact that the object offset changes only
sample value while direction, PDF, and roughness stay unchanged.

`ClosureTypeMask` is a host/JIT reachability set, not a device-side enum or a
second closure representation. If `R` is the scene-proven reachable set, the
recorded dispatcher is formally:

```text
D_R(type) = D(type), type in R
          = Cycles default, otherwise
```

The caller invariant is `stored_type in R`, so `D_R(stored_type) =
D(stored_type)`. The retained ClosureType and Fresnel tag remain runtime
values. The complete Cycles microfacet predicate includes the virtual
Multi-GGX authoring values 14 and 26 so this pruning commutes with the source
classification even though setup normalizes both before storage.

The focused test has a second shader whose ClosureType is deliberately dynamic
(`type_diffuse + dispatch_id.x`) while `R={Diffuse}`. This prevents whole-kernel
constant folding and measures the effectiveness of host reachability pruning:

| Backend artifact | Multi-family regression | Dynamic Diffuse-only mask |
| --- | ---: | ---: |
| HIP generated AMDGPU | 115,068 B | 4,976 B |
| HIP linked code object | 113,856 B | 3,904 B |
| native XIR SPIR-V before optimization | 175,656 words | 3,049 words |
| native XIR SPIR-V after optimization | 159,669 words | 1,519 words |

The HIP LLVM generation times in that uncached run were 188.77 ms and 30.24
ms respectively. These are focused code-generation measurements, not a
full-scene rendering performance claim.

## Backend validation

The exact frozen streams pass on fallback, HIP, and strict native Vulkan
XIR-to-SPIR-V:

```sh
cmake --build build --parallel 32 \
  --target psycles_luisa_cycles_svm_closure_tests \
           psycles_luisa_cycles_svm_principled_tests \
           psycles_luisa_cycles_svm_principled_sheen_tests \
           psycles_luisa_cycles_svm_principled_coat_tests \
           psycles_luisa_cycles_svm_principled_metallic_tests \
           psycles_luisa_cycles_svm_principled_transmission_tests \
           psycles_luisa_cycles_svm_principled_thin_wall_tests \
           psycles_luisa_cycles_svm_principled_subsurface_tests \
           psycles_luisa_cycles_svm_standalone_bssrdf_tests \
           psycles_luisa_cycles_svm_standalone_sheen_tests \
           psycles_luisa_cycles_svm_standalone_toon_tests \
           psycles_luisa_cycles_svm_standalone_ray_portal_tests \
           psycles_luisa_cycles_svm_standalone_hair_tests \
           psycles_luisa_cycles_svm_hair_scattering_tests \
           psycles_luisa_cycles_svm_simple_scattering_tests \
           psycles_luisa_cycles_svm_toon_scattering_tests \
           psycles_luisa_cycles_svm_sheen_scattering_tests \
           psycles_luisa_cycles_svm_microfacet_scattering_tests \
           psycles_luisa_cycles_svm_ashikhmin_shirley_scattering_tests \
           psycles_luisa_cycles_svm_bsdf_dispatch_tests \
           psycles_luisa_cycles_svm_principled_hair_chiang_tests \
           psycles_luisa_cycles_svm_principled_hair_chiang_scattering_tests \
           psycles_luisa_cycles_svm_principled_hair_huang_tests \
           psycles_luisa_cycles_svm_principled_hair_huang_scattering_tests \
           psycles_cycles_svm_compiler_tests \
           psycles_cycles_svm_ray_portal_compiler_tests \
           psycles_cycles_svm_hair_compiler_tests \
           psycles_blender_ray_portal_import_tests \
           psycles_blender_hair_svm_import_tests \
           psycles_luisa_thin_film_fresnel_tests \
           psycles_luisa_thin_film_surface_tests

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_closure_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_sheen_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_coat_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_metallic_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_transmission_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_thin_wall_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_subsurface_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_standalone_bssrdf_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_standalone_sheen_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_standalone_toon_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_standalone_ray_portal_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_standalone_hair_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_hair_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_simple_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_toon_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_sheen_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_microfacet_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_ashikhmin_shirley_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_bsdf_dispatch_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_hair_chiang_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_hair_chiang_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_hair_huang_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_cycles_svm_principled_hair_huang_scattering_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  'psycles\.luisa_thin_film_(fresnel|surface)_(fallback|hip|vk)$'

ctest --test-dir build --output-on-failure -R \
  '^psycles\.cycles_svm(_ray_portal|_hair)?_compiler$'

ctest --test-dir build --output-on-failure -R \
  '^psycles\.blender_(ray_portal|hair_svm)_import$'
```

Result: closure 3/3, Principled 3/3, Principled Sheen 3/3, Principled Coat
3/3, Principled Metallic 3/3, Principled thick Transmission 3/3, Principled
Thin Wall 3/3, Principled Subsurface 3/3, standalone BSSRDF 3/3, standalone
Sheen/Velvet 3/3, standalone Toon 3/3, standalone Ray Portal 3/3, thin-film
6/6, standalone Hair setup 3/3, Hair scattering 3/3, simple scattering 3/3,
Toon scattering 3/3, Sheen/Velvet scattering 3/3,
Microfacet scattering 3/3, Ashikhmin-Shirley scattering 3/3,
unified BSDF dispatch 3/3,
Principled Hair Chiang population/setup 3/3,
Principled Hair Chiang scattering 3/3,
Principled Hair Huang population/setup 3/3,
Principled Hair Huang scattering 3/3,
compiler 3/3, and Ray Portal/Hair bundle imports 2/2 passed. The scattering
kernels were built with the production `enable_fast_math=true` setting. The
compiler test locks the standalone graph-to-word-image mapping independently of the device
interpreter tests, including statically pruned Metallic Fresnel payloads and
Multi-GGX-only fields. After this checkpoint, `cmake --build build --parallel
32` completed successfully and the full 32-way CTest run passed 528/528 tests
in 18.96 seconds on the warm build tree. After the Huang scattering
checkpoint, the corresponding full run passed 531/531 tests in 16.15 seconds.
After the Chiang scattering checkpoint and shared hair-math extraction, the
full run passed 534/534 tests in 16.92 seconds.
After the unified BSDF dispatch checkpoint, the full build completed with
`--parallel 32`; the final warm-tree run passed 537/537 tests in 23.22
seconds after the last focused rerun.
The Vulkan test environment is
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

This checkpoint establishes ordinary and extra closure allocation, the copied
closure population/setup families, their direct scattering functions, and the
unified Cycles `bsdf.h` consumer dispatch inside the exact interpreter. The
subsequent `cycles-5.2-svm-surface-shader` checkpoint copies the collection
selection and one-sample-model fold above this dispatch. Production still
calls the old custom `SurfaceProgram` path. The next required structural step
is the scene-backed `KernelGlobals` adapter and explicit shade-surface route
switch, followed by deletion of the obsolete route after whole-program
equivalence is proved. The Cycles-style global shader linker is already
staged, but this document makes no production-render parity or full-scene
performance claim.
