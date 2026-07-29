# Cycles Filter Glossy alignment — 2026-07-29

This directory records the diagnosis, Luisa implementation, red/green probe,
and current Lone Monk result for Cycles' path-dependent Filter Glossy
contract. Cycles is the only rendering oracle. No CPU renderer, sampler,
BSDF mirror, material pre-bake, exposure fit, or image-space correction was
used.

## Root-cause isolation

The canonical `indirect_principled` probe contains mixed diffuse and glossy
Principled closures, a directly lit first surface, and subsequent scattering.
It explicitly sets Blender `scene.cycles.blur_glossy = 1.0`.

A bounce-depth sweep at 128×128/4096 spp separated the boundary:

- with no secondary scattering, Diffuse Direct and Glossy Direct already
  matched Cycles within `0.003%`;
- the energy error appeared only after a second surface was shaded;
- the old Psycles result with Cycles filtering enabled measured Combined
  `1.014460839×`, Diffuse Indirect `1.075783859×`, and Glossy Indirect
  `1.055669475×` Cycles;
- disabling only `blur_glossy` in Cycles made the old Psycles path close:
  Combined `1.000014457×`, Diffuse Indirect `0.999962155×`, and Glossy
  Indirect `1.000168502×`.

This causal ablation ruled out the first-surface GGX evaluation, PDF, light
selection, and exposure as the source. The Blender scene contract simply had
not exported `blur_glossy`, and the Luisa path state did not implement it.

The current Cycles source used for the contract is clean Blender
`main@4fe17ef6be5d46251fa5e7dbff9018efb1c719d5`:

- Blender sync copies `blur_glossy` to `Integrator::filter_glossy`;
- scene synchronization stores `FLT_MAX` when disabled and otherwise the
  reciprocal scene value;
- path state starts `min_ray_pdf` at `FLT_MAX` and updates it with the
  minimum non-transparent, unguided BSDF PDF;
- before shading each later surface, Cycles computes
  `sqrt(1 - filter_glossy * min_ray_pdf) * 0.5` when the product is below one;
- microfacet blur widens each effective alpha with `max`, after closure setup.

## Luisa implementation invariants

Psycles now exports and imports the raw `blur_glossy` value. The host performs
only Cycles' immutable scene-to-device reciprocal normalization. The actual
path state and closure behavior remain Luisa DSL/JIT code.

The implementation follows these path-wide rules:

```text
min_pdf(primary) = FLT_MAX
min_pdf(next) =
    transparent ? min_pdf(current)
                : min(min_pdf(current), sampled_unguided_bsdf_pdf)

blur_pdf = device_filter_scale * min_pdf(current)
filter_alpha = blur_pdf < 1 ? 0.5 * sqrt(1 - blur_pdf) : 0
effective_microfacet_alpha = max(setup_alpha, filter_alpha)
```

The threshold is exclusive. Transparent events do not change `min_pdf`.
Filtering is computed before both next-event evaluation and BSDF sampling at
the surface. It changes only the effective GGX alpha used by evaluation, PDF,
and sampling. Closure sample weights, Principled layering albedo, and
multi-GGX energy compensation retain the original setup roughness, matching
Cycles' post-setup `bsdf_blur` ordering.

There is intentionally no host implementation of the path formula. The host
unit test covers only the production scene-parameter normalization. The
renderer regression is the real Cycles/Psycles probe: its runner rejects
Diffuse Indirect or Glossy Indirect luminance ratios outside `[0.98, 1.02]`.
Synthetic red values prove the gate rejects the pre-fix report, and the actual
Luisa Vulkan render below proves the green path.

## Focused red/green result

The accepted focused run used the current locally built Cycles HIP renderer
and Psycles Vulkan on the same RX 9070 XT at 128×128/4096 fixed spp.
The complete report is
[probe-report-128x128-4096.json](probe-report-128x128-4096.json).
Every pass contains 16,384 valid pixels and zero invalid pixels.

| Pass | Before luminance ratio | After luminance ratio | After RMSE |
|---|---:|---:|---:|
| Combined | `1.014460839` | `0.999998993` | `0.000574057` |
| Diffuse Direct | `1.000029696` | `1.000029696` | `0.000824153` |
| Diffuse Indirect | `1.075783859` | `0.999868152` | `0.001470416` |
| Glossy Direct | `0.999982589` | `0.999982516` | `0.001675427` |
| Glossy Indirect | `1.055669475` | `1.000189027` | `0.001593163` |

The accepted EXRs have SHA-256
`fc1d73ce8684083f74dfb6bdf41eaf4295009df52b2f4152535df85ef123860a`
for Cycles and
`7b7392184647b780517873e41ecdf2b857be78b408c82945f254153eabc3afcf`
for Psycles.

The following real triptychs were opened and inspected:

- [Combined](probe-triptychs/combined.png)
- [Diffuse Indirect](probe-triptychs/diffind.png)
- [Glossy Indirect](probe-triptychs/glossind.png)

At ordinary scale, the Cycles and Psycles panels are visually coincident.
Their difference panels require `431.8×`, `169.6×`, and `140.2×`
amplification respectively and show unstructured finite-sample noise rather
than a systematic gradient or an over-bright secondary lobe.

## Lone Monk 1440×1080/256 result

The source remains scene `daylight`, frame 4, camera `cam.001` from
`lone-monk_cycles_and_exposure-node_demo.blend`, SHA-256
`4250d4205d8d01cefd98c15e81021d6dead540b2923797378bf7b32e96e8b8f7`.
The source sets `blur_glossy = 10.0`, so omitting the setting was material for
this gate.

The scene was re-exported from the same Blender/Cycles revision. It retains
350 geometries, 7,543 instances, 35 original raw material node graphs, and 47
images. The raw nodes, sockets, links, closure topology, and material values
are passed to Psycles; no Cycles material output is evaluated or baked.
`scene.json` is 10,834,037 bytes with SHA-256
`008c20f1c2bd9392ad8bff430a15d79d0715f1a71e959135a79a351d4845c06b`.

The accepted command used strict native XIR-to-SPIR-V and bounded 8-spp
dispatches:

```bash
env LUISA_XIR_DISABLE_OPTIMIZATION=1 \
    LUISA_SPIRV_OPT_LEVEL=0 \
    LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1 \
  python tools/measure_amd_vram.py \
    --output psycles-vulkan-vram.json -- \
  build/bin/psycles_render_blender_scene \
    export-filter-glossy psycles-vulkan.ppm \
    vk 1440 1080 256 8
```

The new callable ABI required a true cold compile. It produced 2,672,746
SPIR-V words, completed the main shader JIT in `261.613 s`, and passed the
64×48/16 spp first-pixel check. The warm quality run reported `0.666024 s`
for 350 geometries / 7,543 instances, `2.36934 s` shader JIT, and
`27.4415 s` render-only. The 119,754,288-byte 40-channel EXR has SHA-256
`342d3988a8b1b68229bba99bf028ddf8d74a2ac4f3f716449afb93d4f8fc330f`.

Cycles HIP used the same RX 9070 XT and remains the sole oracle at
`18.961390479 s` render-only. Psycles Vulkan therefore has `0.690975×`
Cycles throughput and is `1.447230×` slower; there is no speedup claim.
The monitored Psycles command peaked at 5,547,220,992 bytes of VRAM,
1,867,403,264 bytes above its pre-launch baseline. The raw measurement is
[lone-monk-psycles-vulkan-vram-1440x1080-256.json](lone-monk-psycles-vulkan-vram-1440x1080-256.json).

The complete current report is
[lone-monk-report-1440x1080-256.json](lone-monk-report-1440x1080-256.json).
Key results are:

| Pass | Before Filter Glossy ratio | After ratio | After RMSE |
|---|---:|---:|---:|
| Combined | `1.033502098` | `1.033099552` | `0.218974382` |
| Diffuse Direct | `1.021852371` | `1.021852371` | `1.680650711` |
| Diffuse Indirect | `1.039160403` | `1.033018337` | `0.251446038` |
| Glossy Direct | `1.015272347` | `1.015272347` | `0.557576835` |
| Glossy Indirect | `1.020865404` | `1.012504569` | `0.190343052` |

The Filter Glossy correction improves both indirect channels in the expected
direction, but it is not the remaining Lone Monk root cause. The real
full-resolution triptychs were opened and inspected:

- [Combined](lone-monk-triptychs/combined.png)
- [Diffuse Indirect](lone-monk-triptychs/diffind.png)
- [Glossy Indirect](lone-monk-triptychs/glossind.png)

Camera, geometry, instances, material placement, and large-scale lighting
align. The amplified residual is structured on roof tiles, arch and column
curvature, books, visibility edges, and glossy detail rather than being only
sampling noise. Diffuse Color remains `1.003732×`, Glossy Color
`0.999495×`, while the Normal pass RMSE remains `0.0295066`. Direct
illumination is also still 1.5–2.2% high. This evidence ranks complex
normal/bump/tangent evaluation and the affected direct/indirect shading paths
ahead of further sampling-constant changes.

The accepted 32-thread build and test schedule passes all 16 Psycles CTest
tests. The focused real-probe gate also passes. This is an independently
passing implementation boundary, not a claim that Lone Monk or full Cycles
compatibility is complete.
