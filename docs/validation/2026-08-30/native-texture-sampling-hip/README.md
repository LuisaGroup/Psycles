# Cycles-native HIP texture sampling and material-cost controls

This checkpoint removes Psycles' software reconstruction of ordinary Cycles
image filtering. The former implementation issued one explicit image read for
Closest, four for Linear, and sixteen for Cubic/Smart. Cycles 5.2.1 configures
the backend texture object and issues one native lookup for Closest/Linear;
its fast bicubic factorization issues four native bilinear lookups.

## Reference and invariant

The reference source is Blender/Cycles 5.2.1 (`9e2066aef7ef`):

- `intern/cycles/device/hip/device_impl.cpp::HIPDevice::image_alloc` maps
  Repeat/Extend/Clip/Mirror to HIP wrap/clamp/border/mirror and maps Closest to
  point filtering, every other mode to linear filtering.
- `intern/cycles/kernel/device/gpu/image.h::kernel_image_interp` performs one
  texture-object lookup for Closest/Linear.
- `kernel_image_interp_bicubic` factors the separable cubic B-spline into four
  bilinear lookups using `g0/g1` and `h0/h1`.

Psycles now maps its canonical extension codes
`Repeat/Clip/Extend/Mirror` to Luisa
`REPEAT/ZERO/EDGE/MIRROR`. The exact cubic factorization is

`g0 = w0 + w1`, `g1 = w2 + w3`,
`h0 = w1 / g0 - 1`, `h1 = w3 / g1 + 1`.

Each axis therefore groups the four cubic weights into two adjacent pairs. A
native linear sample at `h0` reproduces the normalized `(w0,w1)` pair and one
at `h1` reproduces `(w2,w3)`; the outer `g0/g1` weights restore their original
amplitudes. Applying the identity on both axes yields exactly four bilinear
samples. The XIR regression requires sample counts `1/1/4/4` for
Closest/Linear/Cubic/Smart and zero explicit texel reads.

## Controlled material ablation

Both controls retain the official Barbershop geometry (1055 meshes, 4 curve
geometries, 1109 instances) and replace all 263 reachable material graphs. The
constant control contains only Diffuse BSDF and Material Output. The procedural
control contains only Generated Coordinate, Noise Texture, Diffuse BSDF, and
Material Output; it has no material image/environment node. Both use
2048x858, 1024 spp, HIP on the RX 9070 XT, Cycles adaptive sampling disabled,
and Psycles `wavefront-staged` with surface sorting.

| Scene | Cycles HIP | Psycles HIP | Psycles/Cycles |
|---|---:|---:|---:|
| Constant Diffuse control | 95.4055 s | 99.8357 s | 1.04644x |
| Procedural Noise -> Diffuse control | 101.2261 s | 102.6840 s | 1.01440x |
| Original Barbershop, software filtering | 127.6620 s | 217.0220 s | 1.69997x |
| Original Barbershop, native filtering | same reference | 212.2730 s | 1.66277x |

The controls prove that geometry, base path scheduling, Diffuse closure setup,
and the tested Noise value path are within 1--5% of Cycles for this scene. They
only localize the original 70% gap to the removed complex-material domain;
they do **not** attribute that entire gap to image sampling. The direct
software-to-native A/B measures the texture change itself: 2.19% lower render
time (1.02237x), leaving a 66.28% gap to Cycles. Most remaining cost is thus in
the current complex-material SVM/closure execution model, which is being
replaced wholesale by the Cycles 5.2.1 model.

## Correctness and visual inspection

For original Barbershop, native sampling versus Cycles has Combined luminance
mean ratio `1.002741`, RMSE `0.0087372`, and relative RMSE `0.039967`. Native
versus the previous Psycles software path has luminance mean ratio `0.9999974`
and relative RMSE `0.003242`; the latter residual has the shape of independent
path-sampling noise. All images contain zero invalid pixels.

Manual inspection found no structural change between the native and software
Psycles renders. Against Cycles, the floor, ceiling, walls, cabinet, picture
frames, and lighting retain the previously accepted alignment; the amplified
difference panel is dominated by sampling noise and the already known small
transport residuals.

- [Constant control triptych](constant-control-combined.png)
- [Procedural control triptych](procedural-control-combined.png)
- [Native Barbershop vs Cycles](barbershop-native-vs-cycles-combined.png)
- [Native vs former software filtering](barbershop-native-vs-software-combined.png)

Machine-readable reports are stored beside this document.

## Commands

The focused build and HIP regression were:

```bash
cmake --build build \
  --target psycles_luisa_texture_sampling_callable_tests \
  -j "$(nproc)"
./build/bin/psycles_luisa_texture_sampling_callable_tests hip
```

The controls were generated with `tools/create_blender_surface_cost_probe.py`
in `constant-diffuse` and `procedural-diffuse` modes, exported without material
image nodes, then run with:

```bash
python tools/run_scene_benchmark.py \
  --blender /home/mike/Projects/blender-install-5.2-hiprt/blender \
  --psycles-render build/bin/psycles_render_blender_scene \
  --blend CONTROL.blend --bundle CONTROL_EXPORT --reuse-export \
  --output-dir OUTPUT --cycles-gpu-device HIP \
  --cycles-gpu-device-name "RX 9070 XT" --skip-cycles-cpu \
  --psycles-backends hip --psycles-schedulers wavefront-staged \
  --staged-surface-sorting --fast-math \
  --width 2048 --height 858 --samples 1024 \
  --max-samples-per-dispatch 64 \
  --wavefront-frame-capacity 1048576
```

