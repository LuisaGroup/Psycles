# Per-(pixel, sample) path-dispatch checkpoint

This checkpoint changes coroutine scheduling from one coroutine per pixel with
an internal sample loop to one coroutine per `(pixel, sample)`. The actual
launch is three-dimensional:

```text
dispatch(width, row_count, samples_in_batch)
```

The absolute sample is `sample_first + dispatch_z()`. The serial megakernel
retains its per-pixel sample loop, and `megakernel-per-sample` supplies a
non-coroutine 3D baseline. All three assembly paths call one host-recorded path
wrapper with a runtime `UInt sub_spp_index`; only the outer loop/launch mapping
differs. Coroutine construction remains a host choice, with no device-side
scheduler branch.

Per-sample execution follows Cycles GPU film semantics: Combined, Normal,
Albedo, every light pass, volume-guiding raw statistics, and sample count use
component-wise atomic accumulation. The serial megakernel keeps its exclusive
pixel writer and ordinary stores. `max_samples_per_dispatch=1` is the
deterministic diagnostic topology because only one contributor per pixel is
in flight between ordered stream barriers.

## Commands

The smoke used the current 37-material Lone Monk export and Luisa fallback:

```text
psycles_render_blender_scene \
  /var/tmp/psycles-lone-monk-transmission-dbdcb17/export \
  <output.ppm> fallback 16 16 4 4 - 8 8 0 0 4 - 1 0 \
  <megakernel|megakernel-per-sample|wavefront|persistent>
```

The same serial and per-sample megakernels were repeated at one sample with
both the total sample count and batch size set to one. OpenImageIO comparison
used:

```text
idiff -a -fail 0 -failpercent 0 serial.exr scheduled.exr
```

Artifacts are retained in
`/var/tmp/psycles-per-sample-smoke-74yXfqUC` on the validation host.

## Results

At one sample, serial and 3D per-sample megakernels are byte-identical for
every PFM pass, and all 46 multilayer EXR channels pass exact comparison. This
locks the pixel mapping, absolute sample index, and the atomic film write path
without an accumulation-order allowance.

At four samples, all four display PPM files have the same SHA-256:
`a0dcfa771396c998feab88b5c6ee5fb5a2248b93ec1bcb5c0cf9d01e48dbf7aa`.
The linear differences are limited to floating-point addition order:

| Scheduled path versus serial | RMS error | Maximum error | Pixels over `1e-6` |
| --- | ---: | ---: | ---: |
| 3D megakernel | `7.62e-8` | `3.8147e-6` | 6 / 256 |
| wavefront | `2.93e-9` | `2.38419e-7` | 0 / 256 |
| persistent | `1.03e-7` | `7.62939e-6` | 5 / 256 |

These are not performance measurements: the image is deliberately tiny and
each scheduler has a different cold cache identity. The observed render-only
times were 13.5 ms serial, 5.16 ms 3D megakernel, 57.9 ms wavefront, and
17.6 ms persistent. Larger paired measurements are required before choosing a
default topology.

Removing the internal sample loop and `path_sample` continuation reduced the
Lone Monk coroutine frame from the prior 1392-byte baseline to 1264 bytes. The
remaining suspend points are `path_bounce` and `surface_shading`; the rejected
1408-byte pure-SSA snapshot prototype is not present.
