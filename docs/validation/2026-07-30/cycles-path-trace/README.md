# Cycles CPU/HIP path-trace checkpoint

This checkpoint validates the version-1 per-event oracle against Cycles
itself before adding the equivalent Luisa output.

Configuration:

- Blender/Cycles: official main `ff404d072bb4`
- device CPU: AMD Ryzen 9 9950X3D, 32 threads
- device HIP: AMD Radeon RX 9070 XT
- scene: `build/diagnostics/minimal-point/point_light.blend`
- full film: 32×32
- traced Cycles pixel: `(17, 16)`, lower-left origin
- sample: 0 of 1
- sampler: Tabulated Sobol, scrambling distance 1

Result:

- 43 discrete state fields: exact
- 16 sampled random fields: exact
- 84 continuous fields: within the schema float32 bound
- 3 topology fields: equal for this non-edge pixel
- failures: 0
- maximum continuous absolute error: `4.76837158203125e-7`

The maximum occurs in the reconstructed surface position. A separate
center-pixel check deliberately lands on the plane's shared triangle edge:
Cycles CPU and HIP select different triangle IDs/barycentrics but reconstruct
the same object, shader, surface point, geometric normal, and shading normal.
The formal topology-invariant rule accepts that event; it does not relax RNG,
closure, light, BSDF, PDF, or path-state gates.

The diagnostic build was also rendered with only one custom AOV. The complete
schema-size guard correctly disabled tracing and the render completed, covering
the former film-buffer-overrun hazard.
