# Surface inlining and texture storage audit

Neither a missing major inline boundary nor float32 expansion of the bound
8-bit textures was established. These are read-only checks, not a performance
repair or an explanation of the remaining Barbershop surface-time gap.

This snapshot uses Psycles `124cee8c` (render implementation `ffb3f7f2`) and
Luisa `da8fff856`. The native source is Cycles
`cb168525138fecc792cc393f94afc39582b0103c`. The actual code object is the
[preceding final Voronoi control](../voronoi-octave/README.md), identified by
its surface entry symbol, not a presumed dump number. The original object
is the same-device Cycles compute binary. [Results](results.json) retain
the source hashes and counting boundaries.

## Function boundaries

Comparing only the two main functions is misleading: the actual main already
includes substantial work that remains in outlined original functions.

| Counting boundary | Functions | Instruction sites | ELF function bytes |
| --- | ---: | ---: | ---: |
| Actual main surface | 1 | 158,118 | 843,220 |
| Original `integrate_surface<1979>` alone | 1 | 82,632 | 441,580 |
| Actual object, all functions | 4 | 161,496 | 860,908 |
| Original observed function-address closure | 138 | 287,152 | 1,547,676 |

The last row is **not a complete proven call-site graph**. It follows observed
`s_getpc_b64` plus signed PC-relative low-word address materializations. The
diagnostic does not prove high-word carry, dominance, or every indirect
transfer. A materialized function address is not a dynamic call count. The
original object contains 334 functions; not all belong to this entry point.
Counts exclude instruction alignment padding using ELF symbol extents.

Nor are these rows equal semantic coverage: the original precompiled module
contains node paths unreachable from this scene, while the actual JIT omits
unused node cases. They do not prove that either renderer executes fewer
instructions or has lower instruction-cache traffic.

The actual object has 49 `s_swappc_b64` sites. Its three ordinary helpers are
`cycles_signed_noise_4d`, `callable.20` (1D fractal Noise), and
`__ocml_tan_f32`. The scene's 3D Noise, main SVM and main microfacet paths are
already inlined. This does not prescribe an inlining policy. Earlier
[full-module inlining controls](../../2026-09-08/lamp-routing-and-surface/README.md)
must not be replaced with guesses based on an isolated main-function size.

Scratch-site totals are also not comparable as memory traffic: original
outlined code uses generic flat loads/stores for private pointers. Its scalar
loads include repeated module-global accesses, and inspected Math payload
loads are flat loads, not evidence for a uniform/scalar SVM optimization.
No hardware-counter attribution to spills, occupancy, or instruction cache
is established by this audit.

## Texture storage

The production image decoder and alpha processing were executed on the
exported bundle, then joined by the
[verified binding identities](../resource-identities/README.md). Of 174 bound
images, two are missing. All 172 valid bindings match the original dimensions
and storage class: 169 `BYTE4` and three `FLOAT4`, totaling 157,005,312 pixel
payload bytes (149.73 MiB).

The reviewed upload path selects byte RGBA for the former and float RGBA for
the latter; HIP maps the byte format to a four-channel unsigned-int8 array.
This rules out a fourfold float32 pixel-storage expansion at that boundary.
It does not measure driver pitch/swizzle allocation, cache behavior, sampler
throughput or bandwidth. No software texture sampler, filtering change,
forced inline/noinline attribute, register cap, or fast-math change was made.
