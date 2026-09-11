# Coroutine frame raw-buffer lowering

The HIP surface kernel had a long-lived frame address value. LLVM kept the
address pair live until continuation writeback, producing address spills even
though each frame access already had an operation-local 32-bit byte offset.

The Luisa fix tags only coroutine frame byte-buffer accesses with the
`luisa.coro.frame.raw` AST/XIR comment. HIP consumes that marker for nonvolatile
scalar `i32`, `u32`, and `f32` reads and writes and emits AMD raw-buffer
operations. Volatile accesses, other types, ordinary buffers, and non-HIP
backends retain their existing lowering. Frame layouts larger than `UINT32_MAX`
are not tagged.

On the original Barbershop module at 2048x858, gfx1201, the exact surface
kernel changed as follows:

| metric | stock | marked raw frame accesses |
| --- | ---: | ---: |
| ELF code bytes | 409,104 | 408,716 |
| private segment bytes | 2,368 | 2,192 |
| VGPRs / SGPRs | 256 / 107 | 256 / 107 |
| VGPR spills | 449 | 407 |
| SGPR spills | 58 | 60 |
| decoded entry lanes | 74,292 | 74,186 |

The render completed successfully in 32.9742 s, 32.9882 s, and 32.9691 s
(mean 32.9772 s), versus 36.5768 s for the frozen stock capture. The output
comparison differed in only a handful of channels, with maximum absolute
difference 2 on 8-bit output; the tiny difference is within the existing GPU
render nondeterminism envelope and was not used as a performance claim.

Validation completed before publishing:

* root and nested Luisa all-target builds with 32-way parallelism;
* 13-test HIP coroutine layout suite (126 assertions);
* 13-test fallback coroutine layout suite (126 assertions);
* XIR coroutine CFG distillation (59 tests, 563 assertions);
* HIP byte-buffer suite (8 assertions);
* exact-SDK driver: 6 host, 17 HIP, 17 fallback, and 17 strict native Vulkan
  tests, all passing. Vulkan produced successful SPIR-V compilations with no
  DXC/DXIL matches.

The Luisa change is published directly on `next` as `7d1f44cb6`; the Psycles
gitlink is published on `main` as `80878c80`. The static counts above are
decoded instruction lanes, not dynamic issue counters. The earlier offline
raw-IR scratch-delta experiment is retracted because its diagnostic did not
accumulate nested GEP offsets correctly; only the live tagged lowering and the
full-module render are evidence for this result.

The post-lowering instruction audit identifies the next target. In the exact
surface entry, Psycles has 3,116 cndmask lanes, 4,585 compares, and 9,013
integer/hash lanes, versus 1,333, 3,291, and 8,256 in Cycles'
`integrate_surface<1979>` entry. Branches and waits are already lower in
Psycles. The cndmask count is unchanged by raw lowering, and disabling every
tested LLVM IfConverter mode was byte-identical, so these masks originate in
coroutine/state-machine selects or later lowering. Raw lowering removes 85
64-bit address-spill instructions while adding 85 scalar b32 scratch
operations; total decoded scratch lanes remain flat.

The stronger remaining pressure signal is register spilling: the raw surface
still uses 256 VGPRs with 407 VGPR spill slots, while Cycles' surface metadata
reports 192 VGPRs and 2 VGPR spills. The current LLVM contains long-lived
aggregate values such as a 60x4-word register object and a 33-float object.
Reducing those continuation live ranges or promoting only proven narrow fields
is the next generic direction; blanket if-conversion changes would not address
this pressure.
