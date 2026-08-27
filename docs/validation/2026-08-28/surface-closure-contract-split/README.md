# Surface closure contract split

## Result

The HIP closure oracle now separates the conditional sampler `p(w | i)` from
Psycles' current selected-closure contract, which samples a direction and then
independently reconstructs the selected closure's finite-direction value and
PDF. This is diagnostic instrumentation only; it does not change renderer
semantics.

The split proves that the regular-GGX arithmetic sampler is not the remaining
local bottleneck. At 1,048,576 lanes and 1,024 iterations per lane, the
conditional Psycles sampler takes about 7.96 ms, versus 8.74 ms for the Cycles
5.2 sampler. Adding Psycles' independent evaluator raises the same probe to
about 13.52 ms. Thus the replay adds roughly 70% over the conditional sampler,
and the complete Psycles probe is about 55% slower than Cycles even though its
sampler alone is about 9% faster.

The corresponding diffuse measurements are approximately 3.93 ms conditional,
5.12 ms with replay, and 3.88 ms in Cycles. The singular GGX domain remains a
useful control: it is approximately 2.23 ms conditional, 2.78 ms with the
existing delta composition, and 3.28 ms in Cycles, so Psycles is already about
15% faster there. This agrees with the earlier domain-partition result and
rules out a blanket microfacet rewrite.

## Formal boundary

For selected closure `s`, let `S_s(u)` produce direction `w` and the
conditional witness needed by the sampler. Let `E_s(w)` be the independent
finite-direction evaluator. The two benchmark contracts are

```text
conditional_sample(u)       = S_s(u)
selected_closure_sample(u)  = (S_s(u), E_s(S_s(u).w)).
```

Cycles' closure sampler instead returns `(w, g_s(w), p_s(w))` while the sampled
half-vector and distribution terms are live. The production mixture then uses
that term as its initial value and evaluates only competitors. The benchmark
does not claim that removing `E_s` is automatically profitable in a divergent
full renderer: two earlier seed/skip candidates enlarged Psycles' aggregate
merge and regressed `shade_surface`. It isolates the arithmetic opportunity and
therefore narrows the required representation change.

A profitable implementation must satisfy all of the following at once:

```text
semantic equality:
  sampled_g_s(w) = evaluated_g_s(w)
  sampled_p_s(w) = evaluated_p_s(w)

bounded representation:
  one shared closure-consumer body per semantic handler,
  not one expansion per NEE/sample use site

private-reference locality:
  the physical tagged record remains in per-thread storage,
  and only the selected family payload is read

backend legality:
  HIP may retain a private reference callable;
  native XIR-to-SPIR-V must legally promote or specialize that reference.
```

The next production experiment will therefore use Luisa callable reference
arguments to share the post-population evaluator across its NEE and sampling
consumers. This matches Cycles' private `ShaderClosure *` shape without adding
`noinline`/`alwaysinline` annotations. LLVM remains responsible for the final
inlining decision.

## Reproduction

Build with all host threads:

```sh
cmake --build build --parallel 32 \
  --target psycles_benchmark_surface_closures
```

The retained complete contract remains the default. The new final argument
selects the isolated conditional contract:

```sh
build/bin/psycles_benchmark_surface_closures \
  hip glossy_ggx_regular 1048576 1024 7 conditional_sample

build/bin/psycles_benchmark_surface_closures \
  hip glossy_ggx_regular 1048576 1024 7 selected_closure_sample

/var/tmp/cycles_surface_closure_benchmark \
  glossy_ggx_regular 1048576 1024 7
```

The device was an AMD Radeon RX 9070 XT (`gfx1201`) with ROCm 7.2.4. A longer
probe was run first to bring the device to steady clocks; the figures above are
medians from repeated steady-state runs rather than cold-process first samples.
