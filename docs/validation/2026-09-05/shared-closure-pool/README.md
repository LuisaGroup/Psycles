# Cycles-sized shared closure pool

## Source contract and structural cause

The sole oracle is Cycles 5.2.1 at
`/home/mike/Projects/blender-cycles-trace-5.2`, commit
`cb168525138fecc792cc393f94afc39582b0103c`. Its `kernel/types.h` and
`kernel/closure/` files have no local changes. The device probe includes these
headers and executes the real `closure_alloc` and `closure_alloc_extra` on HIP;
there is no separate reference allocator or CPU renderer.

Cycles uses one 16-byte-aligned array of 80-byte `ShaderClosure` records.
Ordinary closures grow from its front; extra payloads grow from its tail.
The previous projection charged extras against the correct logical budget,
but physically attached their storage to every ordinary slot: nine float4
arrays and two uint arrays, or **152 bytes per slot**. With Lone Monk's
24-slot capacity this reserved 3648 bytes instead of Cycles' 1920 bytes.
Changing the capacity would not repair that structural difference.

For an effective capacity C, initialized ordinary count n and remaining
budget l, ordinary records occupy [0,n), and allocated extras occupy the
tail starting at n+l. The source transitions are:

- Ordinary success: `(n,l) -> (n+1,l-1)`, returning slot n.
- k-slot extra success: `(n,l) -> (n,l-k)`, returning slot n+l-k.
- Extra failure: `(n,l) -> (n-1,l+1)`, rolling back the immediately preceding
  ordinary allocation.

The implementation now uses one local uint4 array, five rows per record.
Typed accessors use Cycles' actual HIP field offsets. Microfacet's pointer
field at byte 56 and Huang hair's pointer field at byte 64 hold projected
pool-slot indices; the remaining native pointer padding is not observed.
Generalized Schlick, conductor, F82 tint and Huang extras are stored in the
borrowed tail records, not in storage attached to each owner.

Address arithmetic widens the slot index before multiplying by the record
stride, as native pointer arithmetic does. Scalar lvalue accesses avoid
reading or overwriting adjacent uninitialized fields. The pool is not cleared;
ordinary scalar/vector default initialization and the existing explicit
sample-weight initialization are unchanged.

SVM words, stack addressing, PC/dispatch, feature masks, closure counts,
Fresnel-discriminated lazy reads, and coroutine suspension state are unchanged.
No application `noinline`, software floating-point path, Luisa source edit,
or submodule gitlink change is part of this patch.

## Independent permanent regression

`tools/cycles_svm_closure_pool_oracle.hip` emits:

- 91 device `sizeof`/`offsetof` values for the common record and typed payloads;
- 63 actual allocator transitions, covering capacities 0 through 8 and zero-,
  one- and two-slot extra requests, including exhaustion and rollback;
- 12 uint4 output records from three full mixed-family pools. Each has a
  varying ordinary prefix, a microfacet extra and a Huang extra. All remaining
  ordinary slots are written before either extra is read back.

Reproduce the fixture with the external source, not a local expected allocator:

```sh
/opt/rocm/bin/hipcc --offload-arch=gfx1201 -DHIPCC -std=c++17 -O3 \
  -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles \
  tools/cycles_svm_closure_pool_oracle.hip -o /tmp/closure-pool-oracle
/tmp/closure-pool-oracle
```

The output exactly matches `tests/data/cycles_svm_closure_pool.txt`.
`tests/test_luisa_cycles_svm_closure_pool.cpp` independently checks the
declaration footprint, every layout entry, allocator transitions, and mixed
tail payloads on the selected backend. The initial footprint regression failed
on the old implementation: 8 slots reserved **1216 B**, versus the oracle's
**640 B**; allocator state already agreed. After the storage change, the
footprint, allocator states and mixed payloads all pass.

The integrated, already-dirty closure test had a shape assertion requiring
exactly nine float4 arrays and two uint arrays. That was an implementation
workaround, not a Cycles semantic requirement. Its shape assertion was removed
only after the full original HIP module compiled without IR growth and the
full scene passed image comparison. Its runtime round-trip checks remain.
That previously pending test block and unrelated API migrations are not
included in this selective commit; permanent pool coverage is in the new,
independent test above.

## Validation

Evidence directory: `/var/tmp/psycles-closure-pool-layout-aaprOJ`.

- Full build with `cmake --build build --parallel 32`: passed.
- All 149 HIP backend tests: passed.
- All 151 fallback backend tests: passed.
- All 51 `psycles.cycles_*` core tests: passed.
- Six strict native-XIR Vulkan canaries: background sun sampling, NEE setup,
  NEE evaluation, wireframe, bump state, and the new shared-pool regression.

The Vulkan runs require all three environment flags:
`LUISA_VULKAN_USE_XIR=1`,
`LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1`,
`LUISA_VULKAN_DISABLE_DXC=1`. DXC fallback is not permitted.

Authoritative logs are `full-build-final.log`, `full-hip-final.log`,
`full-fallback-final.log`, `core-final.log`, and `strict-native-vk-final.log`.
The original red footprint evidence is `red-hip-confirmed.log` (exit 1).
These are integrated-worktree validations; they are not a claim of a full
build from an otherwise clean checkout. Unrelated renderer edits and the
unfinished Luisa checkout/gitlink remain excluded from the commit.

## Original Lone Monk HIP module and render

The original exported scene is
`/var/tmp/psycles-lone-monk-f297ec53-20260904/export`.
Both post-change runs use 1440x1080, 256 spp, seed 0, staged wavefront,
the native SVM gate, disabled shader cache, and forced LLVM IR dumps.
No other build or GPU test runs concurrently with either profile.

| Quantity | Before shared pool | Run 1 | Run 2 |
| --- | ---: | ---: | ---: |
| Closure storage | 3648 B | 1920 B | 1920 B |
| Surface final IR lines | 135612 | 135250 | 135250 |
| Surface allocas | 12 | 2 | 2 |
| Surface VGPR count | 256 | 256 | 256 |
| Surface private scratch | 5072 B | 3424 B | 3424 B |
| Surface kernel time | 11.581827 s | 10.899365 s | 10.918939 s |
| Full render time | 19.5275 s | 18.8910 s | 18.8344 s |

The two post-change surface modules are byte-identical (SHA-256
`efe377767ce1134ae4e96bdb8db5dd35be048b54acd93fc04993e830de27e47d`).
Their only allocas are the 25-float SVM stack and the 1920-byte closure pool.
The former aliasing/IR-growth concern does not reproduce in this full module.
The persistent coroutine frame remains 88 fields, 448 B AoS / 444 B actual
SoA; this storage reduction is shader-local, not a coroutine-frame shrink.

Evidence directories:

- Before: `/var/tmp/psycles-surface-late-sroa-aTH1UV`.
- Run 1: `/var/tmp/psycles-closure-pool-lone-monk-RkdjyS`.
- Run 2: `/var/tmp/psycles-closure-pool-lone-monk-repeat-2CHlP1`.
- Cycles HIP: `/var/tmp/psycles-lone-monk-cycles-hip-profile-ulQYKD`.

The original Cycles multilayer EXR and build metadata are verified by
`tools/compare_cycles.py`. Combined relative RMSE is 1.113612% and 1.113822%
for the two new renders, versus 1.113156% before. All eight compared passes
have zero nonfinite pixels. DiffInd and GlossInd remain about 14.11% and
16.20%; this is not a claim of full image identity.

The observed full-render time is 3.3-3.5% lower than the preceding measurement;
surface time is about 5.7-5.9% lower. This is an unpaired before/after
comparison, not an established general speedup. Cycles' corresponding HIP
render was 16.4504 s and its surface kernel 5.9491 s. Psycles is still roughly
14.5-14.8% slower end to end; the surface kernel remains the main gap.
Complete SVM coverage and performance parity remain unfinished.
