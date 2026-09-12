# Retracted switch table lowering: narrow GEP counterexample

The original dispatch has one constant integer PHI, two forwarding arms, one case (0), and selector width 1. It passes the retracted transform's structural/density checks and the later max+1 overflow guard (maximum_case is 0, below the i1 maximum 1).

`lowered-i1.ll` manually reproduces the removed pass's emitted instructions, including unreachable forwarding arms. The selected default index is i1 1. LLVM GEP sign-extends integer indices, so that index denotes -1 and its inbounds pointer is poison. The original default value is 10.

The alwaysinline dispatch and constant caller expose the semantic difference with LLVM's ordinary optimizer. These are diagnostic IR fixtures, not an execution of the removed pass. No GPU execution is needed.

Reproduce from this directory:

```sh
opt -passes=always-inline,instcombine,simplifycfg -verify-each -S original-i1.ll -o original-i1.optimized.ll
opt -passes=always-inline,instcombine,simplifycfg -verify-each -S lowered-i1.ll -o lowered-i1.optimized.ll
```

The original `default_case` returns 10. The lowered `default_case` returns poison. Both inputs are verifier-valid: verifier validity alone does not detect this semantic error.

Separately, the removed implementation called `ConstantInt::getZExtValue` on all cases before rejecting selector widths over 32 bits. A structurally eligible i128 switch containing case 18446744073709551616 triggers APInt's assertion in assertion-enabled builds. The width guard must precede the case scan. This second issue is source-audit evidence only; no process crash was executed.

CFG audit found no additional concrete corruption: the one-PHI restriction means `removePredecessor` operates after the sole PHI has been removed; the dead forwarding blocks stay valid. Duplicate direct head-to-merge edges are conservatively rejected by PHI incoming-count validation.
