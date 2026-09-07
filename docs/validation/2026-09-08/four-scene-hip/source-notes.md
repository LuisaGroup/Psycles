# Report audit notes

Audience: technical. Delivery: the user's requested repository Markdown
documentation, not an additional HTML/MCP report or external publication.
The build-report technical structure is mapped to summary, matched scope,
render findings, compilation, pass residuals, and reproduction/remaining work.
Definitions precede the tables; uncertainty and questions are kept beside the
findings they qualify. No inferential or causal performance model is fitted.

Visual/table contract: exact row lookup across four fixed scene extents is
the purpose of the three neutral Markdown tables. They separate rendering,
compilation/frame state and image error rather than merging unlike units or
adding an aggregate score. Three observations per scene are insufficient for
a trend plot or meaningful distribution chart. Existing original-resolution
image triptychs carry the visual quality evidence and have been inspected.
No decorative quantitative chart or color-based performance verdict is added.

QA completed before archiving: 12/12 schema-v2 manifests complete; command,
output and metadata hashes and Cycles main-loop log evidence revalidated via
the runner; all bundle hashes match; all sources use original Blender build
9e2066aef7ef and exactly one named HIP 9070 XT device. All 46 actual channels
were independently read and checked finite, not inferred from RGB-only pass
reports. All 12 reports have 15 passes and zero actual-invalid pixels. The
renderer/implementation-library hashes were verified unchanged before the
next build. Tables were checked against the saved numerical summary.

The original Cycles Classroom direct-pass non-finite values are retained in
the report rather than hidden by the finite-pixel comparison mask. JIT cache
scope, rotation/order, observed ranges, interrupted scheduling shell and the
distinction between current baseline and later unmeasured changes are explicit.
