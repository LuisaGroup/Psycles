# Evidence and presentation audit

This technical checkpoint uses the requested repository Markdown format.
The technical-report structure separates formal cause, red/green evidence,
full-suite qualifications, timing scope and unresolved correctness work.
Tables support exact four-scene lookup; no aggregate score or decorative
chart mixes initialization, rendering, frame storage and image error.

Three post-change Psycles observations per scene use the prior campaign's
unchanged original Cycles run-1 image. They are not new paired reference
measurements. The three archived JSON files preserve every command, log and
output hash, implementation hashes, frame layout, channels and pass metric.
Original source/export/reference hashes were revalidated before rendering.
The executable/library identity stayed fixed throughout all twelve renders.

Numerical tables were checked against those files. All 46 actual EXR
channels were independently read and checked finite; all 15 reported passes
have zero actual-invalid pixels. Original Classroom direct-pass invalid
pixels remain explicitly reported. Four first-run Combined triptychs were
visually inspected at original resolution. Initialization timing is correctly
labeled as the complete create_session span, and its slow first observations
are retained separately from repetitions. Earlier stalled suite attempts and
the two pre-existing complete-suite failures remain qualified, not hidden.
