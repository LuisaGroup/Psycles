# Lone Monk bring-up — 2026-07-29

This directory is the durable record for the first full-scene gate. The
machine-readable [bring-up report](bringup.json) records the exact scene hash,
export scale, backend attempts, resource samples, diagnostic bounds, and why
no image comparison is claimed yet.

The source scene was exported as evaluated geometry and instances plus the
original Blender node graphs, links, sockets, and material metadata. No
Cycles material result was baked into the export.

Neither initial Psycles backend reached a first pixel:

- Vulkan spent more than 20 minutes in the cold main-kernel JIT. Disabling
  optional XIR and SPIR-V optimization did not change the two-thread,
  multi-gigabyte code-generation profile.
- HIP spent more than 10 minutes building the 350 HIPRT bottom-level
  acceleration structures at 100% GPU utilization and did not reach the main
  render-kernel JIT.

There are intentionally no placeholder triptychs. Cycles / Psycles /
amplified-difference triptychs will be added here only after both renderers
produce comparable linear EXR passes at the same frame, seed, sample count,
integrator settings, and RX 9070 XT device.
