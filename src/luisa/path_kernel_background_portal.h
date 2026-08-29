#pragma once

#include "path_tracer_internal.h"

namespace psycles::luisa_backend::detail {

struct BackgroundPortalSample {
  Float3 direction;
  Float pdf;
  Bool valid;
};

struct BackgroundPortalPdf {
  Float pdf;
  Bool possible;
};

// Position-dependent Cycles background-portal proposal. Portal records live
// in the shared light buffer after regular lights; runtime offset/count bounds
// keep scene population out of shader cache identity and prevent loop unroll.
class BackgroundPortalSampling {

public:
  [[nodiscard]] UInt count_possible(const Buffer<LightGpu> &lights,
                                    UInt portal_offset, UInt portal_count,
                                    Float3 reference) const noexcept;

  [[nodiscard]] BackgroundPortalSample
  sample(const Buffer<LightGpu> &lights, UInt portal_offset, UInt portal_count,
         Float3 reference, Float2 random) const noexcept;

  [[nodiscard]] Float pdf(const Buffer<LightGpu> &lights, UInt portal_offset,
                          UInt portal_count, Float3 reference,
                          Float3 direction) const noexcept;

  [[nodiscard]] BackgroundPortalPdf
  evaluate_pdf(const Buffer<LightGpu> &lights, UInt portal_offset,
               UInt portal_count, Float3 reference,
               Float3 direction) const noexcept;
};

} // namespace psycles::luisa_backend::detail
