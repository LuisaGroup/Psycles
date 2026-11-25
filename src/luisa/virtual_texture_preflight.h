#pragma once

#include "path_tracer_lighting.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>

#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>

namespace psycles::luisa_backend::detail {

struct LuisaSceneData;
struct SessionKernelSignature;
struct VtPreflightShaderCache;

struct VtPreflightConfig {
  bool overlap_pso{true};
  std::uint32_t stride{2u};
  std::uint32_t bounces{1u};
  std::uint32_t width{1u};
  std::uint32_t height{1u};
  luisa::compute::ShaderOption shader_option{};
};

class PreflightOverlapHandle final {
public:
  struct State;

  PreflightOverlapHandle() noexcept = default;
  ~PreflightOverlapHandle() noexcept;
  PreflightOverlapHandle(PreflightOverlapHandle &&other) noexcept = default;
  PreflightOverlapHandle &
  operator=(PreflightOverlapHandle &&other) noexcept = default;
  PreflightOverlapHandle(const PreflightOverlapHandle &) = delete;
  PreflightOverlapHandle &
  operator=(const PreflightOverlapHandle &) = delete;

  void join() noexcept;
  [[nodiscard]] bool active() const noexcept { return _state != nullptr; }

private:
  friend PreflightOverlapHandle start_vt_preflight(
      const std::shared_ptr<LuisaSceneData> &scene,
      const SessionKernelSignature &signature,
      const luisa::compute::Buffer<float> &filter_table,
      const RenderKernelParameters &parameters,
      const SafeNormalizeCallable &safe_normalize,
      const VtPreflightConfig &config) noexcept;

  std::shared_ptr<State> _state;
};

[[nodiscard]] PreflightOverlapHandle start_vt_preflight(
    const std::shared_ptr<LuisaSceneData> &scene,
    const SessionKernelSignature &signature,
    const luisa::compute::Buffer<float> &filter_table,
    const RenderKernelParameters &parameters,
    const SafeNormalizeCallable &safe_normalize,
    const VtPreflightConfig &config) noexcept;

} // namespace psycles::luisa_backend::detail
