#include "path_kernel_film.h"

#include <array>
#include <optional>

namespace psycles::luisa_backend::detail {
namespace {

// Cycles film_write_{direct_light,emission_or_background_pass}: select a
// destination before each write. Surface and volume are disjoint branches;
// glossy/transmission precede the common final diffuse/volume/visible write.
// This is host DSL recording, not a device callable or a six-RGB film value.
template <typename Emit>
void route_light_passes(Float3 contribution, UInt flags, Float3 diffuse,
                        Float3 glossy, Bool direct,
                        std::optional<LightPassBuffer> visible, Emit &&emit) {
  UInt pass = light_pass_buffer_count;
  auto select_pass = [&](LightPassBuffer d, LightPassBuffer i) {
    return select(light_pass_index(i), light_pass_index(d), direct);
  };
  auto scattered = [&] {
    $if((flags & cycles_path_state::flag_surface_pass) != 0u) {
      emit(select_pass(LightPassBuffer::glossy_direct, LightPassBuffer::glossy_indirect),
           glossy * contribution);
      const auto transmission = make_float3(1.0f) - diffuse - glossy;
      emit(select_pass(LightPassBuffer::transmission_direct,
                        LightPassBuffer::transmission_indirect), transmission * contribution);
      pass = select_pass(LightPassBuffer::diffuse_direct, LightPassBuffer::diffuse_indirect);
      contribution *= diffuse;
    }
    $elif((flags & cycles_path_state::flag_volume_pass) != 0u) {
      pass = select_pass(LightPassBuffer::volume_direct, LightPassBuffer::volume_indirect);
    };
  };
  if (visible) {
    $if((flags & cycles_path_state::flag_any_pass) == 0u) { pass = light_pass_index(*visible); }
    $else { scattered(); };
  } else {
    scattered();
  }
  $if(pass != light_pass_buffer_count) { emit(pass, contribution); };
}
} // namespace

void PathSampleContext::accumulate_light_pass(UInt pass, Float3 contribution) noexcept {
  if (invocation.film_accumulation == PathFilmAccumulation::atomic) {
    atomic_accumulate_float3(invocation.light_passes, invocation.light_pass_base + pass,
                             contribution);
    return;
  }
  // Serial scheduling keeps its existing named per-sample fields. No local
  // array or path-lifetime temporary film structure is introduced on device.
  const std::array fields{&sample_diffuse_direct, &sample_diffuse_indirect,
                          &sample_glossy_direct, &sample_glossy_indirect,
                          &sample_transmission_direct, &sample_transmission_indirect,
                          &sample_volume_direct, &sample_volume_indirect,
                          &sample_emission, &sample_environment,
                          &sample_glossy_color, &sample_transmission_color};
  static_assert(fields.size() == light_pass_buffer_count);
  $switch(pass) {
    for (unsigned i = 0; i < fields.size(); ++i) {
      $case(i) { *fields[i] += contribution; };
    }
  };
}

void PathSampleContext::accumulate_scattered_light_at_state(
    Float3 contribution, UInt flags, Float3 diffuse, Float3 glossy, Bool direct) noexcept {
  route_light_passes(contribution, flags, diffuse, glossy, direct, std::nullopt,
                     [&](UInt pass, Float3 value) { accumulate_light_pass(pass, value); });
}

void PathSampleContext::accumulate_emission_or_background(
    Float3 contribution, LightPassBuffer pass) noexcept {
  route_light_passes(contribution, path_flags, path_diffuse_weight, path_glossy_weight,
                     path_depth == 1u, pass,
                     [&](UInt offset, Float3 value) { accumulate_light_pass(offset, value); });
}

void atomic_accumulate_scattered_light_passes(
    const BufferFloat4 &light_passes, UInt light_pass_base,
    Float3 contribution, UInt flags, Float3 diffuse, Float3 glossy, Bool direct) noexcept {
  route_light_passes(contribution, flags, diffuse, glossy, direct, std::nullopt,
                     [&](UInt pass, Float3 value) {
                       atomic_accumulate_float3(light_passes, light_pass_base + pass, value);
                     });
}

} // namespace psycles::luisa_backend::detail
