#include <psycles/luisa/homogeneous_volume_segment.h>

#include <luisa/luisa-compute.h>

#include <array>
#include <iostream>

namespace {
using namespace luisa::compute;
using namespace psycles::luisa_backend;

constexpr unsigned extinction = 1u << 0u, scattering = 1u << 1u,
                   emission = 1u << 2u, terminated = 1u << 3u,
                   direct = 1u << 4u, transmit = 1u << 5u,
                   zero_scatter = 1u << 6u, zero_length = 1u << 7u,
                   empty_interval = 1u << 8u, invalid_light = 1u << 9u;

// Controlled ShaderData coefficients, not a second shader or transport oracle.
// All transport/phase operations below are the production Luisa implementation.
class FixtureShader final : public VolumeShaderEvaluator {
private:
  UInt _flags;

public:
  explicit FixtureShader(UInt flags) noexcept : _flags{flags} {}
  VolumeCoefficients evaluate(const VolumeStack &, Float3,
                              VolumePhaseSet *phases) const noexcept override {
    const auto has_scatter = (_flags & scattering) != 0u;
    if (phases != nullptr) {
      $if(has_scatter) {
        phases->add(cycles_volume_phase::henyey_greenstein(0.25f),
                    make_float3(0.5f));
      };
    }
    return {
        .sigma_t = make_float3(select(0.0f, 1.0f, (_flags & extinction) != 0u)),
        .sigma_s = make_float3(select(0.0f, 0.5f, has_scatter)),
        .emission = make_float3(select(0.0f, 1.0f, (_flags & emission) != 0u)),
        .has_extinction = (_flags & extinction) != 0u,
        .has_scatter = (_flags & (scattering | zero_scatter)) != 0u,
        .has_emission = (_flags & emission) != 0u};
  }
};

class CountingLight final : public VolumeDirectLightProvider {
private:
  const BufferUInt &_counts;
  UInt _base;
  Bool _valid;

public:
  CountingLight(const BufferUInt &counts, UInt base, Bool valid) noexcept
      : _counts{counts}, _base{base}, _valid{valid} {}
  VolumeDirectDirectionSample sample_direction(Float) const noexcept override {
    _counts.atomic(_base).fetch_add(1u);
    return {.direction = make_float3(1.0f, 0.0f, 0.0f), .valid = _valid};
  }
  void evaluate_constant_emission() const noexcept override {
    _counts.atomic(_base + 1u).fetch_add(1u);
  }
  void
  evaluate_deferred_emission(Bool receiving_nonzero) const noexcept override {
    $if(receiving_nonzero) { _counts.atomic(_base + 2u).fetch_add(1u); };
  }
};

bool run(const char *program, const char *backend) {
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  constexpr auto medium = extinction | scattering;
  constexpr std::array cases{0u,
                             unsigned(emission),
                             unsigned(extinction),
                             medium | terminated,
                             medium | direct,
                             medium | direct | transmit,
                             medium,
                             extinction | zero_scatter | direct,
                             medium | direct | zero_length,
                             medium | direct | empty_interval,
                             medium | direct | invalid_light,
                             medium | emission | direct};
  constexpr auto fields = 6u;
  auto inputs = device.create_buffer<unsigned>(cases.size());
  auto output = device.create_buffer<unsigned>(cases.size() * fields);
  const auto segment = make_homogeneous_volume_segment_component(1u);
  Kernel1D kernel = [&](BufferUInt flags_buffer, BufferUInt counts,
                        UInt method) {
    const auto row = dispatch_x();
    const auto flags = flags_buffer.read(row);
    const auto distance = select(2.0f, 0.0f, (flags & zero_length) != 0u);
    const FixtureShader shader{flags};
    const VolumeStack stack{1u};
    const CountingLight light{counts, row * fields,
                              (flags & invalid_light) == 0u};
    const auto result = segment->emit(
        shader, stack, make_float3(0.0f), make_float3(0.0f, 0.0f, -1.0f),
        distance, make_float3(1.0f),
        select(0.2f, 0.999f, (flags & transmit) != 0u), 0.4f,
        make_float2(0.3f, 0.6f), (flags & terminated) != 0u,
        {.scattered_radiance = make_float3(0.0f),
         .transmitted_radiance = make_float3(0.0f),
         .majorant_optical_depth = 0.0f,
         .enabled = false},
        {.requested_method = method,
         .light_position = make_float3(1.0f, 0.0f, 1.0f),
         .interval = {.minimum = 0.0f,
                      .maximum = select(distance, 0.0f,
                                        (flags & empty_interval) != 0u)},
         .enabled = (flags & direct) != 0u},
        &light);
    counts.write(row * fields + 3u, cast<unsigned>(result.phase.valid));
    counts.write(row * fields + 4u, cast<unsigned>(result.scattered));
    counts.write(row * fields + 5u, cast<unsigned>(result.phase_failed));
  };
  auto shader = device.compile(
      kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
  auto failures = 0u;
  for (const auto method :
       {volume_sample_distance, volume_sample_equiangular, volume_sample_mis}) {
    std::array<unsigned, cases.size() * fields> actual{};
    stream << inputs.copy_from(cases.data()) << output.copy_from(actual.data())
           << shader(inputs, output, method).dispatch(cases.size())
           << output.copy_to(actual.data()) << synchronize();
    for (auto row = 0u; row < cases.size(); ++row) {
      const auto flags = cases[row];
      // Cycles 5.2.1 shade_volume.h: volume_integrate_homogeneous returns
      // after attenuation for PATH_RAY_TERMINATE or zero sigma_s. Its caller
      // invokes direct light / phase continuation only for the respective
      // sampled scatter event; a failed light resample returns before emission.
      const bool eligible =
          (flags & scattering) && !(flags & (terminated | zero_length));
      const bool direct_event =
          eligible && (flags & direct) && !(flags & empty_interval);
      const bool valid_light = direct_event && !(flags & invalid_light);
      const bool indirect_event = eligible && !(flags & transmit);
      const std::array<unsigned, fields> expected{
          unsigned(direct_event),   unsigned(valid_light),
          unsigned(valid_light),    unsigned(indirect_event),
          unsigned(indirect_event), 0u};
      for (auto field = 0u; field < fields; ++field) {
        if (actual[row * fields + field] != expected[field]) {
          ++failures;
          std::cerr << "method=" << method << " case=" << row
                    << " field=" << field
                    << " actual=" << actual[row * fields + field]
                    << " expected=" << expected[field] << '\n';
        }
      }
    }
  }
  std::cout << "Volume work: " << cases.size() * fields * 3u << " checks, "
            << failures << " failures\n";
  return failures == 0u;
}
} // namespace

int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
