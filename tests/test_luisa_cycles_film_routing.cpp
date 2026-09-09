#include "cycles_film_routing_test_support.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace {
using namespace luisa::compute;
namespace d = psycles::test_support::film_routing;
namespace f = psycles::test_support::film_routing_fixture;
struct Oracle {
  std::array<unsigned, 3> state{};
  std::array<float, 6> weights{};
  std::array<float, f::film_lanes> film{};
  std::array<unsigned, f::film_lanes> writes{};
};
bool near(float a, float b) { return std::isfinite(a) && std::abs(a - b) <= 2e-6f; }

bool run(const char *program, const char *backend) {
  std::array<Oracle, f::cases> oracle{};
  std::ifstream input{PSYCLES_FILM_ROUTING_ORACLE};
  for (unsigned i = 0; i < f::cases; ++i) {
    char tag{};
    unsigned index{};
    input >> tag >> index;
    for (auto &v : oracle[i].state) { input >> v; }
    for (auto &v : oracle[i].weights) { input >> v; }
    for (auto &v : oracle[i].film) { input >> v; }
    for (auto &v : oracle[i].writes) { input >> v; }
    if (!input || tag != 'F' || index != i) { return false; }
  }
  std::string extra;
  if (input >> extra) { return false; }
  Context context{program};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  auto film = device.create_buffer<luisa::float4>(f::film_lanes / 4);
  auto weights = device.create_buffer<luisa::float4>(2);
  auto state = device.create_buffer<unsigned>(3);
  auto unused = device.create_buffer<float>(1);
  auto config = psycles::test_support::surface_emission::config(false, false);
  auto evaluator = d::make_direct_light_task_evaluator(config);
  unsigned passed = 0, total = 0, state_passed = 0, state_total = 0;
  for (unsigned op = 0; op < f::operations; ++op) {
    for (unsigned sink = 0; sink < (op == f::surface_nee ? 3u : 2u); ++sink) {
      const auto mode = sink == 0 ? d::PathFilmAccumulation::serial
                                  : d::PathFilmAccumulation::atomic;
      Kernel1D kernel = [&](BufferFloat4 combined, BufferFloat4 film,
                            BufferFloat4 weights, BufferUInt state, BufferFloat unused,
                            UInt flags, UInt depth, Float3 contribution,
                            Float3 diffuse, Float3 glossy, Float3 bd, Float3 bg, Float3 bs) {
        d::record(config, evaluator, op, mode, sink == 2, combined, film, weights,
                   state, unused, flags, depth, contribution, diffuse, glossy, bd, bg, bs);
      };
      auto shader = device.compile(kernel, ShaderOption{.enable_cache = false,
                                                       .enable_fast_math = true});
      for (unsigned i = 0; i < f::cases; ++i) {
        const auto &in = f::inputs[i];
        if (in.operation != op) { continue; }
        std::array<luisa::float4, f::film_lanes / 4> actual{};
        std::array<luisa::float4, 2> actual_weights{};
        std::array<unsigned, 3> actual_state{};
        stream << film.copy_from(actual.data())
               << shader(film, film, weights, state, unused, d::path_flags(in.route),
                         in.bounce, d::rgb(in.contribution), d::rgb(in.diffuse_weight),
                         d::rgb(in.glossy_weight), d::rgb(in.bsdf_diffuse),
                         d::rgb(in.bsdf_glossy), d::rgb(in.bsdf_sum)).dispatch(1u)
               << film.copy_to(actual.data()) << weights.copy_to(actual_weights.data())
               << state.copy_to(actual_state.data()) << synchronize();
        bool good = true;
        for (unsigned lane = 0; lane < f::film_lanes; ++lane) {
          good &= near(actual[lane / 4][lane % 4], oracle[i].film[lane]);
        }
        if (!good) {
          std::cerr << "film op=" << op << " sink=" << sink << " case=" << i;
          for (unsigned lane = 0; lane < f::film_lanes; ++lane) {
            const auto v = actual[lane / 4][lane % 4];
            if (!near(v, oracle[i].film[lane])) {
              std::cerr << " lane=" << lane << " actual=" << v
                        << " native=" << oracle[i].film[lane];
            }
          }
          std::cerr << '\n';
        }
        passed += good;
        ++total;
        if (op == f::surface_nee) {
          bool matches = actual_state == oracle[i].state;
          for (unsigned lane = 0; lane < 6; ++lane) {
            matches &= near(actual_weights[lane / 3][lane % 3], oracle[i].weights[lane]);
          }
          if (!matches) {
            std::cerr << "shadow state sink=" << sink << " case=" << i
                      << " actual flag=" << actual_state[0]
                      << " native=" << oracle[i].state[0] << '\n';
          }
          state_passed += matches;
          ++state_total;
        }
      }
    }
  }
  std::cout << "Native film routes: " << passed << '/' << total
            << "; native surface shadow state: " << state_passed << '/' << state_total << '\n';
  return passed == total && state_passed == state_total;
}
} // namespace
int main(int argc, char **argv) {
  return run(argv[0], argc > 1 ? argv[1] : "hip") ? 0 : 1;
}
