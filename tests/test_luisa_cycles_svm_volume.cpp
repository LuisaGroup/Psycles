#include "cycles_svm_volume.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <luisa/coro/schedulers/wavefront.h>
#include <luisa/dsl/coro_func.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace luisa::compute;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace layout = svm::detail::volume_closure_layout;
constexpr std::array capacities{0u, 1u, 64u};
constexpr std::array domains{abi::SHADER_TYPE_VOLUME, abi::SHADER_TYPE_SURFACE,
                             abi::SHADER_TYPE_DISPLACEMENT};
constexpr unsigned model_count = 20u, state_count = 16u, value_count = 22u;
struct Expected {
  unsigned flag{}, count{}, left{};
  std::array<float, value_count> values{};
};
struct Fixture {
  std::vector<unsigned> words;
  std::array<std::array<std::array<Expected, state_count>, 3u>, 3u> expected{};
};
void require(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error{message};
  }
}

auto read_fixtures() {
  std::array<Fixture, model_count> fixtures{};
  std::ifstream input{PSYCLES_VOLUME_SVM_ORACLE};
  char tag{};
  input >> tag;
  require(tag == 'L', "missing Cycles HIP volume layout");
  constexpr std::array offsets{layout::size,
                               layout::henyey_greenstein_g,
                               layout::draine_g,
                               layout::draine_alpha,
                               layout::fournier_forand_c1,
                               layout::fournier_forand_c2,
                               layout::fournier_forand_c3};
  for (auto expected : offsets) {
    unsigned actual{};
    input >> actual;
    require(actual == expected,
            "volume closure byte offset differs from Cycles HIP");
  }
  for (auto model = 0u; model < model_count; ++model) {
    unsigned id{}, size{};
    input >> tag >> id >> size;
    require(tag == 'W' && id == model && size >= 4u && size <= 512u,
            "invalid external Cycles volume word stream");
    auto &fixture = fixtures[model];
    fixture.words.resize(size);
    input >> std::hex;
    for (auto &word : fixture.words) {
      input >> word;
    }
    input >> std::dec;
    for (auto capacity = 0u; capacity < capacities.size(); ++capacity) {
      for (auto domain = 0u; domain < domains.size(); ++domain) {
        for (auto state = 0u; state < state_count; ++state) {
          unsigned row_model{}, row_capacity{}, row_domain{}, row_state{};
          input >> tag >> row_model >> row_capacity >> row_domain >> row_state;
          require(tag == 'R' && row_model == model &&
                      row_capacity == capacities[capacity] &&
                      row_domain == domain && row_state == state,
                  "misordered Cycles volume oracle row");
          auto &expected = fixture.expected[capacity][domain][state];
          input >> expected.flag >> expected.count >> expected.left;
          for (auto &value : expected.values) {
            input >> value;
          }
          require(bool(input), "truncated Cycles HIP volume oracle");
        }
      }
    }
  }
  input >> std::ws;
  require(input.eof(), "unexpected trailing volume oracle data");
  return fixtures;
}

// These are the oracle's input tables, not a second implementation of its
// result. All attribute lookup and conversion still execute in native SVM.
class Globals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  Expr<Buffer<abi::KernelObject>> _objects;
  Expr<Buffer<abi::AttributeMap>> _attributes;
  Expr<Buffer<float>> _floats;
  Expr<Buffer<abi::packed_float3>> _colors;

public:
  Globals(Expr<Buffer<abi::KernelObject>> objects,
          Expr<Buffer<abi::AttributeMap>> attributes,
          Expr<Buffer<float>> floats,
          Expr<Buffer<abi::packed_float3>> colors) noexcept
      : _objects{objects}, _attributes{attributes}, _floats{floats},
        _colors{colors} {}
  std::optional<Float>
  object_volume_density(Expr<unsigned> object) const noexcept override {
    Float density = 1.0f;
    $if(object != svm::object_none) {
      density = _objects.read(object).volume_density;
    };
    return density;
  }
  UInt
  object_attribute_map_offset(Expr<unsigned> object) const noexcept override {
    return _objects.read(object).attribute_map_offset;
  }
  Var<abi::AttributeMap>
  attribute_map(Expr<unsigned> offset) const noexcept override {
    return _attributes.read(offset);
  }
  Float attribute_float(Expr<int> offset) const noexcept override {
    return _floats.read(offset);
  }
  Var<abi::packed_float3>
  attribute_float3(Expr<int> offset) const noexcept override {
    return _colors.read(offset);
  }
};

template <bool suspend>
auto make_program(unsigned capacity, abi::ShaderType domain, bool validated,
                  std::uint64_t kernel_features = svm::kernel_feature_volume) {
  std::array<bool, abi::NODE_NUM> used{};
  for (auto node : {abi::NODE_SHADER_JUMP, abi::NODE_END, abi::NODE_VALUE_F,
                    abi::NODE_VALUE_V, abi::NODE_CLOSURE_SET_WEIGHT,
                    abi::NODE_CLOSURE_VOLUME, abi::NODE_VOLUME_COEFFICIENTS,
                    abi::NODE_PRINCIPLED_VOLUME}) {
    used[node] = true;
  }
  auto body = [=](BufferUInt words, BufferVar<abi::KernelObject> objects,
                  BufferVar<abi::AttributeMap> attributes, BufferFloat floats,
                  BufferVar<abi::packed_float3> colors, BufferFloat values,
                  BufferUInt meta) {
    const Globals kg{objects, attributes, floats, colors};
    const UInt i = dispatch_x();
    const auto object = select(i % 4u - 1u, svm::object_none, i % 4u == 0u);
    const auto flag =
        svm::shader_data_is_volume_shader_eval |
        select(0u, svm::shader_data_extinction | svm::shader_data_emission,
               (i & 8u) != 0u);
    const auto identity = make_float4x4(1.0f);
    svm::ClosurePool pool{capacity};
    svm::ShaderData sd{make_float3(0.0f),
                       make_float3(0.0f),
                       make_float3(0.0f),
                       make_float3(0.0f),
                       svm::primitive_volume,
                       0u,
                       flag,
                       0u,
                       svm::primitive_none,
                       0.0f,
                       0.0f,
                       object,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       0.0f,
                       make_float3(0.0f),
                       make_float3(0.0f),
                       identity,
                       identity,
                       0u,
                       &pool};
    sd.closure_transparent_extinction = make_float3(-1.0f, -2.0f, -3.0f);
    sd.closure_emission_background = make_float3(0.125f, 0.25f, 0.5f);
    const svm::TransformState transforms{identity, identity, identity,
                                         identity};
    const svm::PathState path{select(svm::path_ray_visibility_camera,
                                     svm::path_ray_visibility_shadow,
                                     (i & 4u) != 0u),
                              0u};
    svm::EvaluationResult status;
    if (validated) {
      svm::eval_nodes_assume_valid(kg, words, domain, kernel_features,
                                   svm::kernel_feature_node_mask_volume, used,
                                   transforms, sd, path, 5u);
    } else {
      svm::eval_nodes(kg, words, domain, kernel_features,
                      svm::kernel_feature_node_mask_volume, used, transforms,
                      sd, path, status, 5u);
    }
    if constexpr (suspend) {
      $suspend("volume_ready");
    }
    const auto put3 = [&](unsigned offset, Expr<luisa::float3> value) {
      values.write(i * value_count + offset, value.x);
      values.write(i * value_count + offset + 1u, value.y);
      values.write(i * value_count + offset + 2u, value.z);
    };
    put3(0u, sd.closure_transparent_extinction);
    put3(3u, sd.closure_emission_background);
    for (auto slot = 0u; slot < 2u; ++slot) {
      const auto base = 6u + slot * 8u;
      for (auto j = 0u; j < 8u; ++j) {
        values.write(i * value_count + base + j, 0.0f);
      }
      $if(slot < pool.count()) {
        const auto sc = pool.volume_common(slot);
        values.write(i * value_count + base, cast<float>(sc.type));
        put3(base + 1u, sc.weight);
        values.write(i * value_count + base + 4u, sc.sample_weight);
        $if(sc.type ==
            static_cast<unsigned>(abi::CLOSURE_VOLUME_HENYEY_GREENSTEIN_ID)) {
          values.write(i * value_count + base + 5u,
                       pool.volume_henyey_greenstein_g(slot));
        }
        $elif(sc.type == static_cast<unsigned>(abi::CLOSURE_VOLUME_DRAINE_ID)) {
          const auto p = pool.volume_draine_parameters(slot);
          values.write(i * value_count + base + 5u, p.x);
          values.write(i * value_count + base + 6u, p.y);
        }
        $elif(sc.type ==
              static_cast<unsigned>(abi::CLOSURE_VOLUME_FOURNIER_FORAND_ID)) {
          put3(base + 5u, pool.volume_fournier_forand_coefficients(slot));
        };
      };
    }
    meta.write(i * 5u, sd.flag);
    meta.write(i * 5u + 1u, pool.count());
    meta.write(i * 5u + 2u, pool.left());
    meta.write(i * 5u + 3u, status.status);
    meta.write(i * 5u + 4u, status.final_offset);
  };
  if constexpr (suspend) {
    return Coroutine<void(Buffer<unsigned>, Buffer<abi::KernelObject>,
                          Buffer<abi::AttributeMap>, Buffer<float>,
                          Buffer<abi::packed_float3>, Buffer<float>,
                          Buffer<unsigned>)>{body};
  } else {
    return Kernel1D<Buffer<unsigned>, Buffer<abi::KernelObject>,
                    Buffer<abi::AttributeMap>, Buffer<float>,
                    Buffer<abi::packed_float3>, Buffer<float>,
                    Buffer<unsigned>>{body};
  }
}

void run(Device &device) {
  const auto fixtures = read_fixtures();
  auto stream = device.create_stream();
  auto words = device.create_buffer<unsigned>(512u);
  auto objects = device.create_buffer<abi::KernelObject>(3u);
  auto attributes = device.create_buffer<abi::AttributeMap>(10u);
  auto floats = device.create_buffer<float>(2u);
  auto colors = device.create_buffer<abi::packed_float3>(1u);
  auto values = device.create_buffer<float>(state_count * value_count);
  auto meta = device.create_buffer<unsigned>(state_count * 5u);
  std::array<abi::KernelObject, 3u> object_data{};
  object_data[0].volume_density = 0.0f;
  object_data[1].volume_density = 0.5f;
  object_data[2].volume_density = 2.0f;
  object_data[2].attribute_map_offset = 2u;
  std::array<abi::AttributeMap, 10u> attribute_data{};
  for (auto j = 0u; j < 3u; ++j) {
    auto &a = attribute_data[2u + 2u * j];
    a.id = 1000u + j;
    a.element = abi::ATTR_ELEMENT_MESH;
    a.offset = j == 2u ? 1 : 0;
    a.type = j == 1u ? abi::NODE_ATTR_FLOAT3 : abi::NODE_ATTR_FLOAT;
  }
  const std::array float_data{1.5f, 1.1f};
  const std::array<abi::packed_float3, 1u> color_data{{{0.5f, 0.8f, 0.3f}}};
  stream << objects.copy_from(object_data.data())
         << attributes.copy_from(attribute_data.data())
         << floats.copy_from(float_data.data())
         << colors.copy_from(color_data.data());
  std::array<float, state_count * value_count> actual{};
  std::array<unsigned, state_count * 5u> actual_meta{};
  auto checked = 0u;
  const auto verify = [&](auto &&dispatch, unsigned capacity, unsigned domain,
                          bool validated, bool suspended) {
    for (auto model = 0u; model < fixtures.size(); ++model) {
      const auto &fixture = fixtures[model];
      stream << words.view(0u, fixture.words.size())
                    .copy_from(fixture.words.data());
      dispatch();
      stream << values.copy_to(actual.data())
             << meta.copy_to(actual_meta.data()) << synchronize();
      for (auto state = 0u; state < state_count; ++state) {
        const auto &expected = fixture.expected[capacity][domain][state];
        const auto *m = actual_meta.data() + state * 5u;
        bool same = m[0] == expected.flag && m[1] == expected.count &&
                    m[2] == expected.left;
        if (!validated) {
          same &= m[3] == static_cast<unsigned>(svm::EvaluationStatus::ended) &&
                  m[4] == fixture.words.size();
        }
        for (auto lane = 0u; lane < value_count; ++lane) {
          const auto a = actual[state * value_count + lane],
                     e = expected.values[lane];
          if (!std::isfinite(a) ||
              std::abs(a - e) > 3e-5f * std::max(1.0f, std::abs(e))) {
            std::cerr << " lane=" << lane << " actual=" << a << " Cycles=" << e;
            same = false;
          }
        }
        if (!same) {
          std::cerr << " model=" << model
                    << " capacity=" << capacities[capacity]
                    << " domain=" << domain << " state=" << state
                    << " validated=" << validated << " coroutine=" << suspended
                    << " flag/count/left/status/PC=" << m[0] << '/' << m[1]
                    << '/' << m[2] << '/' << m[3] << '/' << m[4] << '\n';
          throw std::runtime_error{
              "native volume SVM differs from original Cycles HIP"};
        }
        ++checked;
      }
    }
  };
  for (const bool validated : {false, true}) {
    for (auto capacity = 0u; capacity < capacities.size(); ++capacity) {
      for (auto domain = 0u; domain < domains.size(); ++domain) {
        auto shader = device.compile(
            make_program<false>(capacities[capacity], domains[domain],
                                validated),
            ShaderOption{.enable_cache = false, .enable_fast_math = true});
        verify(
            [&] {
              stream << shader(words, objects, attributes, floats, colors,
                               values, meta)
                            .dispatch(state_count);
            },
            capacity, domain, validated, false);
      }
    }
  }
  for (auto capacity = 0u; capacity < capacities.size(); ++capacity) {
    // Without Cycles' global __VOLUME__ feature the volume-domain handler
    // consumes its typed payload but has the same no-op state as Surface.
    auto shader = device.compile(
        make_program<false>(capacities[capacity], abi::SHADER_TYPE_VOLUME,
                            false, 0u),
        ShaderOption{.enable_cache = false, .enable_fast_math = true});
    verify(
        [&] {
          stream << shader(words, objects, attributes, floats, colors, values,
                           meta)
                        .dispatch(state_count);
        },
        capacity, 1u, false, false);
  }
  using Scheduler =
      coro::WavefrontCoroScheduler<Buffer<unsigned>, Buffer<abi::KernelObject>,
                                   Buffer<abi::AttributeMap>, Buffer<float>,
                                   Buffer<abi::packed_float3>, Buffer<float>,
                                   Buffer<unsigned>>;
  for (auto capacity = 0u; capacity < capacities.size(); ++capacity) {
    // Read the active phase payload only after a real wavefront suspension.
    // No frame exports, initialization markers, or renderer-specific handler
    // are supplied to make the Local lifetime proof succeed.
    auto coroutine =
        make_program<true>(capacities[capacity], abi::SHADER_TYPE_VOLUME, true);
    require(coroutine.subroutine_count() == 2u,
            "volume continuation was not split");
    std::cout << "Volume coroutine: capacity=" << capacities[capacity]
              << " frame_bytes=" << coroutine.frame().frame_type()->size()
              << '\n';
    coro::WavefrontCoroSchedulerConfig config;
    config.thread_count =
        8u; // Force admission of more tasks than available frames.
    config.shader_option = {.enable_cache = false, .enable_fast_math = true};
    Scheduler scheduler{device, coroutine, config};
    verify(
        [&] {
          stream << scheduler(words, objects, attributes, floats, colors,
                              values, meta)
                        .dispatch(state_count);
        },
        capacity, 0u, true, true);
  }
  std::cout
      << checked
      << " Cycles HIP volume/domain/allocator cases passed with fast math\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "hip");
    run(device);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
