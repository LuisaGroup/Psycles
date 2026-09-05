#include <psycles/luisa/cycles_svm.h>
#include "cycles_svm_closure_layout.h"

#include <luisa/dsl/sugar.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>

namespace {
using namespace luisa::compute;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace abi = psycles::compiler::cycles_svm;
constexpr auto capacity = 8u;
constexpr auto steps = 7u;
constexpr auto cases = capacity + 1u;
constexpr auto mixed_cases = 3u;
constexpr auto mixed_rows = 4u;
constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();

struct Oracle {
  std::unordered_map<std::string, std::uint32_t> layout;
  std::array<luisa::uint4, cases * steps> allocations{};
  std::array<luisa::uint4, mixed_cases * mixed_rows> payloads{};
};

bool read_oracle(Oracle &oracle) {
  std::ifstream input{PSYCLES_CLOSURE_POOL_ORACLE_PATH};
  if (!input) {
    std::cerr << "Cannot read external Cycles closure-pool fixture\n";
    return false;
  }
  std::string kind;
  std::size_t allocation_count = 0u;
  std::size_t payload_count = 0u;
  while (input >> kind) {
    if (kind == "layout") {
      std::string name;
      std::uint32_t value{};
      if (!(input >> name >> value) ||
          !oracle.layout.emplace(name, value).second) {
        return false;
      }
    } else if (kind == "allocation") {
      std::uint32_t c{}, step{};
      luisa::uint4 value{};
      if (!(input >> c >> step >> value.x >> value.y >> value.z >> value.w) ||
          c >= cases || step >= steps || c * steps + step != allocation_count) {
        return false;
      }
      // The public projection returns a validity predicate, not a machine
      // pointer. Compare it with the oracle's actual null/non-null result.
      value.y = static_cast<std::uint32_t>(value.y != invalid);
      oracle.allocations[allocation_count++] = value;
    } else if (kind == "payload") {
      std::uint32_t row{};
      luisa::uint4 value{};
      if (!(input >> row >> value.x >> value.y >> value.z >> value.w) ||
          row != payload_count || row >= oracle.payloads.size()) {
        return false;
      }
      oracle.payloads[payload_count++] = value;
    } else {
      return false;
    }
  }
  return input.eof() && allocation_count == cases * steps &&
         payload_count == mixed_cases * mixed_rows &&
         oracle.layout.contains("ShaderClosure.sizeof");
}

auto allocation_kernel() {
  return Kernel1D<Buffer<luisa::uint4>>{[](BufferUInt4 output) noexcept {
    svm::ClosurePool pool{capacity};
    const auto c = dispatch_x();
    pool.set_left(c);
    for (auto step = 0u; step < steps; ++step) {
      const auto ordinary = pool.allocate(
          static_cast<std::uint32_t>(abi::CLOSURE_BSDF_DIFFUSE_ID),
          make_float3(float(step + 1u)));
      const auto extra = pool.allocate_extra(ordinary, step % 3u);
      output.write(c * steps + step,
                   make_uint4(select(invalid, ordinary.index, ordinary.valid),
                              cast<luisa::uint>(extra), pool.count(),
                              pool.left()));
    }
  }};
}

auto mixed_payload_kernel() {
  return Kernel1D<Buffer<luisa::uint4>>{[](BufferUInt4 output) noexcept {
    svm::ClosurePool pool{capacity};
    const auto prefix = dispatch_x();
    $for(i, prefix) {
      static_cast<void>(pool.allocate(
          static_cast<std::uint32_t>(abi::CLOSURE_NONE_ID), make_float3(0.0f)));
    };
    const auto micro = pool.allocate(
        static_cast<std::uint32_t>(abi::CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID),
        make_float3(2.0f, 3.0f, 4.0f));
    const auto micro_extra = pool.allocate_extra(micro, 1u);
    const auto hair = pool.allocate(
        static_cast<std::uint32_t>(abi::CLOSURE_BSDF_HAIR_HUANG_ID),
        make_float3(12.0f, 13.0f, 14.0f));
    const auto hair_extra = pool.allocate_extra(
        hair, 1u, svm::ClosurePool::ExtraPayload::huang_hair);
    $if(micro.valid & micro_extra & hair.valid & hair_extra) {
      pool.set_sample_weight(micro.index, 5.0f);
      pool.set_normal(micro.index, make_float3(6.0f, 7.0f, 8.0f));
      svm::MicrofacetParam micro_param{};
      micro_param.fresnel_type = static_cast<std::uint32_t>(
          svm::MicrofacetFresnel::generalized_schlick);
      pool.set_microfacet_param(micro.index, micro_param);
      svm::FresnelGeneralizedSchlick fresnel{};
      fresnel.f0 = make_float3(9.0f + cast<float>(prefix), 10.0f, 11.0f);
      pool.set_generalized_schlick(micro.index, fresnel);

      pool.set_sample_weight(hair.index, 15.0f);
      pool.set_normal(hair.index, make_float3(16.0f, 17.0f, 18.0f));
      svm::HuangHairParam hair_param{};
      svm::HuangHairExtra extra{};
      extra.Y = make_float3(19.0f, 20.0f, 21.0f);
      extra.R = 22.0f;
      pool.set_huang_hair(hair.index, hair_param, extra);

      $while(pool.left() != 0u) {
        const auto v = 100.0f + cast<float>(pool.count());
        const auto filler = pool.allocate(
            static_cast<std::uint32_t>(abi::CLOSURE_BSDF_MICROFACET_GGX_ID),
            make_float3(v));
        pool.set_sample_weight(filler.index, v);
        pool.set_normal(filler.index, make_float3(v));
        pool.set_microfacet_param(filler.index,
                                  {.alpha_x = v,
                                   .alpha_y = v,
                                   .ior = v,
                                   .energy_scale = v,
                                   .fresnel_type = static_cast<std::uint32_t>(
                                       svm::MicrofacetFresnel::none),
                                   .T = make_float3(v)});
      };
      const auto m = pool.common(micro.index);
      const auto f = pool.generalized_schlick(micro.index);
      const auto h = pool.huang_hair(hair.index);
      output.write(prefix * mixed_rows,
                   make_float4(m.weight, m.sample_weight).as<luisa::uint4>());
      output.write(
          prefix * mixed_rows + 1u,
          make_float4(f.f0, cast<float>(pool.left())).as<luisa::uint4>());
      output.write(prefix * mixed_rows + 2u,
                   make_float4(h.common.weight, h.common.sample_weight)
                       .as<luisa::uint4>());
      output.write(prefix * mixed_rows + 3u,
                   make_float4(h.extra.Y, h.extra.R).as<luisa::uint4>());
    }
    $else {
      for (auto row = 0u; row < mixed_rows; ++row) {
        output.write(prefix * mixed_rows + row, make_uint4(invalid));
      }
    };
  }};
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: closure-pool-tests <backend>\n";
    return 1;
  }
  Oracle oracle;
  if (!read_oracle(oracle)) {
    std::cerr << "Invalid external Cycles closure-pool fixture\n";
    return 1;
  }
  const auto kernel = allocation_kernel();
  std::size_t array_bytes = 0u;
  for (const auto &local : kernel.function()->local_variables()) {
    if (local.type()->is_array()) {
      array_bytes += local.type()->size();
    }
  }
  const auto expected_bytes =
      capacity * oracle.layout.at("ShaderClosure.sizeof");
  auto passed = array_bytes == expected_bytes;
  if (!passed) {
    std::cerr << "ClosurePool array storage: " << array_bytes << " B, Cycles "
              << expected_bytes << " B for " << capacity << " slots\n";
  }
#define PSYCLES_CHECK_CLOSURE_OFFSET(symbol, name, value)                      \
  if (!oracle.layout.contains(name) ||                                         \
      oracle.layout.at(name) != svm::detail::closure_layout::symbol) {         \
    std::cerr << "Closure field layout differs from Cycles: " << name << '\n'; \
    passed = false;                                                            \
  }
  PSYCLES_CYCLES_CLOSURE_LAYOUT_FIELDS(PSYCLES_CHECK_CLOSURE_OFFSET)
#undef PSYCLES_CHECK_CLOSURE_OFFSET

  Context context{argv[0]};
  auto device = context.create_device(argv[1]);
  auto stream = device.create_stream();
  auto shader = device.compile(kernel, ShaderOption{.enable_cache = false});
  auto output = device.create_buffer<luisa::uint4>(cases * steps);
  std::array<luisa::uint4, cases * steps> actual{};
  stream << shader(output).dispatch(cases) << output.copy_to(actual.data())
         << synchronize();
  for (auto i = 0u; i < actual.size(); ++i) {
    const auto a = actual[i];
    const auto e = oracle.allocations[i];
    if (a.x != e.x || a.y != e.y || a.z != e.z || a.w != e.w) {
      std::cerr << "Allocator differs from Cycles at capacity " << i / steps
                << ", step " << i % steps << '\n';
      passed = false;
    }
  }
  auto mixed_shader = device.compile(mixed_payload_kernel(),
                                     ShaderOption{.enable_cache = false});
  auto mixed_output =
      device.create_buffer<luisa::uint4>(oracle.payloads.size());
  std::array<luisa::uint4, mixed_cases * mixed_rows> mixed_actual{};
  stream << mixed_shader(mixed_output).dispatch(mixed_cases)
         << mixed_output.copy_to(mixed_actual.data()) << synchronize();
  for (auto i = 0u; i < mixed_actual.size(); ++i) {
    const auto a = mixed_actual[i];
    const auto e = oracle.payloads[i];
    if (a.x != e.x || a.y != e.y || a.z != e.z || a.w != e.w) {
      std::cerr << "Mixed closure tail payload differs from Cycles at row " << i
                << '\n';
      passed = false;
    }
  }
  if (!passed) {
    return 1;
  }
  std::cout << "Cycles closure-pool footprint and " << cases * steps
            << " allocation transitions, " << mixed_cases
            << " mixed-family full pools passed on " << argv[1] << '\n';
}
