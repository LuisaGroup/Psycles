#include <psycles/luisa/cycles_svm.h>

#include "luisa_cycles_svm_test_kernel_globals.h"
#include "path_tracer_bsdf_tables.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;

/* Compact surface images copied word-for-word from the final global buffers
 * of unmodified Cycles 5.2.1 standalone Sheen probes. Only the four-word
 * shader jump is relocated from (89, 102, 103) to (4, 17, 18). */
constexpr std::array<std::uint32_t, 19u> microfiber_words{
    0x00000001u, 0x00000004u, 0x00000011u, 0x00000012u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3ec28f5cu, 0x3f451eb8u,
    0x3e23d70au, 0x00000002u, 0x00000007u, 0x000000ffu, 0x3edc28f6u,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

constexpr std::array<std::uint32_t, 19u> ashikhmin_words{
    0x00000001u, 0x00000004u, 0x00000011u, 0x00000012u, 0x0000000bu,
    0x00000001u, 0x00000000u, 0x00000005u, 0x3f51eb85u, 0x3e428f5cu,
    0x3f11eb85u, 0x00000002u, 0x00000010u, 0x000000ffu, 0x3e75c28fu,
    0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u};

[[nodiscard]] consteval std::array<std::uint32_t, 22u>
make_mixed_microfiber_words() noexcept {
  std::array<std::uint32_t, 22u> result{};
  result[0u] = microfiber_words[0u];
  result[1u] = 4u;
  result[2u] = 20u;
  result[3u] = 21u;
  result[4u] = static_cast<std::uint32_t>(NODE_VALUE_F);
  result[5u] = std::bit_cast<std::uint32_t>(0.25f);
  /* The following external NODE_GEOMETRY owns stack [0, 3). */
  result[6u] = 3u;
  for (auto index = 4u; index < microfiber_words.size(); index++) {
    result[index + 3u] = microfiber_words[index];
  }
  /* SVMNodeClosureBsdf::mix_weight_offset is its low byte. */
  result[16u] = 3u;
  return result;
}

constexpr auto mixed_microfiber_words = make_mixed_microfiber_words();
inline constexpr std::uint32_t roughness_word = 14u;
inline constexpr std::uint32_t output_stride = 5u;
inline constexpr std::uint32_t meta_stride = 6u;
inline constexpr std::uint32_t scenario_count = 7u;

class TableKernelGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
private:
  const BufferFloat *_table;
  Bool _valid;

public:
  TableKernelGlobals(const BufferFloat &table, Expr<bool> valid) noexcept
      : _table{&table}, _valid{valid} {}

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return select(0.0f, _table->read(index), _valid);
  }
};

[[nodiscard]] device_svm::TransformState identity_transform_state() noexcept {
  const auto identity = make_float4x4(1.0f);
  return {identity, identity, identity, identity};
}

[[nodiscard]] device_svm::ShaderData
make_shader_data(device_svm::ClosurePool *closures) noexcept {
  const auto identity = make_float4x4(1.0f);
  return {make_float3(0.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          make_float3(0.0f, 0.0f, 1.0f),
          device_svm::primitive_triangle,
          0u,
          0u,
          0u,
          0u,
          0.25f,
          0.25f,
          0u,
          0.0f,
          1.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          0.0f,
          make_float3(1.0f, 0.0f, 0.0f),
          make_float3(0.0f, 1.0f, 0.0f),
          identity,
          identity,
          0u,
          closures};
}

[[nodiscard]] std::array<bool, NODE_NUM> node_types() noexcept {
  std::array<bool, NODE_NUM> result{};
  result[NODE_END] = true;
  result[NODE_SHADER_JUMP] = true;
  result[NODE_GEOMETRY] = true;
  result[NODE_VALUE_F] = true;
  result[NODE_CLOSURE_SET_WEIGHT] = true;
  result[NODE_CLOSURE_BSDF] = true;
  return result;
}

template<std::size_t Capacity> [[nodiscard]] auto transition_kernel() {
  const auto used = node_types();
  return Kernel1D<Buffer<std::uint32_t>, Buffer<float>, Buffer<luisa::float4>,
                  Buffer<std::uint32_t>, std::uint32_t, bool>{
      [used](BufferUInt words, BufferFloat table, BufferFloat4 output,
             BufferUInt meta, UInt scenario, Bool valid_table) noexcept {
        const TableKernelGlobals kernel_globals{table, valid_table};
        device_svm::ClosurePool closures{Capacity};
        auto shader_data = make_shader_data(&closures);
        const device_svm::PathState path_state{
            device_svm::path_ray_visibility_camera, 0u};
        device_svm::EvaluationResult result;
        device_svm::eval_nodes(kernel_globals, words, SHADER_TYPE_SURFACE, 0u,
                               device_svm::kernel_feature_node_bsdf, used,
                               identity_transform_state(), shader_data,
                               path_state, result);

        const auto output_base = scenario * output_stride;
        const auto meta_base = scenario * meta_stride;
        for (auto index = 0u; index < output_stride; index++) {
          output.write(output_base + index, make_float4(0.0f));
        }
        meta.write(meta_base + 0u,
                   static_cast<std::uint32_t>(CLOSURE_NONE_ID));
        $if(closures.count() != 0u) {
          const auto common = closures.common(0u);
          const auto sheen = closures.sheen(0u);
          const auto velvet = closures.velvet(0u);
          const auto is_velvet =
              common.type == static_cast<std::uint32_t>(
                                 CLOSURE_BSDF_ASHIKHMIN_VELVET_ID);
          output.write(output_base + 0u,
                       make_float4(common.weight, common.sample_weight));
          output.write(output_base + 1u, make_float4(common.N, 0.0f));
          output.write(output_base + 2u,
                       make_float4(select(sheen.param.roughness,
                                          velvet.param.sigma, is_velvet),
                                   select(sheen.param.transform_a,
                                          velvet.param.invsigma2, is_velvet),
                                   select(sheen.param.transform_b, 0.0f,
                                          is_velvet),
                                   0.0f));
          output.write(output_base + 3u,
                       make_float4(sheen.param.T, 0.0f));
          output.write(output_base + 4u,
                       make_float4(sheen.param.B, 0.0f));
          meta.write(meta_base + 0u, common.type);
        };
        meta.write(meta_base + 1u, closures.count());
        meta.write(meta_base + 2u, closures.left());
        meta.write(meta_base + 3u, shader_data.flag);
        meta.write(meta_base + 4u, result.status);
        meta.write(meta_base + 5u, result.final_offset);
      }};
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 5.0e-5f) noexcept {
  return std::isfinite(actual) &&
         std::abs(actual - expected) <=
             tolerance * std::max(1.0f, std::abs(expected));
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 5.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] bool run(std::string_view backend, char **argv) {
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table_values =
      psycles::luisa_backend::detail::make_cycles_bsdf_table_values(
          psycles::contract::ShaderColorSpace{});

  auto low_microfiber = microfiber_words;
  low_microfiber[roughness_word] =
      std::bit_cast<std::uint32_t>(-0.25f);
  auto low_ashikhmin = ashikhmin_words;
  low_ashikhmin[roughness_word] = std::bit_cast<std::uint32_t>(-0.25f);

  auto table = device.create_buffer<float>(table_values.size());
  auto microfiber =
      device.create_buffer<std::uint32_t>(microfiber_words.size());
  auto ashikhmin =
      device.create_buffer<std::uint32_t>(ashikhmin_words.size());
  auto low_micro =
      device.create_buffer<std::uint32_t>(low_microfiber.size());
  auto low_velvet =
      device.create_buffer<std::uint32_t>(low_ashikhmin.size());
  auto mixed =
      device.create_buffer<std::uint32_t>(mixed_microfiber_words.size());
  auto output =
      device.create_buffer<luisa::float4>(scenario_count * output_stride);
  auto meta = device.create_buffer<std::uint32_t>(scenario_count * meta_stride);
  const auto options =
      ShaderOption{.enable_cache = false, .enable_fast_math = false};
  auto full_shader = device.compile(transition_kernel<8u>(), options);
  auto empty_shader = device.compile(transition_kernel<0u>(), options);

  std::array<luisa::float4, scenario_count * output_stride> actual{};
  std::array<std::uint32_t, scenario_count * meta_stride> actual_meta{};
  stream << table.copy_from(luisa::span{table_values})
         << microfiber.copy_from(microfiber_words.data())
         << ashikhmin.copy_from(ashikhmin_words.data())
         << low_micro.copy_from(low_microfiber.data())
         << low_velvet.copy_from(low_ashikhmin.data())
         << mixed.copy_from(mixed_microfiber_words.data())
         << full_shader(microfiber, table, output, meta, 0u, true).dispatch(1u)
         << full_shader(ashikhmin, table, output, meta, 1u, true).dispatch(1u)
         << full_shader(microfiber, table, output, meta, 2u, false).dispatch(1u)
         << full_shader(low_micro, table, output, meta, 3u, true).dispatch(1u)
         << full_shader(low_velvet, table, output, meta, 4u, true).dispatch(1u)
         << full_shader(mixed, table, output, meta, 5u, true).dispatch(1u)
         << empty_shader(microfiber, table, output, meta, 6u, true).dispatch(1u)
         << output.copy_to(actual.data()) << meta.copy_to(actual_meta.data())
         << synchronize();

  auto value = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual[scenario * output_stride + field];
  };
  auto state = [&](std::uint32_t scenario, std::uint32_t field) {
    return actual_meta[scenario * meta_stride + field];
  };
  constexpr auto expected_flags =
      device_svm::shader_data_bsdf | device_svm::shader_data_bsdf_has_eval;
  constexpr auto ended =
      static_cast<std::uint32_t>(device_svm::EvaluationStatus::ended);
  const auto up = luisa::float3{0.0f, 0.0f, 1.0f};
  const auto basis_t = luisa::float3{0.7071067812f, -0.7071067812f, 0.0f};
  const auto basis_b = luisa::float3{0.7071067812f, 0.7071067812f, 0.0f};
  const auto external_microfiber_weight =
      luisa::float3{0.00288336398f, 0.00584260561f, 0.00121404789f};
  constexpr auto external_microfiber_sample = 0.00331333932f;

  auto valid =
      state(0u, 0u) == static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID) &&
      state(0u, 1u) == 1u && state(0u, 2u) == 7u &&
      state(0u, 3u) == expected_flags && state(0u, 4u) == ended &&
      state(0u, 5u) == 17u &&
      near(value(0u, 0u).xyz(), external_microfiber_weight) &&
      near(value(0u, 0u).w, external_microfiber_sample) &&
      near(value(0u, 1u).xyz(), up) && near(value(0u, 2u).x, 0.43f) &&
      std::abs(value(0u, 2u).y) >= 1.0e-5f &&
      std::isfinite(value(0u, 2u).z) &&
      near(value(0u, 3u).xyz(), basis_t) &&
      near(value(0u, 4u).xyz(), basis_b);

  valid &=
      state(1u, 0u) ==
          static_cast<std::uint32_t>(CLOSURE_BSDF_ASHIKHMIN_VELVET_ID) &&
      state(1u, 1u) == 1u && state(1u, 2u) == 7u &&
      state(1u, 3u) == expected_flags && state(1u, 4u) == ended &&
      state(1u, 5u) == 17u &&
      near(value(1u, 0u).xyz(), luisa::float3{0.82f, 0.19f, 0.57f}) &&
      near(value(1u, 0u).w, 0.526666641f) &&
      near(value(1u, 1u).xyz(), up) && near(value(1u, 2u).x, 0.24f) &&
      near(value(1u, 2u).y, 1.0f / (0.24f * 0.24f));

  /* Cycles consumes the allocated slot before discovering an invalid LTC.
   * Its input weight and typed setup fields remain observable; only the type,
   * sample weight, and BSDF flags are cleared. */
  valid &= state(2u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(2u, 1u) == 1u && state(2u, 2u) == 7u &&
           state(2u, 3u) == 0u && state(2u, 4u) == ended &&
           state(2u, 5u) == 17u &&
           near(value(2u, 0u).xyz(), luisa::float3{0.38f, 0.77f, 0.16f}) &&
           near(value(2u, 0u).w, 0.0f) &&
           near(value(2u, 2u).x, 0.43f) &&
           near(value(2u, 2u).y, 0.0f) &&
           near(value(2u, 2u).z, 0.0f) &&
           near(value(2u, 3u).xyz(), basis_t) &&
           near(value(2u, 4u).xyz(), basis_b);

  /* Saturation first maps the authored negative roughness to zero, then
   * bsdf_sheen_setup canonicalizes it to 1e-3. Cycles' real LTC endpoint is
   * invalid there, so the canonicalized payload remains in a consumed NONE
   * slot with the unscaled input weight. */
  valid &= state(3u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(3u, 1u) == 1u && state(3u, 2u) == 7u &&
           state(3u, 3u) == 0u &&
           near(value(3u, 0u).xyz(),
                luisa::float3{0.38f, 0.77f, 0.16f}) &&
           near(value(3u, 0u).w, 0.0f) &&
           near(value(3u, 2u).x, 1.0e-3f) &&
           near(value(3u, 2u).y, 0.0f) &&
           near(value(3u, 2u).z, 0.0f);

  valid &= state(4u, 0u) ==
               static_cast<std::uint32_t>(
                   CLOSURE_BSDF_ASHIKHMIN_VELVET_ID) &&
           state(4u, 3u) == expected_flags &&
           near(value(4u, 2u).x, 0.0f) &&
           near(value(4u, 2u).y, 10000.0f);

  valid &=
      state(5u, 0u) == static_cast<std::uint32_t>(CLOSURE_BSDF_SHEEN_ID) &&
      state(5u, 1u) == 1u && state(5u, 2u) == 7u &&
      state(5u, 3u) == expected_flags && state(5u, 4u) == ended &&
      state(5u, 5u) == 20u &&
      near(value(5u, 0u).xyz(), external_microfiber_weight * 0.25f) &&
      near(value(5u, 0u).w, external_microfiber_sample * 0.25f);

  valid &= state(6u, 0u) == static_cast<std::uint32_t>(CLOSURE_NONE_ID) &&
           state(6u, 1u) == 0u && state(6u, 2u) == 0u &&
           state(6u, 3u) == 0u && state(6u, 4u) == ended &&
           state(6u, 5u) == 17u;

  if (!valid) {
    std::cerr << "Cycles standalone Sheen transition mismatch on " << backend
              << '\n';
    for (auto scenario = 0u; scenario < scenario_count; scenario++) {
      const auto common = value(scenario, 0u);
      const auto payload = value(scenario, 2u);
      std::cerr << "scenario " << scenario << ": type="
                << state(scenario, 0u) << ", count=" << state(scenario, 1u)
                << ", left=" << state(scenario, 2u)
                << ", flags=" << state(scenario, 3u)
                << ", status=" << state(scenario, 4u)
                << ", offset=" << state(scenario, 5u) << ", common=("
                << common.x << ", " << common.y << ", " << common.z << ", "
                << common.w << "), payload=(" << payload.x << ", "
                << payload.y << ", " << payload.z << ")\n";
    }
  }
  return valid;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  return run(backend, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
