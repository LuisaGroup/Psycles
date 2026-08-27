#include "path_tracer_bsdf_tables.h"
#include "thin_film_fresnel.h"

#include "luisa_surface_test_support.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::contract;
using namespace psycles::luisa_backend;
using namespace psycles::luisa_backend::detail;
using psycles::test_support::approximately_equal;
using psycles::test_support::compile_named_kernel;

class TableShaderServices final : public ShaderServices {

private:
  const BufferFloat &_table;

public:
  explicit TableShaderServices(const BufferFloat &table) noexcept
      : _table{table} {}

  [[nodiscard]] Float
  parameter_float(Expr<std::uint32_t>,
                  Expr<std::uint32_t>) const noexcept override {
    return 0.0f;
  }

  [[nodiscard]] Float3
  parameter_float3(Expr<std::uint32_t>,
                   Expr<std::uint32_t>) const noexcept override {
    return make_float3(0.0f);
  }

  [[nodiscard]] ULong
  parameter_uint64(Expr<std::uint32_t>,
                   Expr<std::uint32_t>) const noexcept override {
    return 0u;
  }

  [[nodiscard]] Float4 texture_2d(Expr<std::uint32_t>, Expr<luisa::float2>,
                                  Expr<luisa::float2>, Expr<luisa::float2>,
                                  std::uint32_t,
                                  std::uint32_t) const noexcept override {
    return make_float4(0.0f);
  }

  [[nodiscard]] ShaderAttribute
  attribute(Expr<luisa::ulong>, const SurfacePoint &) const noexcept override {
    return ShaderAttribute::missing();
  }

  [[nodiscard]] Float
  cycles_bsdf_data(Expr<std::uint32_t> index) const noexcept override {
    return _table->read(index);
  }

  [[nodiscard]] Float3
  xyz_to_rgb(Expr<luisa::float3> value) const noexcept override {
    return Float3{value};
  }

  [[nodiscard]] Float3
  rec709_to_rgb(Expr<luisa::float3> value) const noexcept override {
    return Float3{value};
  }

  [[nodiscard]] Float3 nishita_sky(Expr<std::uint32_t>, std::uint32_t,
                                   Expr<luisa::float3>, Expr<float>,
                                   Expr<float>, Expr<float>,
                                   Expr<float>) const noexcept override {
    return make_float3(0.0f);
  }
};

struct ThinFilmCase {
  luisa::float4 optical;
  luisa::float3 dielectric_f0;
  luisa::float3 metallic_f0;
  luisa::float3 metallic_b;
  luisa::float4 expected_dielectric;
  luisa::float3 expected_metallic;
};

// These values are emitted by the Blender 5.2 Cycles kernel itself at source
// revision 9e2066aef7ef7e20c142ad7bd3303138a4304c93. The oracle directly calls
// generalized_schlick_fresnel() and microfacet_fresnel() with the production
// 512x6 table; it is not an independently maintained host Fresnel model.
constexpr std::array cases{
    ThinFilmCase{{0.1001f, 1.33f, 1.5f, 1.0f},
                 {0.02f, 0.04f, 0.08f},
                 {0.80f, 0.40f, 0.10f},
                 {3.15088773f, 5.98027658f, 7.29211044f},
                 {0.0199998431f, 0.0399995707f, 0.0799986795f, -1.0f},
                 {0.799999237f, 0.399996728f, 0.0999967828f}},
    ThinFilmCase{{250.0f, 1.33f, 1.5f, 0.8f},
                 {0.08f, 0.04f, 0.02f},
                 {0.72f, 0.31f, 0.08f},
                 {10.4969635f, 3.33207846f, 0.892544568f},
                 {0.0879736319f, 0.0415383764f, 0.0085341651f, -0.916515172f},
                 {0.721437275f, 0.273250937f, 0.00750176283f}},
    ThinFilmCase{{400.0f, 1.33f, 1.5f, 0.1f},
                 {0.10f, 0.50f, 0.90f},
                 {0.95f, 0.62f, 0.22f},
                 {1.71771514f, 8.42831516f, 9.22797394f},
                 {0.954137385f, 3.51393056f, 6.10428381f, -0.748331487f},
                 {0.976734161f, 0.782455146f, 0.553069115f}},
    ThinFilmCase{{1000.0f, 1.33f, 1.5f, -0.5f},
                 {0.65f, 0.12f, 0.03f},
                 {0.61f, 0.44f, 0.19f},
                 {8.37140656f, 4.93597603f, 1.99375236f},
                 {0.755036831f, 0.182022080f, 0.0394877233f, -0.816496611f},
                 {0.471261263f, 0.375289261f, 0.138133168f}},
    ThinFilmCase{{250.0f, 1.33f, 0.66f, 0.5f},
                 {0.15f, 0.35f, 0.75f},
                 {0.33f, 0.66f, 0.91f},
                 {5.64831114f, 7.21328974f, 8.39887905f},
                 {1.0f, 1.0f, 1.0f, 0.0f},
                 {0.318843365f, 0.647480011f, 0.900322080f}},
    ThinFilmCase{{250.0f, 0.70f, 1.5f, 0.1f},
                 {0.03f, 0.25f, 0.85f},
                 {0.88f, 0.57f, 0.12f},
                 {13.2105732f, 2.71459079f, 5.58289957f},
                 {1.0f, 1.0f, 1.0f, 0.0f},
                 {1.0f, 1.0f, 1.0f}},
};

constexpr std::uint32_t case_stride = 4u;
constexpr std::uint32_t output_stride = 2u;

[[nodiscard]] bool finite(luisa::float4 value) noexcept {
  return std::isfinite(value.x) && std::isfinite(value.y) &&
         std::isfinite(value.z) && std::isfinite(value.w);
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "fallback"};
  Kernel1D evaluate = [](BufferFloat table, BufferFloat4 inputs,
                         BufferFloat4 output) noexcept {
    const auto index = dispatch_x();
    const auto base = index * case_stride;
    const auto optical = inputs.read(base);
    const auto dielectric_f0 = inputs.read(base + 1u).xyz();
    const auto metallic_f0 = inputs.read(base + 2u).xyz();
    const auto metallic_b = inputs.read(base + 3u).xyz();
    const TableShaderServices services{table};
    const auto dielectric = thin_film_dielectric_fresnel(
        services, optical.x, optical.y, optical.z, dielectric_f0, optical.w);
    const auto metallic = thin_film_f82_fresnel(
        services, optical.x, optical.y, metallic_f0, metallic_b, optical.w);
    output.write(
        index * output_stride,
        make_float4(dielectric.reflectance, dielectric.cosine_transmitted));
    output.write(index * output_stride + 1u, make_float4(metallic, 0.0f));
  };

  std::array<luisa::float4, cases.size() * case_stride> inputs{};
  for (auto i = std::size_t{0u}; i < cases.size(); ++i) {
    inputs[i * case_stride] = cases[i].optical;
    inputs[i * case_stride + 1u] =
        luisa::float4{cases[i].dielectric_f0.x, cases[i].dielectric_f0.y,
                      cases[i].dielectric_f0.z, 0.0f};
    inputs[i * case_stride + 2u] =
        luisa::float4{cases[i].metallic_f0.x, cases[i].metallic_f0.y,
                      cases[i].metallic_f0.z, 0.0f};
    inputs[i * case_stride + 3u] =
        luisa::float4{cases[i].metallic_b.x, cases[i].metallic_b.y,
                      cases[i].metallic_b.z, 0.0f};
  }

  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  const auto table = make_cycles_bsdf_table_values(ShaderColorSpace{});
  auto table_buffer = device.create_buffer<float>(table.size());
  auto input_buffer = device.create_buffer<luisa::float4>(inputs.size());
  auto output_buffer =
      device.create_buffer<luisa::float4>(cases.size() * output_stride);
  auto shader =
      compile_named_kernel(device, "cycles_5_2_thin_film_fresnel", evaluate);
  std::array<luisa::float4, cases.size() * output_stride> actual{};
  stream << table_buffer.copy_from(luisa::span{table})
         << input_buffer.copy_from(luisa::span{inputs})
         << shader(table_buffer, input_buffer, output_buffer)
                .dispatch(cases.size())
         << output_buffer.copy_to(luisa::span{actual}) << synchronize();

  constexpr float tolerance = 8.0e-6f;
  for (auto i = std::size_t{0u}; i < cases.size(); ++i) {
    const auto expected_dielectric = cases[i].expected_dielectric;
    const auto expected_metallic = luisa::float4{
        cases[i].expected_metallic.x, cases[i].expected_metallic.y,
        cases[i].expected_metallic.z, 0.0f};
    const auto actual_dielectric = actual[i * output_stride];
    const auto actual_metallic = actual[i * output_stride + 1u];
    if (!finite(actual_dielectric) || !finite(actual_metallic) ||
        !approximately_equal(actual_dielectric, expected_dielectric,
                             tolerance) ||
        !approximately_equal(actual_metallic, expected_metallic, tolerance)) {
      std::cerr << "Cycles 5.2 thin-film mismatch on " << backend << ", case "
                << i << ": dielectric={" << actual_dielectric.x << ", "
                << actual_dielectric.y << ", " << actual_dielectric.z << ", "
                << actual_dielectric.w << "}, expected={"
                << expected_dielectric.x << ", " << expected_dielectric.y
                << ", " << expected_dielectric.z << ", "
                << expected_dielectric.w << "}; metallic={" << actual_metallic.x
                << ", " << actual_metallic.y << ", " << actual_metallic.z
                << "}, expected={" << expected_metallic.x << ", "
                << expected_metallic.y << ", " << expected_metallic.z << "}\n";
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}
