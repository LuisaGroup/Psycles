#include <psycles/compiler/core_nodes.h>
#include <psycles/compiler/cycles_svm_scene.h>

#include "cycles_svm_image_sampling.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
using namespace luisa::compute;
using namespace psycles::contract;
using namespace psycles::compiler;
namespace abi = psycles::compiler::cycles_svm;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace scene_detail = psycles::luisa_backend::detail;
using Binding = scene_detail::CyclesSvmImageBindingGpu;
constexpr unsigned shader_count = 48u;
constexpr unsigned oracle_count = 60u;

void require(bool value, const std::string &message) {
  if (!value) {
    throw std::runtime_error{message};
  }
}

bool within_one_ulp(float value, float expected) {
  // Fast-math alpha unassociation may use an approximate reciprocal.
  // Keep that native path; the regression checks structure, not bit parity.
  return value == expected ||
         value == std::nextafter(expected, std::numeric_limits<float>::infinity()) ||
         value == std::nextafter(expected, -std::numeric_limits<float>::infinity());
}

ShaderGraph make_graph(unsigned index) {
  constexpr std::array<const char *, 4u> extensions{"REPEAT", "EXTEND", "CLIP",
                                                    "MIRROR"};
  constexpr std::array coordinates{psycles::Vec3f{0.25f, 0.75f, 0.0f},
                                   psycles::Vec3f{-2.0f, 3.0f, 0.0f},
                                   psycles::Vec3f{0.0f, 1.0f, 0.0f}};
  ShaderGraph graph;
  const auto image = graph.add_node(node_type::image_texture, "Missing image");
  const auto emission = graph.add_node(node_type::emission, "Emission");
  const auto flags = index % 4u;
  require(
      graph.set_input(image, "Vector",
                      SocketValue::vector(coordinates[(index / 4u) % 3u])) &&
          graph.set_property(image, "Image",
                             SocketValue::unsigned_integer(index / 12u + 1u)) &&
          graph.set_property(image, "Interpolation",
                             SocketValue::string("Linear")) &&
          graph.set_property(image, "Extension",
                             SocketValue::string(extensions[index / 12u])) &&
          graph.set_property(image, "Projection",
                             SocketValue::string("FLAT")) &&
          graph.set_property(image, "ProjectionBlend",
                             SocketValue::floating(0.0f)) &&
          graph.set_property(
              image, "ColorSpace",
              SocketValue::string((flags & 1u) ? "sRGB" : "Non-Color")) &&
          graph.set_property(image, "UnassociateAlpha",
                             SocketValue::boolean((flags & 2u) != 0u)) &&
          graph.connect({image, "Color"}, emission, "Color") &&
          graph.connect({image, "Alpha"}, emission, "Strength"),
      "cannot construct failed-image shader");
  graph.set_root(ShaderDomain::surface, OutputRef{emission, "Closure"});
  return graph;
}

class ImageGlobals final
    : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  Expr<BindlessArray> _textures;
  Expr<Buffer<Binding>> _bindings;

public:
  ImageGlobals(Expr<BindlessArray> textures, Expr<Buffer<Binding>> bindings)
      : _textures{textures}, _bindings{bindings} {}
  Float4
  kernel_image_interp_with_udim(svm::ShaderData &, Expr<int> id,
                                const svm::Dual2 &uv) const noexcept override {
    return svm::detail::sample_scene_image_2d(_textures, _bindings, id, uv);
  }
};
} // namespace

int main(int argc, char **argv) {
  try {
    const std::string_view backend{argc > 1 ? argv[1] : "fallback"};
    Context context{argv[0]};
    auto device = context.create_device(backend);
    auto stream = device.create_stream();
    std::array<ShaderCompilation, shader_count> programs;
    std::array<abi::ShaderTableCompileUnit, shader_count> units;
    const ShaderCompiler compiler{make_core_node_registry()};
    for (auto i = 0u; i < shader_count; ++i) {
      programs[i] = compiler.compile(make_graph(i));
      require(programs[i].ok(), "failed-image graph did not normalize");
      units[i] = {.shader_index = i, .shader = programs[i].program.get()};
    }
    const auto table = abi::compile_shader_table(units);
    require(table.table.valid, table.table.diagnostic);
    require(table.images.size() == 4u,
            "distinct failed image identities were collapsed");
    std::vector<Binding> bindings;
    for (const auto &image : table.images) {
      // Deliberately unbound texture addresses: failed sampling must never
      // attempt a hardware texture access, even for cubic/CLIP dispatch.
      bindings.push_back(scene_detail::make_cycles_svm_image_binding(
          1000u + static_cast<unsigned>(image.resource_id), image.interpolation,
          image.extension, true));
    }
    bindings.push_back(scene_detail::make_cycles_svm_image_binding(
        0u, abi::ImageInterpolation::linear, abi::ImageExtension::clip));
    auto textures = device.create_bindless_array(1u);
    auto image = device.create_image<float>(PixelStorage::FLOAT4, 1u, 1u);
    textures.emplace_on_update(0u, image, Sampler::linear_point_zero());
    const std::array pixel{luisa::float4{1.0f, 0.0f, 1.0f, 1.0f}};
    auto binding_buffer = device.create_buffer<Binding>(bindings.size());
    auto words = device.create_buffer<unsigned>(table.table.words.size());
    auto output = device.create_buffer<luisa::float4>(oracle_count + 1u);
    auto metadata = device.create_buffer<luisa::uint2>(shader_count);
    Kernel1D<BindlessArray, Buffer<Binding>, Buffer<unsigned>,
             Buffer<luisa::float4>, Buffer<luisa::uint2>>
        kernel = [used = table.table.node_types_used,
                  features = table.kernel_features,
                  stack_size = table.table.peak_stack_usage](
                     BindlessVar textures, BufferVar<Binding> bindings,
                     BufferUInt words, BufferFloat4 output,
                     BufferVar<luisa::uint2> metadata) {
          const UInt index = dispatch_x();
          $if(index < shader_count) {
            const auto identity = make_float4x4(1.0f);
            const auto normal = make_float3(0.0f, 0.0f, 1.0f);
            svm::ShaderData sd{make_float3(0.0f),
                               normal,
                               normal,
                               normal,
                               svm::primitive_triangle,
                               index,
                               0u,
                               0u,
                               0u,
                               0.0f,
                               0.0f,
                               0u,
                               0.5f,
                               1.0f,
                               0.0f,
                               0.0f,
                               0.0f,
                               0.0f,
                               0.0f,
                               0.0f,
                               normal,
                               normal,
                               identity,
                               identity};
            const ImageGlobals globals{textures, bindings};
            const svm::TransformState transforms{identity, identity, identity,
                                                 identity};
            const svm::PathState state{0u, svm::path_ray_emission};
            svm::EvaluationResult result;
            svm::eval_nodes(globals, words, abi::SHADER_TYPE_SURFACE, features,
                            svm::kernel_feature_node_mask_surface_light, used,
                            transforms, sd, state, result, stack_size);
            output.write(index,
                         make_float4(sd.closure_emission_background, 1.0f));
            metadata.write(index,
                           make_uint2(result.status, result.final_offset));
          }
          $else {
            const svm::Dual2 uv{.val = make_float2(-2.0f, 3.0f),
                                .dx = make_float2(0.0f),
                                .dy = make_float2(0.0f)};
            const Int image_id = select(std::numeric_limits<int>::max(), 4,
                                        index == oracle_count);
            output.write(index, svm::detail::sample_scene_image_2d(
                                    textures, bindings, image_id, uv));
          };
        };
    auto shader = device.compile(
        kernel, ShaderOption{.enable_cache = false, .enable_fast_math = true});
    std::array<luisa::float4, oracle_count + 1u> actual;
    std::array<luisa::uint2, shader_count> states;
    stream << image.copy_from(pixel.data()) << textures.update()
           << words.copy_from(table.table.words.data());
    // Original failed-image sampling returns before it can inspect the
    // interpolation mode. Exercise all four immutable sampler choices with
    // the same original-Cycles outputs and the same compiled word image.
    for (const auto interpolation : std::array{
             abi::ImageInterpolation::closest, abi::ImageInterpolation::linear,
             abi::ImageInterpolation::cubic, abi::ImageInterpolation::smart}) {
      for (auto i = 0u; i < table.images.size(); ++i) {
        bindings[i] = scene_detail::make_cycles_svm_image_binding(
            1000u + static_cast<unsigned>(table.images[i].resource_id),
            interpolation, table.images[i].extension, true);
      }
      stream << binding_buffer.copy_from(bindings.data())
             << shader(textures, binding_buffer, words, output, metadata)
                    .dispatch(oracle_count + 1u)
             << output.copy_to(actual.data()) << metadata.copy_to(states.data())
             << synchronize();
      std::ifstream oracle{PSYCLES_MISSING_IMAGE_ORACLE};
      for (auto i = 0u; i < oracle_count; ++i) {
        unsigned ordinal;
        luisa::float4 expected;
        require(bool(oracle >> ordinal >> expected.x >> expected.y >>
                     expected.z >> expected.w) &&
                    ordinal == i,
                "invalid original Cycles failed-image oracle");
        const auto value = actual[i];
        std::ostringstream diagnostic;
        diagnostic << "failed-image result differs from original Cycles case "
                   << i << ": actual " << std::hexfloat << value.x << ','
                   << value.y << ',' << value.z << ',' << value.w << " expected "
                   << expected.x << ',' << expected.y << ',' << expected.z << ','
                   << expected.w;
        require(within_one_ulp(value.x, expected.x) &&
                    within_one_ulp(value.y, expected.y) &&
                    within_one_ulp(value.z, expected.z) &&
                    within_one_ulp(value.w, expected.w),
                diagnostic.str());
        if (i < shader_count) {
          require(states[i].x ==
                          static_cast<unsigned>(svm::EvaluationStatus::ended) &&
                      states[i].y == table.table.words[i * 4u + 2u],
                  "failed-image full SVM stream has wrong PC/status");
        }
      }
      const auto control = actual.back();
      require(
          control.x == 0.0f && control.y == 0.0f && control.z == 0.0f &&
              control.w == 0.0f,
          "a loaded CLIP texture was incorrectly treated as a failed image");
    }
    std::cout << "Original Cycles failed-image state passed on " << backend
              << '\n';
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
