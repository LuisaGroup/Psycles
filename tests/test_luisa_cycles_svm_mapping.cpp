#include <psycles/luisa/cycles_svm.h>
#include <psycles/compiler/cycles_svm_bytecode.h>

#include "cycles_svm_internal.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

#include <luisa/luisa-compute.h>

namespace {

using namespace luisa::compute;
using namespace psycles::compiler::cycles_svm;
namespace device_svm = psycles::luisa_backend::cycles_svm;
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

constexpr auto input_offset = std::uint32_t{0u};
constexpr auto output_offset = std::uint32_t{16u};
constexpr auto mapping_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeMapping) / sizeof(std::uint32_t));
constexpr auto texture_mapping_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeTextureMapping) / sizeof(std::uint32_t));
constexpr auto min_max_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeMinMax) / sizeof(std::uint32_t));
constexpr auto vector_math_word_count = static_cast<std::uint32_t>(
    sizeof(SVMNodeVectorMath) / sizeof(std::uint32_t));

template<typename T>
void append_payload(std::vector<std::uint32_t> &words,
                    const T &payload) {
  const auto encoded =
      std::bit_cast<std::array<std::uint32_t,
                               sizeof(T) / sizeof(std::uint32_t)>>(payload);
  words.insert(words.end(), encoded.begin(), encoded.end());
}

[[nodiscard]] bool near(float actual, float expected,
                        float tolerance = 8.0e-5f) noexcept {
  return std::abs(actual - expected) <= tolerance;
}

[[nodiscard]] bool near(luisa::float3 actual, luisa::float3 expected,
                        float tolerance = 8.0e-5f) noexcept {
  return near(actual.x, expected.x, tolerance) &&
         near(actual.y, expected.y, tolerance) &&
         near(actual.z, expected.z, tolerance);
}

[[nodiscard]] auto mapping_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float3>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat3 output,
         BufferUInt cursors) noexcept {
        const UInt index = dispatch_x();
        svm_detail::Stack plain_stack;
        svm_detail::stack_store_float3(
            plain_stack, input_offset,
            make_float3(0.17f, 0.31f, 0.47f));
        UInt plain_cursor = index * mapping_word_count;
        svm_detail::Cursor plain_words{words, plain_cursor};
        svm_detail::node_mapping(plain_words, plain_stack, false);
        output.write(index * 4u,
                     svm_detail::stack_load_float3(plain_stack,
                                                   output_offset));

        svm_detail::Stack dual_stack;
        svm_detail::stack_store_dual3(
            dual_stack, input_offset,
            {.val = make_float3(0.17f, 0.31f, 0.47f),
             .dx = make_float3(0.013f, -0.021f, 0.007f),
             .dy = make_float3(-0.017f, 0.009f, 0.023f)});
        UInt dual_cursor = index * mapping_word_count;
        svm_detail::Cursor dual_words{words, dual_cursor};
        svm_detail::node_mapping(dual_words, dual_stack, true);
        output.write(index * 4u + 1u,
                     svm_detail::stack_load_float3(dual_stack,
                                                   output_offset));
        output.write(index * 4u + 2u,
                     svm_detail::stack_load_float3(dual_stack,
                                                   output_offset + 3u));
        output.write(index * 4u + 3u,
                     svm_detail::stack_load_float3(dual_stack,
                                                   output_offset + 6u));
        cursors.write(index * 2u, plain_cursor - index * mapping_word_count);
        cursors.write(index * 2u + 1u,
                      dual_cursor - index * mapping_word_count);
      }};
}

[[nodiscard]] auto texture_mapping_kernel() {
  return Kernel1D<Buffer<std::uint32_t>, Buffer<luisa::float3>,
                  Buffer<std::uint32_t>>{
      [](BufferUInt words, BufferFloat3 output,
         BufferUInt cursors) noexcept {
        svm_detail::Stack stack;
        svm_detail::stack_store_dual3(
            stack, input_offset,
            {.val = make_float3(0.17f, 0.31f, 0.47f),
             .dx = make_float3(0.013f, -0.021f, 0.007f),
             .dy = make_float3(-0.017f, 0.009f, 0.023f)});
        UInt cursor_offset = 0u;
        svm_detail::Cursor cursor{words, cursor_offset};
        svm_detail::node_texture_mapping(cursor, stack, true);
        output.write(0u, svm_detail::stack_load_float3(stack, output_offset));
        output.write(1u,
                     svm_detail::stack_load_float3(stack, output_offset + 3u));
        output.write(2u,
                     svm_detail::stack_load_float3(stack, output_offset + 6u));
        svm_detail::node_min_max(cursor, stack);
        output.write(3u, svm_detail::stack_load_float3(stack, output_offset));
        output.write(4u,
                     svm_detail::stack_load_float3(stack, output_offset + 3u));
        output.write(5u,
                     svm_detail::stack_load_float3(stack, output_offset + 6u));
        cursors.write(0u, cursor_offset);

        svm_detail::Stack normalize_stack;
        svm_detail::stack_store_dual3(
            normalize_stack, input_offset,
            {.val = make_float3(3.0f, 4.0f, 0.0f),
             .dx = make_float3(1.0f, 0.0f, 0.0f),
             .dy = make_float3(0.0f, 1.0f, 0.0f)});
        UInt normalize_offset =
            texture_mapping_word_count + min_max_word_count;
        svm_detail::Cursor normalize_cursor{words, normalize_offset};
        svm_detail::node_vector_math(normalize_cursor, normalize_stack, true);
        output.write(6u,
                     svm_detail::stack_load_float3(normalize_stack,
                                                   output_offset));
        output.write(7u,
                     svm_detail::stack_load_float3(normalize_stack,
                                                   output_offset + 3u));
        output.write(8u,
                     svm_detail::stack_load_float3(normalize_stack,
                                                   output_offset + 6u));
        cursors.write(1u, normalize_offset);
      }};
}

[[nodiscard]] bool test_mapping(Device &device, Stream &stream,
                                std::string_view backend) {
  std::vector<std::uint32_t> words;
  for (const auto type : {NODE_MAPPING_TYPE_POINT, NODE_MAPPING_TYPE_TEXTURE,
                          NODE_MAPPING_TYPE_VECTOR,
                          NODE_MAPPING_TYPE_NORMAL}) {
    append_payload(
        words,
        SVMNodeMapping{
            .mapping_type = type,
            .vector = input_float3(
                static_cast<SVMStackOffset>(input_offset)),
            .location = input_float3(0.05f, 0.04f, 0.03f),
            .rotation = input_float3(0.27f, -0.19f, 0.33f),
            .scale = input_float3(1.1f, 0.8f, 1.3f),
            .result_offset = static_cast<SVMStackOffset>(output_offset),
            ._pad = {0u, 0u, 0u}});
  }
  static constexpr std::array expected{
      std::array{luisa::float3{0.0820550483f, 0.131359007f, 0.708542560f},
                 luisa::float3{0.0185518315f, -0.0133260583f, 0.0069127213f},
                 luisa::float3{-0.0225286543f, -0.00881183904f,
                               0.0266524151f}},
      std::array{luisa::float3{0.254997421f, 0.392280840f, 0.247753715f},
                 luisa::float3{0.00610617887f, -0.0270628653f,
                               0.00926753855f},
                 luisa::float3{-0.00780512707f, 0.0252534235f,
                               0.0157107007f}},
      std::array{luisa::float3{0.0320550483f, 0.0913590073f, 0.678542560f},
                 luisa::float3{0.0185518315f, -0.0133260583f, 0.0069127213f},
                 luisa::float3{-0.0225286543f, -0.00881183904f,
                               0.0266524151f}},
      std::array{luisa::float3{-0.0487564127f, 0.513744392f, 0.856556777f},
                 luisa::float3{0.0351446111f, -0.0280165197f, 0.0188041884f},
                 luisa::float3{-0.0347535541f, -0.0147433493f,
                               0.00686452386f}}};

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float3>(16u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(8u);
  auto shader = device.compile(
      mapping_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float3, 16u> output{};
  std::array<std::uint32_t, 8u> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer).dispatch(4u)
         << output_buffer.copy_to(luisa::span{output})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();
  for (auto type = std::size_t{}; type < expected.size(); ++type) {
    if (!near(output[type * 4u], expected[type][0u]) ||
        !near(output[type * 4u + 1u], expected[type][0u]) ||
        !near(output[type * 4u + 2u], expected[type][1u]) ||
        !near(output[type * 4u + 3u], expected[type][2u]) ||
        cursors[type * 2u] != mapping_word_count ||
        cursors[type * 2u + 1u] != mapping_word_count) {
      std::cerr << "Cycles Mapping value/dual/cursor mismatch for type "
                << type << " on " << backend << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool test_texture_mapping(Device &device, Stream &stream,
                                        std::string_view backend) {
  std::vector<std::uint32_t> words;
  append_payload(
      words,
      SVMNodeTextureMapping{
          .vec_offset = static_cast<SVMStackOffset>(input_offset),
          .out_offset = static_cast<SVMStackOffset>(output_offset),
          ._pad = {0u, 0u},
          .tfm = {.x = {std::bit_cast<float>(0x3f1c1575u),
                        std::bit_cast<float>(0xbe916e85u),
                        std::bit_cast<float>(0x3d5d8cb2u),
                        std::bit_cast<float>(0x3e051eb8u)},
                  .y = {std::bit_cast<float>(0x3e122f18u),
                        std::bit_cast<float>(0x3f8f1488u),
                        std::bit_cast<float>(0x3e1d1728u),
                        std::bit_cast<float>(0xbe570a3du)},
                  .z = {std::bit_cast<float>(0x3d8da3eeu),
                        std::bit_cast<float>(0x3e49780bu),
                        std::bit_cast<float>(0xbf4b22bbu),
                        std::bit_cast<float>(0x3ebd70a4u)}}});
  append_payload(words,
                 SVMNodeMinMax{.vec_offset =
                                   static_cast<SVMStackOffset>(output_offset),
                               .out_offset =
                                   static_cast<SVMStackOffset>(output_offset),
                               ._pad = {0u, 0u},
                               .mn = {-0.1f, 0.2f, 0.0f},
                               .mx = {0.15f, 0.23f, 0.08f}});
  append_payload(
      words,
      SVMNodeVectorMath{
          .math_type = NODE_VECTOR_MATH_NORMALIZE,
          .a = input_float3(static_cast<SVMStackOffset>(input_offset)),
          .b = {},
          .c = {},
          .param1 = {},
          .value_offset = SVM_STACK_INVALID,
          .vector_offset = static_cast<SVMStackOffset>(output_offset),
          ._pad = {0u, 0u}});

  auto word_buffer = device.create_buffer<std::uint32_t>(words.size());
  auto output_buffer = device.create_buffer<luisa::float3>(9u);
  auto cursor_buffer = device.create_buffer<std::uint32_t>(2u);
  auto shader = device.compile(
      texture_mapping_kernel(),
      ShaderOption{.enable_cache = false, .enable_fast_math = false});
  std::array<luisa::float3, 9u> output{};
  std::array<std::uint32_t, 2u> cursors{};
  stream << word_buffer.copy_from(luisa::span{words})
         << shader(word_buffer, output_buffer, cursor_buffer).dispatch(1u)
         << output_buffer.copy_to(luisa::span{output})
         << cursor_buffer.copy_to(luisa::span{cursors}) << synchronize();
  static constexpr std::array expected{
      luisa::float3{0.171016995f, 0.232893252f, 0.0698044407f},
      luisa::float3{0.0142697289f, -0.0205443838f, -0.0087870934f},
      luisa::float3{-0.011677305f, 0.0111618433f, -0.0176554726f},
      luisa::float3{0.15f, 0.23f, 0.0698044407f},
      luisa::float3{0.0142697289f, -0.0205443838f, -0.0087870934f},
      luisa::float3{-0.011677305f, 0.0111618433f, -0.0176554726f},
      luisa::float3{0.6f, 0.8f, 0.0f},
      luisa::float3{0.128f, -0.096f, 0.0f},
      luisa::float3{-0.096f, 0.072f, 0.0f}};
  for (auto index = std::size_t{}; index < expected.size(); ++index) {
    if (!near(output[index], expected[index])) {
      std::cerr << "Cycles TextureMapping result " << index
                << " differs on " << backend << '\n';
      return false;
    }
  }
  const auto mapping_and_minmax_words =
      texture_mapping_word_count + min_max_word_count;
  const auto normalize_end =
      mapping_and_minmax_words + vector_math_word_count;
  if (cursors[0u] != mapping_and_minmax_words ||
      cursors[1u] != normalize_end) {
    std::cerr << "Cycles TextureMapping cursor mismatch on " << backend
              << '\n';
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  const auto backend = std::string_view{argc > 1 ? argv[1] : "hip"};
  Context context{argv[0]};
  auto device = context.create_device(backend);
  auto stream = device.create_stream();
  return test_mapping(device, stream, backend) &&
                 test_texture_mapping(device, stream, backend)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
