#include <psycles/compiler/cycles_svm_bytecode.h>
#include <psycles/luisa/cycles_svm.h>

#include "cycles_svm_internal.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

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
namespace svm_detail = psycles::luisa_backend::cycles_svm::detail;

[[nodiscard]] constexpr std::uint32_t float_bits(float value) noexcept {
  return std::bit_cast<std::uint32_t>(value);
}

[[nodiscard]] constexpr std::uint32_t pack_bytes(std::uint32_t b0,
                                                 std::uint32_t b1 = 0u,
                                                 std::uint32_t b2 = 0u,
                                                 std::uint32_t b3 = 0u) noexcept {
  return (b0 & 0xffu) | ((b1 & 0xffu) << 8u) | ((b2 & 0xffu) << 16u) |
         ((b3 & 0xffu) << 24u);
}

[[nodiscard]] bool approx_eq(float actual, float expected, float eps = 1e-5f) noexcept {
  return std::fabs(actual - expected) <= eps;
}

bool run_nested_stack_test(Device &device, Stream &stream) {
  auto out_buf = device.create_buffer<luisa::float4>(8u);
  auto kernel = device.compile<1u>([&](BufferFloat4 out) noexcept {
    const UInt i = dispatch_x();
    auto stack = svm_detail::Stack::direct_jit(8u);

    // Root-scope literal, runtime SSA, and unwritten (implicit 0.0f) slots.
    svm_detail::stack_store_float(stack, 0u, 10.0f);
    svm_detail::stack_store_float(stack, 1u, i.cast<float>() + 100.0f);
    // Slot 2 intentionally left unwritten before nested branch.

    const Bool cond_a = (i & 1u) != 0u;
    const Bool cond_b = (i & 2u) != 0u;
    const Bool cond_c = (i & 4u) != 0u;

    $if(cond_a) {
      svm_detail::stack_store_float(stack, 0u, 20.0f);
      svm_detail::stack_store_float(stack, 1u, 200.0f);
      svm_detail::stack_store_float(stack, 2u, 30.0f);
      $if(cond_b) {
        svm_detail::stack_store_float(stack, 0u, 25.0f);
      };
    };

    const Float s0_after_a = svm_detail::stack_load_float(stack, 0u);
    const Float s1_after_a = svm_detail::stack_load_float(stack, 1u);
    const Float s2_after_a = svm_detail::stack_load_float(stack, 2u);

    // Overwrite slot 0 at root scope (var_expr != nullptr, use_var -> false),
    // then conditionally overwrite inside a second branch. Non-taken path must
    // preserve 77.0f rather than stale 20.0f / 25.0f from the first branch.
    svm_detail::stack_store_float(stack, 0u, 77.0f);
    $if(cond_c) {
      svm_detail::stack_store_float(stack, 0u, 88.0f);
    };
    const Float s0_after_c = svm_detail::stack_load_float(stack, 0u);

    out.write(i, make_float4(s0_after_a, s1_after_a, s2_after_a, s0_after_c));
  });

  std::vector<luisa::float4> host(8u);
  stream << kernel(out_buf).dispatch(8u) << out_buf.copy_to(host.data())
         << synchronize();

  for (std::uint32_t i = 0u; i < 8u; ++i) {
    const bool a = (i & 1u) != 0u;
    const bool b = (i & 2u) != 0u;
    const bool c = (i & 4u) != 0u;
    const float exp_s0_a = a ? (b ? 25.0f : 20.0f) : 10.0f;
    const float exp_s1_a = a ? 200.0f : (static_cast<float>(i) + 100.0f);
    const float exp_s2_a = a ? 30.0f : 0.0f;
    const float exp_s0_c = c ? 88.0f : 77.0f;
    if (!approx_eq(host[i].x, exp_s0_a) || !approx_eq(host[i].y, exp_s1_a) ||
        !approx_eq(host[i].z, exp_s2_a) || !approx_eq(host[i].w, exp_s0_c)) {
      std::cerr << "nested_stack mismatch lane " << i << ": got ("
                << host[i].x << ", " << host[i].y << ", " << host[i].z << ", "
                << host[i].w << ") expected (" << exp_s0_a << ", " << exp_s1_a
                << ", " << exp_s2_a << ", " << exp_s0_c << ")\n";
      return false;
    }
  }
  return true;
}

bool run_static_offset_nodes_test(Device &device, Stream &stream) {
  auto dummy_words = device.create_buffer<std::uint32_t>(4u);
  auto out_buf = device.create_buffer<luisa::float4>(2u);
  psycles::test_support::DefaultCyclesSvmKernelGlobals kg{};

  auto kernel = device.compile<1u>(
      [&](BufferUInt words, BufferFloat4 out) noexcept {
        auto stack = svm_detail::Stack::direct_jit(16u);
        // Sentinels on adjacent slots to detect clobbering by scalar convert.
        svm_detail::stack_store_float(stack, 5u, 123.0f);
        svm_detail::stack_store_float(stack, 6u, 456.0f);

        // Combine vector into slots 0, 1, 2 (NODE_COMBINE_VECTOR uses
        // out_off + vector_index; must stay static in DirectJit mode).
        const std::array<std::uint32_t, 2u> comb_x{
            float_bits(2.0f), pack_bytes(0u, 0u)};
        const std::array<std::uint32_t, 2u> comb_y{
            float_bits(4.0f), pack_bytes(1u, 0u)};
        const std::array<std::uint32_t, 2u> comb_z{
            float_bits(6.0f), pack_bytes(2u, 0u)};
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{comb_x}, 0u};
          svm_detail::node_combine_vector(c, stack, false);
        }
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{comb_y}, 0u};
          svm_detail::node_combine_vector(c, stack, false);
        }
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{comb_z}, 0u};
          svm_detail::node_combine_vector(c, stack, false);
        }

        // Convert vector (slot 0..2) -> scalar float at slot 4 (NODE_CONVERT_VF).
        // Must not clobber slots 5 or 6.
        const std::array<std::uint32_t, 2u> conv_vf{
            static_cast<std::uint32_t>(NODE_CONVERT_VF), pack_bytes(0u, 4u)};
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{conv_vf}, 0u};
          svm_detail::node_convert(c, stack, kg, false);
        }

        // Separate vector Y component from slot 0 into slot 7.
        const std::array<std::uint32_t, 4u> sep_y{
            SVM_INPUT_STACK_OFFSET_MASK | 0u, 0u, 0u, pack_bytes(1u, 7u)};
        {
          svm_detail::Cursor c{words, std::span<const std::uint32_t>{sep_y}, 0u};
          svm_detail::node_separate_vector(c, stack, false);
        }

        out.write(0u, make_float4(svm_detail::stack_load_float(stack, 0u),
                                  svm_detail::stack_load_float(stack, 1u),
                                  svm_detail::stack_load_float(stack, 2u),
                                  svm_detail::stack_load_float(stack, 4u)));
        out.write(1u, make_float4(svm_detail::stack_load_float(stack, 5u),
                                  svm_detail::stack_load_float(stack, 6u),
                                  svm_detail::stack_load_float(stack, 7u),
                                  0.0f));
      });

  std::vector<luisa::float4> host(2u);
  stream << kernel(dummy_words, out_buf).dispatch(1u)
         << out_buf.copy_to(host.data()) << synchronize();

  if (!approx_eq(host[0].x, 2.0f) || !approx_eq(host[0].y, 4.0f) ||
      !approx_eq(host[0].z, 6.0f) || !approx_eq(host[0].w, 4.0f) ||
      !approx_eq(host[1].x, 123.0f) || !approx_eq(host[1].y, 456.0f) ||
      !approx_eq(host[1].z, 4.0f)) {
    std::cerr << "static_offset_nodes mismatch: out0=(" << host[0].x << ", "
              << host[0].y << ", " << host[0].z << ", " << host[0].w
              << ") out1=(" << host[1].x << ", " << host[1].y << ", "
              << host[1].z << ")\n";
    return false;
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  try {
    Context context{argv[0]};
    auto device = context.create_device(argc > 1 ? argv[1] : "fallback");
    auto stream = device.create_stream();
    if (!run_nested_stack_test(device, stream)) {
      return EXIT_FAILURE;
    }
    if (!run_static_offset_nodes_test(device, stream)) {
      return EXIT_FAILURE;
    }
    std::cout << "DirectJit stack & static offset tests PASSED\n";
    return EXIT_SUCCESS;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
