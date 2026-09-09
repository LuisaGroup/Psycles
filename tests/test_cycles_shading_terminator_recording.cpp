#include "cycles_svm_bsdf.h"
#include "cycles_svm_simple_closure.h"
#include "luisa_cycles_svm_test_kernel_globals.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {
using namespace luisa::compute;
namespace svm = psycles::luisa_backend::cycles_svm;
namespace closure = psycles::luisa_backend::cycles_closure;
constexpr auto diffuse_mask = svm::detail::ClosureTypeMask{1u} << closure::type_diffuse;

// This is a recording-time counter, not device instrumentation. The returned
// frequency remains a dynamic object-buffer read, so constant folding cannot
// substitute for the required ordinary C++ guard.
class DynamicGlobals : public psycles::test_support::DefaultCyclesSvmKernelGlobals {
  const BufferFloat &_frequencies;
  unsigned &_calls;
public:
  DynamicGlobals(const BufferFloat &frequencies, unsigned &calls) noexcept
      : _frequencies{frequencies}, _calls{calls} {}
  [[nodiscard]] Float object_shadow_terminator_shading_offset(
      Expr<unsigned> object) const noexcept override {
    ++_calls;
    return _frequencies.read(object);
  }
};

class ExplicitGlobals final : public DynamicGlobals {
  bool _enabled;
public:
  ExplicitGlobals(const BufferFloat &frequencies, unsigned &calls, bool enabled) noexcept
      : DynamicGlobals{frequencies, calls}, _enabled{enabled} {}
  [[nodiscard]] bool has_shadow_terminator_shading_offset() const noexcept override {
    return _enabled;
  }
};

struct Recording { unsigned service_calls{}, cosine_calls{}; bool has_acos{}; };

unsigned count_cosines(Function function, std::set<std::uint64_t> &functions,
                       std::set<const Expression *> &expressions) {
  if (!functions.insert(function.hash()).second) return 0u;
  unsigned count = 0u;
  traverse_expressions<true>(function.body(), [&](const Expression *expression) {
    if (!expressions.insert(expression).second || expression->tag() != Expression::Tag::CALL) return;
    const auto *call = static_cast<const CallExpr *>(expression);
    if (call->is_custom()) count += count_cosines(call->custom(), functions, expressions);
    else if (call->op() == CallOp::COS) ++count;
  }, [](auto) {}, [](auto) {});
  return count;
}

Recording record(bool sample, bool explicit_capability, bool enabled,
                 const std::filesystem::path &dump_path) {
  Recording result;
  const Kernel1D<Buffer<float>, Buffer<luisa::float4>> kernel =
      [&](BufferFloat frequencies, BufferFloat4 output) {
    DynamicGlobals default_globals{frequencies, result.service_calls};
    ExplicitGlobals explicit_globals{frequencies, result.service_calls, enabled};
    const svm::KernelGlobals &kg = explicit_capability
        ? static_cast<const svm::KernelGlobals &>(explicit_globals)
        : static_cast<const svm::KernelGlobals &>(default_globals);
    const auto identity = make_float4x4(1.0f);
    svm::ClosurePool pool{1u};
    svm::ShaderData sd{
        make_float3(0.0f), make_float3(0.0f, 0.0f, 1.0f),
        make_float3(0.0f, 0.0f, 1.0f), make_float3(0.0f, 0.0f, 1.0f),
        svm::primitive_triangle, 0u, 0u, 0u, 0u, 0.0f, 0.0f,
        dispatch_x(), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        make_float3(1.0f, 0.0f, 0.0f), make_float3(0.0f, 1.0f, 0.0f),
        identity, identity, 0u, &pool};
    svm::detail::diffuse_setup(sd, sd.N, make_float3(1.0f));
    if (sample) {
      const auto value = svm::detail::bsdf_sample(
          kg, sd, 0u, make_float3(0.125f, 0.5f, 0.75f), diffuse_mask);
      output.write(dispatch_x(), make_float4(value.value, value.pdf));
    } else {
      const auto value = svm::detail::bsdf_eval(
          kg, sd, 0u, make_float3(0.6f, 0.0f, 0.8f), diffuse_mask);
      output.write(dispatch_x(), make_float4(value.value, value.pdf));
    }
  };
  // Inspect the complete entry and reachable callable bodies before backend
  // optimization. Native fast_acosf is sqrt times a cubic, not an ACOS call.
  // The correction adds exactly one COS; the genuine diffuse sampler keeps
  // its own disk-mapping COS even when the correction is omitted. Shared AST
  // nodes must not be counted twice merely because several users reference them.
  std::set<std::uint64_t> functions;
  std::set<const Expression *> expressions;
  result.cosine_calls = count_cosines(kernel.function()->function(), functions, expressions);
  result.has_acos = kernel.function()->function().propagated_builtin_callables().test(CallOp::ACOS);
  if (!dump_path.empty()) {
    std::ofstream dump{dump_path};
    dump << to_json(kernel.function()->function());
    if (!dump) throw std::runtime_error{"failed to preserve complete recorded AST"};
  }
  return result;
}
} // namespace

int main(int argc, char **argv) {
  if (argc > 2) return 2;
  const std::filesystem::path dump_dir = argc == 2 ? argv[1] : "";
  if (!dump_dir.empty()) std::filesystem::create_directories(dump_dir);
  unsigned failures = 0u;
  for (const bool sample : {false, true}) {
    for (unsigned mode = 0u; mode < 3u; ++mode) {
      const bool expected_enabled = mode != 0u;
      const auto dump_path = dump_dir.empty() ? std::filesystem::path{} :
          dump_dir / (std::string{sample ? "sample-" : "eval-"} +
                      (mode == 0u ? "false" : mode == 1u ? "true" : "default") + ".json");
      const auto result = record(sample, mode != 2u, expected_enabled, dump_path);
      const bool valid = result.service_calls == unsigned(expected_enabled) &&
                         result.cosine_calls == unsigned(sample) + unsigned(expected_enabled) &&
                         !result.has_acos;
      std::cout << (sample ? "bsdf_sample" : "bsdf_eval")
                << " capability=" << (mode == 0u ? "false" : mode == 1u ? "true" : "default")
                << " service_calls=" << result.service_calls
                << " cos=" << result.cosine_calls
                << " acos=" << result.has_acos << (valid ? " PASS\n" : " FAIL\n");
      failures += !valid;
    }
  }
  return failures == 0u ? 0 : 1;
}
