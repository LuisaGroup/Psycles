#include "cycles_svm_osl.h"

#include "cycles_svm_bsdf.h"
#include "cycles_svm_internal.h"
#include "cycles_svm_microfacet.h"
#include "cycles_svm_microfacet_fresnel.h"
#include "cycles_svm_sheen.h"
#include "cycles_svm_simple_closure.h"
#include "cycles_svm_toon.h"
#include "microfacet_albedo.h"

#include <psycles/luisa/cycles_bsdf_tables.h>
#include <psycles/luisa/cycles_closure.h>
#include <psycles/luisa/cycles_sample_mapping.h>
#include <psycles/luisa/native_vector_math.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <luisa/ast/function_builder.h>
#include <luisa/core/logging.h>
#include <luisa/dsl/sugar.h>
#include <luisa/osl/oso_parser.h>
#include <luisa/osl/shader.h>
#include <luisa/osl/shader_group.h>
#include <luisa/osl/symbol.h>
#include <luisa/osl/translator.h>
#include <luisa/osl/type.h>
#include <luisa/xir/translators/xir2ast.h>

namespace psycles::luisa_backend::cycles_svm::detail {
namespace {

using namespace luisa::compute;
using namespace compiler::cycles_svm;
namespace osl = luisa::compute::osl;
namespace xir = luisa::compute::xir;
namespace table_detail = ::psycles::luisa_backend::detail;

inline constexpr float half_pi = 1.57079632679489661923f;
inline constexpr float two_pi = 6.28318530717958647692f;

static_assert(static_cast<std::uint32_t>(OslClosureTag::generalized_schlick_bsdf) < 64u);

[[nodiscard]] constexpr std::uint64_t closure_tag_bit(OslClosureTag tag) noexcept {
  return std::uint64_t{1ull} << static_cast<std::uint32_t>(tag);
}

[[nodiscard]] bool type_contains_closure(const osl::Type *t) noexcept {
  if (t == nullptr) {
    return false;
  }
  if (t->tag() == osl::Type::Tag::CLOSURE) {
    return true;
  }
  if (t->tag() == osl::Type::Tag::STRUCT) {
    const auto *st = static_cast<const osl::StructType *>(t);
    for (const auto &f : st->fields()) {
      if (type_contains_closure(f.type)) {
        return true;
      }
    }
  }
  return false;
}

struct OslClosureBounds {
  std::uint32_t emitted_bound{max_osl_closures};
  std::uint32_t translator_bound{max_osl_closures};
};

[[nodiscard]] OslClosureBounds
analyze_osl_shader_closure_bounds(const osl::Shader &shader) noexcept {
  for (const auto &sym : shader.symbols()) {
    if (sym->is_array() && type_contains_closure(sym->type())) {
      return {};
    }
    if (sym->type() != nullptr &&
        sym->type()->tag() == osl::Type::Tag::STRUCT &&
        type_contains_closure(sym->type())) {
      return {};
    }
  }
  const auto insts = shader.instructions();
  for (std::size_t pc = 0u; pc < insts.size(); ++pc) {
    const auto &inst = *insts[pc];
    const auto op = inst.opcode();
    if (op == "for" || op == "while" || op == "dowhile") {
      return {};
    }
    for (const int target : inst.jump_targets()) {
      if (target >= 0 && static_cast<std::size_t>(target) <= pc) {
        return {};
      }
    }
  }

  auto is_closure_sym = [](const osl::Symbol *s) noexcept {
    return s != nullptr && s->type() != nullptr &&
           s->type()->tag() == osl::Type::Tag::CLOSURE;
  };
  std::unordered_map<std::string_view, std::uint32_t> sym_max;
  auto get_sym_bound = [&](const osl::Symbol *s) noexcept -> std::uint32_t {
    if (!is_closure_sym(s)) {
      return 0u;
    }
    const auto it = sym_max.find(s->identifier());
    return it != sym_max.end() ? it->second : 0u;
  };
  std::uint32_t ci_max = 0u;
  std::uint32_t max_expr_bound = 0u;
  auto update_dst = [&](const osl::Symbol *dst, std::uint32_t bound) noexcept {
    const auto clamped = std::min<std::uint32_t>(max_osl_closures, bound);
    max_expr_bound = std::max(max_expr_bound, clamped);
    if (dst->tag() == osl::Symbol::Tag::SYM_GLOBAL &&
        dst->identifier() == "Ci") {
      ci_max = std::min<std::uint32_t>(max_osl_closures, ci_max + clamped);
    } else {
      auto &slot = sym_max[dst->identifier()];
      slot = std::max(slot, clamped);
    }
  };

  for (const auto &inst : insts) {
    const auto args = inst->args();
    if (args.empty()) {
      continue;
    }
    const auto *dst = args[0u];
    if (!is_closure_sym(dst)) {
      continue;
    }
    const auto op = inst->opcode();
    if (op == "closure") {
      update_dst(dst, 1u);
    } else if (op == "assign" && args.size() >= 2u) {
      update_dst(dst, get_sym_bound(args[1u]));
    } else if (op == "mul" && args.size() >= 3u) {
      const auto rhs = is_closure_sym(args[1u]) ? get_sym_bound(args[1u])
                                                 : get_sym_bound(args[2u]);
      update_dst(dst, rhs);
    } else if (op == "add" && args.size() >= 3u) {
      update_dst(dst, get_sym_bound(args[1u]) + get_sym_bound(args[2u]));
    } else {
      update_dst(dst, max_osl_closures);
    }
  }

  std::uint32_t total = ci_max;
  for (const auto &sym : shader.symbols()) {
    if (sym->tag() == osl::Symbol::Tag::SYM_OUTPUT_PARAM && sym->is_root() &&
        sym->type() != nullptr && sym->type()->tag() == osl::Type::Tag::CLOSURE) {
      const auto it = sym_max.find(sym->identifier());
      if (it != sym_max.end()) {
        total = std::min<std::uint32_t>(max_osl_closures, total + it->second);
      }
    }
  }
  return {
      .emitted_bound = std::min<std::uint32_t>(max_osl_closures, total),
      .translator_bound = std::clamp<std::uint32_t>(
          std::max(total, max_expr_bound), 1u, max_osl_closures)};
}

struct CompiledOslScript {
  luisa::shared_ptr<const luisa::compute::detail::FunctionBuilder> ast;
  std::uint64_t emitted_closure_mask{0ull};
  std::uint32_t max_emitted_closures{max_osl_closures};
  bool uses_transforms{false};
  bool uses_object_info{false};
  bool writes_N{false};
};

[[nodiscard]] Float average(Expr<luisa::float3> value) noexcept {
  return (value.x + value.y + value.z) * (1.0f / 3.0f);
}

[[nodiscard]] Float square(Expr<float> value) noexcept { return value * value; }

[[nodiscard]] std::string
script_cache_key(const OslScriptBinding &script) noexcept {
  std::ostringstream os;
  os << script.script_name.size() << ':' << script.script_name << '|'
     << script.oso_bytecode.size() << ':' << script.oso_bytecode << '|'
     << (script.has_closure_output ? '1' : '0')
     << (script.outputs_have_derivatives ? '1' : '0') << '|'
     << script.params.size() << ';';
  for (const auto &p : script.params) {
    os << p.name.size() << ':' << p.name << ','
       << static_cast<unsigned>(p.kind) << ',' << (p.is_output ? '1' : '0')
       << (p.is_linked ? '1' : '0') << (p.has_derivatives ? '1' : '0')
       << (p.use_oso_init ? '1' : '0') << ','
       << std::bit_cast<std::uint32_t>(p.default_float3[0]) << ','
       << std::bit_cast<std::uint32_t>(p.default_float3[1]) << ','
       << std::bit_cast<std::uint32_t>(p.default_float3[2]) << ','
       << p.default_int << ',' << p.default_string.size() << ':'
       << p.default_string << ';';
  }
  os << script.attributes.size() << ';';
  for (const auto &a : script.attributes) {
    os << a.name.size() << ':' << a.name << ';';
  }
  return os.str();
}

[[nodiscard]] bool is_param_differentiable(OslParamKind kind) noexcept {
  switch (kind) {
    case OslParamKind::floating:
    case OslParamKind::color:
    case OslParamKind::vector:
    case OslParamKind::point:
    case OslParamKind::normal:
      return true;
    case OslParamKind::integer:
    case OslParamKind::closure:
    case OslParamKind::string:
      return false;
  }
  return false;
}

[[nodiscard]] const Type *param_xir_type(OslParamKind kind) noexcept {
  switch (kind) {
    case OslParamKind::floating:
      return Type::of<float>();
    case OslParamKind::integer:
      return Type::of<std::int32_t>();
    case OslParamKind::color:
    case OslParamKind::vector:
    case OslParamKind::point:
    case OslParamKind::normal:
      return Type::of<luisa::float3>();
    case OslParamKind::closure:
      return Type::of<OslClosureList>();
    case OslParamKind::string:
      return Type::of<std::uint32_t>();
  }
  return Type::of<float>();
}

[[nodiscard]] const Type *osl_simple_xir_type(const osl::Type *t) noexcept {
  if (t == nullptr || t->tag() != osl::Type::Tag::SIMPLE) {
    return nullptr;
  }
  switch (static_cast<const osl::SimpleType *>(t)->primitive()) {
    case osl::SimpleType::Primitive::INT:
      return Type::of<std::int32_t>();
    case osl::SimpleType::Primitive::FLOAT:
      return Type::of<float>();
    case osl::SimpleType::Primitive::POINT:
    case osl::SimpleType::Primitive::VECTOR:
    case osl::SimpleType::Primitive::NORMAL:
    case osl::SimpleType::Primitive::COLOR:
      return Type::of<luisa::float3>();
    case osl::SimpleType::Primitive::MATRIX:
      return Type::of<luisa::float4x4>();
    case osl::SimpleType::Primitive::STRING:
      return Type::of<std::uint32_t>();
  }
  return nullptr;
}

[[nodiscard]] bool is_triple_osl_type(const osl::Type *t) noexcept {
  if (t == nullptr || t->tag() != osl::Type::Tag::SIMPLE) {
    return false;
  }
  const auto p = static_cast<const osl::SimpleType *>(t)->primitive();
  return p == osl::SimpleType::Primitive::COLOR ||
         p == osl::SimpleType::Primitive::VECTOR ||
         p == osl::SimpleType::Primitive::POINT ||
         p == osl::SimpleType::Primitive::NORMAL;
}

struct CompiledParamArgs {
  const OslParamBinding *binding{nullptr};
  xir::Argument *val_arg{nullptr};
  xir::Argument *dx_arg{nullptr};
  xir::Argument *dy_arg{nullptr};
};

class CyclesOslTranslator final : public osl::OSLTranslator {
private:
  const OslScriptBinding &_script;
  xir::CallableFunction *_fn{nullptr};
  xir::ValueArgument *_arg_globals{nullptr};
  xir::ValueArgument *_arg_transforms{nullptr};
  std::vector<xir::ValueArgument *> _arg_attrs;
  std::vector<CompiledParamArgs> _dyn_inputs;
  std::vector<CompiledParamArgs> _dyn_outputs;
  xir::ReferenceArgument *_arg_mutable{nullptr};
  xir::ReferenceArgument *_arg_closures{nullptr};

  xir::Value *_local_N_ptr{nullptr};
  xir::Value *_local_wrote_N_ptr{nullptr};
  xir::Value *_local_P_ptr{nullptr};
  xir::Value *_local_dPdx_ptr{nullptr};
  xir::Value *_local_dPdy_ptr{nullptr};
  xir::Value *_local_closures_ptr{nullptr};
  std::uint32_t _max_closures_bound{max_osl_closures};
  std::uint64_t _emitted_closure_mask{0ull};
  bool _uses_transforms{false};
  bool _uses_object_info{false};
  bool _writes_N{false};

  [[nodiscard]] xir::Value *extract_global(std::uint32_t field_idx,
                                           const Type *ft) noexcept {
    return xir_builder().call(ft, xir::ArithmeticOp::EXTRACT,
                              {_arg_globals, const_uint(field_idx)});
  }

  [[nodiscard]] xir::Value *extract_transform(std::uint32_t field_idx) noexcept {
    _uses_transforms = true;
    return xir_builder().call(Type::of<luisa::float4x4>(),
                              xir::ArithmeticOp::EXTRACT,
                              {_arg_transforms, const_uint(field_idx)});
  }

  [[nodiscard]] xir::Value *to_float_val(xir::Value *v) noexcept {
    if (v == nullptr) {
      return const_float(0.0f);
    }
    if (v->type() == Type::of<float>()) {
      return v;
    }
    if (v->type() == Type::of<std::int32_t>() ||
        v->type() == Type::of<std::uint32_t>()) {
      return xir_builder().static_cast_(Type::of<float>(), v);
    }
    return const_float(0.0f);
  }

  [[nodiscard]] xir::Value *to_float3_val(xir::Value *v) noexcept {
    if (v == nullptr) {
      return const_float3(luisa::float3{0.0f});
    }
    if (v->type() == Type::of<luisa::float3>()) {
      return v;
    }
    auto *fv = to_float_val(v);
    return xir_builder().call(Type::of<luisa::float3>(),
                              xir::ArithmeticOp::AGGREGATE, {fv, fv, fv});
  }

  [[nodiscard]] xir::Value *
  convert_float_to_dst(xir::Value *fv, const osl::Type *dst_type) noexcept {
    const auto *dst_xt = osl_simple_xir_type(dst_type);
    if (dst_xt == Type::of<float>()) {
      return fv;
    }
    if (dst_xt == Type::of<std::int32_t>()) {
      return xir_builder().static_cast_(Type::of<std::int32_t>(), fv);
    }
    if (dst_xt == Type::of<luisa::float3>()) {
      return xir_builder().call(Type::of<luisa::float3>(),
                                xir::ArithmeticOp::AGGREGATE, {fv, fv, fv});
    }
    return zero_of(dst_xt ? dst_xt : Type::of<float>());
  }

  [[nodiscard]] xir::Value *
  convert_int_to_dst(xir::Value *iv, const osl::Type *dst_type) noexcept {
    const auto *dst_xt = osl_simple_xir_type(dst_type);
    if (dst_xt == Type::of<std::int32_t>()) {
      return iv;
    }
    if (dst_xt == Type::of<float>()) {
      return xir_builder().static_cast_(Type::of<float>(), iv);
    }
    if (dst_xt == Type::of<luisa::float3>()) {
      auto *fv = xir_builder().static_cast_(Type::of<float>(), iv);
      return xir_builder().call(Type::of<luisa::float3>(),
                                xir::ArithmeticOp::AGGREGATE, {fv, fv, fv});
    }
    return zero_of(dst_xt ? dst_xt : Type::of<std::int32_t>());
  }

  [[nodiscard]] osl::DualValue
  convert_float3_dual_to_dst(xir::Value *v3, xir::Value *dx3, xir::Value *dy3,
                             const osl::Type *dst_type,
                             bool want_deriv) noexcept {
    const auto *dst_xt = osl_simple_xir_type(dst_type);
    auto &b = xir_builder();
    if (dst_xt == Type::of<luisa::float3>()) {
      osl::DualValue out{.val = v3};
      if (want_deriv) {
        out.dx = dx3;
        out.dy = dy3;
      }
      return out;
    }
    if (dst_xt == Type::of<float>()) {
      auto *c0 = b.call(Type::of<float>(), xir::ArithmeticOp::EXTRACT,
                        {v3, const_uint(0u)});
      osl::DualValue out{.val = c0};
      if (want_deriv) {
        if (dx3) {
          out.dx = b.call(Type::of<float>(), xir::ArithmeticOp::EXTRACT,
                          {dx3, const_uint(0u)});
        }
        if (dy3) {
          out.dy = b.call(Type::of<float>(), xir::ArithmeticOp::EXTRACT,
                          {dy3, const_uint(0u)});
        }
      }
      return out;
    }
    if (dst_xt == Type::of<std::int32_t>()) {
      auto *c0 = b.call(Type::of<float>(), xir::ArithmeticOp::EXTRACT,
                        {v3, const_uint(0u)});
      return {.val = b.static_cast_(Type::of<std::int32_t>(), c0)};
    }
    return {.val = zero_of(dst_xt ? dst_xt : Type::of<float>())};
  }

  [[nodiscard]] xir::Value *merge_closure_lists(xir::Value *lhs,
                                                xir::Value *rhs) noexcept {
    auto &b = xir_builder();
    const auto *u_t = Type::of<std::uint32_t>();
    const auto *b_t = Type::of<bool>();
    const auto *entry_t = Type::of<OslClosureEntry>();
    const auto *list_t = Type::of<OslClosureList>();
    const auto *arr_t = Type::array(entry_t, max_osl_closures);
    const std::uint32_t bound =
        std::clamp<std::uint32_t>(_max_closures_bound, 1u, max_osl_closures);

    auto *lcount = b.call(u_t, xir::ArithmeticOp::EXTRACT, {lhs, const_uint(0u)});
    auto *rcount = b.call(u_t, xir::ArithmeticOp::EXTRACT, {rhs, const_uint(0u)});
    auto *larr = b.call(arr_t, xir::ArithmeticOp::EXTRACT, {lhs, const_uint(1u)});

    auto *arr_ptr = b.alloca_local(arr_t);
    b.store(arr_ptr, larr);

    for (std::uint32_t k = 0u; k < bound; ++k) {
      auto *kv = const_uint(k);
      auto *r_lt = b.call(b_t, xir::ArithmeticOp::BINARY_LESS, {kv, rcount});
      auto *sum_idx =
          b.call(u_t, xir::ArithmeticOp::BINARY_ADD, {lcount, kv});
      auto *cap_ok = b.call(b_t, xir::ArithmeticOp::BINARY_LESS,
                            {sum_idx, const_uint(max_osl_closures)});
      auto *cond =
          b.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {r_lt, cap_ok});
      auto *then_bb = _fn->create_basic_block();
      auto *merge_bb = _fn->create_basic_block();
      b.cond_br(cond, then_bb, merge_bb);
      b.set_insertion_point(then_bb);
      auto *rhs_k = b.call(entry_t, xir::ArithmeticOp::EXTRACT,
                           {rhs, const_uint(1u), kv});
      auto *safe_idx = b.call(u_t, xir::ArithmeticOp::MIN,
                              {sum_idx, const_uint(max_osl_closures - 1u)});
      auto *dst_p = b.gep(entry_t, arr_ptr, {safe_idx});
      b.store(dst_p, rhs_k);
      b.br(merge_bb);
      b.set_insertion_point(merge_bb);
    }

    auto *new_arr = b.load(arr_t, arr_ptr);
    auto *total = b.call(u_t, xir::ArithmeticOp::BINARY_ADD, {lcount, rcount});
    auto *new_cnt = b.call(u_t, xir::ArithmeticOp::MIN,
                           {total, const_uint(max_osl_closures)});
    return b.call(list_t, xir::ArithmeticOp::AGGREGATE, {new_cnt, new_arr});
  }

  [[nodiscard]] xir::Value *make_single_closure_list(
      OslClosureTag tag, xir::Value *N, xir::Value *T, xir::Value *color_arg,
      xir::Value *color_arg2, xir::Value *p0, xir::Value *p1, xir::Value *p2,
      xir::Value *p3, xir::Value *p4) noexcept {
    _emitted_closure_mask |= closure_tag_bit(tag);
    auto &b = xir_builder();
    const auto *entry_t = Type::of<OslClosureEntry>();
    const auto *list_t = Type::of<OslClosureList>();
    const auto *arr_t = Type::array(entry_t, max_osl_closures);
    auto *entry = b.call(
        entry_t, xir::ArithmeticOp::AGGREGATE,
        {const_uint(static_cast<std::uint32_t>(tag)),
         const_float3(luisa::float3{1.0f}),
         N ? N : const_float3(luisa::float3{0.0f}),
         T ? T : const_float3(luisa::float3{0.0f}),
         color_arg ? color_arg : const_float3(luisa::float3{1.0f}),
         color_arg2 ? color_arg2 : const_float3(luisa::float3{0.0f}),
         p0 ? p0 : const_float(0.0f), p1 ? p1 : const_float(0.0f),
         p2 ? p2 : const_float(0.0f), p3 ? p3 : const_float(0.0f),
         p4 ? p4 : const_float(0.0f)});
    auto *zero_e = zero_of(entry_t);
    auto *arr = b.call(arr_t, xir::ArithmeticOp::AGGREGATE,
                       {entry, zero_e, zero_e, zero_e, zero_e, zero_e, zero_e,
                        zero_e});
    return b.call(list_t, xir::ArithmeticOp::AGGREGATE, {const_uint(1u), arr});
  }

  void bind_closure_outputs(xir::Value *&closures_val,
                            const osl::OutputBinding &binding) noexcept {
    if (!binding.fields.empty()) {
      for (const auto &f : binding.fields) {
        bind_closure_outputs(closures_val, f);
      }
      return;
    }
    if (!binding.elements.empty()) {
      for (const auto &el : binding.elements) {
        bind_closure_outputs(closures_val, el);
      }
      return;
    }
    const auto *bt =
        binding.type ? binding.type
                     : (binding.symbol ? binding.symbol->type() : nullptr);
    if (bt != nullptr && bt->tag() == osl::Type::Tag::CLOSURE &&
        binding.closure.valid()) {
      closures_val = closures_val != nullptr
                         ? merge_closure_lists(closures_val, binding.closure.value)
                         : binding.closure.value;
    }
  }

public:
  CyclesOslTranslator(
      const OslScriptBinding &script, xir::CallableFunction *fn,
      std::uint32_t max_closures_bound = max_osl_closures) noexcept
      : _script{script},
        _fn{fn},
        _max_closures_bound{std::clamp<std::uint32_t>(
            max_closures_bound, 1u, max_osl_closures)} {
    _arg_globals = fn->create_value_argument(Type::of<OslEvalGlobals>());
    _arg_transforms = fn->create_value_argument(Type::of<OslTransformGlobals>());
    _arg_attrs.reserve(script.attributes.size());
    for (std::size_t i = 0u; i < script.attributes.size(); ++i) {
      _arg_attrs.push_back(fn->create_value_argument(Type::of<OslAttrInput>()));
    }
    for (const auto &p : script.params) {
      if (p.is_output || !p.is_linked || p.kind == OslParamKind::string ||
          p.kind == OslParamKind::closure) {
        continue;
      }
      const auto *xt = param_xir_type(p.kind);
      CompiledParamArgs cpa{.binding = &p,
                            .val_arg = fn->create_value_argument(xt)};
      if (p.has_derivatives && is_param_differentiable(p.kind)) {
        cpa.dx_arg = fn->create_value_argument(xt);
        cpa.dy_arg = fn->create_value_argument(xt);
      }
      _dyn_inputs.push_back(cpa);
    }
    _arg_mutable = fn->create_reference_argument(Type::of<OslMutableState>());
    for (const auto &p : script.params) {
      if (!p.is_output || p.kind == OslParamKind::closure ||
          p.kind == OslParamKind::string) {
        continue;
      }
      const auto *xt = param_xir_type(p.kind);
      CompiledParamArgs cpa{.binding = &p,
                            .val_arg = fn->create_reference_argument(xt)};
      if (p.has_derivatives && is_param_differentiable(p.kind)) {
        cpa.dx_arg = fn->create_reference_argument(xt);
        cpa.dy_arg = fn->create_reference_argument(xt);
      }
      _dyn_outputs.push_back(cpa);
    }
    if (script.has_closure_output) {
      _arg_closures =
          fn->create_reference_argument(Type::of<OslClosureList>());
    }
  }

  [[nodiscard]] std::uint64_t emitted_closure_mask() const noexcept {
    return _emitted_closure_mask;
  }
  [[nodiscard]] bool uses_transforms() const noexcept {
    return _uses_transforms;
  }
  [[nodiscard]] bool uses_object_info() const noexcept {
    return _uses_object_info;
  }
  [[nodiscard]] bool writes_N() const noexcept {
    return _writes_N;
  }

protected:
  [[nodiscard]] const Type *closure_xir_type() const noexcept override {
    return Type::of<OslClosureList>();
  }

  void on_begin_group(const osl::ShaderGroup &) noexcept override {
    auto &b = xir_builder();
    const auto *f3_t = Type::of<luisa::float3>();
    const auto *u_t = Type::of<std::uint32_t>();
    _local_N_ptr = b.alloca_local(f3_t);
    _local_wrote_N_ptr = b.alloca_local(u_t);
    _local_P_ptr = b.alloca_local(f3_t);
    _local_dPdx_ptr = b.alloca_local(f3_t);
    _local_dPdy_ptr = b.alloca_local(f3_t);
    b.store(_local_N_ptr, extract_global(6u, f3_t));
    b.store(_local_wrote_N_ptr, const_uint(0u));
    b.store(_local_P_ptr, extract_global(0u, f3_t));
    b.store(_local_dPdx_ptr, extract_global(1u, f3_t));
    b.store(_local_dPdy_ptr, extract_global(2u, f3_t));
  }

  void on_end_group(
      const osl::ShaderGroup &,
      luisa::span<const osl::OutputBinding> outputs) noexcept override {
    auto &b = xir_builder();
    const auto *f3_t = Type::of<luisa::float3>();
    const auto *u_t = Type::of<std::uint32_t>();
    auto *mut = b.call(Type::of<OslMutableState>(),
                       xir::ArithmeticOp::AGGREGATE,
                       {b.load(f3_t, _local_N_ptr),
                        b.load(u_t, _local_wrote_N_ptr)});
    b.store(_arg_mutable, mut);

    for (const auto &out : outputs) {
      if (out.symbol == nullptr) {
        continue;
      }
      for (const auto &dyn_out : _dyn_outputs) {
        if (dyn_out.binding->name != out.symbol->identifier() ||
            !out.value.valid()) {
          continue;
        }
        const auto *target_t = dyn_out.val_arg->type();
        auto *val = out.value.val;
        auto *dx = out.value.dx;
        auto *dy = out.value.dy;
        if (val->type() != target_t) {
          if (target_t == Type::of<float>()) {
            val = to_float_val(val);
            if (dx) {
              dx = to_float_val(dx);
            }
            if (dy) {
              dy = to_float_val(dy);
            }
          } else if (target_t == Type::of<luisa::float3>()) {
            val = to_float3_val(val);
            if (dx) {
              dx = to_float3_val(dx);
            }
            if (dy) {
              dy = to_float3_val(dy);
            }
          } else if (target_t == Type::of<std::int32_t>()) {
            val = b.static_cast_(target_t, to_float_val(val));
          }
        }
        b.store(dyn_out.val_arg, val);
        if (dyn_out.dx_arg != nullptr) {
          b.store(dyn_out.dx_arg, dx ? dx : zero_of(target_t));
        }
        if (dyn_out.dy_arg != nullptr) {
          b.store(dyn_out.dy_arg, dy ? dy : zero_of(target_t));
        }
      }
    }

    if (_arg_closures != nullptr) {
      const auto *list_t = Type::of<OslClosureList>();
      xir::Value *closures = (_local_closures_ptr != nullptr)
                                 ? b.load(list_t, _local_closures_ptr)
                                 : nullptr;
      for (const auto &out : outputs) {
        bind_closure_outputs(closures, out);
      }
      b.store(_arg_closures, closures != nullptr ? closures : zero_of(list_t));
    }
  }

  [[nodiscard]] osl::DualValue emit_global_read(osl::GlobalSymbol g,
                                                bool want_deriv) noexcept override {
    auto &b = xir_builder();
    const auto *f3_t = Type::of<luisa::float3>();
    const auto *f_t = Type::of<float>();
    switch (g) {
      case osl::GlobalSymbol::P:
      case osl::GlobalSymbol::Ps: {
        osl::DualValue dv{.val = b.load(f3_t, _local_P_ptr)};
        if (want_deriv) {
          dv.dx = b.load(f3_t, _local_dPdx_ptr);
          dv.dy = b.load(f3_t, _local_dPdy_ptr);
        }
        return dv;
      }
      case osl::GlobalSymbol::I: {
        osl::DualValue dv{.val = extract_global(3u, f3_t)};
        if (want_deriv) {
          dv.dx = extract_global(4u, f3_t);
          dv.dy = extract_global(5u, f3_t);
        }
        return dv;
      }
      case osl::GlobalSymbol::N:
        return {.val = b.load(f3_t, _local_N_ptr)};
      case osl::GlobalSymbol::Ng:
        return {.val = extract_global(7u, f3_t)};
      case osl::GlobalSymbol::dPdu:
        return {.val = extract_global(8u, f3_t)};
      case osl::GlobalSymbol::dPdv:
        return {.val = extract_global(9u, f3_t)};
      case osl::GlobalSymbol::u: {
        osl::DualValue dv{.val = extract_global(10u, f_t)};
        if (want_deriv) {
          dv.dx = extract_global(11u, f_t);
          dv.dy = extract_global(12u, f_t);
        }
        return dv;
      }
      case osl::GlobalSymbol::v: {
        osl::DualValue dv{.val = extract_global(13u, f_t)};
        if (want_deriv) {
          dv.dx = extract_global(14u, f_t);
          dv.dy = extract_global(15u, f_t);
        }
        return dv;
      }
      case osl::GlobalSymbol::time:
        return {.val = extract_global(16u, f_t)};
      case osl::GlobalSymbol::dtime:
        return {.val = extract_global(17u, f_t)};
      case osl::GlobalSymbol::dPdtime:
        return {.val = const_float3(luisa::float3{0.0f})};
      default:
        report_error("Unsupported OSL global read.");
        return {};
    }
  }

  void emit_global_write(osl::GlobalSymbol g,
                         const osl::DualValue &value) noexcept override {
    auto &b = xir_builder();
    const auto *f3_t = Type::of<luisa::float3>();
    if (g == osl::GlobalSymbol::N) {
      _writes_N = true;
      b.store(_local_N_ptr, value.val);
      b.store(_local_wrote_N_ptr, const_uint(1u));
      return;
    }
    if (g == osl::GlobalSymbol::P) {
      b.store(_local_P_ptr, value.val);
      b.store(_local_dPdx_ptr, value.dx ? value.dx : zero_of(f3_t));
      b.store(_local_dPdy_ptr, value.dy ? value.dy : zero_of(f3_t));
      return;
    }
    report_error("Unsupported OSL global write.");
  }

  void emit_closure_output(const osl::ClosureValue &ci) noexcept override {
    if (_arg_closures == nullptr || !ci.valid()) {
      return;
    }
    auto &b = xir_builder();
    const auto *list_t = Type::of<OslClosureList>();
    if (_local_closures_ptr == nullptr) {
      _local_closures_ptr = b.alloca_local(list_t);
      b.store(_local_closures_ptr, ci.value);
      return;
    }
    auto *cur = b.load(list_t, _local_closures_ptr);
    b.store(_local_closures_ptr, merge_closure_lists(cur, ci.value));
  }

  [[nodiscard]] xir::Value *emit_backfacing() noexcept override {
    return extract_global(28u, Type::of<std::int32_t>());
  }

  [[nodiscard]] xir::Value *
  emit_raytype(luisa::string_view type_name) noexcept override {
    std::uint32_t mask = 0u;
    if (type_name == "camera") {
      mask = path_ray_visibility_camera;
    } else if (type_name == "shadow") {
      mask = path_ray_visibility_shadow;
    } else if (type_name == "diffuse") {
      mask = path_ray_visibility_diffuse;
    } else if (type_name == "glossy" || type_name == "reflection") {
      mask = path_ray_visibility_glossy;
    } else if (type_name == "refraction" || type_name == "transmission") {
      mask = path_ray_visibility_transmit;
    } else if (type_name == "volume_scatter") {
      mask = path_ray_visibility_volume_scatter;
    } else if (type_name == "transparent") {
      mask = path_ray_visibility_shadow_transparent;
    } else {
      return const_int(0);
    }
    auto &b = xir_builder();
    const auto *u_t = Type::of<std::uint32_t>();
    auto *vis = extract_global(19u, u_t);
    auto *anded = b.call(u_t, xir::ArithmeticOp::BINARY_BIT_AND,
                         {vis, const_uint(mask)});
    auto *nz = b.call(Type::of<bool>(), xir::ArithmeticOp::BINARY_NOT_EQUAL,
                      {anded, const_uint(0u)});
    return b.call(Type::of<std::int32_t>(), xir::ArithmeticOp::SELECT,
                  {const_int(0), const_int(1), nz});
  }

  [[nodiscard]] osl::DualValue emit_dynamic_input(
      std::uint32_t, luisa::string_view param_name, const osl::Type *, int,
      bool want_deriv) noexcept override {
    for (const auto &in : _dyn_inputs) {
      if (in.binding->name == param_name) {
        osl::DualValue dv{.val = in.val_arg};
        if (want_deriv && in.dx_arg != nullptr && in.dy_arg != nullptr) {
          dv.dx = in.dx_arg;
          dv.dy = in.dy_arg;
        }
        return dv;
      }
    }
    report_error(
        luisa::format("Dynamic input '{}' missing from signature.", param_name));
    return {};
  }

  [[nodiscard]] osl::ClosureValue
  emit_dynamic_closure_input(std::uint32_t, luisa::string_view) noexcept override {
    return emit_closure_zero();
  }

  [[nodiscard]] xir::Value *emit_getmatrix(luisa::string_view from_space,
                                           luisa::string_view to_space,
                                           xir::Value *&out_matrix) noexcept override {
    auto &b = xir_builder();
    auto *id_m = const_float4x4(luisa::make_float4x4(1.0f));
    auto space_kind = [](luisa::string_view s) noexcept -> int {
      if (s == "common" || s == "world") {
        return 0;
      }
      if (s == "object" || s == "shader") {
        return 1;
      }
      if (s == "camera") {
        return 2;
      }
      return -1;
    };
    const int from_k = space_kind(from_space);
    const int to_k = space_kind(to_space);
    if (from_k < 0 || to_k < 0) {
      out_matrix = id_m;
      return const_int(0);
    }
    if (from_k == to_k) {
      out_matrix = id_m;
      return const_int(1);
    }
    xir::Value *from_to_world = id_m;
    if (from_k == 1) {
      from_to_world = extract_transform(0u);
    } else if (from_k == 2) {
      from_to_world = extract_transform(2u);
    }
    xir::Value *world_to_to = id_m;
    if (to_k == 1) {
      world_to_to = extract_transform(1u);
    } else if (to_k == 2) {
      world_to_to = extract_transform(3u);
    }
    if (from_k == 0) {
      out_matrix = world_to_to;
    } else if (to_k == 0) {
      out_matrix = from_to_world;
    } else {
      out_matrix = b.call(Type::of<luisa::float4x4>(),
                          xir::ArithmeticOp::MATRIX_LINALG_MUL,
                          {world_to_to, from_to_world});
    }
    return const_int(1);
  }

  [[nodiscard]] xir::Value *emit_getattribute(
      luisa::optional<luisa::string_view> object_name,
      luisa::string_view attr_name, const osl::Type *dst_type,
      int dst_array_len, bool want_deriv,
      osl::DualValue &out_value) noexcept override {
    const auto *dst_xt = osl_simple_xir_type(dst_type);
    if (dst_array_len != 0 || dst_xt == nullptr) {
      return const_int(0);
    }
    auto &b = xir_builder();
    const auto *f3_t = Type::of<luisa::float3>();
    const auto *f4_t = Type::of<luisa::float4>();
    const auto *f_t = Type::of<float>();
    const auto *u_t = Type::of<std::uint32_t>();
    const auto *i_t = Type::of<std::int32_t>();
    const auto *b_t = Type::of<bool>();

    if (!object_name.has_value()) {
      for (std::size_t i = 0u; i < _script.attributes.size(); ++i) {
        const auto &a = _script.attributes[i];
        const bool match =
            (a.name == attr_name) ||
            ((attr_name == "uv" || attr_name == "geom:uv") &&
             (a.name == "uv" || a.name == "geom:uv"));
        if (!match) {
          continue;
        }
        auto *attr_arg = _arg_attrs[i];
        auto *val4 = b.call(f4_t, xir::ArithmeticOp::EXTRACT,
                            {attr_arg, const_uint(0u)});
        auto *dx4 = b.call(f4_t, xir::ArithmeticOp::EXTRACT,
                           {attr_arg, const_uint(1u)});
        auto *dy4 = b.call(f4_t, xir::ArithmeticOp::EXTRACT,
                           {attr_arg, const_uint(2u)});
        auto *found = b.call(i_t, xir::ArithmeticOp::EXTRACT,
                             {attr_arg, const_uint(3u)});
        auto extract_xyz = [&](xir::Value *v4) -> xir::Value * {
          auto *x = b.call(f_t, xir::ArithmeticOp::EXTRACT, {v4, const_uint(0u)});
          auto *y = b.call(f_t, xir::ArithmeticOp::EXTRACT, {v4, const_uint(1u)});
          auto *z = b.call(f_t, xir::ArithmeticOp::EXTRACT, {v4, const_uint(2u)});
          return b.call(f3_t, xir::ArithmeticOp::AGGREGATE, {x, y, z});
        };
        out_value = convert_float3_dual_to_dst(
            extract_xyz(val4), want_deriv ? extract_xyz(dx4) : nullptr,
            want_deriv ? extract_xyz(dy4) : nullptr, dst_type, want_deriv);
        return found;
      }

      if (attr_name == "geom:P" || attr_name == "P") {
        out_value = convert_float3_dual_to_dst(
            b.load(f3_t, _local_P_ptr),
            want_deriv ? b.load(f3_t, _local_dPdx_ptr) : nullptr,
            want_deriv ? b.load(f3_t, _local_dPdy_ptr) : nullptr, dst_type,
            want_deriv);
        return const_int(1);
      }
      if (attr_name == "geom:N" || attr_name == "N") {
        out_value = convert_float3_dual_to_dst(
            b.load(f3_t, _local_N_ptr), nullptr, nullptr, dst_type, false);
        return const_int(1);
      }
      if (attr_name == "geom:Ng" || attr_name == "Ng") {
        out_value = convert_float3_dual_to_dst(
            extract_global(7u, f3_t), nullptr, nullptr, dst_type, false);
        return const_int(1);
      }
      if (attr_name == "geom:I" || attr_name == "I") {
        out_value = convert_float3_dual_to_dst(
            extract_global(3u, f3_t),
            want_deriv ? extract_global(4u, f3_t) : nullptr,
            want_deriv ? extract_global(5u, f3_t) : nullptr, dst_type,
            want_deriv);
        return const_int(1);
      }
      if (attr_name == "geom:is_smooth") {
        auto *sh = extract_global(20u, u_t);
        auto *bit = b.call(u_t, xir::ArithmeticOp::BINARY_BIT_AND,
                           {sh, const_uint(shader_smooth_normal)});
        auto *nz = b.call(b_t, xir::ArithmeticOp::BINARY_NOT_EQUAL,
                          {bit, const_uint(0u)});
        auto *iv = b.call(i_t, xir::ArithmeticOp::SELECT,
                          {const_int(0), const_int(1), nz});
        out_value = {.val = convert_int_to_dst(iv, dst_type)};
        return const_int(1);
      }
      if (attr_name == "geom:is_curve" || attr_name == "geom:is_point") {
        const std::uint32_t pmask =
            attr_name == "geom:is_curve" ? primitive_curve : primitive_point;
        auto *pt = extract_global(22u, u_t);
        auto *bit = b.call(u_t, xir::ArithmeticOp::BINARY_BIT_AND,
                           {pt, const_uint(pmask)});
        auto *nz = b.call(b_t, xir::ArithmeticOp::BINARY_NOT_EQUAL,
                          {bit, const_uint(0u)});
        auto *iv = b.call(i_t, xir::ArithmeticOp::SELECT,
                          {const_int(0), const_int(1), nz});
        out_value = {.val = convert_int_to_dst(iv, dst_type)};
        return const_int(1);
      }
      if (attr_name == "object:index") {
        _uses_object_info = true;
        out_value = {
            .val = convert_float_to_dst(extract_global(29u, f_t), dst_type)};
        return const_int(1);
      }
      if (attr_name == "object:random") {
        _uses_object_info = true;
        out_value = {
            .val = convert_float_to_dst(extract_global(30u, f_t), dst_type)};
        return const_int(1);
      }
      if (attr_name == "object:alpha") {
        _uses_object_info = true;
        out_value = {
            .val = convert_float_to_dst(extract_global(31u, f_t), dst_type)};
        return const_int(1);
      }
      if (attr_name == "material:index" || attr_name == "shader:index") {
        _uses_object_info = true;
        out_value = {
            .val = convert_float_to_dst(extract_global(32u, f_t), dst_type)};
        return const_int(1);
      }
      if (attr_name == "object:location") {
        _uses_object_info = true;
        out_value = convert_float3_dual_to_dst(
            extract_global(33u, f3_t), nullptr, nullptr, dst_type, false);
        return const_int(1);
      }
      if (attr_name == "object:color") {
        _uses_object_info = true;
        out_value = convert_float3_dual_to_dst(
            extract_global(34u, f3_t), nullptr, nullptr, dst_type, false);
        return const_int(1);
      }
      if (attr_name == "hit:dist" || attr_name == "path:ray_length") {
        out_value = {
            .val = convert_float_to_dst(extract_global(18u, f_t), dst_type)};
        return const_int(1);
      }
      if (attr_name == "path:ray_depth" || attr_name == "path:diffuse_depth" ||
          attr_name == "path:glossy_depth" ||
          attr_name == "path:transmission_depth" ||
          attr_name == "path:transparent_depth") {
        const std::uint32_t idx =
            attr_name == "path:ray_depth"           ? 23u
            : attr_name == "path:diffuse_depth"     ? 24u
            : attr_name == "path:glossy_depth"      ? 25u
            : attr_name == "path:transmission_depth" ? 26u
                                                    : 27u;
        auto *uv = extract_global(idx, u_t);
        auto *iv = b.static_cast_(i_t, uv);
        out_value = {.val = convert_int_to_dst(iv, dst_type)};
        return const_int(1);
      }
    }

    out_value = {.val = zero_of(dst_xt)};
    if (want_deriv &&
        (dst_xt == Type::of<float>() || dst_xt == Type::of<luisa::float3>())) {
      out_value.dx = zero_of(dst_xt);
      out_value.dy = zero_of(dst_xt);
    }
    return const_int(0);
  }

  [[nodiscard]] std::size_t closure_positional_arg_count(
      luisa::string_view name,
      luisa::span<const osl::Type *const> arg_types) const noexcept override {
    const std::size_t n = arg_types.size();
    auto has_string_at = [&](std::size_t idx) noexcept {
      return idx < n && arg_types[idx] != nullptr &&
             arg_types[idx]->tag() == osl::Type::Tag::SIMPLE &&
             static_cast<const osl::SimpleType *>(arg_types[idx])
                     ->primitive() == osl::SimpleType::Primitive::STRING;
    };
    if (name == "dielectric_bsdf") {
      return std::min<std::size_t>(n, 8u);
    }
    if (name == "conductor_bsdf") {
      return std::min<std::size_t>(n, 7u);
    }
    if (name == "generalized_schlick_bsdf") {
      return std::min<std::size_t>(n, 10u);
    }
    if (name == "oren_nayar_diffuse_bsdf" || name == "sheen_bsdf" ||
        name == "diffuse_toon" || name == "glossy_toon" ||
        name == "microfacet_multi_ggx" || name == "microfacet_ggx_refraction" ||
        name == "microfacet_beckmann_refraction") {
      return std::min<std::size_t>(n, 3u);
    }
    if (name == "microfacet_multi_ggx_glass") {
      return std::min<std::size_t>(n, 4u);
    }
    if (name == "microfacet_multi_ggx_aniso" || name == "microfacet_ggx_aniso" ||
        name == "microfacet_beckmann_aniso") {
      return std::min<std::size_t>(n, 5u);
    }
    if (name == "ashikhmin_velvet" || name == "translucent_bsdf") {
      return std::min<std::size_t>(n, 2u);
    }
    if (name == "transparent_bsdf") {
      return 0u;
    }
    std::size_t min_pos =
        OSLTranslator::closure_positional_arg_count(name, arg_types);
    if (name == "microfacet_ggx" || name == "microfacet_beckmann") {
      min_pos = std::min<std::size_t>(n, 2u);
    }
    if (name == "microfacet_f82_tint") {
      min_pos = std::min<std::size_t>(n, 7u);
    }
    (void)has_string_at;
    return min_pos;
  }

  [[nodiscard]] osl::ClosureValue emit_closure_zero() noexcept override {
    return {.value = zero_of(Type::of<OslClosureList>())};
  }

  [[nodiscard]] osl::ClosureValue
  emit_closure_scale(const osl::ClosureValue &cl,
                     xir::Value *color_weight) noexcept override {
    auto &b = xir_builder();
    const auto *u_t = Type::of<std::uint32_t>();
    const auto *f3_t = Type::of<luisa::float3>();
    const auto *entry_t = Type::of<OslClosureEntry>();
    const auto *list_t = Type::of<OslClosureList>();
    const auto *arr_t = Type::array(entry_t, max_osl_closures);
    const std::uint32_t bound =
        std::clamp<std::uint32_t>(_max_closures_bound, 1u, max_osl_closures);

    auto *cnt =
        b.call(u_t, xir::ArithmeticOp::EXTRACT, {cl.value, const_uint(0u)});
    auto *arr =
        b.call(arr_t, xir::ArithmeticOp::EXTRACT, {cl.value, const_uint(1u)});
    auto *w3 = to_float3_val(color_weight);

    for (std::uint32_t k = 0u; k < bound; ++k) {
      auto *kv = const_uint(k);
      auto *e = b.call(entry_t, xir::ArithmeticOp::EXTRACT, {arr, kv});
      auto *old_w = b.call(f3_t, xir::ArithmeticOp::EXTRACT, {e, const_uint(1u)});
      auto *new_w = b.call(f3_t, xir::ArithmeticOp::BINARY_MUL, {old_w, w3});
      auto *ne =
          b.call(entry_t, xir::ArithmeticOp::INSERT, {e, new_w, const_uint(1u)});
      arr = b.call(arr_t, xir::ArithmeticOp::INSERT, {arr, ne, kv});
    }
    return {.value = b.call(list_t, xir::ArithmeticOp::AGGREGATE, {cnt, arr})};
  }

  [[nodiscard]] osl::ClosureValue
  emit_closure_add(const osl::ClosureValue &lhs,
                   const osl::ClosureValue &rhs) noexcept override {
    return {.value = merge_closure_lists(lhs.value, rhs.value)};
  }

  [[nodiscard]] osl::ClosureValue
  emit_closure(luisa::string_view name,
               luisa::span<const osl::HookArg> args,
               luisa::span<const osl::NamedOption> opts) noexcept override {
    auto arg_triple = [&](std::size_t idx,
                          luisa::float3 fallback = {0.0f, 0.0f, 0.0f}) noexcept {
      if (idx < args.size() && args[idx].value.valid()) {
        return to_float3_val(args[idx].value.val);
      }
      return const_float3(fallback);
    };
    auto arg_float = [&](std::size_t idx, float fallback = 0.0f) noexcept {
      if (idx < args.size() && args[idx].value.valid()) {
        return to_float_val(args[idx].value.val);
      }
      return const_float(fallback);
    };
    auto arg_int = [&](std::size_t idx, int fallback = 0) noexcept -> int {
      if (idx < args.size() && args[idx].constant.has_value()) {
        if (const auto *iv = luisa::get_if<std::int32_t>(&*args[idx].constant)) {
          return *iv;
        }
        if (const auto *fv = luisa::get_if<float>(&*args[idx].constant)) {
          return static_cast<int>(*fv);
        }
      }
      return fallback;
    };
    auto arg_str = [&](std::size_t idx) noexcept -> luisa::string_view {
      if (idx < args.size() && args[idx].constant.has_value()) {
        if (const auto *sv =
                luisa::get_if<luisa::string_view>(&*args[idx].constant)) {
          return *sv;
        }
      }
      return {};
    };
    auto opt_float = [&](luisa::string_view key,
                         xir::Value *fallback) noexcept -> xir::Value * {
      for (const auto &opt : opts) {
        if (!opt.is_output && opt.key == key && opt.value.valid()) {
          return to_float_val(opt.value.val);
        }
      }
      return fallback;
    };

    if (name == "diffuse") {
      return {.value = make_single_closure_list(
                  OslClosureTag::diffuse, arg_triple(0u), nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "oren_nayar") {
      return {.value = make_single_closure_list(
                  OslClosureTag::oren_nayar, arg_triple(0u), nullptr,
                  const_float3(luisa::float3{1.0f}), nullptr, arg_float(1u),
                  nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "oren_nayar_diffuse_bsdf") {
      return {.value = make_single_closure_list(
                  OslClosureTag::oren_nayar, arg_triple(0u), nullptr,
                  arg_triple(1u, {1.0f, 1.0f, 1.0f}), nullptr, arg_float(2u),
                  nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "translucent") {
      return {.value = make_single_closure_list(
                  OslClosureTag::translucent, arg_triple(0u), nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "translucent_bsdf") {
      auto *base = make_single_closure_list(
          OslClosureTag::translucent, arg_triple(0u), nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, nullptr, nullptr);
      return emit_closure_scale({.value = base},
                                arg_triple(1u, {1.0f, 1.0f, 1.0f}));
    }
    if (name == "transparent" || name == "transparent_bsdf") {
      return {.value = make_single_closure_list(
                  OslClosureTag::transparent, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "emission") {
      return {.value = make_single_closure_list(
                  OslClosureTag::emission, nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "background") {
      return {.value = make_single_closure_list(
                  OslClosureTag::background, nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "holdout") {
      return {.value = make_single_closure_list(
                  OslClosureTag::holdout, nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "reflection") {
      auto *eta = args.size() >= 2u ? arg_float(1u) : opt_float("ior", const_float(0.0f));
      return {.value = make_single_closure_list(
                  OslClosureTag::reflection, arg_triple(0u), nullptr, nullptr,
                  nullptr, eta, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "refraction") {
      auto *eta = args.size() >= 2u ? arg_float(1u, 1.5f)
                                    : opt_float("ior", const_float(1.5f));
      return {.value = make_single_closure_list(
                  OslClosureTag::refraction, arg_triple(0u), nullptr, nullptr,
                  nullptr, eta, nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "sheen" || name == "sheen_bsdf") {
      if (name == "sheen_bsdf" && args.size() >= 3u) {
        auto *base = make_single_closure_list(
            OslClosureTag::sheen, arg_triple(0u), nullptr, nullptr, nullptr,
            arg_float(2u), nullptr, nullptr, nullptr, nullptr);
        return emit_closure_scale({.value = base},
                                  arg_triple(1u, {1.0f, 1.0f, 1.0f}));
      }
      return {.value = make_single_closure_list(
                  OslClosureTag::sheen, arg_triple(0u), nullptr, nullptr,
                  nullptr, arg_float(1u), nullptr, nullptr, nullptr, nullptr)};
    }
    if (name == "ashikhmin_velvet") {
      return {.value = make_single_closure_list(
                  OslClosureTag::ashikhmin_velvet, arg_triple(0u), nullptr,
                  nullptr, nullptr, arg_float(1u), nullptr, nullptr, nullptr,
                  nullptr)};
    }
    if (name == "diffuse_toon") {
      return {.value = make_single_closure_list(
                  OslClosureTag::diffuse_toon, arg_triple(0u), nullptr, nullptr,
                  nullptr, arg_float(1u, 0.5f), arg_float(2u, 0.0f), nullptr,
                  nullptr, nullptr)};
    }
    if (name == "glossy_toon") {
      return {.value = make_single_closure_list(
                  OslClosureTag::glossy_toon, arg_triple(0u), nullptr, nullptr,
                  nullptr, arg_float(1u, 0.5f), arg_float(2u, 0.0f), nullptr,
                  nullptr, nullptr)};
    }
    if (name == "microfacet") {
      const auto dist = arg_str(0u);
      const bool aniso = args.size() >= 7u;
      xir::Value *n = arg_triple(1u);
      xir::Value *t =
          aniso ? arg_triple(2u) : const_float3(luisa::float3{0.0f});
      xir::Value *ax = aniso ? arg_float(3u) : arg_float(2u);
      xir::Value *ay = aniso ? arg_float(4u) : ax;
      xir::Value *eta = aniso ? arg_float(5u) : arg_float(3u);
      const int refr = aniso ? arg_int(6u, 0) : arg_int(4u, 0);
      OslClosureTag tag = OslClosureTag::microfacet_ggx;
      if (dist == "ashikhmin_shirley") {
        tag = OslClosureTag::ashikhmin_shirley;
      } else if (dist == "beckmann") {
        tag = refr == 2   ? OslClosureTag::microfacet_beckmann_glass
              : refr == 1 ? OslClosureTag::microfacet_beckmann_refraction
                          : OslClosureTag::microfacet_beckmann;
      } else if (dist == "multi_ggx" || dist == "Multiscatter GGX") {
        tag = refr == 2 ? OslClosureTag::microfacet_multi_ggx_glass
                        : OslClosureTag::microfacet_multi_ggx;
      } else {
        tag = refr == 2   ? OslClosureTag::microfacet_ggx_glass
              : refr == 1 ? OslClosureTag::microfacet_ggx_refraction
                          : OslClosureTag::microfacet_ggx;
      }
      return {.value = make_single_closure_list(
                  tag, n, t, const_float3(luisa::float3{1.0f}), nullptr, ax, ay,
                  eta, nullptr, nullptr)};
    }
    if (name == "microfacet_ggx" || name == "microfacet_beckmann") {
      const auto tag = (name == "microfacet_beckmann")
                           ? OslClosureTag::microfacet_beckmann
                           : OslClosureTag::microfacet_ggx;
      auto *a = arg_float(1u);
      return {.value = make_single_closure_list(
                  tag, arg_triple(0u), nullptr, nullptr, nullptr, a, a,
                  const_float(0.0f), nullptr, nullptr)};
    }
    if (name == "microfacet_ggx_aniso" || name == "microfacet_beckmann_aniso") {
      const auto tag = (name == "microfacet_beckmann_aniso")
                           ? OslClosureTag::microfacet_beckmann
                           : OslClosureTag::microfacet_ggx;
      return {.value = make_single_closure_list(
                  tag, arg_triple(0u), arg_triple(1u), nullptr, nullptr,
                  arg_float(2u), arg_float(3u), const_float(0.0f), nullptr,
                  nullptr)};
    }
    if (name == "microfacet_ggx_refraction" ||
        name == "microfacet_beckmann_refraction") {
      const auto tag = (name == "microfacet_beckmann_refraction")
                           ? OslClosureTag::microfacet_beckmann_refraction
                           : OslClosureTag::microfacet_ggx_refraction;
      auto *a = arg_float(1u);
      return {.value = make_single_closure_list(
                  tag, arg_triple(0u), nullptr, nullptr, nullptr, a, a,
                  arg_float(2u, 1.5f), nullptr, nullptr)};
    }
    if (name == "microfacet_multi_ggx") {
      auto *a = arg_float(1u);
      return {.value = make_single_closure_list(
                  OslClosureTag::microfacet_multi_ggx, arg_triple(0u), nullptr,
                  arg_triple(2u, {1.0f, 1.0f, 1.0f}), nullptr, a, a,
                  const_float(0.0f), nullptr, nullptr)};
    }
    if (name == "microfacet_multi_ggx_aniso") {
      return {.value = make_single_closure_list(
                  OslClosureTag::microfacet_multi_ggx, arg_triple(0u),
                  arg_triple(1u), arg_triple(4u, {1.0f, 1.0f, 1.0f}), nullptr,
                  arg_float(2u), arg_float(3u), const_float(0.0f), nullptr,
                  nullptr)};
    }
    if (name == "microfacet_multi_ggx_glass") {
      auto *a = arg_float(1u);
      return {.value = make_single_closure_list(
                  OslClosureTag::microfacet_multi_ggx_glass, arg_triple(0u),
                  nullptr, arg_triple(3u, {1.0f, 1.0f, 1.0f}), nullptr, a, a,
                  arg_float(2u, 1.5f), nullptr, nullptr)};
    }
    if (name == "ashikhmin_shirley") {
      return {.value = make_single_closure_list(
                  OslClosureTag::ashikhmin_shirley, arg_triple(0u),
                  arg_triple(1u), nullptr, nullptr, arg_float(2u),
                  arg_float(3u), nullptr, nullptr, nullptr)};
    }
    if (name == "dielectric_bsdf") {
      auto *tf_t = opt_float("thinfilm_thickness", const_float(0.0f));
      auto *tf_i = opt_float("thinfilm_ior", const_float(1.33f));
      return {.value = make_single_closure_list(
                  OslClosureTag::dielectric_bsdf, arg_triple(0u),
                  arg_triple(1u), arg_triple(2u, {1.0f, 1.0f, 1.0f}),
                  arg_triple(3u, {1.0f, 1.0f, 1.0f}), arg_float(4u),
                  arg_float(5u), arg_float(6u, 1.5f), tf_t, tf_i)};
    }
    if (name == "conductor_bsdf") {
      auto *tf_t = opt_float("thinfilm_thickness", const_float(0.0f));
      auto *tf_i = opt_float("thinfilm_ior", const_float(1.33f));
      return {.value = make_single_closure_list(
                  OslClosureTag::conductor_bsdf, arg_triple(0u), arg_triple(1u),
                  arg_triple(4u), arg_triple(5u), arg_float(2u), arg_float(3u),
                  nullptr, tf_t, tf_i)};
    }
    if (name == "generalized_schlick_bsdf") {
      auto *tf_t = opt_float("thinfilm_thickness", const_float(0.0f));
      auto *tf_i = opt_float("thinfilm_ior", const_float(1.33f));
      return {.value = make_single_closure_list(
                  OslClosureTag::generalized_schlick_bsdf, arg_triple(0u),
                  arg_triple(1u), arg_triple(2u, {1.0f, 1.0f, 1.0f}),
                  arg_triple(3u, {0.0f, 0.0f, 0.0f}), arg_float(4u),
                  arg_float(5u), arg_float(8u, 5.0f), tf_t, tf_i)};
    }
    report_error(
        luisa::format("CyclesOslTranslator: unsupported closure '{}'.", name));
    return {};
  }

  [[nodiscard]] osl::DualValue emit_noise(
      luisa::string_view noise_type, osl::NoiseOp op,
      luisa::span<const osl::DualValue> coords,
      luisa::span<const osl::DualValue> periods,
      const osl::Type *dst_type, bool want_deriv,
      luisa::span<const osl::NamedOption> opts) noexcept override {
    const bool unsupported_op =
        !periods.empty() || op == osl::NoiseOp::CELLNOISE ||
        op == osl::NoiseOp::HASHNOISE || op == osl::NoiseOp::PNOISE ||
        op == osl::NoiseOp::PSNOISE;
    const bool supported_type =
        noise_type.empty() || noise_type == "noise" || noise_type == "snoise" ||
        noise_type == "perlin" || noise_type == "uperlin";
    if (unsupported_op || !supported_type) {
      return osl::OSLTranslator::emit_noise(noise_type, op, coords, periods,
                                            dst_type, want_deriv, opts);
    }

    auto &b = xir_builder();
    const auto *f_t = Type::of<float>();
    const auto *f3_t = Type::of<luisa::float3>();
    const auto *i_t = Type::of<std::int32_t>();
    const auto *u_t = Type::of<std::uint32_t>();
    const auto *b_t = Type::of<bool>();

    luisa::vector<osl::DualValue> c_scalars;
    c_scalars.reserve(4u);
    for (const auto &c : coords) {
      if (c.val != nullptr && c.val->type() == f3_t) {
        for (std::uint32_t k = 0u; k < 3u; ++k) {
          auto *comp = b.call(f_t, xir::ArithmeticOp::EXTRACT,
                              {c.val, const_uint(k)});
          xir::Value *dx = (want_deriv && c.dx != nullptr)
                               ? b.call(f_t, xir::ArithmeticOp::EXTRACT,
                                        {c.dx, const_uint(k)})
                               : nullptr;
          xir::Value *dy = (want_deriv && c.dy != nullptr)
                               ? b.call(f_t, xir::ArithmeticOp::EXTRACT,
                                        {c.dy, const_uint(k)})
                               : nullptr;
          c_scalars.push_back({.val = comp, .dx = dx, .dy = dy});
        }
      } else {
        c_scalars.push_back(
            {.val = to_float_val(c.val),
             .dx = want_deriv ? c.dx : nullptr,
             .dy = want_deriv ? c.dy : nullptr});
      }
    }
    const std::size_t dim = c_scalars.size();
    if (dim == 0u || dim > 4u) {
      return osl::OSLTranslator::emit_noise(noise_type, op, coords, periods,
                                            dst_type, want_deriv, opts);
    }

    auto u_op = [&](xir::ArithmeticOp aop, xir::Value *x,
                    xir::Value *y) noexcept {
      return b.call(u_t, aop, {x, y});
    };
    auto f_op = [&](xir::ArithmeticOp aop, xir::Value *x,
                    xir::Value *y) noexcept {
      return b.call(f_t, aop, {x, y});
    };
    auto f_sel = [&](xir::Value *fv, xir::Value *tv,
                     xir::Value *cond) noexcept {
      return b.call(f_t, xir::ArithmeticOp::SELECT, {fv, tv, cond});
    };
    auto u_cmp = [&](xir::ArithmeticOp aop, xir::Value *x,
                     xir::Value *y) noexcept {
      return b.call(b_t, aop, {x, y});
    };
    auto b_and = [&](xir::Value *x, xir::Value *y) noexcept {
      return b.call(b_t, xir::ArithmeticOp::BINARY_BIT_AND, {x, y});
    };
    auto b_or = [&](xir::Value *x, xir::Value *y) noexcept {
      return b.call(b_t, xir::ArithmeticOp::BINARY_BIT_OR, {x, y});
    };
    auto u_rotl = [&](xir::Value *v, std::uint32_t k) noexcept {
      return u_op(xir::ArithmeticOp::BINARY_BIT_OR,
                  u_op(xir::ArithmeticOp::BINARY_SHIFT_LEFT, v, const_uint(k)),
                  u_op(xir::ArithmeticOp::BINARY_SHIFT_RIGHT, v,
                       const_uint(32u - k)));
    };
    auto bjfinal = [&](xir::Value *a, xir::Value *bv,
                       xir::Value *c) noexcept -> xir::Value * {
      c = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, c, bv);
      c = u_op(xir::ArithmeticOp::BINARY_SUB, c, u_rotl(bv, 14u));
      a = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, a, c);
      a = u_op(xir::ArithmeticOp::BINARY_SUB, a, u_rotl(c, 11u));
      bv = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, bv, a);
      bv = u_op(xir::ArithmeticOp::BINARY_SUB, bv, u_rotl(a, 25u));
      c = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, c, bv);
      c = u_op(xir::ArithmeticOp::BINARY_SUB, c, u_rotl(bv, 16u));
      a = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, a, c);
      a = u_op(xir::ArithmeticOp::BINARY_SUB, a, u_rotl(c, 4u));
      bv = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, bv, a);
      bv = u_op(xir::ArithmeticOp::BINARY_SUB, bv, u_rotl(a, 14u));
      c = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, c, bv);
      c = u_op(xir::ArithmeticOp::BINARY_SUB, c, u_rotl(bv, 24u));
      return c;
    };
    auto bjmix = [&](xir::Value *&a, xir::Value *&bv,
                     xir::Value *&c) noexcept {
      a = u_op(xir::ArithmeticOp::BINARY_SUB, a, c);
      a = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, a, u_rotl(c, 4u));
      c = u_op(xir::ArithmeticOp::BINARY_ADD, c, bv);
      bv = u_op(xir::ArithmeticOp::BINARY_SUB, bv, a);
      bv = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, bv, u_rotl(a, 6u));
      a = u_op(xir::ArithmeticOp::BINARY_ADD, a, c);
      c = u_op(xir::ArithmeticOp::BINARY_SUB, c, bv);
      c = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, c, u_rotl(bv, 8u));
      bv = u_op(xir::ArithmeticOp::BINARY_ADD, bv, a);
      a = u_op(xir::ArithmeticOp::BINARY_SUB, a, c);
      a = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, a, u_rotl(c, 16u));
      c = u_op(xir::ArithmeticOp::BINARY_ADD, c, bv);
      bv = u_op(xir::ArithmeticOp::BINARY_SUB, bv, a);
      bv = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, bv, u_rotl(a, 19u));
      a = u_op(xir::ArithmeticOp::BINARY_ADD, a, c);
      c = u_op(xir::ArithmeticOp::BINARY_SUB, c, bv);
      c = u_op(xir::ArithmeticOp::BINARY_BIT_XOR, c, u_rotl(bv, 4u));
      bv = u_op(xir::ArithmeticOp::BINARY_ADD, bv, a);
    };
    auto hash_coords =
        [&](luisa::span<xir::Value *const> ucoords) noexcept -> xir::Value * {
      xir::Value *init =
          const_uint(0xdeadbeefu + (static_cast<std::uint32_t>(dim) << 2u) +
                     13u);
      xir::Value *a = init;
      xir::Value *bv = init;
      xir::Value *c = init;
      a = u_op(xir::ArithmeticOp::BINARY_ADD, a, ucoords[0]);
      if (dim >= 2u) {
        bv = u_op(xir::ArithmeticOp::BINARY_ADD, bv, ucoords[1]);
      }
      if (dim >= 3u) {
        c = u_op(xir::ArithmeticOp::BINARY_ADD, c, ucoords[2]);
      }
      if (dim == 4u) {
        bjmix(a, bv, c);
        a = u_op(xir::ArithmeticOp::BINARY_ADD, a, ucoords[3]);
      }
      return bjfinal(a, bv, c);
    };

    bool any_input_deriv = false;
    for (const auto &cs : c_scalars) {
      any_input_deriv |= cs.has_derivs();
    }
    const bool need_grad = want_deriv && any_input_deriv;

    luisa::vector<xir::Value *> u0(dim), u1(dim), tx(dim), tx_m1(dim), fade(dim),
        inv_fade(dim), dfade(dim);
    for (std::size_t d = 0u; d < dim; ++d) {
      xir::Value *fl = b.call(f_t, xir::ArithmeticOp::FLOOR, {c_scalars[d].val});
      xir::Value *ix0 = b.static_cast_(i_t, fl);
      xir::Value *ix1 =
          b.call(i_t, xir::ArithmeticOp::BINARY_ADD, {ix0, const_int(1)});
      u0[d] = b.static_cast_(u_t, ix0);
      u1[d] = b.static_cast_(u_t, ix1);
      xir::Value *t = f_op(xir::ArithmeticOp::BINARY_SUB, c_scalars[d].val, fl);
      tx[d] = t;
      tx_m1[d] = f_op(xir::ArithmeticOp::BINARY_SUB, t, const_float(1.0f));
      xir::Value *t2 = f_op(xir::ArithmeticOp::BINARY_MUL, t, t);
      xir::Value *t3 = f_op(xir::ArithmeticOp::BINARY_MUL, t2, t);
      xir::Value *poly = f_op(
          xir::ArithmeticOp::BINARY_ADD,
          f_op(xir::ArithmeticOp::BINARY_MUL, t,
               f_op(xir::ArithmeticOp::BINARY_SUB,
                    f_op(xir::ArithmeticOp::BINARY_MUL, t, const_float(6.0f)),
                    const_float(15.0f))),
          const_float(10.0f));
      fade[d] = f_op(xir::ArithmeticOp::BINARY_MUL, t3, poly);
      inv_fade[d] =
          f_op(xir::ArithmeticOp::BINARY_SUB, const_float(1.0f), fade[d]);
      if (need_grad) {
        xir::Value *tm1 = tx_m1[d];
        dfade[d] = f_op(
            xir::ArithmeticOp::BINARY_MUL, const_float(30.0f),
            f_op(xir::ArithmeticOp::BINARY_MUL, t2,
                 f_op(xir::ArithmeticOp::BINARY_MUL, tm1, tm1)));
      }
    }

    const std::size_t corners = std::size_t{1u} << dim;
    luisa::vector<xir::Value *> corner_hashes(corners);
    luisa::vector<xir::Value *> uc(dim);
    for (std::size_t mask = 0u; mask < corners; ++mask) {
      for (std::size_t d = 0u; d < dim; ++d) {
        uc[d] = (mask & (std::size_t{1u} << d)) ? u1[d] : u0[d];
      }
      corner_hashes[mask] = hash_coords(uc);
    }

    const bool is_triple = is_triple_osl_type(dst_type);
    const std::uint32_t channels = is_triple ? 3u : 1u;
    const bool is_unsigned =
        noise_type == "uperlin" || noise_type == "noise" ||
        (noise_type.empty() && op != osl::NoiseOp::SNOISE);
    const float dim_scale = dim == 1u   ? 0.25f
                            : dim == 2u ? 0.6616f
                            : dim == 3u ? 0.9820f
                                        : 0.8344f;
    const float total_scale = dim_scale * (is_unsigned ? 0.5f : 1.0f);

    auto eval_channel = [&](std::uint32_t ch) noexcept -> osl::DualValue {
      struct Node {
        xir::Value *val{nullptr};
        luisa::vector<xir::Value *> grad;
      };
      luisa::vector<Node> cur(corners);
      for (std::size_t mask = 0u; mask < corners; ++mask) {
        xir::Value *h =
            (ch == 0u) ? corner_hashes[mask]
                       : u_op(xir::ArithmeticOp::BINARY_SHIFT_RIGHT,
                              corner_hashes[mask], const_uint(ch * 8u));
        luisa::vector<xir::Value *> gvec(dim, const_float(0.0f));
        if (dim == 1u) {
          xir::Value *hh =
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, h, const_uint(15u));
          xir::Value *gabs = f_op(
              xir::ArithmeticOp::BINARY_ADD, const_float(1.0f),
              b.static_cast_(f_t, u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh,
                                       const_uint(7u))));
          xir::Value *neg = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(8u)),
              const_uint(0u));
          gvec[0] = f_sel(gabs,
                          b.call(f_t, xir::ArithmeticOp::UNARY_MINUS, {gabs}),
                          neg);
        } else if (dim == 2u) {
          xir::Value *hh =
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, h, const_uint(7u));
          xir::Value *lt4 =
              u_cmp(xir::ArithmeticOp::BINARY_LESS, hh, const_uint(4u));
          xir::Value *neg_u = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(1u)),
              const_uint(0u));
          xir::Value *neg_v = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(2u)),
              const_uint(0u));
          xir::Value *su = f_sel(const_float(1.0f), const_float(-1.0f), neg_u);
          xir::Value *sv = f_sel(const_float(2.0f), const_float(-2.0f), neg_v);
          gvec[0] = f_sel(sv, su, lt4);
          gvec[1] = f_sel(su, sv, lt4);
        } else if (dim == 3u) {
          xir::Value *hh =
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, h, const_uint(15u));
          xir::Value *neg_u = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(1u)),
              const_uint(0u));
          xir::Value *neg_v = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(2u)),
              const_uint(0u));
          xir::Value *su = f_sel(const_float(1.0f), const_float(-1.0f), neg_u);
          xir::Value *sv = f_sel(const_float(1.0f), const_float(-1.0f), neg_v);
          xir::Value *lt8 =
              u_cmp(xir::ArithmeticOp::BINARY_LESS, hh, const_uint(8u));
          xir::Value *ge8 =
              u_cmp(xir::ArithmeticOp::BINARY_GREATER_EQUAL, hh, const_uint(8u));
          xir::Value *lt4 =
              u_cmp(xir::ArithmeticOp::BINARY_LESS, hh, const_uint(4u));
          xir::Value *ge4 =
              u_cmp(xir::ArithmeticOp::BINARY_GREATER_EQUAL, hh, const_uint(4u));
          xir::Value *h12_14 = b_or(
              u_cmp(xir::ArithmeticOp::BINARY_EQUAL, hh, const_uint(12u)),
              u_cmp(xir::ArithmeticOp::BINARY_EQUAL, hh, const_uint(14u)));
          xir::Value *not_h12_14 = b_and(
              u_cmp(xir::ArithmeticOp::BINARY_NOT_EQUAL, hh, const_uint(12u)),
              u_cmp(xir::ArithmeticOp::BINARY_NOT_EQUAL, hh, const_uint(14u)));
          xir::Value *use_z = b_and(ge4, not_h12_14);
          gvec[0] = f_op(xir::ArithmeticOp::BINARY_ADD,
                         f_sel(const_float(0.0f), su, lt8),
                         f_sel(const_float(0.0f), sv, h12_14));
          gvec[1] = f_op(xir::ArithmeticOp::BINARY_ADD,
                         f_sel(const_float(0.0f), su, ge8),
                         f_sel(const_float(0.0f), sv, lt4));
          gvec[2] = f_sel(const_float(0.0f), sv, use_z);
        } else {
          xir::Value *hh =
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, h, const_uint(31u));
          xir::Value *neg_u = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(1u)),
              const_uint(0u));
          xir::Value *neg_v = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(2u)),
              const_uint(0u));
          xir::Value *neg_s = u_cmp(
              xir::ArithmeticOp::BINARY_NOT_EQUAL,
              u_op(xir::ArithmeticOp::BINARY_BIT_AND, hh, const_uint(4u)),
              const_uint(0u));
          xir::Value *su = f_sel(const_float(1.0f), const_float(-1.0f), neg_u);
          xir::Value *sv = f_sel(const_float(1.0f), const_float(-1.0f), neg_v);
          xir::Value *ss = f_sel(const_float(1.0f), const_float(-1.0f), neg_s);
          xir::Value *lt24 =
              u_cmp(xir::ArithmeticOp::BINARY_LESS, hh, const_uint(24u));
          xir::Value *ge24 = u_cmp(xir::ArithmeticOp::BINARY_GREATER_EQUAL, hh,
                                   const_uint(24u));
          xir::Value *lt16 =
              u_cmp(xir::ArithmeticOp::BINARY_LESS, hh, const_uint(16u));
          xir::Value *ge16 = u_cmp(xir::ArithmeticOp::BINARY_GREATER_EQUAL, hh,
                                   const_uint(16u));
          xir::Value *lt8 =
              u_cmp(xir::ArithmeticOp::BINARY_LESS, hh, const_uint(8u));
          xir::Value *ge8 =
              u_cmp(xir::ArithmeticOp::BINARY_GREATER_EQUAL, hh, const_uint(8u));
          gvec[0] = f_sel(const_float(0.0f), su, lt24);
          gvec[1] = f_op(xir::ArithmeticOp::BINARY_ADD,
                         f_sel(const_float(0.0f), su, ge24),
                         f_sel(const_float(0.0f), sv, lt16));
          gvec[2] = f_op(xir::ArithmeticOp::BINARY_ADD,
                         f_sel(const_float(0.0f), sv, ge16),
                         f_sel(const_float(0.0f), ss, lt8));
          gvec[3] = f_sel(const_float(0.0f), ss, ge8);
        }

        xir::Value *dot_v = nullptr;
        for (std::size_t d = 0u; d < dim; ++d) {
          xir::Value *delta =
              (mask & (std::size_t{1u} << d)) ? tx_m1[d] : tx[d];
          xir::Value *term =
              f_op(xir::ArithmeticOp::BINARY_MUL, gvec[d], delta);
          dot_v = dot_v ? f_op(xir::ArithmeticOp::BINARY_ADD, dot_v, term)
                        : term;
        }
        cur[mask].val = dot_v;
        if (need_grad) {
          cur[mask].grad = std::move(gvec);
        }
      }

      for (std::size_t d = 0u; d < dim; ++d) {
        const std::size_t half = cur.size() >> 1u;
        luisa::vector<Node> next(half);
        for (std::size_t j = 0u; j < half; ++j) {
          const auto &a_n = cur[2u * j];
          const auto &b_n = cur[2u * j + 1u];
          next[j].val = f_op(
              xir::ArithmeticOp::BINARY_ADD,
              f_op(xir::ArithmeticOp::BINARY_MUL, inv_fade[d], a_n.val),
              f_op(xir::ArithmeticOp::BINARY_MUL, fade[d], b_n.val));
          if (need_grad) {
            next[j].grad.resize(dim);
            xir::Value *diff =
                f_op(xir::ArithmeticOp::BINARY_SUB, b_n.val, a_n.val);
            for (std::size_t g = 0u; g < dim; ++g) {
              xir::Value *dg = f_op(
                  xir::ArithmeticOp::BINARY_ADD,
                  f_op(xir::ArithmeticOp::BINARY_MUL, inv_fade[d],
                       a_n.grad[g]),
                  f_op(xir::ArithmeticOp::BINARY_MUL, fade[d], b_n.grad[g]));
              if (g == d) {
                dg = f_op(xir::ArithmeticOp::BINARY_ADD, dg,
                          f_op(xir::ArithmeticOp::BINARY_MUL, dfade[d], diff));
              }
              next[j].grad[g] = dg;
            }
          }
        }
        cur = std::move(next);
      }

      xir::Value *val =
          f_op(xir::ArithmeticOp::BINARY_MUL, cur[0].val, const_float(total_scale));
      if (is_unsigned) {
        val = f_op(xir::ArithmeticOp::BINARY_ADD, val, const_float(0.5f));
      }
      if (!need_grad) {
        return {.val = val};
      }
      xir::Value *dx_sum = nullptr;
      xir::Value *dy_sum = nullptr;
      for (std::size_t d = 0u; d < dim; ++d) {
        xir::Value *gd = f_op(xir::ArithmeticOp::BINARY_MUL, cur[0].grad[d],
                              const_float(total_scale));
        if (c_scalars[d].dx != nullptr) {
          xir::Value *term =
              f_op(xir::ArithmeticOp::BINARY_MUL, gd, c_scalars[d].dx);
          dx_sum =
              dx_sum ? f_op(xir::ArithmeticOp::BINARY_ADD, dx_sum, term) : term;
        }
        if (c_scalars[d].dy != nullptr) {
          xir::Value *term =
              f_op(xir::ArithmeticOp::BINARY_MUL, gd, c_scalars[d].dy);
          dy_sum =
              dy_sum ? f_op(xir::ArithmeticOp::BINARY_ADD, dy_sum, term) : term;
        }
      }
      return {.val = val, .dx = dx_sum, .dy = dy_sum};
    };

    if (!is_triple) {
      return eval_channel(0u);
    }
    const osl::DualValue c0 = eval_channel(0u);
    const osl::DualValue c1 = eval_channel(1u);
    const osl::DualValue c2 = eval_channel(2u);
    osl::DualValue out;
    out.val = b.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                     {c0.val, c1.val, c2.val});
    if (need_grad &&
        (c0.has_derivs() || c1.has_derivs() || c2.has_derivs())) {
      auto dv = [&](xir::Value *v) noexcept {
        return v != nullptr ? v : const_float(0.0f);
      };
      out.dx = b.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                      {dv(c0.dx), dv(c1.dx), dv(c2.dx)});
      out.dy = b.call(f3_t, xir::ArithmeticOp::AGGREGATE,
                      {dv(c0.dy), dv(c1.dy), dv(c2.dy)});
    }
    return out;
  }
};

[[nodiscard]] CompiledOslScript
compile_osl_callable(const OslScriptBinding &script) {
  static std::mutex cache_mutex;
  static std::unordered_map<std::string, CompiledOslScript> cache;

  const auto key = script_cache_key(script);
  {
    const std::scoped_lock lock{cache_mutex};
    if (const auto iter = cache.find(key); iter != cache.end()) {
      return iter->second;
    }
  }

  auto shader = osl::OSOParser::parse(script.oso_bytecode);
  LUISA_ASSERT(shader != nullptr,
               "Failed to parse exported .oso bytecode for OSL script '{}'.",
               script.script_name);
  const auto bounds = analyze_osl_shader_closure_bounds(*shader);

  osl::ShaderGroup group;
  const std::uint32_t layer =
      group.add_layer(script.script_name, shader.get());
  for (const auto &p : script.params) {
    if (p.is_output) {
      if (p.kind != OslParamKind::closure && p.kind != OslParamKind::string) {
        group.mark_output(layer, p.name, p.has_derivatives);
      }
      continue;
    }
    if (p.kind == OslParamKind::string) {
      group.set_parameter(layer, p.name, osl::ParameterValue{p.default_string});
      continue;
    }
    if (p.kind == OslParamKind::closure) {
      continue;
    }
    if (p.is_linked) {
      group.mark_dynamic_input(layer, p.name, p.has_derivatives);
    } else if (!p.use_oso_init) {
      switch (p.kind) {
        case OslParamKind::floating:
          group.set_parameter(layer, p.name,
                              osl::ParameterValue{p.default_float3[0]});
          break;
        case OslParamKind::integer:
          group.set_parameter(layer, p.name,
                              osl::ParameterValue{p.default_int});
          break;
        case OslParamKind::color:
        case OslParamKind::vector:
        case OslParamKind::point:
        case OslParamKind::normal:
          group.set_parameter(
              layer, p.name,
              osl::ParameterValue{luisa::float3{p.default_float3[0],
                                                p.default_float3[1],
                                                p.default_float3[2]}});
          break;
        case OslParamKind::closure:
        case OslParamKind::string:
          break;
      }
    }
  }
  if (script.has_closure_output) {
    for (const auto &sym : shader->symbols()) {
      if (sym->tag() == osl::Symbol::Tag::SYM_OUTPUT_PARAM &&
          sym->is_root() && sym->type() != nullptr &&
          sym->type()->tag() == osl::Type::Tag::CLOSURE) {
        group.mark_output(layer, sym->identifier(), false);
      }
    }
  }

  xir::Module module;
  auto *fn = module.create_callable(nullptr);
  std::string xir_name = "cycles_osl_" + script.script_name;
  for (char &c : xir_name) {
    const auto uc = static_cast<unsigned char>(c);
    const bool ok = (uc >= '0' && uc <= '9') || (uc >= 'A' && uc <= 'Z') ||
                    (uc >= 'a' && uc <= 'z') || uc == '_';
    if (!ok) {
      c = '_';
    }
  }
  fn->set_name(xir_name);
  CyclesOslTranslator translator{script, fn, bounds.translator_bound};
  osl::TranslatorOptions opts;
  opts.verify_xir = true;
  opts.run_xir_passes = true;
  opts.restructure_for_ast = true;
  opts.prune_dead_layers = false;
  const auto res = translator.translate(group, fn, opts);
  if (!res.succeeded()) {
    std::string msg;
    for (const auto &err : res.errors) {
      if (!msg.empty()) {
        msg += "; ";
      }
      msg += err.message;
    }
    LUISA_ERROR_WITH_LOCATION("OSL translation failed for '{}': {}",
                              script.script_name, msg);
  }
  auto ast = xir::xir_to_ast_translate(*fn, {});
  const auto mask = translator.emitted_closure_mask();
  CompiledOslScript compiled{
      .ast = std::move(ast),
      .emitted_closure_mask = mask,
      .max_emitted_closures =
          mask == 0ull
              ? 0u
              : std::clamp<std::uint32_t>(bounds.emitted_bound, 1u,
                                            max_osl_closures),
      .uses_transforms = translator.uses_transforms(),
      .uses_object_info = translator.uses_object_info(),
      .writes_N = translator.writes_N()};

  const std::scoped_lock lock{cache_mutex};
  auto [iter, inserted] = cache.emplace(key, std::move(compiled));
  (void)inserted;
  return iter->second;
}

[[nodiscard]] Var<OslAttrInput>
fetch_osl_attribute(const KernelGlobals &kernel_globals,
                    const ShaderData &shader_data, const OslAttrBinding &attr,
                    bool want_derivs) noexcept {
  Var<OslAttrInput> result;
  result.val = make_float4(0.0f);
  result.dx = make_float4(0.0f);
  result.dy = make_float4(0.0f);
  result.found = 0;

  const auto summary = kernel_globals.attribute_summary_for_id(attr.attr_id);
  const auto is_uv =
      attr.attr_id == static_cast<std::uint64_t>(ATTR_STD_UV) ||
      attr.name == "geom:uv" || attr.name == "uv";
  const auto is_generated =
      attr.attr_id == static_cast<std::uint64_t>(ATTR_STD_GENERATED) ||
      attr.name == "geom:generated";

  if (summary.element_mask == 0u) {
    if (is_uv) {
      $if(shader_data.type == primitive_lamp) {
        result.val = make_float4(1.0f - shader_data.u - shader_data.v,
                                 shader_data.u, 0.0f, 1.0f);
        if (want_derivs) {
          result.dx =
              make_float4(-shader_data.du.dx - shader_data.dv.dx,
                          shader_data.du.dx, 0.0f, 0.0f);
          result.dy =
              make_float4(-shader_data.du.dy - shader_data.dv.dy,
                          shader_data.du.dy, 0.0f, 0.0f);
        }
        result.found = 1;
      };
    } else if (is_generated) {
      if (want_derivs) {
        const auto dP = shading_position_dual(shader_data);
        const auto gen =
            kernel_globals.object_inverse_position_transform_if_object_derivative(
                shader_data, dP);
        result.val = make_float4(gen.val, 1.0f);
        result.dx = make_float4(gen.dx, 0.0f);
        result.dy = make_float4(gen.dy, 0.0f);
      } else {
        const auto gen =
            kernel_globals.object_inverse_position_transform_if_object(
                shader_data, shader_data.P);
        result.val = make_float4(gen, 1.0f);
      }
      result.found = 1;
    }
    return result;
  }

  const auto descriptor =
      find_attribute(kernel_globals, shader_data, Expr<luisa::ulong>{attr.attr_id});
  const bool may_float = (summary.type_mask & (1u << NODE_ATTR_FLOAT)) != 0u;
  const bool may_float2 = (summary.type_mask & (1u << NODE_ATTR_FLOAT2)) != 0u;
  const bool may_float4 =
      (summary.type_mask & ((1u << NODE_ATTR_FLOAT4) | (1u << NODE_ATTR_RGBA))) !=
      0u;
  const bool may_float3 = (summary.type_mask & (1u << NODE_ATTR_FLOAT3)) != 0u;

  auto eval_found = [&]() noexcept {
    Bool handled = false;
    if (may_float) {
      $if(descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT)) {
        if (want_derivs) {
          const auto d = primitive_surface_attribute_float_derivative(
              kernel_globals, shader_data, descriptor);
          result.val = make_float4(d.val, 0.0f, 0.0f, 1.0f);
          result.dx = make_float4(d.dx, 0.0f, 0.0f, 0.0f);
          result.dy = make_float4(d.dy, 0.0f, 0.0f, 0.0f);
        } else {
          const auto v = primitive_surface_attribute_float(
              kernel_globals, shader_data, descriptor);
          result.val = make_float4(v, 0.0f, 0.0f, 1.0f);
        }
        handled = true;
      };
    }
    if (may_float2) {
      $if((!handled) &
          (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT2))) {
        if (want_derivs) {
          const auto d = primitive_surface_attribute_float2_derivative(
              kernel_globals, shader_data, descriptor);
          result.val = make_float4(d.val, 0.0f, 1.0f);
          result.dx = make_float4(d.dx, 0.0f, 0.0f);
          result.dy = make_float4(d.dy, 0.0f, 0.0f);
        } else {
          const auto v = primitive_surface_attribute_float2(
              kernel_globals, shader_data, descriptor);
          result.val = make_float4(v, 0.0f, 1.0f);
        }
        handled = true;
      };
    }
    if (may_float4) {
      $if((!handled) &
          ((descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_FLOAT4)) |
           (descriptor.type == static_cast<std::uint32_t>(NODE_ATTR_RGBA)))) {
        if (want_derivs) {
          const auto d = primitive_surface_attribute_float4_derivative(
              kernel_globals, shader_data, descriptor);
          result.val = d.val;
          result.dx = d.dx;
          result.dy = d.dy;
        } else {
          result.val = primitive_surface_attribute_float4(
              kernel_globals, shader_data, descriptor);
        }
        handled = true;
      };
    }
    if (may_float3) {
      $if(!handled) {
        if (want_derivs) {
          const auto d = primitive_surface_attribute_float3_derivative(
              kernel_globals, shader_data, descriptor);
          result.val = make_float4(d.val, 1.0f);
          result.dx = make_float4(d.dx, 0.0f);
          result.dy = make_float4(d.dy, 0.0f);
        } else {
          const auto v = primitive_surface_attribute_float3(
              kernel_globals, shader_data, descriptor);
          result.val = make_float4(v, 1.0f);
        }
      };
    }
  };

  $if(descriptor.offset != static_cast<std::int32_t>(ATTR_STD_NOT_FOUND)) {
    eval_found();
    result.found = 1;
  }
  $else {
    if (is_uv) {
      $if(shader_data.type == primitive_lamp) {
        result.val = make_float4(1.0f - shader_data.u - shader_data.v,
                                 shader_data.u, 0.0f, 1.0f);
        if (want_derivs) {
          result.dx =
              make_float4(-shader_data.du.dx - shader_data.dv.dx,
                          shader_data.du.dx, 0.0f, 0.0f);
          result.dy =
              make_float4(-shader_data.du.dy - shader_data.dv.dy,
                          shader_data.du.dy, 0.0f, 0.0f);
        }
        result.found = 1;
      };
    } else if (is_generated) {
      if (want_derivs) {
        const auto dP = shading_position_dual(shader_data);
        const auto gen =
            kernel_globals.object_inverse_position_transform_if_object_derivative(
                shader_data, dP);
        result.val = make_float4(gen.val, 1.0f);
        result.dx = make_float4(gen.dx, 0.0f);
        result.dy = make_float4(gen.dy, 0.0f);
      } else {
        const auto gen =
            kernel_globals.object_inverse_position_transform_if_object(
                shader_data, shader_data.P);
        result.val = make_float4(gen, 1.0f);
      }
      result.found = 1;
    }
  };
  return result;
}

void emit_osl_closures(const KernelGlobals &kernel_globals,
                       ShaderData &shader_data, const PathState &path_state,
                       std::uint32_t kernel_features,
                       std::uint32_t node_feature_mask,
                       Expr<float> mix_weight,
                       const Var<OslClosureList> &closures,
                       const EvaluationTransition &,
                       std::uint64_t allowed_tag_mask = ~0ull,
                       std::uint32_t max_closures = max_osl_closures) noexcept {
  const auto capped_closures =
      std::min<std::uint32_t>(max_closures, max_osl_closures);
  if (allowed_tag_mask == 0ull || capped_closures == 0u) {
    return;
  }

  constexpr std::uint64_t kNeedSpecNMask =
      closure_tag_bit(OslClosureTag::translucent) |
      closure_tag_bit(OslClosureTag::microfacet_ggx) |
      closure_tag_bit(OslClosureTag::microfacet_beckmann) |
      closure_tag_bit(OslClosureTag::ashikhmin_shirley) |
      closure_tag_bit(OslClosureTag::microfacet_multi_ggx) |
      closure_tag_bit(OslClosureTag::reflection) |
      closure_tag_bit(OslClosureTag::microfacet_ggx_refraction) |
      closure_tag_bit(OslClosureTag::microfacet_beckmann_refraction) |
      closure_tag_bit(OslClosureTag::refraction) |
      closure_tag_bit(OslClosureTag::microfacet_ggx_glass) |
      closure_tag_bit(OslClosureTag::microfacet_beckmann_glass) |
      closure_tag_bit(OslClosureTag::microfacet_multi_ggx_glass) |
      closure_tag_bit(OslClosureTag::dielectric_bsdf) |
      closure_tag_bit(OslClosureTag::generalized_schlick_bsdf) |
      closure_tag_bit(OslClosureTag::conductor_bsdf);
  constexpr std::uint64_t kNeedPlainNMask =
      closure_tag_bit(OslClosureTag::diffuse) |
      closure_tag_bit(OslClosureTag::oren_nayar) |
      closure_tag_bit(OslClosureTag::sheen) |
      closure_tag_bit(OslClosureTag::ashikhmin_velvet) |
      closure_tag_bit(OslClosureTag::diffuse_toon) |
      closure_tag_bit(OslClosureTag::glossy_toon);
  const bool need_norm_N =
      (allowed_tag_mask & (kNeedSpecNMask | kNeedPlainNMask)) != 0ull;
  const bool need_spec_N = (allowed_tag_mask & kNeedSpecNMask) != 0ull;

  const Bool bsdf_allowed = (node_feature_mask & kernel_feature_node_bsdf) != 0u;
  const Bool emission_allowed =
      (node_feature_mask & kernel_feature_node_emission) != 0u;
  const Bool diffuse_vis =
      (path_state.visibility & path_ray_visibility_diffuse) != 0u;
  const Bool reflective_caustics =
      kernel_globals.caustics_reflective() | !diffuse_vis;
  const Bool refractive_caustics =
      kernel_globals.caustics_refractive() | !diffuse_vis;

  auto emit_entry = [&](const Var<OslClosureEntry> &entry) noexcept {
    const Float3 raw_w = max(entry.weight * mix_weight, make_float3(0.0f));
    const Float3 N =
        need_norm_N
            ? native_vector_math::safe_normalize_nonzero_or(entry.N, shader_data.N)
            : shader_data.N;
    const Float3 spec_N =
        need_spec_N ? maybe_ensure_valid_specular_reflection(shader_data, N)
                    : N;

    auto emit_body = [&](bool use_switch) noexcept {
      auto add_cases = [&](auto &&body, auto... tags) noexcept {
        std::uint64_t group_mask = 0ull;
        ((group_mask |= closure_tag_bit(tags)), ...);
        if ((allowed_tag_mask & group_mask) == 0ull) {
          return;
        }
        if (!use_switch) {
          body();
          return;
        }
        ::luisa::compute::detail::SwitchCaseStmtBuilder::create_with_comment(
            ::luisa::compute::dsl_detail::format_source_location(__FILE__,
                                                                 __LINE__),
            static_cast<std::uint32_t>(tags)...) %
            [&]() noexcept -> void { body(); };
      };

      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed) { diffuse_setup(shader_data, N, raw_w); };
          },
          OslClosureTag::diffuse);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed) {
              const Float3 w = raw_w;
              $if(entry.p0 < 1.0e-5f) { diffuse_setup(shader_data, N, w); }
              $else {
                oren_nayar_setup(shader_data, N, w, entry.p0,
                                 clamp(entry.color_arg, 0.0f, 1.0f));
              };
            };
          },
          OslClosureTag::oren_nayar);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed) { translucent_setup(shader_data, spec_N, raw_w); };
          },
          OslClosureTag::translucent);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed) {
              transparent_setup(shader_data, path_state, raw_w);
            };
          },
          OslClosureTag::transparent);
      add_cases(
          [&]() noexcept {
            $if(emission_allowed) {
              Float3 ew = raw_w;
              $if((shader_data.flag & shader_data_is_volume_shader_eval) != 0u) {
                if (const auto density =
                        kernel_globals.object_volume_density(shader_data.object)) {
                  ew *= *density;
                }
              };
              $if(any(ew > 0.0f)) {
                $if((shader_data.flag & shader_data_emission) != 0u) {
                  shader_data.closure_emission_background += ew;
                }
                $else {
                  shader_data.flag |= shader_data_emission;
                  shader_data.closure_emission_background = ew;
                };
              };
            };
          },
          OslClosureTag::emission);
      add_cases(
          [&]() noexcept {
            $if(emission_allowed & any(raw_w > 0.0f)) {
              $if((shader_data.flag & shader_data_emission) != 0u) {
                shader_data.closure_emission_background += raw_w;
              }
              $else {
                shader_data.flag |= shader_data_emission;
                shader_data.closure_emission_background = raw_w;
              };
            };
          },
          OslClosureTag::background);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed & any(raw_w > 0.0f)) {
              if (shader_data.closure != nullptr) {
                static_cast<void>(shader_data.closure->allocate(
                    static_cast<std::uint32_t>(CLOSURE_HOLDOUT_ID), raw_w));
              }
              shader_data.flag |= static_cast<std::uint32_t>(SD_HOLDOUT);
            };
          },
          OslClosureTag::holdout);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed & reflective_caustics) {
              if (shader_data.closure != nullptr) {
                auto &pool = *shader_data.closure;
                const auto is_refl =
                    entry.tag ==
                    static_cast<std::uint32_t>(OslClosureTag::reflection);
                const auto is_beckmann =
                    entry.tag ==
                    static_cast<std::uint32_t>(
                        OslClosureTag::microfacet_beckmann);
                const auto is_ashikhmin =
                    entry.tag ==
                    static_cast<std::uint32_t>(
                        OslClosureTag::ashikhmin_shirley);
                const auto is_multi =
                    entry.tag ==
                    static_cast<std::uint32_t>(
                        OslClosureTag::microfacet_multi_ggx);

                const Float ax = select(
                    select(clamp(entry.p0, 0.0f, 1.0f),
                           clamp(entry.p0, 1.0e-4f, 1.0f), is_ashikhmin),
                    Float{0.0f}, is_refl);
                const Float ay = select(
                    select(clamp(entry.p1, 0.0f, 1.0f),
                           clamp(entry.p1, 1.0e-4f, 1.0f), is_ashikhmin),
                    Float{0.0f}, is_refl);
                const Float raw_eta = select(entry.p2, entry.p0, is_refl);
                const Bool has_dielectric_fresnel = raw_eta > 0.0f;
                const auto backfacing =
                    (shader_data.flag & shader_data_backfacing) != 0u;
                const Float eta = select(max(raw_eta, 1.0e-5f),
                                         1.0f / max(raw_eta, 1.0e-5f),
                                         backfacing);

                const auto allocated = bsdf_allocate(shader_data, raw_w);
                $if(allocated.valid) {
                  UInt ctype =
                      static_cast<std::uint32_t>(CLOSURE_BSDF_MICROFACET_GGX_ID);
                  $if(is_beckmann) {
                    ctype = static_cast<std::uint32_t>(
                        CLOSURE_BSDF_MICROFACET_BECKMANN_ID);
                  }
                  $elif(is_ashikhmin) {
                    ctype = static_cast<std::uint32_t>(
                        CLOSURE_BSDF_ASHIKHMIN_SHIRLEY_ID);
                  };
                  MicrofacetParam microfacet{
                      .alpha_x = ax,
                      .alpha_y = ay,
                      .ior = select(1.0f, eta, has_dielectric_fresnel),
                      .energy_scale = 1.0f,
                      .fresnel_type = select(
                          static_cast<std::uint32_t>(MicrofacetFresnel::none),
                          static_cast<std::uint32_t>(
                              MicrofacetFresnel::dielectric),
                          has_dielectric_fresnel),
                      .T = entry.T};
                  pool.set_type(allocated.index, ctype);
                  pool.set_normal(allocated.index, spec_N);
                  const auto common = pool.common(allocated.index);
                  $if(has_dielectric_fresnel) {
                    const auto albedo =
                        table_detail::microfacet_dielectric_reflection_albedo(
                            kernel_globals, shader_data.wi, spec_N, ax, ay, eta);
                    pool.set_sample_weight(allocated.index,
                                           common.sample_weight * albedo);
                  };
                  $if(is_multi) {
                    const auto mu = dot(shader_data.wi, spec_N);
                    const auto table_roughness = sqrt(sqrt(ax * ay));
                    const auto energy = table_detail::cycles_table_2d(
                        kernel_globals, table_roughness, mu,
                        UInt{cycles45_tables::ggx_e_offset}, 32u, 32u);
                    const auto missing_factor = (1.0f - energy) / energy;
                    microfacet.energy_scale = 1.0f + missing_factor;
                    const auto c = max(entry.color_arg, make_float3(0.0f));
                    $if(!all(c == make_float3(1.0f))) {
                      const auto avg_e = table_detail::cycles_table_1d(
                          kernel_globals, table_roughness,
                          UInt{cycles45_tables::ggx_eavg_offset}, 32u);
                      const auto ms_fres =
                          c * avg_e / (make_float3(1.0f) - c * (1.0f - avg_e));
                      const auto dark =
                          (make_float3(1.0f) + ms_fres * missing_factor) /
                          microfacet.energy_scale;
                      const auto cur_c = pool.common(allocated.index);
                      pool.set_weight_and_type(allocated.index,
                                               cur_c.weight * dark, ctype);
                      pool.set_sample_weight(allocated.index,
                                             cur_c.sample_weight *
                                                 average(dark));
                    };
                  };
                  pool.set_microfacet_param(allocated.index, microfacet);
                  UInt flags = shader_data_bsdf;
                  $if(is_ashikhmin | ((ax * ay) > 2.0e-10f)) {
                    flags |= shader_data_bsdf_has_eval;
                  };
                  shader_data.flag |= flags;
                };
              }
            };
          },
          OslClosureTag::microfacet_ggx, OslClosureTag::microfacet_beckmann,
          OslClosureTag::ashikhmin_shirley, OslClosureTag::microfacet_multi_ggx,
          OslClosureTag::reflection);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed & refractive_caustics) {
              if (shader_data.closure != nullptr) {
                const auto allocated = bsdf_allocate(shader_data, raw_w);
                $if(allocated.valid) {
                  auto &pool = *shader_data.closure;
                  const auto is_refr =
                      entry.tag ==
                      static_cast<std::uint32_t>(OslClosureTag::refraction);
                  const auto is_beckmann =
                      entry.tag ==
                      static_cast<std::uint32_t>(
                          OslClosureTag::microfacet_beckmann_refraction);
                  const Float ax =
                      select(clamp(entry.p0, 0.0f, 1.0f), Float{0.0f}, is_refr);
                  const Float ay =
                      select(clamp(entry.p1, 0.0f, 1.0f), Float{0.0f}, is_refr);
                  const Float orig_ior =
                      max(select(entry.p2, entry.p0, is_refr), 1.0e-5f);
                  const auto backfacing =
                      (shader_data.flag & shader_data_backfacing) != 0u;
                  const Float eta = select(orig_ior, 1.0f / orig_ior, backfacing);
                  const UInt ctype = select(
                      UInt{static_cast<std::uint32_t>(
                          CLOSURE_BSDF_MICROFACET_GGX_REFRACTION_ID)},
                      UInt{static_cast<std::uint32_t>(
                          CLOSURE_BSDF_MICROFACET_BECKMANN_REFRACTION_ID)},
                      is_beckmann);
                  pool.set_type(allocated.index, ctype);
                  pool.set_normal(allocated.index, spec_N);
                  pool.set_microfacet_param(
                      allocated.index,
                      {.alpha_x = ax,
                       .alpha_y = ay,
                       .ior = eta,
                       .energy_scale = 1.0f,
                       .fresnel_type =
                           static_cast<std::uint32_t>(MicrofacetFresnel::none),
                       .T = entry.T});
                  UInt flags = shader_data_bsdf | shader_data_bsdf_has_transmission;
                  $if((ax * ay) > 2.0e-10f) {
                    flags |= shader_data_bsdf_has_eval;
                  };
                  shader_data.flag |= flags;
                };
              }
            };
          },
          OslClosureTag::microfacet_ggx_refraction,
          OslClosureTag::microfacet_beckmann_refraction,
          OslClosureTag::refraction);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed & (reflective_caustics | refractive_caustics)) {
              if (shader_data.closure != nullptr) {
                auto &pool = *shader_data.closure;
                const auto sample_w = abs(average(raw_w));
                $if((sample_w >= cycles_closure::closure_weight_cutoff) &
                    (sample_w > 0.0f) & (pool.left() >= 2u)) {
                  const auto is_beckmann =
                      entry.tag ==
                      static_cast<std::uint32_t>(
                          OslClosureTag::microfacet_beckmann_glass);
                  const UInt ctype = select(
                      UInt{static_cast<std::uint32_t>(
                          CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)},
                      UInt{static_cast<std::uint32_t>(
                          CLOSURE_BSDF_MICROFACET_BECKMANN_GLASS_ID)},
                      is_beckmann);
                  const auto allocated = pool.allocate(ctype, make_float3(1.0f));
                  const auto extra_ok = pool.allocate_extra(allocated, 1u);
                  $if(extra_ok) {
                    const auto orig_ior = max(entry.p2, 1.0e-5f);
                    const auto backfacing =
                        (shader_data.flag & shader_data_backfacing) != 0u;
                    const auto eta =
                        select(orig_ior, 1.0f / orig_ior, backfacing);
                    const auto ax = clamp(entry.p0, 0.0f, 1.0f);
                    const auto ay = clamp(entry.p1, 0.0f, 1.0f);
                    const FresnelGeneralizedSchlick fresnel{
                        .thin_film = {.thickness = 0.0f, .ior = 1.33f},
                        .reflection_tint =
                            select(make_float3(0.0f), raw_w, reflective_caustics),
                        .transmission_tint =
                            select(make_float3(0.0f), raw_w, refractive_caustics),
                        .f0 = clamp(make_float3(f0_from_ior(orig_ior)), 0.0f,
                                    1.0f),
                        .f90 = make_float3(1.0f),
                        .exponent = -orig_ior};
                    const MicrofacetParam microfacet{
                        .alpha_x = ax,
                        .alpha_y = ay,
                        .ior = eta,
                        .energy_scale = 1.0f,
                        .fresnel_type = static_cast<std::uint32_t>(
                            MicrofacetFresnel::generalized_schlick),
                        .T = entry.T};
                    pool.set_generalized_schlick(allocated.index, fresnel);
                    const auto albedo =
                        table_detail::microfacet_generalized_schlick_albedo(
                            kernel_globals, shader_data.wi, spec_N, ax, ay, eta,
                            0.0f, 1.33f, fresnel.reflection_tint,
                            fresnel.transmission_tint, fresnel.f0, fresnel.f90,
                            fresnel.exponent, true, true,
                            kernel_globals.may_have_thin_film());
                    pool.set_sample_weight_and_normal(allocated.index,
                                                      average(albedo), spec_N);
                    pool.set_microfacet_param(allocated.index, microfacet);
                    UInt flags =
                        shader_data_bsdf | shader_data_bsdf_has_transmission;
                    $if((ax * ay) > 2.0e-10f) {
                      flags |= shader_data_bsdf_has_eval;
                    };
                    shader_data.flag |= flags;
                  };
                };
              }
            };
          },
          OslClosureTag::microfacet_ggx_glass,
          OslClosureTag::microfacet_beckmann_glass,
          OslClosureTag::microfacet_multi_ggx_glass);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed & (reflective_caustics | refractive_caustics)) {
              if (shader_data.closure != nullptr) {
                auto &pool = *shader_data.closure;
                const Float3 r_tint =
                    max(entry.color_arg * raw_w, make_float3(0.0f));
                const Float3 t_tint =
                    max(entry.color_arg2 * raw_w, make_float3(0.0f));
                const Float sample_w = abs(average(r_tint + t_tint));
                $if((sample_w >= cycles_closure::closure_weight_cutoff) &
                    (sample_w > 0.0f) & (pool.left() >= 2u)) {
                  const bool is_gen = false;
                  (void)is_gen;
                  const auto has_t = any(t_tint > 0.0f);
                  const UInt ctype = select(
                      UInt{static_cast<std::uint32_t>(
                          CLOSURE_BSDF_MICROFACET_GGX_ID)},
                      UInt{static_cast<std::uint32_t>(
                          CLOSURE_BSDF_MICROFACET_GGX_GLASS_ID)},
                      has_t);
                  const auto allocated = pool.allocate(ctype, make_float3(1.0f));
                  const auto extra_ok = pool.allocate_extra(allocated, 1u);
                  $if(extra_ok) {
                    const auto is_dielectric =
                        entry.tag ==
                        static_cast<std::uint32_t>(
                            OslClosureTag::dielectric_bsdf);
                    const auto orig_ior =
                        select(1.5f, max(entry.p2, 1.0e-5f), is_dielectric);
                    const auto backfacing =
                        (shader_data.flag & shader_data_backfacing) != 0u;
                    const auto eta =
                        select(orig_ior, 1.0f / orig_ior, backfacing);
                    const auto ax = clamp(square(entry.p0), 0.0f, 1.0f);
                    const auto ay = clamp(square(entry.p1), 0.0f, 1.0f);
                    const FresnelGeneralizedSchlick fresnel{
                        .thin_film = {.thickness = entry.p3,
                                      .ior = select(entry.p4,
                                                    entry.p4 / orig_ior,
                                                    backfacing)},
                        .reflection_tint = select(make_float3(0.0f), r_tint,
                                                  reflective_caustics),
                        .transmission_tint = select(make_float3(0.0f), t_tint,
                                                    refractive_caustics),
                        .f0 = select(clamp(entry.color_arg, 0.0f, 1.0f),
                                     clamp(make_float3(f0_from_ior(orig_ior)),
                                           0.0f, 1.0f),
                                     is_dielectric),
                        .f90 = make_float3(1.0f),
                        .exponent = select(-max(entry.p2, 0.0f), -orig_ior,
                                           is_dielectric)};
                    const MicrofacetParam microfacet{
                        .alpha_x = ax,
                        .alpha_y = ay,
                        .ior = eta,
                        .energy_scale = 1.0f,
                        .fresnel_type = static_cast<std::uint32_t>(
                            MicrofacetFresnel::generalized_schlick),
                        .T = entry.T};
                    pool.set_generalized_schlick(allocated.index, fresnel);
                    const auto albedo =
                        table_detail::microfacet_generalized_schlick_albedo(
                            kernel_globals, shader_data.wi, spec_N, ax, ay, eta,
                            fresnel.thin_film.thickness, fresnel.thin_film.ior,
                            fresnel.reflection_tint, fresnel.transmission_tint,
                            fresnel.f0, fresnel.f90, fresnel.exponent, true,
                            has_t, kernel_globals.may_have_thin_film());
                    pool.set_sample_weight_and_normal(allocated.index,
                                                      average(albedo), spec_N);
                    pool.set_microfacet_param(allocated.index, microfacet);
                    UInt flags = shader_data_bsdf;
                    $if(has_t) { flags |= shader_data_bsdf_has_transmission; };
                    $if((ax * ay) > 2.0e-10f) {
                      flags |= shader_data_bsdf_has_eval;
                    };
                    shader_data.flag |= flags;
                  };
                };
              }
            };
          },
          OslClosureTag::dielectric_bsdf,
          OslClosureTag::generalized_schlick_bsdf);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed & reflective_caustics) {
              if (shader_data.closure != nullptr) {
                auto &pool = *shader_data.closure;
                const auto allocated = bsdf_allocate(shader_data, raw_w);
                $if(allocated.valid) {
                  const auto extra_ok = pool.allocate_extra(allocated, 1u);
                  $if(extra_ok) {
                    const auto ax = clamp(square(entry.p0), 0.0f, 1.0f);
                    const auto ay = clamp(square(entry.p1), 0.0f, 1.0f);
                    const FresnelConductor fresnel{
                        .thin_film = {.thickness = max(entry.p3, 1.0e-5f),
                                      .ior = max(entry.p4, 1.0e-5f)},
                        .ior = max(entry.color_arg, make_float3(0.0f)),
                        .extinction = max(entry.color_arg2, make_float3(0.0f))};
                    const MicrofacetParam microfacet{
                        .alpha_x = ax,
                        .alpha_y = ay,
                        .ior = 1.0f,
                        .energy_scale = 1.0f,
                        .fresnel_type = static_cast<std::uint32_t>(
                            MicrofacetFresnel::conductor),
                        .T = entry.T};
                    pool.set_type(allocated.index,
                                  static_cast<std::uint32_t>(
                                      CLOSURE_BSDF_MICROFACET_GGX_ID));
                    pool.set_normal(allocated.index, spec_N);
                    pool.set_fresnel_conductor(allocated.index, fresnel);
                    const auto albedo =
                        table_detail::microfacet_conductor_albedo(
                            kernel_globals, shader_data.wi, spec_N,
                            fresnel.thin_film.thickness, fresnel.thin_film.ior,
                            fresnel.ior, fresnel.extinction,
                            kernel_globals.may_have_thin_film());
                    const auto common = pool.common(allocated.index);
                    pool.set_sample_weight(allocated.index,
                                           common.sample_weight *
                                               average(albedo));
                    pool.set_microfacet_param(allocated.index, microfacet);
                    UInt flags = shader_data_bsdf;
                    $if((ax * ay) > 2.0e-10f) {
                      flags |= shader_data_bsdf_has_eval;
                    };
                    shader_data.flag |= flags;
                  };
                };
              }
            };
          },
          OslClosureTag::conductor_bsdf);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed) {
              if (shader_data.closure != nullptr) {
                const auto allocated = bsdf_allocate(shader_data, raw_w);
                $if(allocated.valid) {
                  auto &pool = *shader_data.closure;
                  const auto rough = clamp(entry.p0, 1.0e-3f, 1.0f);
                  const auto cosine = dot(N, shader_data.wi);
                  const auto basis =
                      cycles_sample_mapping::make_orthonormals_safe_tangent(
                          N, shader_data.wi);
                  const auto ta = table_detail::cycles_table_2d(
                      kernel_globals, cosine, rough,
                      UInt{cycles45_tables::sheen_ltc_offset}, 32u, 32u);
                  const auto tb = table_detail::cycles_table_2d(
                      kernel_globals, cosine, rough,
                      UInt{cycles45_tables::sheen_ltc_offset + 32u * 32u}, 32u,
                      32u);
                  const auto alb = table_detail::cycles_table_2d(
                      kernel_globals, cosine, rough,
                      UInt{cycles45_tables::sheen_ltc_offset + 2u * 32u * 32u},
                      32u, 32u);
                  pool.set_normal(allocated.index, N);
                  pool.set_type(allocated.index,
                                static_cast<std::uint32_t>(
                                    CLOSURE_BSDF_SHEEN_ID));
                  pool.set_sheen_param(allocated.index,
                                       {.roughness = rough,
                                        .transform_a = ta,
                                        .transform_b = tb,
                                        .T = basis.tangent,
                                        .B = basis.bitangent});
                  $if((abs(ta) < 1.0e-5f) | (alb < 1.0e-5f)) {
                    pool.set_type(allocated.index,
                                  static_cast<std::uint32_t>(CLOSURE_NONE_ID));
                    pool.set_sample_weight(allocated.index, 0.0f);
                  }
                  $else {
                    const auto common = pool.common(allocated.index);
                    pool.set_weight(allocated.index, common.weight * alb);
                    pool.set_sample_weight(allocated.index,
                                           common.sample_weight * alb);
                    shader_data.flag |=
                        shader_data_bsdf | shader_data_bsdf_has_eval;
                  };
                };
              }
            };
          },
          OslClosureTag::sheen);
      add_cases(
          [&]() noexcept {
            $if(bsdf_allowed) {
              if (shader_data.closure != nullptr) {
                const auto allocated = bsdf_allocate(shader_data, raw_w);
                $if(allocated.valid) {
                  auto &pool = *shader_data.closure;
                  const auto sigma = clamp(entry.p0, 0.0f, 1.0f);
                  const auto cs = max(sigma, 0.01f);
                  pool.set_normal(allocated.index, N);
                  pool.set_velvet_param(
                      allocated.index,
                      {.sigma = sigma, .invsigma2 = 1.0f / (cs * cs)});
                  pool.set_type(allocated.index,
                                static_cast<std::uint32_t>(
                                    CLOSURE_BSDF_ASHIKHMIN_VELVET_ID));
                  shader_data.flag |=
                      shader_data_bsdf | shader_data_bsdf_has_eval;
                };
              }
            };
          },
          OslClosureTag::ashikhmin_velvet);
      add_cases(
          [&]() noexcept {
            const auto is_glossy =
                entry.tag ==
                static_cast<std::uint32_t>(OslClosureTag::glossy_toon);
            $if(bsdf_allowed & (!is_glossy | reflective_caustics)) {
              if (shader_data.closure != nullptr) {
                const auto allocated = bsdf_allocate(shader_data, raw_w);
                $if(allocated.valid) {
                  auto &pool = *shader_data.closure;
                  const auto sz = clamp(entry.p0, 1.0e-5f, 1.0f) * half_pi;
                  const auto sm = clamp(entry.p1, 0.0f, 1.0f) * half_pi;
                  pool.set_normal(allocated.index, N);
                  pool.set_toon_param(allocated.index,
                                      {.size = sz, .smooth = sm});
                  pool.set_type(allocated.index,
                                select(UInt{static_cast<std::uint32_t>(
                                           CLOSURE_BSDF_DIFFUSE_TOON_ID)},
                                       UInt{static_cast<std::uint32_t>(
                                           CLOSURE_BSDF_GLOSSY_TOON_ID)},
                                       is_glossy));
                  shader_data.flag |=
                      shader_data_bsdf | shader_data_bsdf_has_eval;
                };
              }
            };
          },
          OslClosureTag::diffuse_toon, OslClosureTag::glossy_toon);
    };

    if (std::has_single_bit(allowed_tag_mask)) {
      const auto only_tag =
          static_cast<std::uint32_t>(std::countr_zero(allowed_tag_mask));
      $if(entry.tag == only_tag) { emit_body(false); };
    } else {
      $switch(entry.tag) {
        emit_body(true);
        $default {};
      };
    }
  };

  if (capped_closures == 1u) {
    $if(closures.count != 0u) { emit_entry(closures.entries[0u]); };
  } else if (capped_closures <= 4u) {
    for (std::uint32_t k = 0u; k < capped_closures; ++k) {
      $if(k < closures.count) { emit_entry(closures.entries[k]); };
    }
  } else {
    const auto n_closures = min(closures.count, capped_closures);
    $for(i, n_closures) { emit_entry(closures.entries[i]); };
  }
}

void eval_one_osl_script(const OslScriptBinding &script, Stack &stack,
                         const KernelGlobals &kernel_globals,
                         const TransformState &transform_state,
                         ShaderData &shader_data, const PathState &path_state,
                         ShaderType shader_type, std::uint32_t kernel_features,
                         std::uint32_t node_feature_mask,
                         Expr<float> mix_weight,
                         const EvaluationTransition &transition) noexcept {
  const auto compiled = compile_osl_callable(script);

  Var<OslEvalGlobals> globals;
  const auto dP = shading_position_dual(shader_data);
  const auto dI = differential_from_compact(shader_data.wi, shader_data.dI);
  globals.P = dP.val;
  globals.dPdx = dP.dx;
  globals.dPdy = dP.dy;
  // Blender/Cycles OSL sets I = sd->wi (from the shading point toward the
  // viewer), see node_geometry.osl and node_layer_weight.osl.
  globals.I = shader_data.wi;
  globals.dIdx = dI.dx;
  globals.dIdy = dI.dy;
  globals.N = shader_data.N;
  globals.Ng = shader_data.Ng;
  globals.dPdu = shader_data.dPdu;
  globals.dPdv = shader_data.dPdv;
  globals.u = shader_data.u;
  globals.dudx = shader_data.du.dx;
  globals.dudy = shader_data.du.dy;
  globals.v = shader_data.v;
  globals.dvdx = shader_data.dv.dx;
  globals.dvdy = shader_data.dv.dy;
  globals.time = shader_data.time;
  globals.dtime = 0.0f;
  globals.ray_length = shader_data.ray_length;
  globals.ray_visibility = path_state.visibility;
  globals.shader_raw = shader_data.shader;
  globals.sd_flag = shader_data.flag;
  globals.prim_type = shader_data.type;
  globals.bounce = path_state.bounce;
  globals.diffuse_bounce = path_state.diffuse_bounce;
  globals.glossy_bounce = path_state.glossy_bounce;
  globals.transmission_bounce = path_state.transmission_bounce;
  globals.transparent_bounce = path_state.transparent_bounce;
  globals.backfacing =
      select(Int{0}, Int{1}, (shader_data.flag & shader_data_backfacing) != 0u);
  if (compiled.uses_object_info) {
    if (const auto *info = kernel_globals.info_services()) {
      globals.object_index = info->object_pass_id(shader_data.object);
      globals.object_random = info->object_random_number(shader_data.object);
      globals.object_alpha = info->object_alpha(shader_data.object);
      globals.material_index = info->shader_pass_id(shader_data);
      globals.object_location = info->object_location(shader_data);
      globals.object_color = info->object_color(shader_data.object);
    } else {
      globals.object_index = 0.0f;
      globals.object_random = 0.0f;
      globals.object_alpha = 1.0f;
      globals.material_index = 0.0f;
      globals.object_location = make_float3(0.0f);
      globals.object_color = make_float3(1.0f);
    }
  } else {
    globals.object_index = 0.0f;
    globals.object_random = 0.0f;
    globals.object_alpha = 1.0f;
    globals.material_index = 0.0f;
    globals.object_location = make_float3(0.0f);
    globals.object_color = make_float3(1.0f);
  }

  Var<OslTransformGlobals> transforms;
  if (compiled.uses_transforms) {
    Float4x4 o2w = make_float4x4(1.0f);
    Float4x4 w2o = make_float4x4(1.0f);
    $if(shader_data.object != object_none) {
      if ((kernel_features & kernel_feature_object_motion) != 0u) {
        $if((shader_data.object_flag & shader_data_object_motion) != 0u) {
          o2w = shader_data.ob_tfm_motion;
          w2o = shader_data.ob_itfm_motion;
        }
        $else {
          o2w = transform_state.fetch_object_to_world();
          w2o = transform_state.fetch_world_to_object();
        };
      } else {
        o2w = transform_state.fetch_object_to_world();
        w2o = transform_state.fetch_world_to_object();
      }
    };
    transforms.object_to_world = o2w;
    transforms.world_to_object = w2o;
    transforms.camera_to_world = transform_state.camera_to_world;
    transforms.world_to_camera = transform_state.world_to_camera;
  } else {
    const auto id = make_float4x4(1.0f);
    transforms.object_to_world = id;
    transforms.world_to_object = id;
    transforms.camera_to_world = id;
    transforms.world_to_camera = id;
  }

  luisa::vector<Var<OslAttrInput>> attrs;
  attrs.reserve(script.attributes.size());
  for (const auto &attr : script.attributes) {
    attrs.emplace_back(fetch_osl_attribute(
        kernel_globals, shader_data, attr, shader_type != SHADER_TYPE_VOLUME));
  }

  struct OutputStorage {
    const OslParamBinding *param{nullptr};
    luisa::optional<Var<float>> f;
    luisa::optional<Var<float>> f_dx;
    luisa::optional<Var<float>> f_dy;
    luisa::optional<Var<std::int32_t>> i;
    luisa::optional<Var<luisa::float3>> v;
    luisa::optional<Var<luisa::float3>> v_dx;
    luisa::optional<Var<luisa::float3>> v_dy;
  };
  luisa::vector<OutputStorage> outputs;

  luisa::vector<const luisa::compute::Expression *> arg_exprs;
  arg_exprs.push_back(globals.expression());
  arg_exprs.push_back(transforms.expression());
  for (const auto &a : attrs) {
    arg_exprs.push_back(a.expression());
  }

  for (const auto &p : script.params) {
    if (p.is_output || !p.is_linked || p.kind == OslParamKind::string ||
        p.kind == OslParamKind::closure) {
      continue;
    }
    const auto off = static_cast<std::uint32_t>(p.stack_offset);
    switch (p.kind) {
      case OslParamKind::floating: {
        arg_exprs.push_back(stack_load_float(stack, off).expression());
        if (p.has_derivatives) {
          arg_exprs.push_back(stack_load_float(stack, off + 1u).expression());
          arg_exprs.push_back(stack_load_float(stack, off + 2u).expression());
        }
        break;
      }
      case OslParamKind::integer:
        arg_exprs.push_back(stack_load_int(stack, off).expression());
        break;
      case OslParamKind::color:
      case OslParamKind::vector:
      case OslParamKind::point:
      case OslParamKind::normal: {
        arg_exprs.push_back(stack_load_float3(stack, off).expression());
        if (p.has_derivatives) {
          arg_exprs.push_back(stack_load_float3(stack, off + 3u).expression());
          arg_exprs.push_back(stack_load_float3(stack, off + 6u).expression());
        }
        break;
      }
      case OslParamKind::closure:
      case OslParamKind::string:
        break;
    }
  }

  Var<OslMutableState> mut_state;
  mut_state.N = shader_data.N;
  mut_state.wrote_N = 0u;
  arg_exprs.push_back(mut_state.expression());

  for (const auto &p : script.params) {
    if (!p.is_output || p.kind == OslParamKind::closure ||
        p.kind == OslParamKind::string) {
      continue;
    }
    auto &out = outputs.emplace_back();
    out.param = &p;
    switch (p.kind) {
      case OslParamKind::floating:
        out.f.emplace(0.0f);
        arg_exprs.push_back(out.f->expression());
        if (p.has_derivatives) {
          out.f_dx.emplace(0.0f);
          out.f_dy.emplace(0.0f);
          arg_exprs.push_back(out.f_dx->expression());
          arg_exprs.push_back(out.f_dy->expression());
        }
        break;
      case OslParamKind::integer:
        out.i.emplace(0);
        arg_exprs.push_back(out.i->expression());
        break;
      case OslParamKind::color:
      case OslParamKind::vector:
      case OslParamKind::point:
      case OslParamKind::normal:
        out.v.emplace(make_float3(0.0f));
        arg_exprs.push_back(out.v->expression());
        if (p.has_derivatives) {
          out.v_dx.emplace(make_float3(0.0f));
          out.v_dy.emplace(make_float3(0.0f));
          arg_exprs.push_back(out.v_dx->expression());
          arg_exprs.push_back(out.v_dy->expression());
        }
        break;
      case OslParamKind::closure:
      case OslParamKind::string:
        break;
    }
  }

  luisa::optional<Var<OslClosureList>> closures;
  if (script.has_closure_output) {
    closures.emplace();
    closures->count = 0u;
    arg_exprs.push_back(closures->expression());
  }

  luisa::compute::detail::FunctionBuilder::current()->call(
      compiled.ast->function(), arg_exprs);

  if (compiled.writes_N) {
    $if(mut_state.wrote_N != 0u) {
      shader_data.N = safe_normalize_cycles(mut_state.N);
    };
  }

  for (const auto &out : outputs) {
    const auto off = static_cast<std::uint32_t>(out.param->stack_offset);
    if (off == static_cast<std::uint32_t>(SVM_STACK_INVALID)) {
      continue;
    }
    switch (out.param->kind) {
      case OslParamKind::floating:
        if (out.param->has_derivatives && out.f_dx && out.f_dy) {
          stack_store_dual1(
              stack, off,
              {.val = *out.f, .dx = *out.f_dx, .dy = *out.f_dy});
        } else {
          stack_store_float(stack, off, *out.f);
        }
        break;
      case OslParamKind::integer:
        stack_store_int(stack, off, *out.i);
        break;
      case OslParamKind::color:
      case OslParamKind::vector:
      case OslParamKind::point:
      case OslParamKind::normal:
        if (out.param->has_derivatives && out.v_dx && out.v_dy) {
          stack_store_dual3(
              stack, off,
              {.val = *out.v, .dx = *out.v_dx, .dy = *out.v_dy});
        } else {
          stack_store_float3(stack, off, *out.v);
        }
        break;
      case OslParamKind::closure:
      case OslParamKind::string:
        break;
    }
  }

  if (script.has_closure_output && shader_type == SHADER_TYPE_SURFACE &&
      closures && compiled.emitted_closure_mask != 0ull &&
      compiled.max_emitted_closures != 0u) {
    $if(mix_weight != 0.0f) {
      emit_osl_closures(kernel_globals, shader_data, path_state,
                        kernel_features, node_feature_mask, mix_weight,
                        *closures, transition, compiled.emitted_closure_mask,
                        compiled.max_emitted_closures);
    };
  }
}

} // namespace

void node_osl_script(Cursor &cursor, Stack &stack,
                     const KernelGlobals &kernel_globals,
                     const TransformState &transform_state,
                     ShaderData &shader_data, const PathState &path_state,
                     ShaderType shader_type, std::uint32_t kernel_features,
                     std::uint32_t node_feature_mask,
                     const EvaluationTransition &transition) noexcept {
  const auto script_id = cursor.word();
  const auto packed = cursor.word();
  const auto mix_weight_offset = cursor.byte(packed, 0u);
  const auto scripts = kernel_globals.osl_scripts();
  if (scripts.empty()) {
    transition.unsupported();
    return;
  }

  auto run_script = [&](const OslScriptBinding &script) noexcept {
    const Float mix_weight =
        script.has_closure_output
            ? stack_load_float_default(stack, mix_weight_offset, 1.0f)
            : Float{1.0f};
    eval_one_osl_script(script, stack, kernel_globals, transform_state,
                        shader_data, path_state, shader_type, kernel_features,
                        node_feature_mask, mix_weight, transition);
  };

  if (const auto const_id = try_const_uint(script_id)) {
    if (*const_id >= scripts.size()) {
      transition.unsupported();
      return;
    }
    run_script(scripts[*const_id]);
    return;
  }

  $switch(script_id) {
    for (std::size_t i = 0u; i < scripts.size(); ++i) {
      $case(static_cast<std::uint32_t>(i)) { run_script(scripts[i]); };
    }
    $default { transition.unsupported(); };
  };
}

} // namespace psycles::luisa_backend::cycles_svm::detail
