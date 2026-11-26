#include "cycles_svm_osl_nodes.h"

#include "cycles_svm_compiler_internal.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace psycles::compiler::cycles_svm {
namespace {

[[nodiscard]] std::optional<std::string_view>
string_property(const GraphNode *node, std::string_view name) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::string) {
    return std::nullopt;
  }
  if (const auto *value = std::get_if<std::string>(&iter->second.value)) {
    return *value;
  }
  return std::nullopt;
}

[[nodiscard]] bool boolean_property(const GraphNode *node,
                                    std::string_view name,
                                    bool fallback = false) noexcept {
  const auto iter = node->properties.find(name);
  if (iter == node->properties.end() ||
      iter->second.type != contract::SocketType::boolean) {
    return fallback;
  }
  if (const auto *value = std::get_if<bool>(&iter->second.value)) {
    return *value;
  }
  return fallback;
}

[[nodiscard]] std::vector<std::pair<std::string, std::string>>
parse_escaped_param_lines(std::string_view encoded) {
  std::vector<std::pair<std::string, std::string>> entries;
  std::size_t pos = 0u;
  while (pos < encoded.size()) {
    const auto nl = encoded.find('\n', pos);
    const auto line = encoded.substr(
        pos, nl == std::string_view::npos ? encoded.size() - pos : nl - pos);
    std::string key;
    std::string val;
    bool in_val = false;
    bool escaped = false;
    for (const char ch : line) {
      if (escaped) {
        char decoded = ch;
        if (ch == 'n') { decoded = '\n'; }
        else if (ch == 'r') { decoded = '\r'; }
        (in_val ? val : key).push_back(decoded);
        escaped = false;
      } else if (ch == '\\') {
        escaped = true;
      } else if (ch == '=' && !in_val) {
        in_val = true;
      } else {
        (in_val ? val : key).push_back(ch);
      }
    }
    if (in_val && !key.empty()) {
      entries.emplace_back(std::move(key), std::move(val));
    }
    if (nl == std::string_view::npos) {
      break;
    }
    pos = nl + 1u;
  }
  return entries;
}

[[nodiscard]] OslParamKind kind_from_graph_socket(
    GraphSocketType type) noexcept {
  switch (type) {
    case GraphSocketType::floating:
      return OslParamKind::floating;
    case GraphSocketType::integer:
      return OslParamKind::integer;
    case GraphSocketType::color:
      return OslParamKind::color;
    case GraphSocketType::vector:
      return OslParamKind::vector;
    case GraphSocketType::point:
      return OslParamKind::point;
    case GraphSocketType::normal:
      return OslParamKind::normal;
    case GraphSocketType::closure:
      return OslParamKind::closure;
  }
  return OslParamKind::floating;
}

[[nodiscard]] std::optional<OslParamKind> kind_from_oso_type(
    std::string_view type_name) noexcept {
  if (type_name == "float") { return OslParamKind::floating; }
  if (type_name == "int") { return OslParamKind::integer; }
  if (type_name == "color") { return OslParamKind::color; }
  if (type_name == "vector") { return OslParamKind::vector; }
  if (type_name == "point") { return OslParamKind::point; }
  if (type_name == "normal") { return OslParamKind::normal; }
  if (type_name == "closure color") { return OslParamKind::closure; }
  if (type_name == "string") { return OslParamKind::string; }
  return std::nullopt;
}

[[nodiscard]] bool is_builtin_osl_attribute(std::string_view name) noexcept {
  return name.empty() || name.starts_with("geom:") ||
         name.starts_with("object:") || name.starts_with("camera:") ||
         name.starts_with("shader:") || name.starts_with("path:") ||
         name.starts_with("hit:") || name.starts_with("material:");
}

[[nodiscard]] std::string unquote_oso_string(std::string_view token) {
  if (token.size() >= 2u && token.front() == '"' && token.back() == '"') {
    token = token.substr(1u, token.size() - 2u);
  }
  std::string out;
  out.reserve(token.size());
  for (std::size_t i = 0u; i < token.size(); ++i) {
    if (token[i] == '\\' && i + 1u < token.size()) {
      const char next = token[++i];
      if (next == 'n') { out.push_back('\n'); }
      else if (next == 't') { out.push_back('\t'); }
      else { out.push_back(next); }
    } else {
      out.push_back(token[i]);
    }
  }
  return out;
}

struct OsoParamDecl {
  OslParamKind kind{OslParamKind::floating};
  bool is_output{};
  bool has_init{};
};

struct OsoScanResult {
  std::unordered_map<std::string, OsoParamDecl> params;
  std::vector<std::string> attribute_names;
  std::uint32_t closure_calls{};
  std::uint32_t closure_slots{};
  bool has_loop{};
  bool has_transparent{};
  bool has_emission{};
  bool has_trace{};
  bool has_getattribute{};
  bool needs_uv{};
  bool needs_generated{};
};

[[nodiscard]] std::uint32_t oso_closure_slot_cost(
    std::string_view closure_name) noexcept {
  if (closure_name == "diffuse" || closure_name == "oren_nayar" ||
      closure_name == "oren_nayar_diffuse_bsdf" ||
      closure_name == "translucent" || closure_name == "translucent_bsdf" ||
      closure_name == "transparent" || closure_name == "transparent_bsdf" ||
      closure_name == "emission" || closure_name == "background" ||
      closure_name == "holdout" || closure_name == "reflection" ||
      closure_name == "refraction" || closure_name == "sheen" ||
      closure_name == "sheen_bsdf" || closure_name == "ashikhmin_velvet" ||
      closure_name == "ashikhmin_shirley" || closure_name == "diffuse_toon" ||
      closure_name == "glossy_toon" || closure_name == "microfacet_ggx" ||
      closure_name == "microfacet_beckmann" ||
      closure_name == "microfacet_ggx_aniso" ||
      closure_name == "microfacet_beckmann_aniso" ||
      closure_name == "microfacet_ggx_refraction" ||
      closure_name == "microfacet_beckmann_refraction" ||
      closure_name == "microfacet_multi_ggx" ||
      closure_name == "microfacet_multi_ggx_aniso") {
    return 1u;
  }
  // Dual-slot / extra-payload BSDFs (dielectric_bsdf, conductor_bsdf,
  // generalized_schlick_bsdf, microfacet glass, generic microfacet, unknown).
  return 2u;
}

[[nodiscard]] OsoScanResult scan_oso_bytecode(std::string_view oso) {
  OsoScanResult scan;
  std::unordered_map<std::string, std::string> string_consts;
  bool in_code_section = false;
  std::uint32_t code_pc = 0u;
  auto add_attr = [&](std::string attr) {
    if (attr == "uv" || attr == "geom:uv") {
      scan.needs_uv = true;
    } else if (attr == "geom:generated") {
      scan.needs_generated = true;
    }
    if (is_builtin_osl_attribute(attr)) {
      return;
    }
    if (std::find(scan.attribute_names.begin(), scan.attribute_names.end(),
                  attr) == scan.attribute_names.end()) {
      scan.attribute_names.emplace_back(std::move(attr));
    }
  };

  std::size_t pos = 0u;
  while (pos < oso.size()) {
    const auto nl = oso.find('\n', pos);
    auto line = oso.substr(
        pos, nl == std::string_view::npos ? oso.size() - pos : nl - pos);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1u);
    }
    pos = nl == std::string_view::npos ? oso.size() : nl + 1u;

    const auto first = line.find_first_not_of(" \t");
    if (first == std::string_view::npos) {
      continue;
    }
    line = line.substr(first);
    if (line.starts_with('#')) {
      continue;
    }

    // Tokenize keeping quoted literals together and splitting tabs/spaces
    std::vector<std::string_view> tokens;
    std::size_t p = 0u;
    while (p < line.size()) {
      while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) {
        ++p;
      }
      if (p >= line.size()) {
        break;
      }
      if (line[p] == '%') {
        break; // trailing %hint metadata
      }
      if (line[p] == '"') {
        std::size_t q = p + 1u;
        while (q < line.size()) {
          if (line[q] == '\\' && q + 1u < line.size()) {
            q += 2u;
            continue;
          }
          if (line[q] == '"') {
            ++q;
            break;
          }
          ++q;
        }
        tokens.emplace_back(line.substr(p, q - p));
        p = q;
      } else {
        std::size_t q = p;
        while (q < line.size() && line[q] != ' ' && line[q] != '\t') {
          ++q;
        }
        tokens.emplace_back(line.substr(p, q - p));
        p = q;
      }
    }
    if (tokens.empty()) {
      continue;
    }

    std::size_t head_idx = 0u;
    if (tokens[0].ends_with(':')) {
      head_idx = 1u;
      if (head_idx >= tokens.size()) {
        continue;
      }
    }
    const auto head = tokens[head_idx];
    if ((head == "param" || head == "oparam") &&
        head_idx + 2u < tokens.size()) {
      std::string_view type_tok = tokens[head_idx + 1u];
      std::size_t name_idx = head_idx + 2u;
      if (type_tok == "closure" && head_idx + 3u < tokens.size() &&
          tokens[head_idx + 2u] == "color") {
        type_tok = "closure color";
        name_idx = head_idx + 3u;
      }
      if (const auto kind = kind_from_oso_type(type_tok)) {
        scan.params.insert_or_assign(
            std::string{tokens[name_idx]},
            OsoParamDecl{
                .kind = *kind,
                .is_output = head == "oparam",
                .has_init = line.find("%init{") != std::string_view::npos ||
                            line.find("%initexpr") != std::string_view::npos,
            });
      }
    } else if (head == "code" && head_idx + 1u < tokens.size()) {
      in_code_section = true;
      if (tokens[head_idx + 1u] != "___main___") {
        if (const auto it = scan.params.find(std::string{tokens[head_idx + 1u]});
            it != scan.params.end()) {
          it->second.has_init = true;
        }
      }
    } else if (head == "end") {
      break;
    } else if (head == "const" && head_idx + 3u < tokens.size() &&
               tokens[head_idx + 1u] == "string") {
      string_consts.insert_or_assign(
          std::string{tokens[head_idx + 2u]},
          unquote_oso_string(tokens[head_idx + 3u]));
    } else if (head == "closure") {
      ++scan.closure_calls;
      std::string resolved_name;
      if (head_idx + 2u < tokens.size()) {
        const auto name_tok = tokens[head_idx + 2u];
        if (name_tok.starts_with('"')) {
          resolved_name = unquote_oso_string(name_tok);
        } else if (const auto it = string_consts.find(std::string{name_tok});
                   it != string_consts.end()) {
          resolved_name = it->second;
        }
      }
      scan.closure_slots += oso_closure_slot_cost(resolved_name);
      for (std::size_t i = head_idx + 1u; i < tokens.size(); ++i) {
        std::string resolved;
        if (tokens[i].starts_with('"')) {
          resolved = unquote_oso_string(tokens[i]);
        } else if (const auto it = string_consts.find(std::string{tokens[i]});
                   it != string_consts.end()) {
          resolved = it->second;
        }
        if (resolved == "transparent") {
          scan.has_transparent = true;
        } else if (resolved == "emission" || resolved == "background") {
          scan.has_emission = true;
        }
      }
    } else if (head == "trace") {
      scan.has_trace = true;
    } else if (head == "getattribute") {
      scan.has_getattribute = true;
      for (std::size_t i = head_idx + 1u; i < tokens.size(); ++i) {
        if (tokens[i].starts_with('"')) {
          add_attr(unquote_oso_string(tokens[i]));
        } else if (const auto it = string_consts.find(std::string{tokens[i]});
                   it != string_consts.end()) {
          add_attr(it->second);
        }
      }
    } else if (head == "for" || head == "while" || head == "dowhile") {
      scan.has_loop = true;
    } else if (head == "jump" || head == "if") {
      for (std::size_t i = head_idx + 1u; i < tokens.size(); ++i) {
        const auto tok = tokens[i];
        if (!tok.empty() && tok.front() >= '0' && tok.front() <= '9') {
          bool all_digits = true;
          std::uint64_t target = 0u;
          for (const char ch : tok) {
            if (ch < '0' || ch > '9') {
              all_digits = false;
              break;
            }
            target = target * 10u + static_cast<std::uint64_t>(ch - '0');
          }
          if (all_digits && target <= static_cast<std::uint64_t>(code_pc)) {
            scan.has_loop = true;
          }
        }
      }
    }
    if (in_code_section && head != "code") {
      ++code_pc;
    }
  }
  return scan;
}

class OslScriptGraphNode final : public GraphNode {
public:
  [[nodiscard]] ShaderNodeType shader_node_type() const noexcept override {
    return NODE_OSL_SCRIPT;
  }

  [[nodiscard]] bool has_spatial_varying() const noexcept override {
    return true;
  }

  [[nodiscard]] bool has_attribute_dependency() const noexcept override {
    const auto oso = string_property(this, "Bytecode");
    if (!oso || oso->empty()) {
      return false;
    }
    const auto scan = scan_oso_bytecode(*oso);
    return scan.has_getattribute || !scan.attribute_names.empty() ||
           scan.needs_uv;
  }

  [[nodiscard]] bool has_surface_transparent() const noexcept override {
    if (!boolean_property(this, "HasClosureOutput")) {
      return false;
    }
    const auto oso = string_property(this, "Bytecode");
    if (!oso || oso->empty()) {
      return false;
    }
    const auto scan = scan_oso_bytecode(*oso);
    return scan.has_transparent || scan.closure_calls == 0u;
  }

  [[nodiscard]] bool has_surface_emission() const noexcept override {
    if (!boolean_property(this, "HasClosureOutput")) {
      return false;
    }
    const auto oso = string_property(this, "Bytecode");
    if (!oso || oso->empty()) {
      return false;
    }
    const auto scan = scan_oso_bytecode(*oso);
    return scan.has_emission || scan.closure_calls == 0u;
  }

  [[nodiscard]] std::uint32_t get_surface_closure_slots() const noexcept override {
    if (!boolean_property(this, "HasClosureOutput")) {
      return 0u;
    }
    const auto oso = string_property(this, "Bytecode");
    if (!oso || oso->empty()) {
      return 8u;
    }
    const auto scan = scan_oso_bytecode(*oso);
    if (scan.has_loop || scan.closure_calls == 0u) {
      return 8u;
    }
    return std::clamp<std::uint32_t>(scan.closure_slots, 1u, 8u);
  }

  [[nodiscard]] ClosureUsage get_surface_closure_usage() const noexcept override {
    if (!boolean_property(this, "HasClosureOutput")) {
      return ClosureUsage::none();
    }
    ClosureUsage usage{};
    usage.principled_mask = 0u;
    return usage;
  }

  [[nodiscard]] std::uint32_t get_feature() const noexcept override {
    std::uint32_t features = 0u;
    if (boolean_property(this, "HasClosureOutput")) {
      features |= kernel_feature_node_bsdf | kernel_feature_node_emission;
    }
    const auto oso = string_property(this, "Bytecode");
    if (oso && !oso->empty()) {
      const auto scan = scan_oso_bytecode(*oso);
      if (scan.has_trace) {
        features |= kernel_feature_node_raytrace;
      }
    }
    return features;
  }

  void attributes(const GraphAttributeContext &context,
                  AttributeRequestSet &requests) const override {
    const auto oso = string_property(this, "Bytecode");
    if (oso && !oso->empty()) {
      const auto scan = scan_oso_bytecode(*oso);
      for (const auto &attr : scan.attribute_names) {
        requests.add_standard(attr);
      }
      if (scan.needs_uv) {
        requests.add(ATTR_STD_UV);
      }
      if (scan.needs_generated) {
        requests.add(ATTR_STD_GENERATED);
      }
    }
    GraphNode::attributes(context, requests);
  }

  void compile(SVMCompiler &compiler) override {
    const auto oso = string_property(this, "Bytecode");
    if (!oso || oso->empty()) {
      compiler.fail("OSL script node has empty .oso bytecode");
      return;
    }
    const auto scan = scan_oso_bytecode(*oso);
    const auto has_closure_output = boolean_property(this, "HasClosureOutput");

    OslScriptBinding binding;
    binding.script_name =
        std::string{string_property(this, "ScriptName").value_or("osl_script")};
    binding.oso_bytecode = std::string{*oso};
    binding.has_closure_output = has_closure_output;
    binding.outputs_have_derivatives =
        (compiler.output_type() != SHADER_TYPE_VOLUME) &&
        outputs_have_derivatives();

    for (const auto &in : inputs) {
      if (in.name == "SurfaceMixWeight" || in.name == "VolumeMixWeight") {
        continue;
      }
      OslParamBinding param;
      param.name = in.name;
      param.is_output = false;
      const auto sym_it = scan.params.find(in.name);
      param.kind = sym_it != scan.params.end()
                       ? sym_it->second.kind
                       : kind_from_graph_socket(in.type);
      const bool oso_has_init =
          sym_it != scan.params.end() && sym_it->second.has_init;
      if (in.link != nullptr || in.constant_folded_in) {
        param.is_linked = true;
        param.has_derivatives =
            (compiler.output_type() != SHADER_TYPE_VOLUME) &&
            (in.link != nullptr && in.link->parent != nullptr &&
             in.link->parent->outputs_have_derivatives());
        param.stack_offset = compiler.input_link(in.name);
      } else {
        param.is_linked = false;
        param.use_oso_init = oso_has_init;
        if (!oso_has_init && in.value) {
          if (const auto *v3 = std::get_if<Vec3f>(&in.value->value)) {
            param.default_float3 = {v3->x, v3->y, v3->z};
          } else if (const auto *f = std::get_if<float>(&in.value->value)) {
            param.default_float3 = {*f, *f, *f};
          } else if (const auto *i =
                         std::get_if<std::int64_t>(&in.value->value)) {
            param.default_int = static_cast<std::int32_t>(*i);
            param.default_float3 = {static_cast<float>(*i),
                                    static_cast<float>(*i),
                                    static_cast<float>(*i)};
          } else if (const auto *u =
                         std::get_if<std::uint64_t>(&in.value->value)) {
            param.default_int = static_cast<std::int32_t>(*u);
          } else if (const auto *b = std::get_if<bool>(&in.value->value)) {
            param.default_int = *b ? 1 : 0;
          }
        }
      }
      binding.params.push_back(std::move(param));
    }

    if (const auto str_params = string_property(this, "StringParams")) {
      for (auto &[key, val] : parse_escaped_param_lines(*str_params)) {
        OslParamBinding param;
        param.name = std::move(key);
        param.kind = OslParamKind::string;
        param.default_string = std::move(val);
        binding.params.push_back(std::move(param));
      }
    }

    for (const auto &out : outputs) {
      if (out.links.empty() || out.type == GraphSocketType::closure) {
        continue;
      }
      OslParamBinding param;
      param.name = out.name;
      param.is_output = true;
      param.is_linked = true;
      const auto sym_it = scan.params.find(out.name);
      param.kind = sym_it != scan.params.end()
                       ? sym_it->second.kind
                       : kind_from_graph_socket(out.type);
      param.has_derivatives = binding.outputs_have_derivatives;
      param.stack_offset = compiler.output(out.name);
      binding.params.push_back(std::move(param));
    }

    for (const auto &attr_name : scan.attribute_names) {
      const auto attr_id = compiler.attribute_standard(attr_name);
      binding.attributes.push_back(OslAttrBinding{
          .name = attr_name,
          .attr_id = attr_id,
          .attr_i32 = static_cast<std::int32_t>(attr_id),
      });
    }
    if (scan.needs_uv) {
      const auto uv_id = compiler.attribute(ATTR_STD_UV);
      binding.attributes.push_back(OslAttrBinding{
          .name = "geom:uv",
          .attr_id = uv_id,
          .attr_i32 = static_cast<std::int32_t>(uv_id),
      });
    }
    if (scan.needs_generated) {
      const auto gen_id = compiler.attribute(ATTR_STD_GENERATED);
      binding.attributes.push_back(OslAttrBinding{
          .name = "geom:generated",
          .attr_id = gen_id,
          .attr_i32 = static_cast<std::int32_t>(gen_id),
      });
    }

    const auto script_id = compiler.add_osl_script(std::move(binding));
    const auto mix_weight_offset =
        has_closure_output ? compiler.closure_mix_weight_offset()
                           : SVM_STACK_INVALID;
    compiler.add_node(
        this, NODE_OSL_SCRIPT,
        SVMNodeOslScript{
            .script_id = script_id,
            .mix_weight_offset = mix_weight_offset,
            ._pad = {0u, 0u, 0u},
        });
  }
};

} // namespace

std::unique_ptr<GraphNode> make_osl_graph_node(std::string_view type) {
  if (type == node_type::osl_script) {
    return std::make_unique<OslScriptGraphNode>();
  }
  return nullptr;
}

} // namespace psycles::compiler::cycles_svm
