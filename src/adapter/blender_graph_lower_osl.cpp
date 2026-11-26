#include "blender_graph_lowering_component.h"

#include <psycles/compiler/core_nodes.h>

#include <algorithm>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace psycles::adapter::detail {
namespace {

using contract::SocketType;
using contract::SocketValue;

[[nodiscard]] std::optional<SocketType> osl_blender_socket_type(
    std::string_view raw_type,
    SocketType closure_kind) noexcept {
  if (raw_type.find("Shader") != std::string_view::npos ||
      raw_type == "SHADER") {
    return closure_kind;
  }
  if (raw_type.find("Color") != std::string_view::npos || raw_type == "RGBA") {
    return SocketType::color;
  }
  if (raw_type.find("Vector") != std::string_view::npos ||
      raw_type == "VECTOR") {
    return SocketType::vector;
  }
  if (raw_type.find("Float") != std::string_view::npos ||
      raw_type == "VALUE") {
    return SocketType::floating;
  }
  if (raw_type.find("Bool") != std::string_view::npos ||
      raw_type == "BOOLEAN") {
    return SocketType::boolean;
  }
  if (raw_type.find("Int") != std::string_view::npos || raw_type == "INT") {
    return SocketType::integer;
  }
  if (raw_type.find("String") != std::string_view::npos ||
      raw_type == "STRING") {
    return SocketType::string;
  }
  return std::nullopt;
}

[[nodiscard]] std::string_view contract_socket_token(SocketType type) noexcept {
  switch (type) {
    case SocketType::floating:
      return "float";
    case SocketType::integer:
      return "int";
    case SocketType::boolean:
      return "bool";
    case SocketType::unsigned_integer:
      return "uint";
    case SocketType::float2:
      return "float2";
    case SocketType::float3:
      return "float3";
    case SocketType::color:
      return "color";
    case SocketType::spectrum:
      return "spectrum";
    case SocketType::vector:
      return "vector";
    case SocketType::point:
      return "point";
    case SocketType::normal:
      return "normal";
    case SocketType::transform:
      return "transform";
    case SocketType::string:
      return "string";
    case SocketType::closure:
      return "closure";
    case SocketType::volume_closure:
      return "volume_closure";
  }
  return "float";
}

void append_escaped_param_line(std::string &dst,
                               std::string_view key,
                               std::string_view value) {
  auto escape = [&](std::string_view src) {
    for (const char ch : src) {
      if (ch == '\\') {
        dst += "\\\\";
      } else if (ch == '\n') {
        dst += "\\n";
      } else if (ch == '\r') {
        dst += "\\r";
      } else if (ch == '=') {
        dst += "\\=";
      } else {
        dst.push_back(ch);
      }
    }
  };
  escape(key);
  dst.push_back('=');
  escape(value);
  dst.push_back('\n');
}

class OslNodeLoweringComponent final : public BlenderNodeLoweringComponent {
public:
  [[nodiscard]] std::optional<TypedOutput> lower(
      BlenderNodeLoweringContext &context,
      const BlenderNodeLoweringRequest &request) const override {
    if (request.type != "SCRIPT") {
      return std::nullopt;
    }

    const auto shared_key = "osl:" + request.socket;
    if (auto cached = context.shared_output(request.node_name, shared_key)) {
      return request.finish(*cached);
    }

    auto *osl_data = member(member(request.node, "special"), "osl");
    const auto oso_text = text(member(osl_data, "oso"));
    const auto script_name = text(member(osl_data, "script_name"));
    if (oso_text.empty()) {
      context.warn_once(
          "osl-empty:" + request.node_name,
          "OSL script node '" + request.node_name +
              "' has no compiled .oso bytecode; using default output");
      return request.finish(context.constant_from_output(
          request.node, request.socket, request.requested));
    }

    const auto closure_kind =
        request.requested == SocketType::volume_closure
            ? SocketType::volume_closure
            : SocketType::closure;

    struct DynamicSocketInfo {
      std::string id;
      SocketType type;
      std::string string_default;
    };

    std::vector<DynamicSocketInfo> inputs;
    std::string input_types;
    std::string string_params;
    if (auto *raw_inputs = member(request.node, "inputs");
        raw_inputs != nullptr && yyjson_is_arr(raw_inputs)) {
      yyjson_arr_iter iter = yyjson_arr_iter_with(raw_inputs);
      while (auto *sock = yyjson_arr_iter_next(&iter)) {
        const auto id = text(member(sock, "identifier"));
        const auto raw_type = text(member(sock, "type"));
        const auto mapped = osl_blender_socket_type(raw_type, closure_kind);
        if (!mapped || id.empty()) {
          continue;
        }
        if (*mapped == SocketType::string) {
          append_escaped_param_line(
              string_params, id, text(member(sock, "default")));
          continue;
        }
        if (!input_types.empty()) {
          input_types.push_back(';');
        }
        input_types += id;
        input_types.push_back('=');
        input_types += contract_socket_token(*mapped);
        inputs.push_back(DynamicSocketInfo{.id = id, .type = *mapped});
      }
    }

    std::vector<DynamicSocketInfo> outputs;
    std::string output_types;
    bool has_closure_output = false;
    if (auto *raw_outputs = member(request.node, "outputs");
        raw_outputs != nullptr && yyjson_is_arr(raw_outputs)) {
      yyjson_arr_iter iter = yyjson_arr_iter_with(raw_outputs);
      while (auto *sock = yyjson_arr_iter_next(&iter)) {
        const auto id = text(member(sock, "identifier"));
        const auto raw_type = text(member(sock, "type"));
        const auto mapped = osl_blender_socket_type(raw_type, closure_kind);
        if (!mapped || id.empty() || *mapped == SocketType::string) {
          continue;
        }
        if (*mapped == SocketType::closure ||
            *mapped == SocketType::volume_closure) {
          has_closure_output = true;
        }
        if (!output_types.empty()) {
          output_types.push_back(';');
        }
        output_types += id;
        output_types.push_back('=');
        output_types += contract_socket_token(*mapped);
        outputs.push_back(DynamicSocketInfo{.id = id, .type = *mapped});
      }
    }

    const auto node_id =
        context.graph().add_node(compiler::node_type::osl_script, request.node_name);
    static_cast<void>(context.graph().set_property(
        node_id, "Bytecode", SocketValue::string(oso_text)));
    static_cast<void>(context.graph().set_property(
        node_id, "ScriptName", SocketValue::string(script_name)));
    static_cast<void>(context.graph().set_property(
        node_id, "InputTypes", SocketValue::string(input_types)));
    static_cast<void>(context.graph().set_property(
        node_id, "OutputTypes", SocketValue::string(output_types)));
    static_cast<void>(context.graph().set_property(
        node_id, "StringParams", SocketValue::string(string_params)));
    static_cast<void>(context.graph().set_property(
        node_id, "HasClosureOutput", SocketValue::boolean(has_closure_output)));

    for (const auto &in : inputs) {
      static_cast<void>(
          context.bind(node_id, in.id, request.node, in.id, in.type));
    }

    std::optional<TypedOutput> matched;
    for (const auto &out : outputs) {
      TypedOutput typed{
          .ref = {.node = node_id, .socket = out.id},
          .type = out.type,
      };
      context.remember_shared_output(
          request.node_name, "osl:" + out.id, typed);
      if (out.id == request.socket) {
        matched = typed;
      }
    }

    if (matched) {
      return request.finish(*matched);
    }
    return request.finish(context.constant_from_output(
        request.node, request.socket, request.requested));
  }
};

} // namespace

std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_osl_lowering_component() {
  return std::make_unique<OslNodeLoweringComponent>();
}

} // namespace psycles::adapter::detail
