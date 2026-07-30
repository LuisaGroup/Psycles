#pragma once

#include "blender_scene_internal.h"

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <psycles/compiler/core_nodes.h>

namespace psycles::adapter::detail {

using contract::ImageAlphaType;
using contract::ImageColorSpace;
using contract::ImageId;
using contract::SocketValue;

using ImageIdMap = std::map<
    std::string,
    contract::ImageId,
    std::less<>>;
using ImageColorSpaceMap = std::map<
    std::string,
    contract::ImageColorSpace,
    std::less<>>;
using ImageAlphaTypeMap = std::map<
    std::string,
    contract::ImageAlphaType,
    std::less<>>;

struct TypedOutput {
    contract::OutputRef ref;
    contract::SocketType type{
        contract::SocketType::floating};
};

struct RawOutputKey {
    std::string node;
    std::string socket;

    auto operator<=>(const RawOutputKey &) const noexcept = default;
};

class BlenderNodeLoweringContext {

public:
    virtual ~BlenderNodeLoweringContext() noexcept = default;

    [[nodiscard]] virtual contract::ShaderGraph &graph()
        noexcept = 0;
    [[nodiscard]] virtual const ImageIdMap &image_ids()
        const noexcept = 0;
    [[nodiscard]] virtual const ImageColorSpaceMap &
    image_color_spaces() const noexcept = 0;
    [[nodiscard]] virtual const ImageAlphaTypeMap &
    image_alpha_types() const noexcept = 0;

    virtual void warn_once(
        std::string key,
        std::string message) = 0;

    [[nodiscard]] virtual yyjson_val *raw_input(
        yyjson_val *node,
        std::string_view identifier) const noexcept = 0;

    [[nodiscard]] virtual std::optional<RawOutputKey>
    input_source(
        yyjson_val *node,
        std::string_view identifier) const = 0;

    [[nodiscard]] virtual bool output_is_linked(
        yyjson_val *node,
        std::string_view identifier) const = 0;

    [[nodiscard]] virtual std::string node_property_text(
        yyjson_val *node,
        const char *name,
        std::string fallback = {}) const = 0;

    [[nodiscard]] virtual float node_property_number(
        yyjson_val *node,
        const char *name,
        float fallback = 0.0f) const noexcept = 0;

    [[nodiscard]] virtual bool node_property_bool(
        yyjson_val *node,
        const char *name,
        bool fallback = false) const noexcept = 0;

    [[nodiscard]] virtual TypedOutput default_image_coordinates() = 0;
    [[nodiscard]] virtual TypedOutput
    default_generated_coordinates() = 0;

    [[nodiscard]] virtual TypedOutput geometry_output(
        std::string socket,
        contract::SocketType type) = 0;

    [[nodiscard]] virtual TypedOutput conversion(
        TypedOutput source,
        contract::SocketType target) = 0;

    [[nodiscard]] virtual bool bind(
        contract::NodeId destination,
        std::string target_socket,
        yyjson_val *raw_destination,
        std::string_view raw_input_name,
        contract::SocketType target_type) = 0;

    [[nodiscard]] virtual std::string color_ramp_table(
        yyjson_val *node) const = 0;

    [[nodiscard]] virtual std::string rgb_curve_table(
        yyjson_val *node) const = 0;

    [[nodiscard]] virtual TypedOutput constant_from_output(
        yyjson_val *node,
        std::string_view socket,
        contract::SocketType type) = 0;
};

struct BlenderNodeLoweringRequest {
    const std::string &node_name;
    const std::string &socket;
    contract::SocketType requested;
    yyjson_val *node;
    const std::string &type;
    const std::function<TypedOutput(TypedOutput)> &finish;
};

// Components are ordinary host objects. They lower Blender JSON into the
// typed Psycles graph and never alter or pre-bake the original closure data.
class BlenderNodeLoweringComponent {

public:
    virtual ~BlenderNodeLoweringComponent() noexcept = default;

    [[nodiscard]] virtual std::optional<TypedOutput> lower(
        BlenderNodeLoweringContext &context,
        const BlenderNodeLoweringRequest &request) const = 0;
};

[[nodiscard]] std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_input_lowering_component();
[[nodiscard]] std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_value_lowering_component();
[[nodiscard]] std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_procedural_lowering_component();
[[nodiscard]] std::unique_ptr<BlenderNodeLoweringComponent>
make_blender_closure_lowering_component();

[[nodiscard]] std::vector<
    std::unique_ptr<BlenderNodeLoweringComponent>>
make_blender_node_lowering_components();

}// namespace psycles::adapter::detail
