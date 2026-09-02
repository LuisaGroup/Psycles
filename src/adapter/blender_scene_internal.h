#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <psycles/adapter/blender_scene.h>

#include <yyjson.h>

namespace psycles::adapter::detail {

struct DocumentDeleter {
    void operator()(yyjson_doc *document) const noexcept;
};

using Document = std::unique_ptr<yyjson_doc, DocumentDeleter>;

[[nodiscard]] yyjson_val *member(
    yyjson_val *object,
    const char *name) noexcept;

[[nodiscard]] std::string text(
    yyjson_val *value,
    std::string fallback = {});

[[nodiscard]] std::uint64_t unsigned_number(
    yyjson_val *value,
    std::uint64_t fallback = 0u) noexcept;

[[nodiscard]] std::optional<std::uint32_t>
optional_unsigned_number(yyjson_val *value) noexcept;

[[nodiscard]] std::int64_t signed_number(
    yyjson_val *value,
    std::int64_t fallback = 0) noexcept;

[[nodiscard]] float number(
    yyjson_val *value,
    float fallback = 0.0f) noexcept;

[[nodiscard]] bool boolean(
    yyjson_val *value,
    bool fallback = false) noexcept;

[[nodiscard]] std::uint32_t ray_visibility_mask(
    yyjson_val *visibility) noexcept;

[[nodiscard]] Vec3f float3(
    yyjson_val *value,
    Vec3f fallback = {}) noexcept;

[[nodiscard]] Vec2f float2(
    yyjson_val *value,
    Vec2f fallback = {}) noexcept;

[[nodiscard]] Mat4f matrix(yyjson_val *value) noexcept;

[[nodiscard]] yyjson_val *find_socket(
    yyjson_val *node,
    const char *collection,
    std::string_view identifier) noexcept;

[[nodiscard]] std::optional<contract::NishitaSkyDesc>
find_simple_world_nishita(yyjson_val *world);

[[nodiscard]] contract::ShaderGraph diffuse_graph(
    Vec3f color,
    float roughness = 0.0f);

[[nodiscard]] contract::ShaderGraph cycles_default_surface_graph();

[[nodiscard]] contract::ShaderGraph emission_graph(
    Vec3f color,
    float strength);

[[nodiscard]] contract::ShaderGraph normalized_material_graph(
    yyjson_val *material,
    const std::map<
        std::string,
        contract::ImageId,
        std::less<>> &image_ids,
    const std::map<
        std::string,
        contract::ImageColorSpace,
        std::less<>> &image_color_spaces,
    const std::map<
        std::string,
        contract::ImageAlphaType,
        std::less<>> &image_alpha_types,
    const std::map<
        std::string,
        yyjson_val *,
        std::less<>> &node_groups,
    std::vector<BlenderSceneDiagnostic> &diagnostics);

}// namespace psycles::adapter::detail
