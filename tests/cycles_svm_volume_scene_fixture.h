#pragma once

#include "path_tracer_cycles_svm_volume.h"
#include <array>
#include <stdexcept>

namespace psycles::test_support {
namespace volume_abi = compiler::cycles_svm;

template<typename T> auto upload_volume_fixture(
    luisa::compute::Device &device, luisa::compute::Stream &stream, const T &values) {
    auto buffer = device.create_buffer<typename T::value_type>(values.size());
    stream << buffer.copy_from(values.data()) << luisa::compute::synchronize();
    return buffer;
}

inline void initialize_volume_fixture(
    const std::shared_ptr<luisa_backend::detail::LuisaSceneData> &scene,
    luisa::compute::Stream &stream, volume_abi::CompiledShaderTable table,
    const std::vector<volume_abi::KernelObject> &objects,
    const std::vector<unsigned> &flags) {
    if (!table.table.valid) { throw std::runtime_error{table.table.diagnostic}; }
    auto &device = scene->device;
    scene->cycles_svm = std::make_unique<luisa_backend::detail::CyclesSvmRuntime>();
    auto &r = *scene->cycles_svm;
    r.geometry = std::make_unique<luisa_backend::detail::CyclesSvmGeometryRuntime>();
    r.objects = std::make_unique<luisa_backend::detail::CyclesSvmObjectRuntime>();
    const auto dummy = [&]<typename T>(luisa::compute::Buffer<T> &buffer) {
        buffer = upload_volume_fixture(device, stream, std::array<T, 1u>{});
    };
    auto &g = *r.geometry;
    dummy(g.attribute_map_buffer); dummy(g.attribute_float_buffer);
    dummy(g.attribute_float2_buffer); dummy(g.attribute_float3_buffer);
    dummy(g.attribute_float4_buffer); dummy(g.attribute_uchar4_buffer);
    dummy(g.attribute_normal_buffer); dummy(g.triangle_vertex_buffer);
    dummy(g.triangle_index_buffer); dummy(g.triangle_shader_buffer);
    dummy(g.curve_key_buffer); dummy(g.curve_buffer); dummy(g.point_buffer);
    r.objects->object_buffer = upload_volume_fixture(device, stream, objects);
    r.objects->object_flag_buffer = upload_volume_fixture(device, stream, flags);
    r.word_buffer = upload_volume_fixture(device, stream, table.table.words);
    r.kernel_shader_buffer = upload_volume_fixture(device, stream, table.kernel_shaders);
    r.kernel_features = table.kernel_features;
    r.compilation = std::move(table);
}

inline void volume_generated_fixture(
    const std::shared_ptr<luisa_backend::detail::LuisaSceneData> &scene,
    luisa::compute::Stream &stream,
    const std::array<volume_abi::packed_float4, 3u> &rows) {
    const std::array maps{
        volume_abi::AttributeMap{volume_abi::ATTR_STD_GENERATED_TRANSFORM, 0,
            volume_abi::ATTR_ELEMENT_MESH, volume_abi::NODE_ATTR_MATRIX, 0},
        volume_abi::AttributeMap{}};
    scene->cycles_svm->geometry->attribute_map_buffer =
        upload_volume_fixture(scene->device, stream, maps);
    scene->cycles_svm->geometry->attribute_float4_buffer =
        upload_volume_fixture(scene->device, stream, rows);
}
} // namespace psycles::test_support
