#include "blender_graph_lowering_component.h"

namespace psycles::adapter::detail {

std::vector<std::unique_ptr<BlenderNodeLoweringComponent>>
make_blender_node_lowering_components() {
    std::vector<
        std::unique_ptr<BlenderNodeLoweringComponent>>
        components;
    components.reserve(4u);
    components.emplace_back(
        make_blender_input_lowering_component());
    components.emplace_back(
        make_blender_value_lowering_component());
    components.emplace_back(
        make_blender_procedural_lowering_component());
    components.emplace_back(
        make_blender_closure_lowering_component());
    return components;
}

}// namespace psycles::adapter::detail
