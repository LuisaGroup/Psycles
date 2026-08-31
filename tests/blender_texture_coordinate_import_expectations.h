#pragma once

namespace psycles::compiler {
class SurfaceProgram;
}

namespace psycles::contract {
struct MaterialDesc;
}

namespace psycles::tests {

void expect_texture_coordinate_import(
    const contract::MaterialDesc &material,
    const compiler::SurfaceProgram &surface_program);

}// namespace psycles::tests
