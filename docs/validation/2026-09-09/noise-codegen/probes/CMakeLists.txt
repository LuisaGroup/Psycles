cmake_minimum_required(VERSION 3.20)
project(NoiseCodegenAudit LANGUAGES CXX)
set(PROBE_ROOT /home/mike/Projects/Psycles-surface-svm)
set(PROBE_LUISA ${PROBE_ROOT}/third_party/LuisaCompute)
add_executable(noise-codegen-probe probe.cpp)
target_compile_features(noise-codegen-probe PRIVATE cxx_std_20)
target_compile_definitions(noise-codegen-probe PRIVATE
    PSYCLES_WITH_LUISA=1 MARL_DLL=1 MARL_USE_SYSTEM_STL=1
    LUISA_ENABLE_DSL=1 LUISA_ENABLE_XIR=1 LUISA_PLATFORM_UNIX=1
    LUISA_USE_SYSTEM_STL=1 FMT_HEADER_ONLY=1 FMT_USE_CONSTEVAL=0 FMT_EXCEPTIONS=0
    FMT_USE_NOEXCEPT=1 SPDLOG_NO_EXCEPTIONS SPDLOG_NO_THREAD_ID XXH_INLINE_ALL)
target_include_directories(noise-codegen-probe PRIVATE
    ${PROBE_ROOT}/include ${PROBE_LUISA}/include
    ${PROBE_LUISA}/src/ext/spdlog/include ${PROBE_LUISA}/src/ext/xxHash
    ${PROBE_LUISA}/src/ext/BTree/include ${PROBE_LUISA}/src/ext/marl/include
    ${PROBE_LUISA}/src/ext/half/include ${PROBE_LUISA}/src/ext/magic_enum/include)
foreach(PROBE_LIBRARY IN ITEMS dsl runtime ast core coro xir)
    target_link_libraries(noise-codegen-probe PRIVATE
        ${PROBE_ROOT}/build/bin/libluisa-${PROBE_LIBRARY}.so)
endforeach()
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cycles)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cycles/original-noise-probe
    COMMAND /opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201
        -DHIPCC -std=c++20 -O3 -ffast-math --save-temps=obj
        -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles
        ${CMAKE_CURRENT_SOURCE_DIR}/oracle.hip -o original-noise-probe
    DEPENDS oracle.hip inputs.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cycles VERBATIM)
add_custom_target(original-noise-probe ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cycles/original-noise-probe)

add_executable(node-codegen-probe node_probe.cpp)
target_compile_features(node-codegen-probe PRIVATE cxx_std_20)
get_target_property(PROBE_DEFINITIONS noise-codegen-probe COMPILE_DEFINITIONS)
get_target_property(PROBE_INCLUDES noise-codegen-probe INCLUDE_DIRECTORIES)
get_target_property(PROBE_LINKS noise-codegen-probe LINK_LIBRARIES)
target_compile_definitions(node-codegen-probe PRIVATE ${PROBE_DEFINITIONS})
target_include_directories(node-codegen-probe PRIVATE ${PROBE_INCLUDES} ${PROBE_ROOT}/src/luisa)
target_link_libraries(node-codegen-probe PRIVATE ${PROBE_LINKS} ${PROBE_ROOT}/build/libpsycles_luisa_runtime.so)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cycles-node)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cycles-node/original-node-probe
    COMMAND /opt/rocm/bin/hipcc -parallel-jobs=32 --offload-arch=gfx1201
        -DHIPCC -std=c++20 -O3 -ffast-math --save-temps=obj
        -I /home/mike/Projects/blender-cycles-trace-5.2/intern/cycles
        ${CMAKE_CURRENT_SOURCE_DIR}/node_oracle.hip -o original-node-probe
    DEPENDS node_oracle.hip node_inputs.h inputs.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cycles-node VERBATIM)
add_custom_target(original-node-probe ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/cycles-node/original-node-probe)
