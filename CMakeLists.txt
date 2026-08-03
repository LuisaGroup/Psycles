cmake_minimum_required(VERSION 3.26)
project(Psycles
    VERSION 0.1.0
    DESCRIPTION "Cycles-compatible renderer contracts on LuisaCompute DSL/JIT"
    LANGUAGES C CXX)

option(PSYCLES_BUILD_TESTS "Build Psycles contract tests" ON)
option(PSYCLES_BUILD_EXAMPLES "Build Psycles executable examples" ON)
option(PSYCLES_ENABLE_OPENIMAGEIO
    "Use OpenImageIO for Cycles-compatible multilayer OpenEXR output" ON)
option(PSYCLES_ENABLE_LUISA "Build the LuisaCompute DSL interface when available" ON)
option(PSYCLES_ENABLE_LUISA_FALLBACK
    "Build LuisaCompute's fallback backend for machines without a GPU backend" ON)
if(APPLE)
    set(_psycles_enable_luisa_metal_default ON)
    set(_psycles_enable_luisa_non_apple_gpu_default OFF)
else()
    set(_psycles_enable_luisa_metal_default OFF)
    set(_psycles_enable_luisa_non_apple_gpu_default ON)
endif()
option(PSYCLES_ENABLE_LUISA_METAL
    "Build LuisaCompute's Metal backend on Apple platforms"
    ${_psycles_enable_luisa_metal_default})
option(PSYCLES_ENABLE_LUISA_HIP
    "Build LuisaCompute's HIP backend"
    ${_psycles_enable_luisa_non_apple_gpu_default})
option(PSYCLES_ENABLE_LUISA_VULKAN
    "Build LuisaCompute's Vulkan backend"
    ${_psycles_enable_luisa_non_apple_gpu_default})
unset(_psycles_enable_luisa_metal_default)
unset(_psycles_enable_luisa_non_apple_gpu_default)
option(PSYCLES_FETCH_LUISA_NEXT
    "Fetch LuisaCompute from its next branch when the bundled submodule is unavailable" OFF)
set(PSYCLES_LUISA_SUBMODULE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/LuisaCompute")

if(PSYCLES_BUILD_TESTS)
    include(CTest)
    include(cmake/PsyclesLuisaBackendTest.cmake)
endif()

add_library(psycles_core
    src/adapter/cycles_shader_graph.cpp
    src/compiler/core_nodes.cpp
    src/compiler/hosek_sky.cpp
    src/compiler/material_library.cpp
    src/compiler/shader_program.cpp
    src/compiler/surface_program.cpp
    src/compiler/surface_program_builder.cpp
    src/compiler/surface_program_lower_closures.cpp
    src/compiler/surface_program_lower_context.cpp
    src/compiler/surface_program_lower_textures.cpp
    src/compiler/surface_program_lower_values.cpp
    src/contract/scene.cpp
    src/contract/shader_graph.cpp
    src/io/image.cpp
    src/sampling/background_distribution.cpp
    src/sampling/light_distribution.cpp
    src/sampling/pixel_filter.cpp
    src/sampling/tabulated_sobol.cpp
    third_party/sky_hosek/sky_hosek.cpp)
add_library(Psycles::core ALIAS psycles_core)

target_include_directories(psycles_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sky_hosek)
target_compile_features(psycles_core PUBLIC cxx_std_20)

if(PSYCLES_ENABLE_OPENIMAGEIO)
    find_package(OpenImageIO CONFIG REQUIRED)
    target_sources(psycles_core PRIVATE src/io/openexr.cpp)
    target_compile_definitions(
        psycles_core PUBLIC PSYCLES_WITH_OPENIMAGEIO=1)
    target_link_libraries(
        psycles_core PRIVATE OpenImageIO::OpenImageIO)
endif()

if(MSVC)
    target_compile_options(psycles_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(psycles_core PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion)
endif()

if(PSYCLES_ENABLE_LUISA)
    set(_psycles_luisa_source_dir "")
    if(EXISTS "${PSYCLES_LUISA_SUBMODULE_DIR}/CMakeLists.txt")
        set(_psycles_luisa_source_dir "${PSYCLES_LUISA_SUBMODULE_DIR}")
        message(STATUS
            "Using bundled LuisaCompute submodule: "
            "${_psycles_luisa_source_dir}")
    endif()

    if(_psycles_luisa_source_dir)
        set(LUISA_COMPUTE_ENABLE_DSL ON CACHE BOOL "" FORCE)
        # LuisaCompute's legacy Rust CPU backend is retired. The supported
        # no-GPU execution path is the fallback backend, which consumes the
        # same Luisa DSL AST/JIT program as every other device backend.
        set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_FALLBACK
            ${PSYCLES_ENABLE_LUISA_FALLBACK} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_METAL
            ${PSYCLES_ENABLE_LUISA_METAL} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_HIP
            ${PSYCLES_ENABLE_LUISA_HIP} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VULKAN
            ${PSYCLES_ENABLE_LUISA_VULKAN} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_CLANG_CXX OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        add_subdirectory(
            "${_psycles_luisa_source_dir}"
            "${CMAKE_CURRENT_BINARY_DIR}/_deps/luisa-compute"
            EXCLUDE_FROM_ALL)
    elseif(PSYCLES_FETCH_LUISA_NEXT AND NOT TARGET luisa::compute)
        set(LUISA_COMPUTE_ENABLE_DSL ON CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_RUST OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_CPU OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_FALLBACK
            ${PSYCLES_ENABLE_LUISA_FALLBACK} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_METAL
            ${PSYCLES_ENABLE_LUISA_METAL} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_HIP
            ${PSYCLES_ENABLE_LUISA_HIP} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VULKAN
            ${PSYCLES_ENABLE_LUISA_VULKAN} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_CLANG_CXX OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        include(FetchContent)
        FetchContent_Declare(
            psycles_luisa_compute
            GIT_REPOSITORY https://github.com/LuisaGroup/LuisaCompute.git
            GIT_TAG next
            GIT_SHALLOW TRUE
            GIT_PROGRESS TRUE)
        FetchContent_MakeAvailable(psycles_luisa_compute)
    elseif(NOT TARGET luisa::compute AND
           EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.gitmodules")
        message(FATAL_ERROR
            "The bundled LuisaCompute submodule is not initialized. Run "
            "'git submodule update --init --recursive', enable "
            "PSYCLES_FETCH_LUISA_NEXT, or "
            "configure with PSYCLES_ENABLE_LUISA=OFF for the contract core.")
    endif()

    if(PSYCLES_ENABLE_LUISA_FALLBACK AND
       NOT TARGET luisa-compute-backend-fallback)
        message(FATAL_ERROR
            "PSYCLES_ENABLE_LUISA_FALLBACK=ON requires a real Luisa fallback "
            "backend, but LuisaCompute did not create "
            "luisa-compute-backend-fallback. Install compatible LLVM and "
            "Embree development packages, set LLVM_DIR and embree_DIR when "
            "needed, and reconfigure.")
    endif()
    if(PSYCLES_ENABLE_LUISA_METAL AND
       NOT TARGET luisa-compute-backend-metal)
        message(FATAL_ERROR
            "PSYCLES_ENABLE_LUISA_METAL=ON requires LuisaCompute's Metal "
            "backend, but LuisaCompute did not create "
            "luisa-compute-backend-metal. Configure on an Apple platform "
            "with the Metal developer tools available, or configure with "
            "PSYCLES_ENABLE_LUISA_METAL=OFF.")
    endif()
    if(PSYCLES_ENABLE_LUISA_HIP AND
       NOT TARGET luisa-compute-backend-hip)
        message(FATAL_ERROR
            "PSYCLES_ENABLE_LUISA_HIP=ON requires LuisaCompute's HIP "
            "backend, but LuisaCompute did not create "
            "luisa-compute-backend-hip. Install the HIP backend "
            "prerequisites or configure with "
            "PSYCLES_ENABLE_LUISA_HIP=OFF.")
    endif()
    if(PSYCLES_ENABLE_LUISA_VULKAN AND
       NOT TARGET luisa-compute-backend-vk)
        message(FATAL_ERROR
            "PSYCLES_ENABLE_LUISA_VULKAN=ON requires LuisaCompute's "
            "Vulkan backend, but LuisaCompute did not create "
            "luisa-compute-backend-vk. Install the Vulkan SDK/runtime "
            "prerequisites or configure with "
            "PSYCLES_ENABLE_LUISA_VULKAN=OFF.")
    endif()

    if(TARGET luisa::compute)
        add_library(
            psycles_luisa
            STATIC
            src/luisa/graph_surface.cpp
            src/luisa/graph_surface_closures.cpp
            src/luisa/graph_surface_emission.cpp
            src/luisa/graph_surface_implementation.cpp
            src/luisa/graph_surface_physical_closures.cpp
            src/luisa/microfacet_glass_component.cpp
            src/luisa/principled_base_component.cpp
            src/luisa/principled_layer_component.cpp
            src/luisa/graph_surface_scattering.cpp
            src/luisa/graph_surface_state.cpp
            src/luisa/graph_surface_values.cpp
            src/luisa/graph_surface_value_context.cpp
            src/luisa/graph_surface_value_image.cpp
            src/luisa/graph_surface_value_math.cpp
            src/luisa/graph_surface_value_normal.cpp
            src/luisa/graph_surface_value_procedural.cpp
            src/luisa/heterogeneous_volume_candidate.cpp
            src/luisa/heterogeneous_volume_collision.cpp
            src/luisa/heterogeneous_volume_guiding.cpp
            src/luisa/heterogeneous_volume_reservoir.cpp
            src/luisa/heterogeneous_volume_segment.cpp
            src/luisa/heterogeneous_volume_shadow.cpp
            src/luisa/heterogeneous_volume_tracking.cpp
            src/luisa/heterogeneous_volume_transmittance.cpp
            src/luisa/homogeneous_volume_segment.cpp
            src/luisa/homogeneous_volume_transport.cpp
            src/luisa/volume_scatter_probability.cpp
            src/luisa/volume_direct_sampling.cpp
            src/luisa/area_light_sampling.cpp
            src/luisa/triangle_light_sampling.cpp
            src/luisa/path_kernel_area_light.cpp
            src/luisa/volume_analytic_light_sampling.cpp
            src/luisa/volume_light_interval.cpp
            src/luisa/volume_guiding.cpp
            src/luisa/stacked_volume.cpp
            src/luisa/volume_majorant_hierarchy.cpp
            src/luisa/volume_majorant_overlap.cpp
            src/luisa/volume_majorant_prepass.cpp
            src/luisa/volume_majorant_traversal.cpp
            src/luisa/volume_phase_set.cpp
            src/luisa/volume_shadow_interval.cpp
            src/luisa/volume_stack.cpp)
        add_library(Psycles::luisa ALIAS psycles_luisa)
        target_include_directories(psycles_luisa PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_compile_features(psycles_luisa PUBLIC cxx_std_20)
        target_compile_definitions(psycles_luisa PUBLIC PSYCLES_WITH_LUISA=1)
        target_link_libraries(psycles_luisa PUBLIC
            Psycles::core
            luisa::compute)
        message(STATUS "Psycles Luisa DSL surface compiler enabled")
        if(MSVC)
            target_compile_options(
                psycles_luisa
                PRIVATE /W4 /permissive-)
        else()
            target_compile_options(
                psycles_luisa
                PRIVATE
                    -Wall -Wextra -Wpedantic
                    -Wconversion -Wsign-conversion)
        endif()
        add_library(
            psycles_luisa_runtime
            src/adapter/blender_graph_lower_closures.cpp
            src/adapter/blender_graph_lower_inputs.cpp
            src/adapter/blender_graph_lower_procedural.cpp
            src/adapter/blender_graph_lower_values.cpp
            src/adapter/blender_graph_lowering_components.cpp
            src/adapter/blender_graph_normalizer.cpp
            src/adapter/blender_scene.cpp
            src/adapter/blender_scene_json.cpp
            src/luisa/path_tracer.cpp
            src/luisa/path_tracer_camera.cpp
            src/luisa/path_tracer_common.cpp
            src/luisa/path_tracer_environment.cpp
            src/luisa/path_tracer_generated_coordinates.cpp
            src/luisa/path_tracer_geometry.cpp
            src/luisa/path_kernel_background_event.cpp
            src/luisa/path_kernel_bounce_setup.cpp
            src/luisa/path_kernel_closest_event.cpp
            src/luisa/path_kernel_direct_light_trace.cpp
            src/luisa/path_kernel_emissive_triangle.cpp
            src/luisa/path_kernel_environment_light.cpp
            src/luisa/path_kernel_forward_light.cpp
            src/luisa/path_kernel_heterogeneous_volume.cpp
            src/luisa/path_kernel_nee_analytic.cpp
            src/luisa/path_kernel_nee_emissive_mesh.cpp
            src/luisa/path_kernel_nee_environment.cpp
            src/luisa/path_kernel_pipeline.cpp
            src/luisa/path_kernel_setup.cpp
            src/luisa/path_kernel_surface_geometry.cpp
            src/luisa/path_kernel_surface_scatter.cpp
            src/luisa/path_kernel_surface_shading.cpp
            src/luisa/path_kernel_triangle_geometry.cpp
            src/luisa/path_kernel_triangle_primitive.cpp
            src/luisa/path_kernel_volume_boundary.cpp
            src/luisa/path_kernel_volume_direct_light.cpp
            src/luisa/path_kernel_volume_environment_light.cpp
            src/luisa/path_kernel_volume_majorant_provider.cpp
            src/luisa/path_kernel_volume_mesh_light.cpp
            src/luisa/path_kernel_volume_point.cpp
            src/luisa/path_kernel_volume_random.cpp
            src/luisa/path_kernel_volume_segment.cpp
            src/luisa/path_kernel_volume_shadow.cpp
            src/luisa/path_kernel_volume_state.cpp
            src/luisa/path_tracer_kernel.cpp
            src/luisa/path_tracer_image_decode.cpp
            src/luisa/path_tracer_light_distribution.cpp
            src/luisa/path_tracer_lighting.cpp
            src/luisa/path_tracer_scene.cpp
            src/luisa/path_tracer_scene_geometry.cpp
            src/luisa/path_tracer_session.cpp
            src/luisa/path_tracer_surfaces.cpp
            src/luisa/path_tracer_volume_capabilities.cpp
            src/luisa/path_tracer_volume_majorant_scene.cpp
            src/luisa/path_tracer_volume_metadata.cpp
            src/luisa/volume_guiding_filter.cpp)
        add_library(
            Psycles::luisa_runtime
            ALIAS psycles_luisa_runtime)
        target_include_directories(psycles_luisa_runtime
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
        target_compile_features(
            psycles_luisa_runtime
            PUBLIC cxx_std_20)
        target_link_libraries(
            psycles_luisa_runtime
            PUBLIC Psycles::luisa)

        if(MSVC)
            target_compile_options(
                psycles_luisa_runtime
                PRIVATE /W4 /permissive-)
        else()
            target_compile_options(
                psycles_luisa_runtime
                PRIVATE
                    -Wall -Wextra -Wpedantic
                    -Wconversion -Wsign-conversion)
        endif()

        if(PSYCLES_BUILD_TESTS)
            add_executable(
                psycles_luisa_compile_tests
                tests/test_luisa_compile.cpp)
            target_link_libraries(
                psycles_luisa_compile_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_compile_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.luisa_ast
                COMMAND psycles_luisa_compile_tests)

            add_executable(
                psycles_luisa_volume_coefficients_tests
                tests/test_luisa_volume_coefficients.cpp)
            target_link_libraries(
                psycles_luisa_volume_coefficients_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_volume_coefficients_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_coefficients_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_coefficients_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_coefficients_fallback
                    COMMAND
                        psycles_luisa_volume_coefficients_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_coefficients_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_coefficients_hip
                    COMMAND
                        psycles_luisa_volume_coefficients_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_coefficients_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_coefficients_vk
                    COMMAND
                        psycles_luisa_volume_coefficients_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_stacked_volume_tests
                tests/test_luisa_stacked_volume.cpp)
            target_link_libraries(
                psycles_luisa_stacked_volume_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_stacked_volume_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_stacked_volume_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_stacked_volume_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_stacked_volume_fallback
                    COMMAND
                        psycles_luisa_stacked_volume_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_stacked_volume_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_stacked_volume_hip
                    COMMAND
                        psycles_luisa_stacked_volume_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_stacked_volume_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_stacked_volume_vk
                    COMMAND
                        psycles_luisa_stacked_volume_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_phase_tests
                tests/test_luisa_volume_phase.cpp)
            target_link_libraries(
                psycles_luisa_volume_phase_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_volume_phase_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_phase_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_phase_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_phase_fallback
                    COMMAND
                        psycles_luisa_volume_phase_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_phase_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_phase_hip
                    COMMAND
                        psycles_luisa_volume_phase_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_phase_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_phase_vk
                    COMMAND
                        psycles_luisa_volume_phase_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_stack_tests
                tests/test_luisa_volume_stack.cpp)
            target_link_libraries(
                psycles_luisa_volume_stack_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_volume_stack_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_stack_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_stack_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_stack_fallback
                    COMMAND
                        psycles_luisa_volume_stack_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_stack_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_stack_hip
                    COMMAND
                        psycles_luisa_volume_stack_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_stack_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_stack_vk
                    COMMAND
                        psycles_luisa_volume_stack_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_homogeneous_volume_tests
                tests/test_luisa_homogeneous_volume.cpp)
            target_link_libraries(
                psycles_luisa_homogeneous_volume_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_homogeneous_volume_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_homogeneous_volume_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_homogeneous_volume_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_homogeneous_volume_fallback
                    COMMAND
                        psycles_luisa_homogeneous_volume_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_homogeneous_volume_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_homogeneous_volume_hip
                    COMMAND
                        psycles_luisa_homogeneous_volume_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_homogeneous_volume_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_homogeneous_volume_vk
                    COMMAND
                        psycles_luisa_homogeneous_volume_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_heterogeneous_volume_tests
                tests/test_luisa_heterogeneous_volume.cpp)
            target_link_libraries(
                psycles_luisa_heterogeneous_volume_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_heterogeneous_volume_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_heterogeneous_volume_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_heterogeneous_volume_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_heterogeneous_volume_fallback
                    COMMAND
                        psycles_luisa_heterogeneous_volume_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_heterogeneous_volume_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_heterogeneous_volume_hip
                    COMMAND
                        psycles_luisa_heterogeneous_volume_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_heterogeneous_volume_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_heterogeneous_volume_vk
                    COMMAND
                        psycles_luisa_heterogeneous_volume_tests
                        vk)
            endif()

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_heterogeneous_volume_candidate_tests
                SOURCE
                    tests/test_luisa_heterogeneous_volume_candidate.cpp
                TEST_STEM
                    luisa_heterogeneous_volume_candidate
                LIBRARIES Psycles::luisa)

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_heterogeneous_volume_segment_tests
                SOURCE
                    tests/test_luisa_heterogeneous_volume_segment.cpp
                TEST_STEM
                    luisa_heterogeneous_volume_segment
                LIBRARIES Psycles::luisa)

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_heterogeneous_volume_transmittance_tests
                SOURCE
                    tests/test_luisa_heterogeneous_volume_transmittance.cpp
                TEST_STEM
                    luisa_heterogeneous_volume_transmittance
                LIBRARIES Psycles::luisa)

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_heterogeneous_volume_shadow_tests
                SOURCE
                    tests/test_luisa_heterogeneous_volume_shadow.cpp
                TEST_STEM
                    luisa_heterogeneous_volume_shadow
                LIBRARIES Psycles::luisa)

            add_executable(
                psycles_luisa_volume_majorant_tests
                tests/test_luisa_volume_majorant.cpp)
            target_link_libraries(
                psycles_luisa_volume_majorant_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_volume_majorant_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_majorant_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_majorant_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_majorant_fallback
                    COMMAND
                        psycles_luisa_volume_majorant_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_majorant_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_majorant_hip
                    COMMAND
                        psycles_luisa_volume_majorant_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_majorant_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_majorant_vk
                    COMMAND
                        psycles_luisa_volume_majorant_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_majorant_overlap_tests
                tests/test_luisa_volume_majorant_overlap.cpp)
            target_link_libraries(
                psycles_luisa_volume_majorant_overlap_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_volume_majorant_overlap_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_majorant_overlap_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_majorant_overlap_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_majorant_overlap_fallback
                    COMMAND
                        psycles_luisa_volume_majorant_overlap_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_majorant_overlap_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_majorant_overlap_hip
                    COMMAND
                        psycles_luisa_volume_majorant_overlap_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_majorant_overlap_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_majorant_overlap_vk
                    COMMAND
                        psycles_luisa_volume_majorant_overlap_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_majorant_prepass_tests
                tests/test_luisa_volume_majorant_prepass.cpp)
            target_link_libraries(
                psycles_luisa_volume_majorant_prepass_tests
                PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_luisa_volume_majorant_prepass_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_majorant_prepass_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_majorant_prepass_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_majorant_prepass_fallback
                    COMMAND
                        psycles_luisa_volume_majorant_prepass_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_majorant_prepass_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_majorant_prepass_hip
                    COMMAND
                        psycles_luisa_volume_majorant_prepass_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_majorant_prepass_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_majorant_prepass_vk
                    COMMAND
                        psycles_luisa_volume_majorant_prepass_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_majorant_scene_tests
                tests/test_luisa_volume_majorant_scene.cpp)
            target_link_libraries(
                psycles_luisa_volume_majorant_scene_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_luisa_volume_majorant_scene_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_luisa_volume_majorant_scene_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_majorant_scene_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_majorant_scene_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_majorant_scene_fallback
                    COMMAND
                        psycles_luisa_volume_majorant_scene_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_majorant_scene_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_majorant_scene_hip
                    COMMAND
                        psycles_luisa_volume_majorant_scene_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_majorant_scene_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_majorant_scene_vk
                    COMMAND
                        psycles_luisa_volume_majorant_scene_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_guiding_tests
                tests/test_luisa_volume_guiding.cpp)
            target_link_libraries(
                psycles_luisa_volume_guiding_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_luisa_volume_guiding_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_luisa_volume_guiding_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_guiding_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_guiding_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_guiding_fallback
                    COMMAND
                        psycles_luisa_volume_guiding_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_guiding_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_guiding_hip
                    COMMAND
                        psycles_luisa_volume_guiding_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_guiding_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_guiding_vk
                    COMMAND
                        psycles_luisa_volume_guiding_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_triangle_primitive_tests
                tests/test_luisa_triangle_primitive.cpp)
            target_link_libraries(
                psycles_luisa_triangle_primitive_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_luisa_triangle_primitive_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_luisa_triangle_primitive_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_triangle_primitive_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_triangle_primitive_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_triangle_primitive_fallback
                    COMMAND
                        psycles_luisa_triangle_primitive_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_triangle_primitive_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_triangle_primitive_hip
                    COMMAND
                        psycles_luisa_triangle_primitive_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_triangle_primitive_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_triangle_primitive_vk
                    COMMAND
                        psycles_luisa_triangle_primitive_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_camera_volume_stack_tests
                tests/test_luisa_camera_volume_stack.cpp)
            target_link_libraries(
                psycles_luisa_camera_volume_stack_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_luisa_camera_volume_stack_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_luisa_camera_volume_stack_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_camera_volume_stack_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_camera_volume_stack_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_camera_volume_stack_fallback
                    COMMAND
                        psycles_luisa_camera_volume_stack_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_camera_volume_stack_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_camera_volume_stack_hip
                    COMMAND
                        psycles_luisa_camera_volume_stack_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_camera_volume_stack_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_camera_volume_stack_vk
                    COMMAND
                        psycles_luisa_camera_volume_stack_tests
                        vk)
            endif()

            add_executable(
                psycles_luisa_volume_point_tests
                tests/test_luisa_volume_point.cpp)
            target_link_libraries(
                psycles_luisa_volume_point_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_luisa_volume_point_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_luisa_volume_point_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_volume_point_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_volume_point_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_volume_point_fallback
                    COMMAND
                        psycles_luisa_volume_point_tests
                        fallback)
            endif()
            if(TARGET luisa-compute-backend-hip)
                add_dependencies(
                    psycles_luisa_volume_point_tests
                    luisa-compute-backend-hip)
                add_test(
                    NAME psycles.luisa_volume_point_hip
                    COMMAND
                        psycles_luisa_volume_point_tests
                        hip)
            endif()
            if(TARGET luisa-compute-backend-vk)
                add_dependencies(
                    psycles_luisa_volume_point_tests
                    luisa-compute-backend-vk)
                add_test(
                    NAME psycles.luisa_volume_point_vk
                    COMMAND
                        psycles_luisa_volume_point_tests
                        vk)
            endif()

            add_executable(
                psycles_volume_scene_metadata_tests
                tests/test_volume_scene_metadata.cpp)
            target_link_libraries(
                psycles_volume_scene_metadata_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_volume_scene_metadata_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_volume_scene_metadata_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.volume_scene_metadata
                COMMAND
                    psycles_volume_scene_metadata_tests)

            add_executable(
                psycles_generated_coordinates_tests
                tests/test_generated_coordinates.cpp)
            target_link_libraries(
                psycles_generated_coordinates_tests
                PRIVATE Psycles::luisa_runtime)
            target_include_directories(
                psycles_generated_coordinates_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            target_compile_features(
                psycles_generated_coordinates_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.generated_coordinates
                COMMAND
                    psycles_generated_coordinates_tests)

            if(TARGET luisa-compute-backend-fallback)
                add_executable(
                    psycles_luisa_sobol_fallback_tests
                    tests/test_luisa_sobol_fallback.cpp)
                target_link_libraries(
                    psycles_luisa_sobol_fallback_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_sobol_fallback_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_sobol_fallback_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_sobol_fallback_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_sobol_fallback
                    COMMAND psycles_luisa_sobol_fallback_tests)

                add_executable(
                    psycles_luisa_spherical_geometry_tests
                    tests/test_luisa_spherical_geometry.cpp)
                target_link_libraries(
                    psycles_luisa_spherical_geometry_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_spherical_geometry_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_spherical_geometry_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_spherical_geometry_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_spherical_geometry_fallback
                    COMMAND
                        psycles_luisa_spherical_geometry_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_spherical_geometry_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_spherical_geometry_hip
                        COMMAND
                            psycles_luisa_spherical_geometry_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_spherical_geometry_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_spherical_geometry_vk
                        COMMAND
                            psycles_luisa_spherical_geometry_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_triangle_light_sampling_tests
                    tests/test_luisa_triangle_light_sampling.cpp)
                target_link_libraries(
                    psycles_luisa_triangle_light_sampling_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_triangle_light_sampling_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_triangle_light_sampling_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_triangle_light_sampling_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_triangle_light_sampling_fallback
                    COMMAND
                        psycles_luisa_triangle_light_sampling_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_triangle_light_sampling_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_triangle_light_sampling_hip
                        COMMAND
                            psycles_luisa_triangle_light_sampling_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_triangle_light_sampling_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_triangle_light_sampling_vk
                        COMMAND
                            psycles_luisa_triangle_light_sampling_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_analytic_light_sampling_tests
                    tests/test_luisa_analytic_light_sampling.cpp)
                target_link_libraries(
                    psycles_luisa_analytic_light_sampling_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_analytic_light_sampling_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_analytic_light_sampling_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_analytic_light_sampling_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_analytic_light_sampling_fallback
                    COMMAND
                        psycles_luisa_analytic_light_sampling_tests
                        fallback)

                add_executable(
                    psycles_luisa_area_light_forward_tests
                    tests/test_luisa_area_light_forward.cpp)
                target_link_libraries(
                    psycles_luisa_area_light_forward_tests
                    PRIVATE Psycles::luisa_runtime)
                target_compile_features(
                    psycles_luisa_area_light_forward_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_area_light_forward_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_area_light_forward_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_area_light_forward_fallback
                    COMMAND
                        psycles_luisa_area_light_forward_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_area_light_forward_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_area_light_forward_hip
                        COMMAND
                            psycles_luisa_area_light_forward_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_area_light_forward_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_area_light_forward_vk
                        COMMAND
                            psycles_luisa_area_light_forward_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_volume_path_tests
                    tests/test_luisa_volume_path.cpp)
                target_link_libraries(
                    psycles_luisa_volume_path_tests
                    PRIVATE Psycles::luisa_runtime)
                target_compile_features(
                    psycles_luisa_volume_path_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_volume_path_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_volume_path_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_volume_path_fallback
                    COMMAND
                        psycles_luisa_volume_path_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_volume_path_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_volume_path_hip
                        COMMAND
                            psycles_luisa_volume_path_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_volume_path_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_volume_path_vk
                        COMMAND
                            psycles_luisa_volume_path_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_volume_triangle_tests
                    tests/test_luisa_volume_triangle.cpp)
                target_link_libraries(
                    psycles_luisa_volume_triangle_tests
                    PRIVATE Psycles::luisa_runtime)
                target_compile_features(
                    psycles_luisa_volume_triangle_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_volume_triangle_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_volume_triangle_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_volume_triangle_fallback
                    COMMAND
                        psycles_luisa_volume_triangle_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_volume_triangle_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_volume_triangle_hip
                        COMMAND
                            psycles_luisa_volume_triangle_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_volume_triangle_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_volume_triangle_vk
                        COMMAND
                            psycles_luisa_volume_triangle_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_volume_environment_tests
                    tests/test_luisa_volume_environment.cpp)
                target_link_libraries(
                    psycles_luisa_volume_environment_tests
                    PRIVATE Psycles::luisa_runtime)
                target_compile_features(
                    psycles_luisa_volume_environment_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_volume_environment_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_volume_environment_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_volume_environment_fallback
                    COMMAND
                        psycles_luisa_volume_environment_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_volume_environment_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_volume_environment_hip
                        COMMAND
                            psycles_luisa_volume_environment_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_volume_environment_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_volume_environment_vk
                        COMMAND
                            psycles_luisa_volume_environment_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_cycles_sample_mapping_tests
                    tests/test_luisa_cycles_sample_mapping.cpp)
                target_link_libraries(
                    psycles_luisa_cycles_sample_mapping_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_cycles_sample_mapping_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_cycles_sample_mapping_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_cycles_sample_mapping_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_cycles_sample_mapping_fallback
                    COMMAND
                        psycles_luisa_cycles_sample_mapping_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_cycles_sample_mapping_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_cycles_sample_mapping_hip
                        COMMAND
                            psycles_luisa_cycles_sample_mapping_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_cycles_sample_mapping_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_cycles_sample_mapping_vk
                        COMMAND
                            psycles_luisa_cycles_sample_mapping_tests
                            vk)
                endif()

                psycles_add_luisa_backend_test(
                    TARGET
                        psycles_luisa_cycles_closure_tests
                    SOURCE
                        tests/test_luisa_cycles_closure.cpp
                    TEST_STEM
                        luisa_cycles_closure
                    LIBRARIES Psycles::luisa)

                psycles_add_luisa_backend_test(
                    TARGET
                        psycles_luisa_principled_coat_tests
                    SOURCE
                        tests/test_luisa_principled_coat.cpp
                    TEST_STEM
                        luisa_principled_coat
                    LIBRARIES Psycles::luisa)

                psycles_add_luisa_backend_test(
                    TARGET
                        psycles_luisa_principled_transmission_tests
                    SOURCE
                        tests/test_luisa_principled_transmission.cpp
                    TEST_STEM
                        luisa_principled_transmission
                    LIBRARIES Psycles::luisa)

                add_executable(
                    psycles_luisa_cycles_path_state_tests
                    tests/test_luisa_cycles_path_state.cpp)
                target_link_libraries(
                    psycles_luisa_cycles_path_state_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_cycles_path_state_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_cycles_path_state_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_cycles_path_state_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_cycles_path_state_fallback
                    COMMAND
                        psycles_luisa_cycles_path_state_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_cycles_path_state_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_cycles_path_state_hip
                        COMMAND
                            psycles_luisa_cycles_path_state_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_cycles_path_state_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_cycles_path_state_vk
                        COMMAND
                            psycles_luisa_cycles_path_state_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_camera_sampling_tests
                    tests/test_luisa_camera_sampling.cpp)
                target_link_libraries(
                    psycles_luisa_camera_sampling_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_camera_sampling_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_camera_sampling_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_camera_sampling_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_camera_sampling_fallback
                    COMMAND
                        psycles_luisa_camera_sampling_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_camera_sampling_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_camera_sampling_hip
                        COMMAND
                            psycles_luisa_camera_sampling_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_camera_sampling_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_camera_sampling_vk
                        COMMAND
                            psycles_luisa_camera_sampling_tests
                            vk)
                endif()

                add_executable(
                    psycles_luisa_background_sampling_tests
                    tests/test_luisa_background_sampling.cpp)
                target_link_libraries(
                    psycles_luisa_background_sampling_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_background_sampling_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_background_sampling_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_background_sampling_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_background_sampling_fallback
                    COMMAND
                        psycles_luisa_background_sampling_tests
                        fallback)

                add_executable(
                    psycles_luisa_surface_ray_tests
                    tests/test_luisa_surface_ray.cpp)
                target_link_libraries(
                    psycles_luisa_surface_ray_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_surface_ray_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_surface_ray_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_surface_ray_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_surface_ray_fallback
                    COMMAND
                        psycles_luisa_surface_ray_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_surface_ray_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_surface_ray_hip
                        COMMAND
                            psycles_luisa_surface_ray_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_surface_ray_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_surface_ray_vk
                        COMMAND
                            psycles_luisa_surface_ray_tests
                            vk)
                endif()
            endif()

            add_executable(
                psycles_blender_import_tests
                tests/test_blender_import.cpp)
            target_link_libraries(
                psycles_blender_import_tests
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_blender_import_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.blender_import
                COMMAND psycles_blender_import_tests)

            add_executable(
                psycles_sample_dispatch_partition_tests
                tests/test_sample_dispatch_partition.cpp)
            target_compile_features(
                psycles_sample_dispatch_partition_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.sample_dispatch_partition
                COMMAND psycles_sample_dispatch_partition_tests)

            add_executable(
                psycles_cycles_integrator_limits_tests
                tests/test_cycles_integrator_limits.cpp)
            target_compile_features(
                psycles_cycles_integrator_limits_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.cycles_integrator_limits
                COMMAND psycles_cycles_integrator_limits_tests)

            add_executable(
                psycles_cycles_filter_glossy_tests
                tests/test_cycles_filter_glossy.cpp)
            target_compile_features(
                psycles_cycles_filter_glossy_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.cycles_filter_glossy
                COMMAND psycles_cycles_filter_glossy_tests)

            add_executable(
                psycles_cycles_shader_identity_tests
                tests/test_cycles_shader_identity.cpp
                src/luisa/path_tracer_scene_geometry.cpp)
            target_link_libraries(
                psycles_cycles_shader_identity_tests
                PRIVATE Psycles::core)
            target_compile_features(
                psycles_cycles_shader_identity_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.cycles_shader_identity
                COMMAND psycles_cycles_shader_identity_tests)
        endif()

        if(PSYCLES_BUILD_EXAMPLES)
            add_executable(
                psycles_luisa_render_demo
                examples/render_luisa_demo.cpp)
            target_link_libraries(
                psycles_luisa_render_demo
                PRIVATE
                    Psycles::luisa_runtime)
            target_compile_features(
                psycles_luisa_render_demo
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_render_demo
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")

            add_executable(
                psycles_render_blender_scene
                examples/render_blender_scene.cpp)
            target_link_libraries(
                psycles_render_blender_scene
                PRIVATE
                    Psycles::luisa_runtime)
            target_compile_features(
                psycles_render_blender_scene
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_render_blender_scene
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")

            add_executable(
                psycles_probe_nishita
                examples/probe_nishita.cpp)
            target_link_libraries(
                psycles_probe_nishita
                PRIVATE
                    Psycles::luisa_runtime)
            target_compile_features(
                psycles_probe_nishita
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_probe_nishita
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")

            add_executable(
                psycles_inspect_blender_material
                tools/inspect_blender_material.cpp)
            target_link_libraries(
                psycles_inspect_blender_material
                PRIVATE
                    Psycles::luisa_runtime)
            target_compile_features(
                psycles_inspect_blender_material
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_inspect_blender_material
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
        endif()
    else()
        message(STATUS
            "LuisaCompute target luisa::compute is unavailable; "
            "building contract core only")
    endif()
endif()

if(PSYCLES_BUILD_TESTS)
    add_executable(psycles_tests tests/test_main.cpp)
    target_link_libraries(psycles_tests PRIVATE Psycles::core)
    target_compile_features(psycles_tests PRIVATE cxx_std_20)
    add_test(NAME psycles.contracts COMMAND psycles_tests)

    add_executable(
        psycles_tabulated_sobol_tests
        tests/test_tabulated_sobol.cpp)
    target_link_libraries(
        psycles_tabulated_sobol_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_tabulated_sobol_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.tabulated_sobol
        COMMAND psycles_tabulated_sobol_tests)

    add_executable(
        psycles_pixel_filter_tests
        tests/test_pixel_filter.cpp)
    target_link_libraries(
        psycles_pixel_filter_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_pixel_filter_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.pixel_filter
        COMMAND psycles_pixel_filter_tests)

    add_executable(
        psycles_background_distribution_tests
        tests/test_background_distribution.cpp)
    target_link_libraries(
        psycles_background_distribution_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_background_distribution_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.background_distribution
        COMMAND psycles_background_distribution_tests)

    add_executable(
        psycles_light_distribution_tests
        tests/test_light_distribution.cpp)
    target_link_libraries(
        psycles_light_distribution_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_light_distribution_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.light_distribution
        COMMAND psycles_light_distribution_tests)

    if(PSYCLES_ENABLE_OPENIMAGEIO)
        add_executable(
            psycles_openexr_tests
            tests/test_openexr.cpp)
        target_link_libraries(
            psycles_openexr_tests
            PRIVATE
                Psycles::core
                OpenImageIO::OpenImageIO)
        target_compile_features(
            psycles_openexr_tests
            PRIVATE cxx_std_20)
        add_test(
            NAME psycles.openexr
            COMMAND psycles_openexr_tests)
    endif()

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME psycles.source_size
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_source_size.py"
                "${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(
            NAME psycles.cycles_shader_node_inventory
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_cycles_shader_node_coverage.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/cycles-shader-nodes-4.5.10.json"
                --probe-baselines
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/cycles-shader-probe-baselines-4.5.10.json")
        add_test(
            NAME psycles.cycles_integrator_baselines
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_cycles_integrator_baselines.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/cycles-integrator-probe-baselines-4.5.10.json")
        add_test(
            NAME psycles.cycles_light_baselines
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/check_cycles_light_baselines.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/docs/cycles-light-probe-baselines-4.5.10.json")
        add_test(
            NAME psycles.shader_probe_runner_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_shader_probe_runner.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_cycles_shader_probes.py")
        add_test(
            NAME psycles.scene_benchmark_runner_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_scene_benchmark_runner.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/run_scene_benchmark.py")
        add_test(
            NAME psycles.compare_cycles_contract
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_compare_cycles.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_cycles.py")
        add_test(
            NAME psycles.cycles_path_trace_schema
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cycles_path_trace_schema.py")
        add_test(
            NAME psycles.cycles_path_trace_comparison
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cycles_path_trace_comparison.py")
        add_test(
            NAME psycles.cycles_path_trace_decoder
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cycles_path_trace_decoder.py")
    endif()

    find_program(PSYCLES_BLENDER_EXECUTABLE NAMES blender)
    if(PSYCLES_BLENDER_EXECUTABLE)
        add_test(
            NAME psycles.blender_export_geometry_cache
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_geometry_cache.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py")
        add_test(
            NAME psycles.blender_export_attribute_domains
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python-exit-code
                1
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_attribute_domains.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py"
                "$<TARGET_FILE:psycles_inspect_blender_material>")
        add_test(
            NAME psycles.blender_export_linked_images
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python-exit-code
                1
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_linked_images.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py")
        add_test(
            NAME psycles.blender_export_generated_images
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python-exit-code
                1
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_generated_images.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py")
        add_test(
            NAME psycles.blender_export_particle_info
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_particle_info.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py")
        add_test(
            NAME psycles.blender_export_render_settings
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python-exit-code
                1
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_render_settings.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_cycles_shader_probe.py")
        add_test(
            NAME psycles.blender_export_cycles_identity
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python-exit-code
                1
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_cycles_identity.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py")
        add_test(
            NAME psycles.blender_diagnostic_probes
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python-exit-code
                1
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_diagnostic_probes.py"
                --
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_blender_shader_stage_probe.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/probe_cycles_world.py")
        add_test(
            NAME psycles.blender_multilayer_exr_api
            COMMAND
                "${PSYCLES_BLENDER_EXECUTABLE}"
                --background
                --factory-startup
                --python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_multilayer_exr_api.py")
        set_tests_properties(
            psycles.blender_export_geometry_cache
            psycles.blender_export_attribute_domains
            psycles.blender_export_linked_images
            psycles.blender_export_generated_images
            psycles.blender_export_particle_info
            psycles.blender_export_render_settings
            psycles.blender_export_cycles_identity
            psycles.blender_diagnostic_probes
            psycles.blender_multilayer_exr_api
            PROPERTIES
                ENVIRONMENT
                    "SDL_AUDIODRIVER=dummy;ALSOFT_DRIVERS=null")
    endif()
endif()
