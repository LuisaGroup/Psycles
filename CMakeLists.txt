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
option(PSYCLES_ENABLE_LUISA_HIP
    "Build LuisaCompute's HIP backend" ON)
option(PSYCLES_ENABLE_LUISA_VULKAN
    "Build LuisaCompute's Vulkan backend" ON)
option(PSYCLES_FETCH_LUISA_NEXT
    "Fetch LuisaCompute from its next branch when the bundled submodule is unavailable" OFF)
set(PSYCLES_LUISA_SUBMODULE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/LuisaCompute")

if(PSYCLES_BUILD_TESTS)
    include(CTest)
endif()

add_library(psycles_core
    src/adapter/cycles_shader_graph.cpp
    src/compiler/core_nodes.cpp
    src/compiler/material_library.cpp
    src/compiler/shader_program.cpp
    src/compiler/surface_program.cpp
    src/contract/scene.cpp
    src/contract/shader_graph.cpp
    src/io/image.cpp
    src/sampling/background_distribution.cpp
    src/sampling/light_distribution.cpp
    src/sampling/pixel_filter.cpp
    src/sampling/tabulated_sobol.cpp)
add_library(Psycles::core ALIAS psycles_core)

target_include_directories(psycles_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
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
        add_library(psycles_luisa INTERFACE)
        add_library(Psycles::luisa ALIAS psycles_luisa)
        target_include_directories(psycles_luisa INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>)
        target_compile_features(psycles_luisa INTERFACE cxx_std_20)
        target_compile_definitions(psycles_luisa INTERFACE PSYCLES_WITH_LUISA=1)
        target_link_libraries(psycles_luisa INTERFACE
            Psycles::core
            luisa::compute)
        message(STATUS "Psycles Luisa DSL interface enabled")

        add_library(
            psycles_luisa_runtime
            src/adapter/blender_scene.cpp
            src/luisa/path_tracer.cpp
            src/luisa/path_tracer_camera.cpp
            src/luisa/path_tracer_common.cpp
            src/luisa/path_tracer_environment.cpp
            src/luisa/path_tracer_geometry.cpp
            src/luisa/path_tracer_kernel.cpp
            src/luisa/path_tracer_light_distribution.cpp
            src/luisa/path_tracer_lighting.cpp
            src/luisa/path_tracer_scene.cpp
            src/luisa/path_tracer_session.cpp
            src/luisa/path_tracer_surfaces.cpp)
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

                add_executable(
                    psycles_luisa_cycles_closure_tests
                    tests/test_luisa_cycles_closure.cpp)
                target_link_libraries(
                    psycles_luisa_cycles_closure_tests
                    PRIVATE Psycles::luisa)
                target_compile_features(
                    psycles_luisa_cycles_closure_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_cycles_closure_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_cycles_closure_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_cycles_closure_fallback
                    COMMAND
                        psycles_luisa_cycles_closure_tests
                        fallback)

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
                tests/test_cycles_shader_identity.cpp)
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
    endif()
endif()
