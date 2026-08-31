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
option(PSYCLES_ENABLE_LUISA_SIMD
    "Build LuisaCompute's experimental SIMD CPU backend" OFF)
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
    include(cmake/PsyclesBlenderTest.cmake)
    include(cmake/PsyclesLuisaBackendTest.cmake)
endif()
add_library(psycles_core
    src/adapter/cycles_shader_graph.cpp
    src/compiler/core_nodes.cpp
    src/compiler/cycles_svm_attribute_nodes.cpp
    src/compiler/cycles_svm_bytecode.cpp
    src/compiler/cycles_svm_camera_nodes.cpp
    src/compiler/cycles_svm_closure_nodes.cpp
    src/compiler/cycles_svm_constant_fold.cpp
    src/compiler/cycles_svm_compiler.cpp
    src/compiler/cycles_svm_geometry_nodes.cpp
    src/compiler/cycles_svm_graph.cpp
    src/compiler/cycles_svm_image_nodes.cpp
    src/compiler/cycles_svm_mapping_nodes.cpp
    src/compiler/cycles_svm_noise_nodes.cpp
    src/compiler/cycles_svm_nodes.cpp
    src/compiler/cycles_svm_ramp_nodes.cpp
    src/compiler/cycles_svm_texture_coordinate_nodes.cpp
    src/compiler/cycles_svm_value_nodes.cpp
    src/compiler/cycles_svm_vector_nodes.cpp
    src/compiler/cycles_transform.cpp
    src/compiler/cycles_emission_sampling.cpp
    src/compiler/hosek_sky.cpp
    src/compiler/material_library.cpp
    src/compiler/shader_program.cpp
    src/compiler/surface_bump_expansion.cpp
    src/compiler/surface_closure_execution_plan.cpp
    src/compiler/surface_svm_schedule.cpp
    src/compiler/surface_svm_evaluator.cpp
    src/compiler/surface_svm_lowering.cpp
    src/compiler/surface_svm_scene.cpp
    src/compiler/surface_svm_storage.cpp
    src/compiler/surface_execution_plan.cpp
    src/compiler/surface_value_variant_interner.cpp
    src/compiler/surface_program.cpp
    src/compiler/surface_program_builder.cpp
    src/compiler/surface_program_compaction.cpp
    src/compiler/surface_program_lower_closures.cpp
    src/compiler/surface_program_lower_context.cpp
    src/compiler/surface_program_lower_textures.cpp
    src/compiler/surface_program_lower_values.cpp
    src/compiler/surface_value_definition_liveness.cpp
    src/compiler/surface_value_dependencies.cpp
    src/contract/cycles_pointiness.cpp
    src/contract/scene.cpp
    src/contract/shader_graph.cpp
    src/io/image.cpp
    src/io/progressive_pixel_probe.cpp
    src/sampling/background_distribution.cpp
    src/sampling/light_distribution.cpp
    src/sampling/light_tree.cpp
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
    # Psycles uses the platform C++20 library throughout its host-side graph
    # compiler. Keep the Luisa public ABI on the same STL implementation and
    # avoid exposing bundled EASTL types through Luisa headers.
    set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL
        "Use system STL for Psycles' LuisaCompute dependency" FORCE)
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
        set(LUISA_COMPUTE_ENABLE_SIMD
            ${PSYCLES_ENABLE_LUISA_SIMD} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_METAL
            ${PSYCLES_ENABLE_LUISA_METAL} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_HIP
            ${PSYCLES_ENABLE_LUISA_HIP} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VULKAN
            ${PSYCLES_ENABLE_LUISA_VULKAN} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV
            ${PSYCLES_ENABLE_LUISA_VULKAN} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV
            OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY
            OFF CACHE BOOL "" FORCE)
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
        set(LUISA_COMPUTE_ENABLE_SIMD
            ${PSYCLES_ENABLE_LUISA_SIMD} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_METAL
            ${PSYCLES_ENABLE_LUISA_METAL} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_HIP
            ${PSYCLES_ENABLE_LUISA_HIP} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VULKAN
            ${PSYCLES_ENABLE_LUISA_VULKAN} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_XIR_SPIRV
            ${PSYCLES_ENABLE_LUISA_VULKAN} CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_ENABLE_VK_AST_LLVM_SPIRV
            OFF CACHE BOOL "" FORCE)
        set(LUISA_COMPUTE_VULKAN_ENABLE_DXC_COMPATIBILITY
            OFF CACHE BOOL "" FORCE)
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
    if(PSYCLES_ENABLE_LUISA_SIMD AND
       NOT TARGET luisa-compute-backend-simd)
        message(FATAL_ERROR
            "PSYCLES_ENABLE_LUISA_SIMD=ON requires LuisaCompute's SIMD "
            "backend, but LuisaCompute did not create "
            "luisa-compute-backend-simd. Install compatible LLVM and "
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
        include(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PsyclesLuisaSources.cmake")
        add_library(
            psycles_luisa
            STATIC
            ${PSYCLES_LUISA_SOURCES})
        # The Luisa DSL archive is linked into psycles_luisa_runtime, which is
        # a shared library when BUILD_SHARED_LIBS is enabled.  Keep every
        # object in the archive position-independent; in particular,
        # Type::of<T>() owns function-local TLS whose local-exec relocations
        # cannot be promoted by the shared-library link step.
        set_target_properties(
            psycles_luisa
            PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
            ${PSYCLES_LUISA_RUNTIME_SOURCES})
        add_library(
            Psycles::luisa_runtime
            ALIAS psycles_luisa_runtime)
        target_include_directories(psycles_luisa_runtime
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
        target_include_directories(psycles_luisa_runtime
            SYSTEM PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party/mikktspace)
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
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_luisa_compile_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.luisa_ast
                COMMAND psycles_luisa_compile_tests)

            add_executable(
                psycles_luisa_path_scheduler_tests
                tests/test_luisa_path_scheduler.cpp)
            target_link_libraries(
                psycles_luisa_path_scheduler_tests
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_luisa_path_scheduler_tests
                PRIVATE cxx_std_20)
            target_include_directories(
                psycles_luisa_path_scheduler_tests
                PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            add_test(
                NAME psycles.luisa_path_scheduler
                COMMAND psycles_luisa_path_scheduler_tests)

            add_executable(
                psycles_luisa_sample_dispatch_film_tests
                tests/test_luisa_sample_dispatch_film.cpp)
            target_link_libraries(
                psycles_luisa_sample_dispatch_film_tests
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_luisa_sample_dispatch_film_tests
                PRIVATE cxx_std_20)
            set_target_properties(
                psycles_luisa_sample_dispatch_film_tests
                PROPERTIES
                    RUNTIME_OUTPUT_DIRECTORY
                        "${CMAKE_BINARY_DIR}/bin")
            if(TARGET luisa-compute-backend-fallback)
                add_dependencies(
                    psycles_luisa_sample_dispatch_film_tests
                    luisa-compute-backend-fallback)
                add_test(
                    NAME psycles.luisa_sample_dispatch_film_fallback
                    COMMAND
                        psycles_luisa_sample_dispatch_film_tests
                        fallback)
            endif()

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
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_analytic_light_sampling_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_analytic_light_sampling_hip
                        COMMAND
                            psycles_luisa_analytic_light_sampling_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_analytic_light_sampling_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_analytic_light_sampling_vk
                        COMMAND
                            psycles_luisa_analytic_light_sampling_tests
                            vk)
                endif()

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
                    psycles_luisa_emissive_triangle_forward_pdf_tests
                    tests/test_luisa_emissive_triangle_forward_pdf.cpp)
                target_link_libraries(
                    psycles_luisa_emissive_triangle_forward_pdf_tests
                    PRIVATE Psycles::luisa_runtime)
                target_compile_features(
                    psycles_luisa_emissive_triangle_forward_pdf_tests
                    PRIVATE cxx_std_20)
                add_dependencies(
                    psycles_luisa_emissive_triangle_forward_pdf_tests
                    luisa-compute-backend-fallback)
                set_target_properties(
                    psycles_luisa_emissive_triangle_forward_pdf_tests
                    PROPERTIES
                        RUNTIME_OUTPUT_DIRECTORY
                            "${CMAKE_BINARY_DIR}/bin")
                add_test(
                    NAME psycles.luisa_emissive_triangle_forward_pdf_fallback
                    COMMAND
                        psycles_luisa_emissive_triangle_forward_pdf_tests
                        fallback)
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_emissive_triangle_forward_pdf_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_emissive_triangle_forward_pdf_hip
                        COMMAND
                            psycles_luisa_emissive_triangle_forward_pdf_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_emissive_triangle_forward_pdf_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_emissive_triangle_forward_pdf_vk
                        COMMAND
                            psycles_luisa_emissive_triangle_forward_pdf_tests
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

                include(cmake/PsyclesCurveTests.cmake)

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
                    PRIVATE Psycles::luisa_runtime)
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
                if(TARGET luisa-compute-backend-hip)
                    add_dependencies(
                        psycles_luisa_background_sampling_tests
                        luisa-compute-backend-hip)
                    add_test(
                        NAME psycles.luisa_background_sampling_hip
                        COMMAND
                            psycles_luisa_background_sampling_tests
                            hip)
                endif()
                if(TARGET luisa-compute-backend-vk)
                    add_dependencies(
                        psycles_luisa_background_sampling_tests
                        luisa-compute-backend-vk)
                    add_test(
                        NAME psycles.luisa_background_sampling_vk
                        COMMAND
                            psycles_luisa_background_sampling_tests
                            vk)
                endif()

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

            # These fixtures register independently for every enabled Luisa
            # backend. Keep them outside the fallback-only block so a native
            # Metal/HIP/Vulkan build does not silently lose surface coverage.
            include(cmake/PsyclesSurfaceBackendTests.cmake)
            include(cmake/PsyclesBlenderImportTests.cmake)

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_pointiness_scene_tests
                SOURCE
                    tests/test_luisa_pointiness_scene.cpp
                TEST_STEM
                    luisa_pointiness_scene
                LIBRARIES Psycles::luisa_runtime)

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_displacement_scene_tests
                SOURCE
                    tests/test_luisa_displacement_scene.cpp
                TEST_STEM
                    luisa_displacement_scene
                LIBRARIES Psycles::luisa_runtime)

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_surface_nee_normal_tests
                SOURCE
                    tests/test_luisa_surface_nee_normal.cpp
                TEST_STEM
                    luisa_surface_nee_normal
                LIBRARIES Psycles::luisa_runtime)
            target_include_directories(
                psycles_luisa_surface_nee_normal_tests
                PRIVATE
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
            foreach(_backend IN ITEMS fallback simd metal hip vk)
                if(TEST psycles.luisa_surface_nee_normal_${_backend})
                    set_tests_properties(
                        psycles.luisa_surface_nee_normal_${_backend}
                        PROPERTIES TIMEOUT 180)
                endif()
            endforeach()

            psycles_add_luisa_backend_test(
                TARGET
                    psycles_luisa_curve_path_tests
                SOURCE
                    tests/test_luisa_curve_path.cpp
                TEST_STEM
                    luisa_curve_path
                LIBRARIES Psycles::luisa_runtime)

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
                src/luisa/path_tracer_displacement_plan.cpp
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

            add_executable(
                psycles_cycles_instance_support_tests
                tests/test_cycles_instance_support.cpp)
            target_link_libraries(
                psycles_cycles_instance_support_tests
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_cycles_instance_support_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.cycles_instance_support
                COMMAND psycles_cycles_instance_support_tests)

            add_executable(
                psycles_cycles_tangent_space_tests
                tests/test_cycles_tangent_space.cpp)
            target_link_libraries(
                psycles_cycles_tangent_space_tests
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_cycles_tangent_space_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.cycles_tangent_space
                COMMAND psycles_cycles_tangent_space_tests)

            add_executable(
                psycles_attribute_residency_tests
                tests/test_attribute_residency.cpp)
            target_link_libraries(
                psycles_attribute_residency_tests
                PRIVATE Psycles::luisa_runtime)
            target_compile_features(
                psycles_attribute_residency_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.attribute_residency
                COMMAND psycles_attribute_residency_tests)

            add_executable(
                psycles_normal_map_semantics_tests
                tests/test_normal_map_semantics.cpp)
            target_link_libraries(
                psycles_normal_map_semantics_tests
                PRIVATE Psycles::core)
            target_compile_features(
                psycles_normal_map_semantics_tests
                PRIVATE cxx_std_20)
            add_test(
                NAME psycles.normal_map_semantics
                COMMAND psycles_normal_map_semantics_tests)
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

            add_executable(
                psycles_benchmark_surface_closures
                tools/benchmark_surface_closures.cpp)
            target_link_libraries(
                psycles_benchmark_surface_closures PRIVATE Psycles::luisa)
            target_compile_features(
                psycles_benchmark_surface_closures PRIVATE cxx_std_20)
            set_target_properties(
                psycles_benchmark_surface_closures
                PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
            if(PSYCLES_BUILD_TESTS)
                add_test(
                    NAME psycles.inspect_blender_material_producer_partition
                    COMMAND
                        psycles_inspect_blender_material
                        --self-test-producer-partition)
                add_test(
                    NAME psycles.inspect_blender_material_operand_index_partition
                    COMMAND
                        psycles_inspect_blender_material
                        --self-test-operand-index-partition)
            endif()
        endif()
    else()
        message(STATUS
            "LuisaCompute target luisa::compute is unavailable; "
            "building contract core only")
    endif()
endif()

include(cmake/PsyclesCoreTests.cmake)
