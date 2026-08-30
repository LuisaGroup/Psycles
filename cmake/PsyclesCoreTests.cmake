if(PSYCLES_BUILD_TESTS)
    add_executable(psycles_tests tests/test_main.cpp)
    target_link_libraries(psycles_tests PRIVATE Psycles::core)
    target_compile_features(psycles_tests PRIVATE cxx_std_20)
    add_test(NAME psycles.contracts COMMAND psycles_tests)

    add_executable(
        psycles_graph_material_scene_tests
        tests/test_graph_material_scene.cpp)
    target_link_libraries(
        psycles_graph_material_scene_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_graph_material_scene_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.graph_material_scene
        COMMAND psycles_graph_material_scene_tests)

    add_executable(
        psycles_surface_program_metadata_tests
        tests/test_surface_program_metadata.cpp
        tests/surface_program_metadata_closure_tests.cpp)
    target_link_libraries(
        psycles_surface_program_metadata_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_program_metadata_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_program_metadata
        COMMAND psycles_surface_program_metadata_tests)

    add_executable(
        psycles_surface_svm_math_immediate_tests
        tests/test_surface_svm_math_immediate.cpp)
    target_link_libraries(
        psycles_surface_svm_math_immediate_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_svm_math_immediate_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_svm_math_immediate
        COMMAND psycles_surface_svm_math_immediate_tests)

    add_executable(
        psycles_surface_svm_vector_math_immediate_tests
        tests/test_surface_svm_vector_math_immediate.cpp)
    target_link_libraries(
        psycles_surface_svm_vector_math_immediate_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_svm_vector_math_immediate_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_svm_vector_math_immediate
        COMMAND psycles_surface_svm_vector_math_immediate_tests)

    add_executable(
        psycles_surface_svm_record_immediate_tests
        tests/test_surface_svm_record_immediates.cpp)
    target_link_libraries(
        psycles_surface_svm_record_immediate_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_svm_record_immediate_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_svm_record_immediates
        COMMAND psycles_surface_svm_record_immediate_tests)

    add_executable(
        psycles_surface_closure_execution_plan_tests
        tests/test_surface_closure_execution_plan.cpp)
    target_link_libraries(
        psycles_surface_closure_execution_plan_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_closure_execution_plan_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_closure_execution_plan
        COMMAND psycles_surface_closure_execution_plan_tests)

    add_executable(
        psycles_surface_svm_schedule_tests
        tests/test_surface_svm_schedule.cpp)
    target_link_libraries(
        psycles_surface_svm_schedule_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_svm_schedule_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_svm_schedule
        COMMAND psycles_surface_svm_schedule_tests)

    add_executable(
        psycles_surface_svm_scene_tests
        tests/test_surface_svm_scene.cpp)
    target_link_libraries(
        psycles_surface_svm_scene_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_svm_scene_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_svm_scene
        COMMAND psycles_surface_svm_scene_tests)

    add_executable(
        psycles_cycles_svm_abi_tests
        tests/test_cycles_svm_abi.cpp)
    target_link_libraries(
        psycles_cycles_svm_abi_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_abi_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_abi
        COMMAND psycles_cycles_svm_abi_tests)

    add_executable(
        psycles_cycles_svm_bytecode_tests
        tests/test_cycles_svm_bytecode.cpp)
    target_link_libraries(
        psycles_cycles_svm_bytecode_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_bytecode_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_bytecode
        COMMAND psycles_cycles_svm_bytecode_tests)

    add_executable(
        psycles_cycles_svm_compiler_tests
        tests/test_cycles_svm_compiler.cpp)
    target_link_libraries(
        psycles_cycles_svm_compiler_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_compiler_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_compiler
        COMMAND psycles_cycles_svm_compiler_tests)

    add_executable(
        psycles_cycles_svm_modern_mix_tests
        tests/test_cycles_svm_modern_mix.cpp)
    target_link_libraries(
        psycles_cycles_svm_modern_mix_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_modern_mix_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_modern_mix
        COMMAND psycles_cycles_svm_modern_mix_tests)

    add_executable(
        psycles_cycles_svm_vector_tests
        tests/test_cycles_svm_vector.cpp)
    target_link_libraries(
        psycles_cycles_svm_vector_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_vector_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_vector
        COMMAND psycles_cycles_svm_vector_tests)

    add_executable(
        psycles_cycles_svm_vector_rotate_tests
        tests/test_cycles_svm_vector_rotate.cpp)
    target_link_libraries(
        psycles_cycles_svm_vector_rotate_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_vector_rotate_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_vector_rotate
        COMMAND psycles_cycles_svm_vector_rotate_tests)

    add_executable(
        psycles_cycles_svm_vector_transform_tests
        tests/test_cycles_svm_vector_transform.cpp)
    target_link_libraries(
        psycles_cycles_svm_vector_transform_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_vector_transform_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_vector_transform
        COMMAND psycles_cycles_svm_vector_transform_tests)

    add_executable(
        psycles_cycles_svm_wireframe_tests
        tests/test_cycles_svm_wireframe.cpp)
    target_link_libraries(
        psycles_cycles_svm_wireframe_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_wireframe_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_wireframe_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_wireframe
        COMMAND psycles_cycles_svm_wireframe_tests)

    add_executable(
        psycles_progressive_pixel_probe_tests
        tests/test_progressive_pixel_probe.cpp)
    target_link_libraries(
        psycles_progressive_pixel_probe_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_progressive_pixel_probe_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.progressive_pixel_probe
        COMMAND psycles_progressive_pixel_probe_tests)

    add_executable(
        psycles_hair_info_graph_tests
        tests/test_hair_info_graph.cpp)
    target_link_libraries(
        psycles_hair_info_graph_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_hair_info_graph_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.hair_info_graph
        COMMAND psycles_hair_info_graph_tests)

    add_executable(
        psycles_curve_scene_contract_tests
        tests/test_curve_scene_contract.cpp)
    target_link_libraries(
        psycles_curve_scene_contract_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_curve_scene_contract_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.curve_scene_contract
        COMMAND psycles_curve_scene_contract_tests)

    add_executable(
        psycles_cycles_pointiness_tests
        tests/test_cycles_pointiness.cpp)
    target_link_libraries(
        psycles_cycles_pointiness_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_pointiness_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_pointiness
        COMMAND psycles_cycles_pointiness_tests)

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

    add_executable(
        psycles_light_tree_tests
        tests/test_light_tree.cpp)
    target_link_libraries(
        psycles_light_tree_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_light_tree_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.light_tree
        COMMAND psycles_light_tree_tests)

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
        set(blender_exporter
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/export_psycles_scene.py")
        set(blender_inspector
            "$<TARGET_FILE:psycles_inspect_blender_material>")
        psycles_add_blender_test(
            NAME psycles.blender_export_geometry_cache
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_geometry_cache.py"
            NO_EXIT_CODE ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_attribute_domains
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_attribute_domains.py"
            ARGUMENTS "${blender_exporter}" "${blender_inspector}")
        psycles_add_blender_test(
            NAME psycles.blender_export_smooth_normals
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_smooth_normals.py"
            NO_EXIT_CODE ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_triangle_order
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_triangle_order.py"
            NO_EXIT_CODE ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_pointiness_source
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_pointiness_source.py"
            ARGUMENTS "${blender_exporter}" "${blender_inspector}")
        foreach(test IN ITEMS
                linked_images
                generated_images
                particle_hair
                cycles_identity
                cycles_output
                muted_nodes)
            psycles_add_blender_test(
                NAME "psycles.blender_export_${test}"
                SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_${test}.py"
                ARGUMENTS "${blender_exporter}")
        endforeach()
        psycles_add_blender_test(
            NAME psycles.blender_export_texture_coordinate_object
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_texture_coordinate_object.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_texture_mapping
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_texture_mapping.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_particle_info
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_particle_info.py"
            NO_EXIT_CODE ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_render_settings
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_render_settings.py"
            ARGUMENTS
                "${blender_exporter}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_cycles_shader_probe.py")
        psycles_add_blender_test(
            NAME psycles.blender_diagnostic_probes
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_diagnostic_probes.py"
            ARGUMENTS
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_blender_shader_stage_probe.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/probe_cycles_world.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/create_blender_surface_cost_probe.py")
        psycles_add_blender_test(
            NAME psycles.blender_multilayer_exr_api
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_multilayer_exr_api.py"
            NO_EXIT_CODE)
    endif()
endif()
