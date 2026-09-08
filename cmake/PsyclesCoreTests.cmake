if(PSYCLES_BUILD_TESTS)
    add_executable(psycles_path_trace_scheduler_comparison_tests
        tests/test_path_trace_scheduler_comparison.cpp)
    target_include_directories(psycles_path_trace_scheduler_comparison_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../include")
    target_compile_features(psycles_path_trace_scheduler_comparison_tests PRIVATE cxx_std_20)
    add_test(NAME psycles.path_trace_scheduler_comparison
        COMMAND psycles_path_trace_scheduler_comparison_tests)

    add_executable(psycles_cycles_camera_projection_tests
        tests/test_cycles_camera_projection.cpp)
    target_link_libraries(psycles_cycles_camera_projection_tests PRIVATE Psycles::core)
    target_compile_features(psycles_cycles_camera_projection_tests PRIVATE cxx_std_20)
    target_compile_definitions(psycles_cycles_camera_projection_tests PRIVATE
        PSYCLES_CAMERA_PROJECTION_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_camera_projection.txt")
    add_test(NAME psycles.cycles_camera_projection COMMAND psycles_cycles_camera_projection_tests)

    add_executable(psycles_cycles_static_normal_tests
        tests/test_cycles_static_normal.cpp)
    target_link_libraries(psycles_cycles_static_normal_tests PRIVATE Psycles::core)
    target_compile_features(psycles_cycles_static_normal_tests PRIVATE cxx_std_20)
    target_compile_definitions(psycles_cycles_static_normal_tests PRIVATE
        PSYCLES_STATIC_NORMAL_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_static_normal.txt")
    add_test(NAME psycles.cycles_static_normal COMMAND psycles_cycles_static_normal_tests)

    add_executable(psycles_cycles_wavefront_policy_tests
        tests/test_cycles_wavefront_policy.cpp)
    target_include_directories(psycles_cycles_wavefront_policy_tests PRIVATE
        "${CMAKE_CURRENT_LIST_DIR}/../src/luisa")
    target_compile_features(psycles_cycles_wavefront_policy_tests PRIVATE cxx_std_20)
    target_compile_definitions(psycles_cycles_wavefront_policy_tests PRIVATE
        PSYCLES_WAVEFRONT_POLICY_ORACLE="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_wavefront_policy.txt")
    add_test(NAME psycles.cycles_wavefront_policy
        COMMAND psycles_cycles_wavefront_policy_tests)

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
        psycles_surface_program_vector_components_tests
        tests/test_surface_program_vector_components.cpp)
    target_link_libraries(
        psycles_surface_program_vector_components_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_surface_program_vector_components_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.surface_program_vector_components
        COMMAND psycles_surface_program_vector_components_tests)

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

    add_executable(psycles_cycles_svm_compiler_color_tests tests/test_cycles_svm_compiler_color.cpp)
    target_link_libraries(psycles_cycles_svm_compiler_color_tests PRIVATE Psycles::core)
    target_compile_features(psycles_cycles_svm_compiler_color_tests PRIVATE cxx_std_20)
    add_test(NAME psycles.cycles_svm_compiler_color COMMAND psycles_cycles_svm_compiler_color_tests)

    add_executable(
        psycles_cycles_svm_default_input_provenance_tests
        tests/test_cycles_svm_default_input_provenance.cpp)
    target_link_libraries(
        psycles_cycles_svm_default_input_provenance_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_default_input_provenance_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_default_input_provenance
        COMMAND psycles_cycles_svm_default_input_provenance_tests)

    add_executable(
        psycles_cycles_svm_ray_portal_compiler_tests
        tests/test_cycles_svm_ray_portal_compiler.cpp)
    target_link_libraries(
        psycles_cycles_svm_ray_portal_compiler_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_ray_portal_compiler_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_ray_portal_compiler
        COMMAND psycles_cycles_svm_ray_portal_compiler_tests)

    add_executable(
        psycles_cycles_svm_hair_compiler_tests
        tests/test_cycles_svm_hair_compiler.cpp)
    target_link_libraries(
        psycles_cycles_svm_hair_compiler_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_hair_compiler_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_hair_compiler
        COMMAND psycles_cycles_svm_hair_compiler_tests)

    add_executable(
        psycles_cycles_svm_scene_tests
        tests/test_cycles_svm_scene.cpp)
    target_link_libraries(
        psycles_cycles_svm_scene_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_scene_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_scene
        COMMAND psycles_cycles_svm_scene_tests)

    add_executable(
        psycles_cycles_svm_kernel_features_tests
        tests/test_cycles_svm_kernel_features.cpp)
    target_link_libraries(
        psycles_cycles_svm_kernel_features_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_kernel_features_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_kernel_features
        COMMAND psycles_cycles_svm_kernel_features_tests)

    add_executable(psycles_cycles_svm_shared_closure_tests
        tests/test_cycles_svm_shared_closure.cpp)
    target_link_libraries(psycles_cycles_svm_shared_closure_tests PRIVATE Psycles::core)
    target_compile_features(psycles_cycles_svm_shared_closure_tests PRIVATE cxx_std_20)
    target_compile_definitions(psycles_cycles_svm_shared_closure_tests PRIVATE
        PSYCLES_SHARED_CLOSURE_WORDS_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_shared_closure_words.txt")
    add_test(NAME psycles.cycles_svm_shared_closure
        COMMAND psycles_cycles_svm_shared_closure_tests)

    add_executable(psycles_cycles_svm_entry_usage_tests
        tests/test_cycles_svm_entry_usage.cpp)
    target_link_libraries(psycles_cycles_svm_entry_usage_tests PRIVATE Psycles::core)
    target_compile_features(psycles_cycles_svm_entry_usage_tests PRIVATE cxx_std_20)
    target_compile_definitions(psycles_cycles_svm_entry_usage_tests PRIVATE
        PSYCLES_SHARED_CLOSURE_WORDS_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_shared_closure_words.txt")
    add_test(NAME psycles.cycles_svm_entry_usage
        COMMAND psycles_cycles_svm_entry_usage_tests)

    add_executable(psycles_cycles_svm_closure_budget_tests
        tests/test_cycles_svm_closure_budget.cpp)
    target_link_libraries(psycles_cycles_svm_closure_budget_tests PRIVATE Psycles::core)
    target_compile_features(psycles_cycles_svm_closure_budget_tests PRIVATE cxx_std_20)
    target_compile_definitions(psycles_cycles_svm_closure_budget_tests PRIVATE
        PSYCLES_CLOSURE_BUDGET_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_closure_budget.txt")
    add_test(NAME psycles.cycles_svm_closure_budget
        COMMAND psycles_cycles_svm_closure_budget_tests)

    add_executable(
        psycles_cycles_svm_object_scene_tests
        tests/test_cycles_svm_object_scene.cpp)
    target_link_libraries(
        psycles_cycles_svm_object_scene_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_object_scene_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_object_scene
        COMMAND psycles_cycles_svm_object_scene_tests)

    add_executable(
        psycles_cycles_svm_geometry_scene_tests
        tests/test_cycles_svm_geometry_scene.cpp)
    target_link_libraries(
        psycles_cycles_svm_geometry_scene_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_geometry_scene_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_geometry_scene
        COMMAND psycles_cycles_svm_geometry_scene_tests)

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

    if(TARGET psycles_luisa_runtime)
        add_executable(psycles_cycles_svm_group_liveness_tests tests/test_cycles_svm_group_liveness.cpp)
        target_link_libraries(psycles_cycles_svm_group_liveness_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_group_liveness_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_group_liveness_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_group_liveness COMMAND psycles_cycles_svm_group_liveness_tests)
        add_executable(psycles_cycles_svm_mapping_declarations_tests tests/test_cycles_svm_mapping_declarations.cpp)
        target_link_libraries(psycles_cycles_svm_mapping_declarations_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_mapping_declarations_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_mapping_declarations_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_mapping_declarations COMMAND psycles_cycles_svm_mapping_declarations_tests)
        add_executable(psycles_cycles_svm_math_expand_tests tests/test_cycles_svm_math_expand.cpp)
        target_link_libraries(psycles_cycles_svm_math_expand_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_math_expand_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_math_expand_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_math_expand COMMAND psycles_cycles_svm_math_expand_tests)
        add_executable(psycles_cycles_svm_group_forward_tests tests/test_cycles_svm_group_forward.cpp)
        target_link_libraries(psycles_cycles_svm_group_forward_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_group_forward_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_group_forward_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_group_forward COMMAND psycles_cycles_svm_group_forward_tests)
        add_executable(psycles_cycles_svm_closure_inputs_tests tests/test_cycles_svm_closure_inputs.cpp)
        target_link_libraries(psycles_cycles_svm_closure_inputs_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_closure_inputs_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_closure_inputs_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_closure_inputs COMMAND psycles_cycles_svm_closure_inputs_tests)
        add_executable(psycles_cycles_svm_socket_types_tests tests/test_cycles_svm_socket_types.cpp)
        target_link_libraries(psycles_cycles_svm_socket_types_tests PRIVATE Psycles::luisa_runtime)
        target_include_directories(psycles_cycles_svm_socket_types_tests PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
        target_compile_features(psycles_cycles_svm_socket_types_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_socket_types_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_socket_types COMMAND psycles_cycles_svm_socket_types_tests)
        add_executable(psycles_cycles_svm_texture_outputs_tests tests/test_cycles_svm_texture_outputs.cpp)
        target_link_libraries(psycles_cycles_svm_texture_outputs_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_texture_outputs_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_texture_outputs_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_texture_outputs COMMAND psycles_cycles_svm_texture_outputs_tests)
        add_executable(psycles_cycles_svm_bump_alias_tests tests/test_cycles_svm_bump_alias.cpp)
        target_link_libraries(psycles_cycles_svm_bump_alias_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_bump_alias_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_bump_alias_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_bump_alias COMMAND psycles_cycles_svm_bump_alias_tests)
        add_executable(psycles_cycles_svm_hidden_socket_tests
            tests/test_cycles_svm_hidden_socket.cpp)
        target_link_libraries(psycles_cycles_svm_hidden_socket_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_hidden_socket_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_hidden_socket_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_hidden_socket
            COMMAND psycles_cycles_svm_hidden_socket_tests)
        add_executable(psycles_cycles_svm_vector_fold_tests tests/test_cycles_svm_vector_fold.cpp)
        target_link_libraries(psycles_cycles_svm_vector_fold_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_vector_fold_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_vector_fold_tests PRIVATE
            PSYCLES_SVM_IMPORT_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_vector_fold COMMAND psycles_cycles_svm_vector_fold_tests)
        add_executable(psycles_cycles_svm_map_range_tests tests/test_cycles_svm_map_range.cpp)
        target_link_libraries(psycles_cycles_svm_map_range_tests PRIVATE Psycles::luisa_runtime)
        target_compile_definitions(psycles_cycles_svm_map_range_tests PRIVATE
            PSYCLES_MAP_RANGE_SCENE="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_map_range_scene.json"
            PSYCLES_MAP_RANGE_WORDS="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_map_range_words.txt"
            PSYCLES_MONSTER_MAP_RANGE_SCENE="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_monster_map_range_scene.json"
            PSYCLES_MONSTER_MAP_RANGE_WORDS="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_monster_map_range_words.txt")
        add_test(NAME psycles.cycles_svm_map_range COMMAND psycles_cycles_svm_map_range_tests)
        add_executable(psycles_cycles_svm_analytic_sky_tests
            tests/test_cycles_svm_analytic_sky.cpp)
        target_link_libraries(psycles_cycles_svm_analytic_sky_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_analytic_sky_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_analytic_sky_tests PRIVATE
            PSYCLES_HOSEK_SCENE="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_hosek_sky_scene.json"
            PSYCLES_HOSEK_WORDS="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_hosek_sky_words.txt"
            PSYCLES_PREETHAM_SCENE="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_preetham_sky_scene.json"
            PSYCLES_PREETHAM_WORDS="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_preetham_sky_words.txt")
        add_test(NAME psycles.cycles_svm_analytic_sky
            COMMAND psycles_cycles_svm_analytic_sky_tests)
        add_executable(psycles_cycles_svm_volume_emission_tests
            tests/test_cycles_svm_volume_emission.cpp)
        target_link_libraries(psycles_cycles_svm_volume_emission_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_volume_emission_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_volume_emission_tests PRIVATE
            PSYCLES_VOLUME_EMISSION_SCENE="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_volume_emission_scene.json"
            PSYCLES_VOLUME_EMISSION_WORDS="${CMAKE_CURRENT_LIST_DIR}/../tests/data/cycles_volume_emission_words.txt")
        add_test(NAME psycles.cycles_svm_volume_emission
            COMMAND psycles_cycles_svm_volume_emission_tests)

        add_executable(psycles_cycles_svm_volume_tests tests/test_cycles_svm_volume.cpp)
        target_link_libraries(psycles_cycles_svm_volume_tests PRIVATE Psycles::luisa_runtime)
        target_compile_features(psycles_cycles_svm_volume_tests PRIVATE cxx_std_20)
        target_compile_definitions(psycles_cycles_svm_volume_tests PRIVATE
            PSYCLES_VOLUME_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/../tests/data")
        add_test(NAME psycles.cycles_svm_volume COMMAND psycles_cycles_svm_volume_tests)

        add_executable(
            psycles_cycles_svm_bump_state_tests
            tests/test_cycles_svm_bump_state.cpp)
        target_link_libraries(
            psycles_cycles_svm_bump_state_tests
            PRIVATE Psycles::luisa_runtime)
        target_compile_features(
            psycles_cycles_svm_bump_state_tests
            PRIVATE cxx_std_20)
        add_test(
            NAME psycles.cycles_svm_bump_state
            COMMAND psycles_cycles_svm_bump_state_tests)

        add_executable(psycles_cycles_svm_bsdf_case_groups_tests
            tests/test_cycles_svm_bsdf_case_groups.cpp)
        target_link_libraries(psycles_cycles_svm_bsdf_case_groups_tests
            PRIVATE Psycles::luisa_runtime)
        target_include_directories(psycles_cycles_svm_bsdf_case_groups_tests
            PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
        target_compile_features(psycles_cycles_svm_bsdf_case_groups_tests PRIVATE cxx_std_20)
        add_test(NAME psycles.cycles_svm_bsdf_case_groups
            COMMAND psycles_cycles_svm_bsdf_case_groups_tests)

        add_executable(psycles_cycles_svm_closure_dispatch_tests
            tests/test_cycles_svm_closure_dispatch.cpp)
        target_link_libraries(psycles_cycles_svm_closure_dispatch_tests
            PRIVATE Psycles::luisa_runtime)
        target_include_directories(psycles_cycles_svm_closure_dispatch_tests
            PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
        target_compile_features(psycles_cycles_svm_closure_dispatch_tests PRIVATE cxx_std_20)
        add_test(NAME psycles.cycles_svm_closure_dispatch
            COMMAND psycles_cycles_svm_closure_dispatch_tests)

        add_executable(
            psycles_cycles_svm_vector_displacement_tests
            tests/test_cycles_svm_vector_displacement.cpp)
        target_link_libraries(
            psycles_cycles_svm_vector_displacement_tests
            PRIVATE Psycles::luisa_runtime)
        target_compile_features(
            psycles_cycles_svm_vector_displacement_tests
            PRIVATE cxx_std_20)
        add_test(
            NAME psycles.cycles_svm_vector_displacement
            COMMAND psycles_cycles_svm_vector_displacement_tests)
    endif()

    add_executable(
        psycles_cycles_svm_vertex_color_tests
        tests/test_cycles_svm_vertex_color.cpp)
    target_link_libraries(
        psycles_cycles_svm_vertex_color_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_vertex_color_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_vertex_color_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_vertex_color
        COMMAND psycles_cycles_svm_vertex_color_tests)

    add_executable(
        psycles_cycles_svm_background_tests
        tests/test_cycles_svm_background.cpp)
    target_link_libraries(
        psycles_cycles_svm_background_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_background_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_background
        COMMAND psycles_cycles_svm_background_tests)

    add_executable(
        psycles_cycles_svm_texture_coordinate_tests
        tests/test_cycles_svm_texture_coordinate.cpp)
    target_link_libraries(
        psycles_cycles_svm_texture_coordinate_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_texture_coordinate_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_texture_coordinate
        COMMAND psycles_cycles_svm_texture_coordinate_tests)

    add_executable(
        psycles_cycles_svm_image_tests
        tests/test_cycles_svm_image.cpp)
    target_link_libraries(
        psycles_cycles_svm_image_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_image_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_image
        COMMAND psycles_cycles_svm_image_tests)

    add_executable(
        psycles_cycles_svm_sky_tests
        tests/test_cycles_svm_sky.cpp)
    target_link_libraries(
        psycles_cycles_svm_sky_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_sky_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_sky
        COMMAND psycles_cycles_svm_sky_tests)

    add_executable(
        psycles_cycles_svm_mapping_tests
        tests/test_cycles_svm_mapping.cpp)
    target_link_libraries(
        psycles_cycles_svm_mapping_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_mapping_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_mapping
        COMMAND psycles_cycles_svm_mapping_tests)

    add_executable(
        psycles_cycles_svm_noise_tests
        tests/test_cycles_svm_noise.cpp)
    target_link_libraries(
        psycles_cycles_svm_noise_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_noise_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_noise
        COMMAND psycles_cycles_svm_noise_tests)

    add_executable(
        psycles_cycles_svm_white_noise_tests
        tests/test_cycles_svm_white_noise.cpp)
    target_link_libraries(
        psycles_cycles_svm_white_noise_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_white_noise_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_white_noise
        COMMAND psycles_cycles_svm_white_noise_tests)

    add_executable(
        psycles_cycles_svm_gradient_tests
        tests/test_cycles_svm_gradient.cpp)
    target_link_libraries(
        psycles_cycles_svm_gradient_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_gradient_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_gradient
        COMMAND psycles_cycles_svm_gradient_tests)

    add_executable(
        psycles_cycles_svm_rgb_ramp_tests
        tests/test_cycles_svm_rgb_ramp.cpp)
    target_link_libraries(
        psycles_cycles_svm_rgb_ramp_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_rgb_ramp_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_rgb_ramp
        COMMAND psycles_cycles_svm_rgb_ramp_tests)

    add_executable(
        psycles_cycles_svm_rgb_curve_tests
        tests/test_cycles_svm_rgb_curve.cpp)
    target_link_libraries(
        psycles_cycles_svm_rgb_curve_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_rgb_curve_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_rgb_curve
        COMMAND psycles_cycles_svm_rgb_curve_tests)

    add_executable(
        psycles_cycles_svm_curve_family_tests
        tests/test_cycles_svm_curve_family.cpp)
    target_link_libraries(
        psycles_cycles_svm_curve_family_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_curve_family_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_curve_family
        COMMAND psycles_cycles_svm_curve_family_tests)

    add_executable(
        psycles_cycles_svm_camera_tests
        tests/test_cycles_svm_camera.cpp)
    target_link_libraries(
        psycles_cycles_svm_camera_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_camera_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_camera
        COMMAND psycles_cycles_svm_camera_tests)

    add_executable(
        psycles_cycles_svm_fresnel_tests
        tests/test_cycles_svm_fresnel.cpp)
    target_link_libraries(
        psycles_cycles_svm_fresnel_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_fresnel_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_fresnel_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_fresnel
        COMMAND psycles_cycles_svm_fresnel_tests)

    add_executable(
        psycles_cycles_svm_light_path_tests
        tests/test_cycles_svm_light_path.cpp)
    target_link_libraries(
        psycles_cycles_svm_light_path_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_light_path_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_light_path_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_light_path
        COMMAND psycles_cycles_svm_light_path_tests)

    add_executable(
        psycles_cycles_svm_info_tests
        tests/test_cycles_svm_info.cpp)
    target_link_libraries(
        psycles_cycles_svm_info_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_info_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_info_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_info
        COMMAND psycles_cycles_svm_info_tests)

    add_executable(
        psycles_cycles_svm_attribute_request_tests
        tests/test_cycles_svm_attribute_requests.cpp)
    target_link_libraries(
        psycles_cycles_svm_attribute_request_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_attribute_request_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_attribute_requests
        COMMAND psycles_cycles_svm_attribute_request_tests)

    add_executable(
        psycles_cycles_svm_normal_tests
        tests/test_cycles_svm_normal.cpp)
    target_link_libraries(
        psycles_cycles_svm_normal_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_normal_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_normal_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_normal
        COMMAND psycles_cycles_svm_normal_tests)

    add_executable(
        psycles_cycles_svm_light_falloff_tests
        tests/test_cycles_svm_light_falloff.cpp)
    target_link_libraries(
        psycles_cycles_svm_light_falloff_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_light_falloff_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_light_falloff_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_light_falloff
        COMMAND psycles_cycles_svm_light_falloff_tests)

    add_executable(
        psycles_cycles_svm_ies_tests
        tests/test_cycles_svm_ies.cpp)
    target_link_libraries(
        psycles_cycles_svm_ies_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_ies_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_ies_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_ies
        COMMAND psycles_cycles_svm_ies_tests)

    add_executable(
        psycles_cycles_svm_procedural_texture_tests
        tests/test_cycles_svm_procedural_texture.cpp)
    target_link_libraries(
        psycles_cycles_svm_procedural_texture_tests
        PRIVATE Psycles::core)
    target_include_directories(
        psycles_cycles_svm_procedural_texture_tests
        PRIVATE ${PROJECT_SOURCE_DIR}/src/compiler)
    target_compile_features(
        psycles_cycles_svm_procedural_texture_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_procedural_texture
        COMMAND psycles_cycles_svm_procedural_texture_tests)

    add_executable(
        psycles_cycles_svm_spectral_tests
        tests/test_cycles_svm_spectral.cpp)
    target_link_libraries(
        psycles_cycles_svm_spectral_tests
        PRIVATE Psycles::core)
    target_compile_features(
        psycles_cycles_svm_spectral_tests
        PRIVATE cxx_std_20)
    add_test(
        NAME psycles.cycles_svm_spectral
        COMMAND psycles_cycles_svm_spectral_tests)

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
            NAME psycles.luisa_shader_performance_policy
            COMMAND
                "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_luisa_shader_performance_policy.py"
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
            NAME psycles.render_pass_contract
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_render_pass_contract.py")
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
        psycles_add_blender_test(
            NAME psycles.blender_cycles_sampler
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_cycles_sampler.py")
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
                missing_images
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
            NAME psycles.blender_export_curve_mapping
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_curve_mapping.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_camera_data
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_camera_data.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_socket_availability
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_socket_availability.py"
            ARGUMENTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/blender_scene_manifest.py")
        psycles_add_blender_test(
            NAME psycles.blender_export_fresnel
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_fresnel.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_light_path
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_light_path.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_info_nodes
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_info_nodes.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_normal_node
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_normal_node.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_light_falloff
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_light_falloff.py"
            ARGUMENTS "${blender_exporter}")
        psycles_add_blender_test(
            NAME psycles.blender_export_ies
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_export_ies.py"
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
            NAME psycles.blender_golden_pass_contract
            SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_blender_golden_pass_contract.py"
            ARGUMENTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/render_cycles_golden.py")
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
