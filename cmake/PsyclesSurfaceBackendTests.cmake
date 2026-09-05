include_guard(GLOBAL)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_shadow_surface_tests
    SOURCE tests/test_luisa_cycles_svm_shadow_surface.cpp
    TEST_STEM luisa_cycles_svm_shadow_surface
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_svm_shadow_surface_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_cycles_svm_shadow_surface_tests PRIVATE
    PSYCLES_SHADOW_SURFACE_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_shadow_surface.txt")
if(TEST psycles.luisa_cycles_svm_shadow_surface_vk)
    set_tests_properties(psycles.luisa_cycles_svm_shadow_surface_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_scene_metadata_tests
    SOURCE tests/test_luisa_cycles_svm_scene_metadata.cpp
    TEST_STEM luisa_cycles_svm_scene_metadata
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_svm_scene_metadata_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_cycles_svm_scene_metadata_vk)
    set_tests_properties(psycles.luisa_cycles_svm_scene_metadata_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_scene_compilation_tests
    SOURCE tests/test_luisa_cycles_svm_scene_compilation.cpp
    TEST_STEM luisa_cycles_svm_scene_compilation
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_svm_scene_compilation_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_cycles_svm_scene_compilation_tests PRIVATE
    PSYCLES_CAMERA_SCENE="${PROJECT_SOURCE_DIR}/tests/data/cycles_camera_data_scene.json"
    PSYCLES_CAMERA_WORDS="${PROJECT_SOURCE_DIR}/tests/data/cycles_camera_data_words.txt")
if(TEST psycles.luisa_cycles_svm_scene_compilation_vk)
    set_tests_properties(psycles.luisa_cycles_svm_scene_compilation_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_camera_transform_tests
    SOURCE tests/test_luisa_cycles_camera_transform.cpp
    TEST_STEM luisa_cycles_camera_transform
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_camera_transform_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_cycles_camera_transform_tests PRIVATE
    PSYCLES_CAMERA_TRANSFORM_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_camera_transform.txt")
if(TEST psycles.luisa_cycles_camera_transform_vk)
    set_tests_properties(psycles.luisa_cycles_camera_transform_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_surface_queue_tests
    SOURCE tests/test_luisa_cycles_surface_queue.cpp
    TEST_STEM luisa_cycles_surface_queue
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_surface_queue_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_cycles_surface_queue_tests PRIVATE
    PSYCLES_SURFACE_QUEUE_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_wavefront_sort.txt")
if(TEST psycles.luisa_cycles_surface_queue_vk)
    set_tests_properties(psycles.luisa_cycles_surface_queue_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_background_sun_sample_tests
    SOURCE tests/test_luisa_background_sun_sample.cpp
    TEST_STEM luisa_background_sun_sample
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_background_sun_sample_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_background_sun_sample_tests PRIVATE
    PSYCLES_BACKGROUND_SUN_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_background_sun_sample.txt")
if(TEST psycles.luisa_background_sun_sample_vk)
    set_tests_properties(psycles.luisa_background_sun_sample_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_nee_setup_tests
    SOURCE tests/test_luisa_cycles_svm_nee_setup.cpp
    TEST_STEM luisa_cycles_svm_nee_setup
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_svm_nee_setup_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_cycles_svm_nee_setup_tests PRIVATE
    PSYCLES_NEE_SETUP_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_nee_setup.txt")
if(TEST psycles.luisa_cycles_svm_nee_setup_vk)
    set_tests_properties(psycles.luisa_cycles_svm_nee_setup_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_nee_tests
    SOURCE tests/test_luisa_cycles_svm_nee.cpp
    TEST_STEM luisa_cycles_svm_nee
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_svm_nee_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(psycles_luisa_cycles_svm_nee_tests PRIVATE
    PSYCLES_LIGHT_EMISSION_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_light_emission.txt")
if(TEST psycles.luisa_cycles_svm_nee_vk)
    set_tests_properties(psycles.luisa_cycles_svm_nee_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_light_emission_tests
    SOURCE tests/test_luisa_cycles_svm_light_emission.cpp
    TEST_STEM luisa_cycles_svm_light_emission
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_light_emission_tests
    PRIVATE "${PROJECT_SOURCE_DIR}/src/luisa")
target_compile_definitions(
    psycles_luisa_cycles_svm_light_emission_tests PRIVATE
    PSYCLES_LIGHT_EMISSION_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_light_emission.txt")
if(TEST psycles.luisa_cycles_svm_light_emission_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_light_emission_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

# Surface and closure fixtures share the same per-backend registration policy.
# Keep the inventory out of the project root so adding one semantic regression
# does not make the top-level build description grow without bound.
psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_roughness_tests
    SOURCE tests/test_luisa_surface_roughness.cpp
    TEST_STEM luisa_surface_roughness
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_surface_roughness_vk)
    set_tests_properties(
        psycles.luisa_surface_roughness_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_closure_tests
    SOURCE tests/test_luisa_cycles_closure.cpp
    TEST_STEM luisa_cycles_closure
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_principled_sheen_tests
    SOURCE tests/test_luisa_principled_sheen.cpp
    TEST_STEM luisa_principled_sheen
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_standalone_sheen_tests
    SOURCE tests/test_luisa_standalone_sheen.cpp
    TEST_STEM luisa_standalone_sheen
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_standalone_sheen_vk)
    set_tests_properties(
        psycles.luisa_standalone_sheen_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_legacy_hair_tests
    SOURCE tests/test_luisa_legacy_hair.cpp
    TEST_STEM luisa_legacy_hair
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_legacy_hair_vk)
    set_tests_properties(
        psycles.luisa_legacy_hair_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_beckmann_glossy_tests
    SOURCE tests/test_luisa_beckmann_glossy.cpp
    TEST_STEM luisa_beckmann_glossy
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_microfacet_anisotropy_tests
    SOURCE tests/test_luisa_microfacet_anisotropy.cpp
    TEST_STEM luisa_microfacet_anisotropy
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_microfacet_anisotropy_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_microfacet_anisotropy_vk)
    set_tests_properties(
        psycles.luisa_microfacet_anisotropy_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_metallic_surface_tests
    SOURCE tests/test_luisa_metallic_surface.cpp
    TEST_STEM luisa_metallic_surface
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_metallic_surface_vk)
    set_tests_properties(
        psycles.luisa_metallic_surface_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_ray_differential_tests
    SOURCE tests/test_luisa_ray_differential.cpp
    TEST_STEM luisa_ray_differential
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_normal_map_tests
    SOURCE tests/test_luisa_normal_map.cpp
    TEST_STEM luisa_normal_map
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_geometry_context_tests
    SOURCE tests/test_luisa_surface_geometry_context.cpp
    TEST_STEM luisa_surface_geometry_context
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_surface_geometry_context_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_surface_geometry_context_vk)
    set_tests_properties(
        psycles.luisa_surface_geometry_context_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_film_light_tests
    SOURCE tests/test_luisa_cycles_film_light.cpp
    TEST_STEM luisa_cycles_film_light
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_bssrdf_tests
    SOURCE tests/test_luisa_bssrdf.cpp
    TEST_STEM luisa_bssrdf
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_subsurface_exit_tests
    SOURCE tests/test_luisa_subsurface_exit.cpp
    TEST_STEM luisa_subsurface_exit
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_subsurface_exit_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_random_walk_tests
    SOURCE tests/test_luisa_random_walk.cpp
    TEST_STEM luisa_random_walk
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_random_walk_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_closure_collection_tests
    SOURCE tests/test_luisa_surface_closure_collection.cpp
    TEST_STEM luisa_surface_closure_collection
    LIBRARIES Psycles::luisa)
target_sources(
    psycles_luisa_surface_closure_collection_tests
    PRIVATE tests/luisa_surface_closure_collection_test_support.cpp)
if(TEST psycles.luisa_surface_closure_collection_vk)
    set_tests_properties(
        psycles.luisa_surface_closure_collection_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_closure_reachability_tests
    SOURCE tests/test_luisa_surface_closure_reachability.cpp
    TEST_STEM luisa_surface_closure_reachability
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_surface_closure_reachability_vk)
    set_tests_properties(
        psycles.luisa_surface_closure_reachability_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_population_tests
    SOURCE tests/test_luisa_surface_population.cpp
    TEST_STEM luisa_surface_population
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_surface_population_vk)
    set_tests_properties(
        psycles.luisa_surface_population_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_compact_surface_preparation_tests
    SOURCE tests/test_luisa_compact_surface_preparation.cpp
    TEST_STEM luisa_compact_surface_preparation
    LIBRARIES Psycles::luisa_runtime)
target_sources(
    psycles_luisa_compact_surface_preparation_tests
    PRIVATE
        tests/compact_surface_program_test_support.cpp
        tests/compact_surface_color_family_test_support.cpp
        tests/compact_surface_procedural_family_test_support.cpp
        tests/compact_surface_context_family_test_support.cpp
        tests/compact_surface_state_family_test_support.cpp)
target_include_directories(
    psycles_luisa_compact_surface_preparation_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_compact_surface_preparation_vk)
    set_tests_properties(
        psycles.luisa_compact_surface_preparation_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1")
endif()
# The default fixture contains Add(Mix(...), sibling) and therefore selects
# full parent-restoring frames. Reuse the same differential executable with a
# root-tail Mix scene so every enabled backend also compiles and executes the
# scalar no-restoration JIT shape.
foreach(_backend IN ITEMS fallback simd metal hip vk)
    if(TARGET luisa-compute-backend-${_backend})
        add_test(
            NAME psycles.luisa_compact_surface_tail_${_backend}
            COMMAND psycles_luisa_compact_surface_preparation_tests
                    ${_backend} tail)
    endif()
endforeach()
if(TEST psycles.luisa_compact_surface_tail_vk)
    set_tests_properties(
        psycles.luisa_compact_surface_tail_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_closure_point_tests
    SOURCE tests/test_luisa_surface_closure_point.cpp
    TEST_STEM luisa_surface_closure_point
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_surface_closure_point_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_surface_closure_point_vk)
    set_tests_properties(
        psycles.luisa_surface_closure_point_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_curve_closure_semantics_tests
    SOURCE tests/test_luisa_curve_closure_semantics.cpp
    TEST_STEM luisa_curve_closure_semantics
    LIBRARIES Psycles::luisa)
if(TEST psycles.luisa_curve_closure_semantics_vk)
    set_tests_properties(
        psycles.luisa_curve_closure_semantics_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_closure_physical_tests
    SOURCE tests/test_luisa_surface_closure_physical.cpp
    TEST_STEM luisa_surface_closure_physical
    LIBRARIES Psycles::luisa_runtime)
if(TEST psycles.luisa_surface_closure_physical_vk)
    set_tests_properties(
        psycles.luisa_surface_closure_physical_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_thin_film_fresnel_tests
    SOURCE tests/test_luisa_thin_film_fresnel.cpp
    TEST_STEM luisa_thin_film_fresnel
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_thin_film_fresnel_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_thin_film_fresnel_vk)
    set_tests_properties(
        psycles.luisa_thin_film_fresnel_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_thin_film_surface_tests
    SOURCE tests/test_luisa_thin_film_surface.cpp
    TEST_STEM luisa_thin_film_surface
    LIBRARIES Psycles::luisa Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_thin_film_surface_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_thin_film_surface_vk)
    set_tests_properties(
        psycles.luisa_thin_film_surface_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_standalone_caustics_tests
    SOURCE tests/test_luisa_standalone_caustics.cpp
    TEST_STEM luisa_standalone_caustics
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_principled_coat_tests
    SOURCE tests/test_luisa_principled_coat.cpp
    TEST_STEM luisa_principled_coat
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_principled_transmission_tests
    SOURCE tests/test_luisa_principled_transmission.cpp
    TEST_STEM luisa_principled_transmission
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_principled_setup_callable_tests
    SOURCE tests/test_luisa_principled_setup_callable.cpp
    TEST_STEM luisa_principled_setup_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_principled_setup_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_texture_sampling_callable_tests
    SOURCE tests/test_luisa_texture_sampling_callable.cpp
    TEST_STEM luisa_texture_sampling_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_texture_sampling_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_attribute_lookup_callable_tests
    SOURCE tests/test_luisa_attribute_lookup_callable.cpp
    TEST_STEM luisa_attribute_lookup_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_attribute_lookup_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_color_transform_callable_tests
    SOURCE tests/test_luisa_color_transform_callable.cpp
    TEST_STEM luisa_color_transform_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_color_transform_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_vector_mapping_callable_tests
    SOURCE tests/test_luisa_vector_mapping_callable.cpp
    TEST_STEM luisa_vector_mapping_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_vector_mapping_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_shader_table_callable_tests
    SOURCE tests/test_luisa_shader_table_callable.cpp
    TEST_STEM luisa_shader_table_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_shader_table_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_normal_map_callable_tests
    SOURCE tests/test_luisa_normal_map_callable.cpp
    TEST_STEM luisa_normal_map_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_normal_map_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_bump_callable_tests
    SOURCE tests/test_luisa_bump_callable.cpp
    TEST_STEM luisa_bump_callable
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_bump_callable_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_displacement_tests
    SOURCE tests/test_luisa_displacement.cpp
    TEST_STEM luisa_displacement
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_displacement_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_noise_callable_tests
    SOURCE tests/test_luisa_noise_callable.cpp
    TEST_STEM luisa_noise_callable
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_mix_svm_tests
    SOURCE tests/test_luisa_surface_mix_svm.cpp
    TEST_STEM luisa_surface_mix_svm
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_surface_mix_svm_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_surface_mix_svm_vk)
    set_tests_properties(
        psycles.luisa_surface_mix_svm_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_math_svm_tests
    SOURCE tests/test_luisa_surface_math_svm.cpp
    TEST_STEM luisa_surface_math_svm
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_surface_math_svm_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_tests
    SOURCE tests/test_luisa_cycles_svm.cpp
    TEST_STEM luisa_cycles_svm
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_wireframe_tests
    SOURCE tests/test_luisa_cycles_svm_wireframe.cpp
    TEST_STEM luisa_cycles_svm_wireframe
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_wireframe_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_bump_state_tests
    SOURCE tests/test_luisa_cycles_svm_bump_state.cpp
    TEST_STEM luisa_cycles_svm_bump_state
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_bump_state_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_bump_state_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_bump_state_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_vector_displacement_tests
    SOURCE tests/test_luisa_cycles_svm_vector_displacement.cpp
    TEST_STEM luisa_cycles_svm_vector_displacement
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_vector_displacement_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_vector_displacement_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_vector_displacement_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_primitive_attribute_tests
    SOURCE tests/test_luisa_cycles_svm_primitive_attribute.cpp
    TEST_STEM luisa_cycles_svm_primitive_attribute
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_primitive_attribute_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)

# This fixture validates the transactional host-side DeviceScene projection;
# it deliberately owns no device so one proof covers every backend upload.
add_executable(
    psycles_luisa_cycles_svm_geometry_scene_tests
    tests/test_luisa_cycles_svm_geometry_scene.cpp)
target_link_libraries(
    psycles_luisa_cycles_svm_geometry_scene_tests
    PRIVATE Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_geometry_scene_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
target_compile_features(
    psycles_luisa_cycles_svm_geometry_scene_tests
    PRIVATE cxx_std_20)
set_target_properties(
    psycles_luisa_cycles_svm_geometry_scene_tests
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
add_test(
    NAME psycles.luisa_cycles_svm_geometry_scene
    COMMAND psycles_luisa_cycles_svm_geometry_scene_tests)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_geometry_runtime_tests
    SOURCE tests/test_luisa_cycles_svm_geometry_runtime.cpp
    TEST_STEM luisa_cycles_svm_geometry_runtime
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_geometry_runtime_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_geometry_runtime_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_geometry_runtime_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_texture_coordinate_tests
    SOURCE tests/test_luisa_cycles_svm_texture_coordinate.cpp
    TEST_STEM luisa_cycles_svm_texture_coordinate
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_texture_coordinate_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_texture_coordinate_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_texture_coordinate_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_image_tests
    SOURCE tests/test_luisa_cycles_svm_image.cpp
    TEST_STEM luisa_cycles_svm_image
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_image_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_image_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_image_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_sky_tests
    SOURCE tests/test_luisa_cycles_svm_sky.cpp
    TEST_STEM luisa_cycles_svm_sky
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_sky_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_sky_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_sky_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_mapping_tests
    SOURCE tests/test_luisa_cycles_svm_mapping.cpp
    TEST_STEM luisa_cycles_svm_mapping
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_mapping_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_mapping_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_mapping_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_vector_math_tests
    SOURCE tests/test_luisa_cycles_svm_vector_math.cpp
    TEST_STEM luisa_cycles_svm_vector_math
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_vector_math_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_vector_math_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_vector_math_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_noise_tests
    SOURCE tests/test_luisa_cycles_svm_noise.cpp
    TEST_STEM luisa_cycles_svm_noise
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_noise_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_noise_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_noise_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_white_noise_tests
    SOURCE tests/test_luisa_cycles_svm_white_noise.cpp
    TEST_STEM luisa_cycles_svm_white_noise
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_white_noise_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_white_noise_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_white_noise_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_gradient_tests
    SOURCE tests/test_luisa_cycles_svm_gradient.cpp
    TEST_STEM luisa_cycles_svm_gradient
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_gradient_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_gradient_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_gradient_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_voronoi_tests
    SOURCE tests/test_luisa_cycles_svm_voronoi.cpp
    TEST_STEM luisa_cycles_svm_voronoi
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_voronoi_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_voronoi_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_voronoi_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_gabor_tests
    SOURCE tests/test_luisa_cycles_svm_gabor.cpp
    TEST_STEM luisa_cycles_svm_gabor
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_gabor_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_gabor_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_gabor_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_spectral_tests
    SOURCE tests/test_luisa_cycles_svm_spectral.cpp
    TEST_STEM luisa_cycles_svm_spectral
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_spectral_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_spectral_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_spectral_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_procedural_texture_tests
    SOURCE tests/test_luisa_cycles_svm_procedural_texture.cpp
    TEST_STEM luisa_cycles_svm_procedural_texture
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_procedural_texture_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_procedural_texture_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_procedural_texture_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_rgb_ramp_tests
    SOURCE tests/test_luisa_cycles_svm_rgb_ramp.cpp
    TEST_STEM luisa_cycles_svm_rgb_ramp
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_rgb_ramp_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_rgb_ramp_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_rgb_ramp_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_rgb_curve_tests
    SOURCE tests/test_luisa_cycles_svm_rgb_curve.cpp
    TEST_STEM luisa_cycles_svm_rgb_curve
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_rgb_curve_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_rgb_curve_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_rgb_curve_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_curve_family_tests
    SOURCE tests/test_luisa_cycles_svm_curve_family.cpp
    TEST_STEM luisa_cycles_svm_curve_family
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_curve_family_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_curve_family_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_curve_family_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_camera_tests
    SOURCE tests/test_luisa_cycles_svm_camera.cpp
    TEST_STEM luisa_cycles_svm_camera
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_camera_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_camera_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_camera_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_fresnel_tests
    SOURCE tests/test_luisa_cycles_svm_fresnel.cpp
    TEST_STEM luisa_cycles_svm_fresnel
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_fresnel_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_fresnel_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_fresnel_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_light_path_tests
    SOURCE tests/test_luisa_cycles_svm_light_path.cpp
    TEST_STEM luisa_cycles_svm_light_path
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_light_path_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_light_path_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_light_path_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_info_tests
    SOURCE tests/test_luisa_cycles_svm_info.cpp
    TEST_STEM luisa_cycles_svm_info
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_info_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_info_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_info_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_normal_tests
    SOURCE tests/test_luisa_cycles_svm_normal.cpp
    TEST_STEM luisa_cycles_svm_normal
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_normal_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_normal_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_normal_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_normal_map_tangent_tests
    SOURCE tests/test_luisa_cycles_svm_normal_map_tangent.cpp
    TEST_STEM luisa_cycles_svm_normal_map_tangent
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_normal_map_tangent_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_normal_map_tangent_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_normal_map_tangent_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_closure_pool_tests
    SOURCE tests/test_luisa_cycles_svm_closure_pool.cpp
    TEST_STEM luisa_cycles_svm_closure_pool
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_closure_pool_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
target_compile_definitions(
    psycles_luisa_cycles_svm_closure_pool_tests PRIVATE
    PSYCLES_CLOSURE_POOL_ORACLE_PATH="${PROJECT_SOURCE_DIR}/tests/data/cycles_svm_closure_pool.txt")
if(TEST psycles.luisa_cycles_svm_closure_pool_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_closure_pool_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_closure_tests
    SOURCE tests/test_luisa_cycles_svm_closure.cpp
    TEST_STEM luisa_cycles_svm_closure
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_closure_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_closure_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_closure_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_tests
    SOURCE tests/test_luisa_cycles_svm_principled.cpp
    TEST_STEM luisa_cycles_svm_principled
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_subsurface_tests
    SOURCE tests/test_luisa_cycles_svm_principled_subsurface.cpp
    TEST_STEM luisa_cycles_svm_principled_subsurface
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_subsurface_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_subsurface_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_subsurface_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_standalone_bssrdf_tests
    SOURCE tests/test_luisa_cycles_svm_standalone_bssrdf.cpp
    TEST_STEM luisa_cycles_svm_standalone_bssrdf
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_standalone_bssrdf_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_standalone_bssrdf_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_standalone_bssrdf_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_standalone_sheen_tests
    SOURCE tests/test_luisa_cycles_svm_standalone_sheen.cpp
    TEST_STEM luisa_cycles_svm_standalone_sheen
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_standalone_sheen_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_standalone_sheen_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_standalone_sheen_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_standalone_toon_tests
    SOURCE tests/test_luisa_cycles_svm_standalone_toon.cpp
    TEST_STEM luisa_cycles_svm_standalone_toon
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_standalone_toon_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_standalone_toon_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_standalone_toon_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_standalone_ray_portal_tests
    SOURCE tests/test_luisa_cycles_svm_standalone_ray_portal.cpp
    TEST_STEM luisa_cycles_svm_standalone_ray_portal
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_standalone_ray_portal_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_standalone_ray_portal_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_standalone_ray_portal_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_standalone_hair_tests
    SOURCE tests/test_luisa_cycles_svm_standalone_hair.cpp
    TEST_STEM luisa_cycles_svm_standalone_hair
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_standalone_hair_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_standalone_hair_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_standalone_hair_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_hair_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_hair_scattering.cpp
    TEST_STEM luisa_cycles_svm_hair_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_hair_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_hair_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_hair_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_simple_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_simple_scattering.cpp
    TEST_STEM luisa_cycles_svm_simple_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_simple_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_simple_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_simple_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_toon_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_toon_scattering.cpp
    TEST_STEM luisa_cycles_svm_toon_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_toon_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_toon_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_toon_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_sheen_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_sheen_scattering.cpp
    TEST_STEM luisa_cycles_svm_sheen_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_sheen_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_sheen_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_sheen_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_microfacet_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_microfacet_scattering.cpp
    TEST_STEM luisa_cycles_svm_microfacet_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_microfacet_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_microfacet_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_microfacet_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_microfacet_union_tests
    SOURCE tests/test_luisa_cycles_svm_microfacet_union.cpp
    TEST_STEM luisa_cycles_svm_microfacet_union
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_microfacet_union_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_microfacet_union_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_microfacet_union_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_ashikhmin_shirley_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_ashikhmin_shirley_scattering.cpp
    TEST_STEM luisa_cycles_svm_ashikhmin_shirley_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_ashikhmin_shirley_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_ashikhmin_shirley_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_ashikhmin_shirley_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_bsdf_dispatch_tests
    SOURCE tests/test_luisa_cycles_svm_bsdf_dispatch.cpp
    TEST_STEM luisa_cycles_svm_bsdf_dispatch
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_bsdf_dispatch_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_bsdf_dispatch_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_bsdf_dispatch_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_surface_shader_tests
    SOURCE tests/test_luisa_cycles_svm_surface_shader.cpp
    TEST_STEM luisa_cycles_svm_surface_shader
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_surface_shader_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_surface_shader_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_surface_shader_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_hair_chiang_tests
    SOURCE tests/test_luisa_cycles_svm_principled_hair_chiang.cpp
    TEST_STEM luisa_cycles_svm_principled_hair_chiang
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_hair_chiang_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_hair_chiang_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_hair_chiang_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_hair_chiang_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_principled_hair_chiang_scattering.cpp
    TEST_STEM luisa_cycles_svm_principled_hair_chiang_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_hair_chiang_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_hair_chiang_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_hair_chiang_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_hair_huang_tests
    SOURCE tests/test_luisa_cycles_svm_principled_hair_huang.cpp
    TEST_STEM luisa_cycles_svm_principled_hair_huang
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_hair_huang_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_hair_huang_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_hair_huang_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_hair_huang_scattering_tests
    SOURCE tests/test_luisa_cycles_svm_principled_hair_huang_scattering.cpp
    TEST_STEM luisa_cycles_svm_principled_hair_huang_scattering
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_hair_huang_scattering_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_hair_huang_scattering_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_hair_huang_scattering_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_sheen_tests
    SOURCE tests/test_luisa_cycles_svm_principled_sheen.cpp
    TEST_STEM luisa_cycles_svm_principled_sheen
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_sheen_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_sheen_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_sheen_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_coat_tests
    SOURCE tests/test_luisa_cycles_svm_principled_coat.cpp
    TEST_STEM luisa_cycles_svm_principled_coat
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_coat_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_coat_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_coat_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_metallic_tests
    SOURCE tests/test_luisa_cycles_svm_principled_metallic.cpp
    TEST_STEM luisa_cycles_svm_principled_metallic
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_metallic_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_metallic_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_metallic_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_transmission_tests
    SOURCE tests/test_luisa_cycles_svm_principled_transmission.cpp
    TEST_STEM luisa_cycles_svm_principled_transmission
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_transmission_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_transmission_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_transmission_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_principled_thin_wall_tests
    SOURCE tests/test_luisa_cycles_svm_principled_thin_wall.cpp
    TEST_STEM luisa_cycles_svm_principled_thin_wall
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_cycles_svm_principled_thin_wall_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_principled_thin_wall_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_principled_thin_wall_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_light_falloff_tests
    SOURCE tests/test_luisa_cycles_svm_light_falloff.cpp
    TEST_STEM luisa_cycles_svm_light_falloff
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_light_falloff_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_light_falloff_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_light_falloff_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_svm_ies_tests
    SOURCE tests/test_luisa_cycles_svm_ies.cpp
    TEST_STEM luisa_cycles_svm_ies
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_cycles_svm_ies_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
if(TEST psycles.luisa_cycles_svm_ies_vk)
    set_tests_properties(
        psycles.luisa_cycles_svm_ies_vk
        PROPERTIES ENVIRONMENT
            "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_light_falloff_tests
    SOURCE tests/test_luisa_light_falloff.cpp
    TEST_STEM luisa_light_falloff
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_light_falloff_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_vector_math_svm_tests
    SOURCE tests/test_luisa_surface_vector_math_svm.cpp
    TEST_STEM luisa_surface_vector_math_svm
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_surface_vector_math_svm_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_direct_lighting_plan_tests
    SOURCE tests/test_luisa_direct_lighting_plan.cpp
    TEST_STEM luisa_direct_lighting_plan
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_direct_lighting_plan_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_principled_thin_wall_tests
    SOURCE tests/test_luisa_principled_thin_wall.cpp
    TEST_STEM luisa_principled_thin_wall
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_refraction_tests
    SOURCE tests/test_luisa_refraction.cpp
    TEST_STEM luisa_refraction
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_attribute_tests
    SOURCE tests/test_luisa_attribute.cpp
    TEST_STEM luisa_attribute
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_texture_coordinate_tests
    SOURCE tests/test_luisa_texture_coordinate.cpp
    TEST_STEM luisa_texture_coordinate
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_fmod_tests
    SOURCE tests/test_luisa_fmod.cpp
    TEST_STEM luisa_fmod
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_magic_texture_tests
    SOURCE tests/test_luisa_magic_texture.cpp
    TEST_STEM luisa_magic_texture
    LIBRARIES Psycles::luisa)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_light_tree_tests
    SOURCE tests/test_luisa_light_tree.cpp
    TEST_STEM luisa_light_tree
    LIBRARIES Psycles::luisa_runtime)
