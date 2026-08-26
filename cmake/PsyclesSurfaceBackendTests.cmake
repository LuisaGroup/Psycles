include_guard(GLOBAL)

# Surface and closure fixtures share the same per-backend registration policy.
# Keep the inventory out of the project root so adding one semantic regression
# does not make the top-level build description grow without bound.
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
    TARGET psycles_luisa_beckmann_glossy_tests
    SOURCE tests/test_luisa_beckmann_glossy.cpp
    TEST_STEM luisa_beckmann_glossy
    LIBRARIES Psycles::luisa)

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

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_compact_surface_preparation_tests
    SOURCE tests/test_luisa_compact_surface_preparation.cpp
    TEST_STEM luisa_compact_surface_preparation
    LIBRARIES Psycles::luisa_runtime)
target_sources(
    psycles_luisa_compact_surface_preparation_tests
    PRIVATE tests/compact_surface_program_test_support.cpp)
target_include_directories(
    psycles_luisa_compact_surface_preparation_tests
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
if(TEST psycles.luisa_compact_surface_preparation_vk)
    set_tests_properties(
        psycles.luisa_compact_surface_preparation_vk
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

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_closure_physical_tests
    SOURCE tests/test_luisa_surface_closure_physical.cpp
    TEST_STEM luisa_surface_closure_physical
    LIBRARIES Psycles::luisa_runtime)

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

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_surface_math_svm_tests
    SOURCE tests/test_luisa_surface_math_svm.cpp
    TEST_STEM luisa_surface_math_svm
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(
    psycles_luisa_surface_math_svm_tests
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
