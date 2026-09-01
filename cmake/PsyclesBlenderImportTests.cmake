add_executable(
    psycles_blender_import_tests
    tests/blender_texture_coordinate_import_expectations.cpp
    tests/test_blender_import.cpp
    tests/test_blender_legacy_mix_import.cpp
    tests/test_blender_vector_import.cpp)
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
    psycles_blender_background_import_tests
    tests/test_blender_background_import.cpp)
target_link_libraries(
    psycles_blender_background_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_background_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_background_import
    COMMAND psycles_blender_background_import_tests)

add_executable(
    psycles_blender_wireframe_import_tests
    tests/test_blender_wireframe_import.cpp)
target_link_libraries(
    psycles_blender_wireframe_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_wireframe_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_wireframe_import
    COMMAND psycles_blender_wireframe_import_tests)

add_executable(
    psycles_blender_attribute_import_tests
    tests/test_blender_attribute_import.cpp)
target_link_libraries(
    psycles_blender_attribute_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_attribute_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_attribute_import
    COMMAND psycles_blender_attribute_import_tests)

add_executable(
    psycles_blender_displacement_import_tests
    tests/test_blender_displacement_import.cpp)
target_link_libraries(
    psycles_blender_displacement_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_displacement_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_displacement_import
    COMMAND psycles_blender_displacement_import_tests)

add_executable(
    psycles_blender_light_falloff_import_tests
    tests/test_blender_light_falloff_import.cpp)
target_link_libraries(
    psycles_blender_light_falloff_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_light_falloff_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_light_falloff_import
    COMMAND psycles_blender_light_falloff_import_tests)

add_executable(
    psycles_blender_ambient_occlusion_import_tests
    tests/test_blender_ambient_occlusion_import.cpp)
target_link_libraries(
    psycles_blender_ambient_occlusion_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_ambient_occlusion_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_ambient_occlusion_import
    COMMAND psycles_blender_ambient_occlusion_import_tests)

add_executable(
    psycles_blender_gradient_import_tests
    tests/test_blender_gradient_import.cpp)
target_link_libraries(
    psycles_blender_gradient_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_gradient_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_gradient_import
    COMMAND psycles_blender_gradient_import_tests)

add_executable(
    psycles_blender_rgb_ramp_import_tests
    tests/test_blender_rgb_ramp_import.cpp)
target_link_libraries(
    psycles_blender_rgb_ramp_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_rgb_ramp_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_rgb_ramp_import
    COMMAND psycles_blender_rgb_ramp_import_tests)

add_executable(
    psycles_blender_rgb_curve_import_tests
    tests/test_blender_rgb_curve_import.cpp)
target_link_libraries(
    psycles_blender_rgb_curve_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_rgb_curve_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_rgb_curve_import
    COMMAND psycles_blender_rgb_curve_import_tests)

add_executable(
    psycles_blender_curve_family_import_tests
    tests/test_blender_curve_family_import.cpp)
target_link_libraries(
    psycles_blender_curve_family_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_curve_family_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_curve_family_import
    COMMAND psycles_blender_curve_family_import_tests)

add_executable(
    psycles_blender_camera_data_import_tests
    tests/test_blender_camera_data_import.cpp)
target_link_libraries(
    psycles_blender_camera_data_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_camera_data_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_camera_data_import
    COMMAND psycles_blender_camera_data_import_tests)

add_executable(
    psycles_blender_fresnel_import_tests
    tests/test_blender_fresnel_import.cpp)
target_link_libraries(
    psycles_blender_fresnel_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_fresnel_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_fresnel_import
    COMMAND psycles_blender_fresnel_import_tests)

add_executable(
    psycles_blender_light_path_import_tests
    tests/test_blender_light_path_import.cpp)
target_link_libraries(
    psycles_blender_light_path_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_light_path_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_light_path_import
    COMMAND psycles_blender_light_path_import_tests)

add_executable(
    psycles_blender_hair_import_tests
    tests/test_blender_hair_import.cpp)
target_link_libraries(
    psycles_blender_hair_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_hair_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_hair_import
    COMMAND psycles_blender_hair_import_tests)

add_executable(
    psycles_blender_muted_node_tests
    tests/test_blender_muted_node.cpp)
target_link_libraries(
    psycles_blender_muted_node_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_muted_node_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_muted_node
    COMMAND psycles_blender_muted_node_tests)

add_executable(
    psycles_blender_curve_import_tests
    tests/test_blender_curve_import.cpp)
target_link_libraries(
    psycles_blender_curve_import_tests
    PRIVATE Psycles::luisa_runtime)
target_compile_features(
    psycles_blender_curve_import_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.blender_curve_import
    COMMAND psycles_blender_curve_import_tests)
