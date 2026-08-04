include_guard(GLOBAL)

psycles_add_luisa_backend_test(
    TARGET
        psycles_luisa_curve_ribbon_tests
    SOURCE
        tests/test_luisa_curve_ribbon.cpp
    TEST_STEM
        luisa_curve_ribbon
    LIBRARIES Psycles::luisa)
target_include_directories(
    psycles_luisa_curve_ribbon_tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")

add_executable(
    psycles_curve_geometry_upload_tests
    tests/test_curve_geometry_upload.cpp)
target_link_libraries(
    psycles_curve_geometry_upload_tests
    PRIVATE Psycles::luisa_runtime)
target_include_directories(
    psycles_curve_geometry_upload_tests
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
target_compile_features(
    psycles_curve_geometry_upload_tests
    PRIVATE cxx_std_20)
add_test(
    NAME psycles.curve_geometry_upload
    COMMAND psycles_curve_geometry_upload_tests)
