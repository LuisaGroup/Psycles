# Recording-only capability witness; no backend or optimizer is involved.
add_executable(psycles_cycles_shading_terminator_recording_tests
    tests/test_cycles_shading_terminator_recording.cpp)
target_link_libraries(psycles_cycles_shading_terminator_recording_tests
    PRIVATE Psycles::luisa_runtime)
target_include_directories(psycles_cycles_shading_terminator_recording_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
add_test(NAME psycles.cycles_shading_terminator_recording
    COMMAND psycles_cycles_shading_terminator_recording_tests)

psycles_add_luisa_backend_test(
    TARGET psycles_luisa_cycles_shading_terminator_tests
    SOURCE tests/test_luisa_cycles_shading_terminator.cpp
    TEST_STEM luisa_cycles_shading_terminator
    LIBRARIES Psycles::luisa_runtime)
target_include_directories(psycles_luisa_cycles_shading_terminator_tests
    PRIVATE ${PROJECT_SOURCE_DIR}/src/luisa)
target_compile_definitions(psycles_luisa_cycles_shading_terminator_tests PRIVATE
    PSYCLES_SHADING_TERMINATOR_ORACLE="${PROJECT_SOURCE_DIR}/tests/data/cycles_shading_terminator.txt")
if(TEST psycles.luisa_cycles_shading_terminator_vk)
    set_tests_properties(psycles.luisa_cycles_shading_terminator_vk PROPERTIES ENVIRONMENT
        "LUISA_VULKAN_USE_XIR=1;LUISA_VULKAN_REQUIRE_NATIVE_XIR_SPIRV=1;LUISA_VULKAN_DISABLE_DXC=1")
endif()
