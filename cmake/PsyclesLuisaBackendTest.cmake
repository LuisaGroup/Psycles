include_guard(GLOBAL)

# Declare one Luisa device fixture and register it independently for every
# backend enabled in the current build. Keeping backend discovery here makes
# fallback/HIP/Vulkan coverage uniform and prevents copy-pasted test blocks
# from silently drifting apart.
function(psycles_add_luisa_backend_test)
    cmake_parse_arguments(
        PARSE_ARGV 0
        PSYCLES_TEST
        ""
        "TARGET;SOURCE;TEST_STEM"
        "LIBRARIES")

    foreach(_required IN ITEMS TARGET SOURCE TEST_STEM)
        if(NOT PSYCLES_TEST_${_required})
            message(FATAL_ERROR
                "psycles_add_luisa_backend_test requires ${_required}")
        endif()
    endforeach()

    add_executable(
        ${PSYCLES_TEST_TARGET}
        ${PSYCLES_TEST_SOURCE})
    target_link_libraries(
        ${PSYCLES_TEST_TARGET}
        PRIVATE ${PSYCLES_TEST_LIBRARIES})
    target_compile_features(
        ${PSYCLES_TEST_TARGET}
        PRIVATE cxx_std_20)
    set_target_properties(
        ${PSYCLES_TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                "${CMAKE_BINARY_DIR}/bin")

    foreach(_backend IN ITEMS fallback hip vk)
        set(_backend_target
            "luisa-compute-backend-${_backend}")
        if(TARGET ${_backend_target})
            add_dependencies(
                ${PSYCLES_TEST_TARGET}
                ${_backend_target})
            add_test(
                NAME
                    "psycles.${PSYCLES_TEST_TEST_STEM}_${_backend}"
                COMMAND
                    ${PSYCLES_TEST_TARGET}
                    ${_backend})
        endif()
    endforeach()
endfunction()
