include_guard(GLOBAL)

include(CMakeParseArguments)

function(psycles_add_blender_test)
    cmake_parse_arguments(
        PSYCLES_BLENDER_TEST
        "NO_EXIT_CODE"
        "NAME;SCRIPT"
        "ARGUMENTS"
        ${ARGN})
    if(NOT PSYCLES_BLENDER_TEST_NAME)
        message(FATAL_ERROR "psycles_add_blender_test requires NAME")
    endif()
    if(NOT PSYCLES_BLENDER_TEST_SCRIPT)
        message(FATAL_ERROR "psycles_add_blender_test requires SCRIPT")
    endif()
    set(command
        "${PSYCLES_BLENDER_EXECUTABLE}"
        --background
        --factory-startup)
    if(NOT PSYCLES_BLENDER_TEST_NO_EXIT_CODE)
        list(APPEND command --python-exit-code 1)
    endif()
    list(APPEND command
        --python
        "${PSYCLES_BLENDER_TEST_SCRIPT}")
    if(PSYCLES_BLENDER_TEST_ARGUMENTS)
        list(APPEND command -- ${PSYCLES_BLENDER_TEST_ARGUMENTS})
    endif()
    add_test(
        NAME "${PSYCLES_BLENDER_TEST_NAME}"
        COMMAND ${command})
    set_tests_properties(
        "${PSYCLES_BLENDER_TEST_NAME}"
        PROPERTIES
            ENVIRONMENT
                "SDL_AUDIODRIVER=dummy;ALSOFT_DRIVERS=null")
endfunction()
