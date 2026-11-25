function(_psycles_add_vt_test target source stem)
    psycles_add_luisa_backend_test(
        TARGET ${target}
        SOURCE ${source}
        TEST_STEM ${stem}
        LIBRARIES Psycles::luisa_runtime)
    if(ARGN)
        target_sources(${target} PRIVATE ${ARGN})
    endif()
    target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/luisa")
    if(PSYCLES_ENABLE_OPENIMAGEIO)
        target_link_libraries(${target} PRIVATE OpenImageIO::OpenImageIO)
    endif()
    foreach(_backend IN ITEMS fallback simd metal metal4 hip vk dx cuda)
        if(TEST "psycles.${stem}_${_backend}")
            set_tests_properties("psycles.${stem}_${_backend}" PROPERTIES TIMEOUT 300)
        endif()
    endforeach()
endfunction()

_psycles_add_vt_test(
    psycles_luisa_virtual_texture_cache_tests
    tests/test_luisa_virtual_texture_cache.cpp
    luisa_virtual_texture_cache)
_psycles_add_vt_test(
    psycles_luisa_vt_async_overlap_tests
    tests/test_luisa_vt_async_overlap.cpp
    luisa_vt_async_overlap)
_psycles_add_vt_test(
    psycles_luisa_vt_multislot_tests
    tests/test_luisa_vt_multislot.cpp
    luisa_vt_multislot)
_psycles_add_vt_test(
    psycles_luisa_vt_bc_tests
    tests/test_luisa_vt_bc.cpp
    luisa_vt_bc
    tests/test_luisa_vt_bc_gpu.cpp)
_psycles_add_vt_test(
    psycles_luisa_vt_preflight_tests
    tests/test_luisa_vt_preflight.cpp
    luisa_vt_preflight)
_psycles_add_vt_test(
    psycles_luisa_vt_mipmap_tests
    tests/test_luisa_vt_mipmap.cpp
    luisa_vt_mipmap)
if(TEST "psycles.luisa_vt_mipmap_simd")
    set_tests_properties("psycles.luisa_vt_mipmap_simd" PROPERTIES TIMEOUT 900)
endif()
