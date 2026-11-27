cmake_minimum_required(VERSION 3.21)



project(vec_math

        VERSION 1.0.0

        DESCRIPTION "Zero-allocation, battle-tested header-only vector math for std::vector"

        LANGUAGES CXX

)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(GNUInstallDirs)

include(CMakePackageConfigHelpers)



# Header-only INTERFACE target

add_library(vec_math INTERFACE)

add_library(vec_math::vec_math ALIAS vec_math)



target_compile_features(vec_math INTERFACE cxx_std_26)

target_include_directories(vec_math

        INTERFACE

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

)



# Only build tests when vec_math is the top-level project (avoids polluting

# downstream projects that consume vec_math via FetchContent/add_subdirectory).

option(VEC_MATH_BUILD_TESTS "Build vec_math unit and zero-malloc tests" ${PROJECT_IS_TOP_LEVEL})
option(VEC_MATH_BUILD_BENCHMARKS "Build vec_math performance benchmarks" ${PROJECT_IS_TOP_LEVEL})

function(vec_math_apply_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX /wd4702 /EHsc)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
            target_compile_options(${target} PRIVATE -Wno-mismatched-new-delete)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(${target} PRIVATE -Wno-pass-failed)
        endif()
    endif()
endfunction()

if(VEC_MATH_BUILD_TESTS)
    enable_testing()
    find_package(OpenMP)

    add_executable(vector_math_test tests/vector_math_test.cpp)
    target_link_libraries(vector_math_test PRIVATE vec_math::vec_math)
    vec_math_apply_warnings(vector_math_test)
    add_test(NAME vector_math_test COMMAND vector_math_test)

    add_executable(audit_math tests/audit_math.cpp)
    target_link_libraries(audit_math PRIVATE vec_math::vec_math)
    vec_math_apply_warnings(audit_math)
    add_test(NAME audit_math COMMAND audit_math)

    add_executable(edge_cases_test tests/edge_cases_test.cpp)
    target_link_libraries(edge_cases_test PRIVATE vec_math::vec_math)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(edge_cases_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    vec_math_apply_warnings(edge_cases_test)
    add_test(NAME edge_cases_test COMMAND edge_cases_test)

    add_executable(no_intrinsics_test tests/edge_cases_test.cpp)
    target_link_libraries(no_intrinsics_test PRIVATE vec_math::vec_math)
    target_compile_definitions(no_intrinsics_test PRIVATE VEC_MATH_NO_INTRINSICS)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(no_intrinsics_test PRIVATE OpenMP::OpenMP_CXX)
    endif()
    vec_math_apply_warnings(no_intrinsics_test)
    add_test(NAME no_intrinsics_test COMMAND no_intrinsics_test)
endif()

if(VEC_MATH_BUILD_BENCHMARKS)
    find_package(OpenMP)
    set(VEC_MATH_ARCH_FLAGS "")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64|AMD64")
        if(MSVC)
            set(VEC_MATH_ARCH_FLAGS "/arch:AVX2")
        else()
            set(VEC_MATH_ARCH_FLAGS "-mavx2" "-mfma")
        endif()
    endif()

    add_executable(benchmark benchmarks/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE vec_math::vec_math)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        target_compile_options(benchmark PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:fast>
            ${VEC_MATH_ARCH_FLAGS}
            /W4
        )
    else()
        target_compile_options(benchmark PRIVATE
            $<$<CONFIG:Release>:-Ofast>
            ${VEC_MATH_ARCH_FLAGS}
            -Wall
            -Wextra
        )
    endif()

    add_executable(regression_bench benchmarks/regression_bench.cpp)
    target_link_libraries(regression_bench PRIVATE vec_math::vec_math)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(regression_bench PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(MSVC)
        target_compile_options(regression_bench PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:fast>
            ${VEC_MATH_ARCH_FLAGS}
            /W4 /WX
        )
    else()
        target_compile_options(regression_bench PRIVATE
            $<$<CONFIG:Release>:-O3>
            ${VEC_MATH_ARCH_FLAGS}
            -Wall -Wextra -Wpedantic -Werror
        )
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            target_compile_options(regression_bench PRIVATE -Wno-pass-failed)
        endif()
    endif()

    # Optimization Comparison Benchmarks (-Ofast vs -O3 vs -Os)
    add_executable(bench_ofast benchmarks/opt_compare.cpp)
    target_link_libraries(bench_ofast PRIVATE vec_math::vec_math)
    target_compile_definitions(bench_ofast PRIVATE OPT_OFAST)

    add_executable(bench_o3 benchmarks/opt_compare.cpp)
    target_link_libraries(bench_o3 PRIVATE vec_math::vec_math)
    target_compile_definitions(bench_o3 PRIVATE OPT_O3)

    add_executable(bench_os benchmarks/opt_compare.cpp)
    target_link_libraries(bench_os PRIVATE vec_math::vec_math)
    target_compile_definitions(bench_os PRIVATE OPT_OS)

    if(MSVC)
        target_compile_options(bench_ofast PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:fast> ${VEC_MATH_ARCH_FLAGS} /W4)
        target_compile_options(bench_o3 PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:precise> ${VEC_MATH_ARCH_FLAGS} /W4)
        target_compile_options(bench_os PRIVATE $<$<CONFIG:Release>:/O1> $<$<CONFIG:Release>:/Os> $<$<CONFIG:Release>:/fp:precise> ${VEC_MATH_ARCH_FLAGS} /W4)
    else()
        target_compile_options(bench_ofast PRIVATE $<$<CONFIG:Release>:-Ofast> ${VEC_MATH_ARCH_FLAGS} -Wall -Wextra)
        target_compile_options(bench_o3 PRIVATE $<$<CONFIG:Release>:-O3> ${VEC_MATH_ARCH_FLAGS} -Wall -Wextra)
        target_compile_options(bench_os PRIVATE $<$<CONFIG:Release>:-Os> ${VEC_MATH_ARCH_FLAGS} -Wall -Wextra)
    endif()
endif()



# Installation & find_package(vec_math) export rules

install(

        TARGETS vec_math

        EXPORT vec_mathTargets

)



install(

        DIRECTORY include/

        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}

)



configure_package_config_file(

        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/vec_mathConfig.cmake.in"

        "${CMAKE_CURRENT_BINARY_DIR}/vec_mathConfig.cmake"

        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vec_math"

)



write_basic_package_version_file(

        "${CMAKE_CURRENT_BINARY_DIR}/vec_mathConfigVersion.cmake"

        VERSION ${PROJECT_VERSION}

        COMPATIBILITY SameMajorVersion

)



install(

        EXPORT vec_mathTargets

        FILE vec_mathTargets.cmake

        NAMESPACE vec_math::

        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vec_math"

)



install(

        FILES

        "${CMAKE_CURRENT_BINARY_DIR}/vec_mathConfig.cmake"

        "${CMAKE_CURRENT_BINARY_DIR}/vec_mathConfigVersion.cmake"

        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/vec_math"

)

