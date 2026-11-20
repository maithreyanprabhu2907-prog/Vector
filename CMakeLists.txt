cmake_minimum_required(VERSION 3.20)



project(vec_math

        VERSION 1.0.0

        DESCRIPTION "Zero-allocation, battle-tested header-only vector math for std::vector"

        LANGUAGES CXX

)


include(GNUInstallDirs)

include(CMakePackageConfigHelpers)



# Header-only INTERFACE target

add_library(vec_math INTERFACE)

add_library(vec_math::vec_math ALIAS vec_math)



target_compile_features(vec_math INTERFACE cxx_std_20)

target_include_directories(vec_math

        INTERFACE

        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>

        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>

)



# Only build tests when vec_math is the top-level project (avoids polluting

# downstream projects that consume vec_math via FetchContent/add_subdirectory).

option(VEC_MATH_BUILD_TESTS "Build vec_math unit and zero-malloc tests" ${PROJECT_IS_TOP_LEVEL})
option(VEC_MATH_BUILD_BENCHMARKS "Build vec_math performance benchmarks" ${PROJECT_IS_TOP_LEVEL})

if(VEC_MATH_BUILD_TESTS)
    enable_testing()
    add_executable(vector_math_test tests/vector_math_test.cpp)
    target_link_libraries(vector_math_test PRIVATE vec_math::vec_math)
    if(MSVC)
        target_compile_options(vector_math_test PRIVATE /W4 /WX)
    else()
        target_compile_options(vector_math_test PRIVATE -Wall -Wextra -Wpedantic -Werror)
    endif()
    add_test(NAME vector_math_test COMMAND vector_math_test)
endif()

if(VEC_MATH_BUILD_BENCHMARKS)
    add_executable(benchmark benchmarks/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE vec_math::vec_math)
    if(MSVC)
        target_compile_options(benchmark PRIVATE
            $<$<CONFIG:Release>:/O2>
            $<$<CONFIG:Release>:/fp:fast>
            $<$<CONFIG:Release>:/openmp>
            /arch:AVX2
            /W4
        )
    else()
        target_compile_options(benchmark PRIVATE
            $<$<CONFIG:Release>:-Ofast>
            -mavx2
            -fopenmp
            -Wall
            -Wextra
        )
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
        target_compile_options(bench_ofast PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:fast> /arch:AVX2 /W4)
        target_compile_options(bench_o3 PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:Release>:/fp:precise> /arch:AVX2 /W4)
        target_compile_options(bench_os PRIVATE $<$<CONFIG:Release>:/O1> $<$<CONFIG:Release>:/Os> $<$<CONFIG:Release>:/fp:precise> /arch:AVX2 /W4)
    else()
        target_compile_options(bench_ofast PRIVATE $<$<CONFIG:Release>:-Ofast> -mavx2 -Wall -Wextra)
        target_compile_options(bench_o3 PRIVATE $<$<CONFIG:Release>:-O3> -mavx2 -Wall -Wextra)
        target_compile_options(bench_os PRIVATE $<$<CONFIG:Release>:-Os> -mavx2 -Wall -Wextra)
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



file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/vec_mathConfig.cmake.in"

        "@PACKAGE_INIT@\ninclude(\"\${CMAKE_CURRENT_LIST_DIR}/vec_mathTargets.cmake\")\ncheck_required_components(vec_math)\n"

)



configure_package_config_file(

        "${CMAKE_CURRENT_BINARY_DIR}/vec_mathConfig.cmake.in"

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

