# CMake Setup & Integration Guide

`keisan++` is a header-only C++26 library exporting the `keisan::keisan` `INTERFACE` target (`cmake_minimum_required(VERSION 3.21)`).

When consumed via `FetchContent` or `add_subdirectory`, `KEISAN_BUILD_TESTS` and `KEISAN_BUILD_BENCHMARKS` automatically default to `OFF` (via `${PROJECT_IS_TOP_LEVEL}`), and `KEISAN_BUILD_EXAMPLES` defaults to `OFF`.

---

## Option 1: `FetchContent` (Recommended)

Add this to your project's `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_solver LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(
  keisan
  GIT_REPOSITORY https://github.com/maithreyanprabhu2907-prog/keisanpp.git
  GIT_TAG        master # or pin a specific commit hash
)
FetchContent_MakeAvailable(keisan)

add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE keisan::keisan)
```

`keisan::keisan` automatically sets `cxx_std_26` on your target.

---

## Option 2: `add_subdirectory` (Git Submodule / Vendored)

If you vendor `keisanpp` into `third_party/keisanpp`:

```cmake
add_subdirectory(third_party/keisanpp)

add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE keisan::keisan)
```

---

## Option 3: System / Prefix Install + `find_package`

Install `keisan++` once:

```bash
cmake -B build -DKEISAN_BUILD_TESTS=OFF -DKEISAN_BUILD_BENCHMARKS=OFF
cmake --install build --prefix /usr/local   # or ~/.local
```

Then consume it in any downstream `CMakeLists.txt`:

```cmake
find_package(keisan CONFIG REQUIRED)

add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE keisan::keisan)
```

If installed to a custom prefix (e.g. `~/.local`), pass `-DCMAKE_PREFIX_PATH=$HOME/.local` when configuring your project.

---

## Optional: OpenMP, SIMD Arch Flags, & Feature Macros

### 1. Enabling OpenMP (`assign_par`, `dot_par`, `sum_par`)
`OpenMP::OpenMP_CXX` is **not** propagated on `keisan::keisan`'s `INTERFACE` so single-threaded consumers don't pull in an OpenMP runtime. Link it on your target if you use `*_par`:

```cmake
find_package(OpenMP REQUIRED)
target_link_libraries(my_solver PRIVATE keisan::keisan OpenMP::OpenMP_CXX)
```

### 2. Unlocking AVX2 + FMA `std::simd` Widths (x86_64)
Without `-mavx2 -mfma` (or `/arch:AVX2` on MSVC), compilers default to baseline SSE2 (`128`-bit) `std::simd` widths:

```cmake
if(MSVC)
  target_compile_options(my_solver PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "amd64|x86_64|AMD64")
  target_compile_options(my_solver PRIVATE $<$<CONFIG:Release>:-mavx2 -mfma>)
endif()
```

### 3. Compile-Time Configuration Macros
Pass any of these via `target_compile_definitions(my_solver PRIVATE ...)`:

| Macro | Effect |
|---|---|
| `KEISAN_STRICT_ZERO_MALLOC` | Forbids heap fallback on large aliased temporaries or `assign_par` (throws `std::bad_alloc` / `std::length_error`, or aborts under `KEISAN_NO_EXCEPTIONS`). |
| `KEISAN_NO_EXCEPTIONS` | Replaces `throw` with `assert` + `std::abort()` (safe with `-fno-exceptions`). |
| `KEISAN_NO_INTRINSICS` | Disables `<immintrin.h>` and forces portable scalar loops (no `<simd>` / intrinsics). |
| `KEISAN_NO_CHECKS` | Strips runtime `KEISAN_CHECK_SIZE_EQ` bounds checks in hot paths. |

---

## CMake Options Reference (Top-Level Builds)

| Option | Default | Description |
|---|---|---|
| `KEISAN_BUILD_TESTS` | `${PROJECT_IS_TOP_LEVEL}` | Build CTest suite (`vector_math_test`, `audit_math`, `edge_cases_test`, `no_intrinsics_test`) |
| `KEISAN_BUILD_BENCHMARKS` | `${PROJECT_IS_TOP_LEVEL}` | Build `benchmark`, `regression_bench`, `bench_ofast`, `bench_o3`, `bench_os` |
| `KEISAN_BUILD_EXAMPLES` | `OFF` | Build runnable examples (`01_fused_expressions`..`05_repro_openmp`) |
