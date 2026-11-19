# Integration & Build Guide

This guide explains how to integrate `vec_math` into downstream C++ projects and build the library's test and benchmark suites.

---

## 1. Integration Methods

### Method 1: CMake `FetchContent` (Recommended)
Add the following to your project's `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
    vec_math
    GIT_REPOSITORY https://github.com/<your-org>/vec_math.git
    GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(vec_math)

# Link your target to vec_math
target_link_libraries(your_target PRIVATE vec_math::vec_math)
```

### Method 2: System Installation & `find_package`
Build and install `vec_math` globally or into a custom prefix:

```bash
# Configure and install
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local
sudo cmake --install build
```

In your application's `CMakeLists.txt`:
```cmake
find_package(vec_math 1.0 REQUIRED)

target_link_libraries(your_target PRIVATE vec_math::vec_math)
```

### Method 3: Header-Only Drop-In
`vec_math` is completely header-only. Copy the `include/vec_math/` directory directly into your project's include path and compile with C++20 support (`-std=c++20` or `/std:c++20`).

---

## 2. Recommended Compiler Flags

To maximize auto-vectorization and memory bandwidth throughput:

### MSVC (Visual Studio)
```cmake
if(MSVC)
    target_compile_options(your_target PRIVATE
        /std:c++20
        /O2          # Maximum speed optimization
        /arch:AVX2   # Enable AVX2 and FMA SIMD instructions
        /fp:fast     # (Optional) Aggressive floating-point reassociation
        /openmp      # (Optional) Enable multi-core parallel helpers (assign_par, dot_par, sum_par)
    )
endif()
```

### Clang / GCC
```cmake
if(NOT MSVC)
    target_compile_options(your_target PRIVATE
        -std=c++20
        -O3             # Aggressive loop optimization & vectorization
        -mavx2          # Enable AVX2 instruction set
        -mfma           # Enable hardware FMA instructions
        -march=native   # (Optional) Optimize for host CPU architecture
        -fopenmp        # (Optional) Enable multi-core parallel helpers (assign_par, dot_par, sum_par)
    )
endif()
```

---

## 3. Configuration Flags

Pass these compile definitions to customize `vec_math` behavior:

### `VEC_MATH_NO_EXCEPTIONS`
Disables C++ exception throwing on dimension mismatch or capacity exhaustion, falling back to `assert` and `std::abort()`. Useful for embedded systems or game engines built with `-fno-exceptions`.

```cmake
target_compile_definitions(your_target PRIVATE VEC_MATH_NO_EXCEPTIONS)
```

---

## 4. Building the Test & Benchmark Suites

### Prerequisites
- CMake 3.20 or newer
- C++20 compatible compiler (MSVC 2022+, GCC 11+, or Clang 13+)

### Build & Run Tests
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build test executable
cmake --build build --config Release --target vector_math_test

# Run tests with CTest
ctest --test-dir build -C Release --output-on-failure
```

### Build & Run Benchmarks
```bash
# Build benchmark target
cmake --build build --config Release --target benchmark

# Run benchmark
./build/Release/benchmark
```
*Note: Running `benchmark` will print live throughput/latency tables to stdout and automatically refresh the graphs in `README.md`.*
