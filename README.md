# vec_math

Zero-allocation, header-only C++20 vector math operators (`+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`, `<<=`) for `std::vector<T, Alloc>`.

## Repository Layout

Place the files into this structure:

```text
vec_math/
├── CMakeLists.txt
├── README.md
├── benchmarks/
│   └── benchmark.cpp
├── cmake/
│   └── vec_mathConfig.cmake.in
├── docs/
│   ├── README.md
│   ├── api_reference.md
│   ├── architecture.md
│   ├── benchmarks.md
│   ├── integration.md
│   └── sycl_guide.md
├── include/
│   └── vec_math/
│       └── vector_math.hpp
└── tests/
    └── vector_math_test.cpp
```

## Documentation

Comprehensive guides and references are available in the [`docs/`](docs/README.md) folder:
* **[Architecture & Design Principles](docs/architecture.md)**: Zero-allocation expression templates, AST lazy evaluation, SIMD unrolling, and memory optimization.
* **[API Reference](docs/api_reference.md)**: Exhaustive catalog of concepts, operators, expressions, views, and math utilities.
* **[SYCL Compatibility Guide](docs/sycl_guide.md)**: Zero-header SYCL compiler compatibility, TriviallyCopyable expressions, and device execution.
* **[Performance & Benchmarking](docs/benchmarks.md)**: In-depth metrics, GFLOPS, memory bandwidth, and scalability analysis.
* **[Integration & Build Guide](docs/integration.md)**: CMake FetchContent, system installation, compiler tuning, and test instructions.

## Integration

### 1. CMake `FetchContent`
```cmake
include(FetchContent)
FetchContent_Declare(
    vec_math
    GIT_REPOSITORY https://github.com/<you>/vec_math.git
    GIT_TAG v1.0.0
)
FetchContent_MakeAvailable(vec_math)

target_link_libraries(your_target PRIVATE vec_math::vec_math)
```

### 2. System Install + `find_package`
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
sudo cmake --install build
```
Then in any project:
```cmake
find_package(vec_math 1.0 REQUIRED)
target_link_libraries(your_target PRIVATE vec_math::vec_math)
```

### 3. Single-Header Drop-In
Copy `vector_math.hpp` into your project's `include/` folder and compile with `-std=c++20` (or `/std:c++20`).

## Usage

```cpp
#include <vec_math/vector_math.hpp>
#include <vector>

int main() {
    using namespace vec_math;

    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    std::vector<double> out(3);

    // Zero-malloc fused expression evaluation into pre-allocated buffer
    out <<= 2.0 * a + b / 2.0 - 1.0;

    // Zero-malloc in-place compound ops
    out += a;
    out *= 0.5;

    // Zero-malloc rvalue buffer recycling
    auto res = 3.0 * std::move(a) + b;
}
```

## Compile-Time Options

* **`VEC_MATH_NO_EXCEPTIONS`**: Define before including `vector_math.hpp` when building with `-fno-exceptions` to use `assert` + `std::abort()` on dimension/capacity errors instead of throwing.


<!-- BEGIN_BENCHMARK_SECTION -->

## Comprehensive Performance Benchmarks (-Ofast Optimization)

Benchmark configuration:
- **Compiler Optimization**: `-Ofast` mode (MSVC `/O2 /fp:fast /arch:AVX2` | GCC/Clang `-Ofast -mavx2`)
- **Workload**: Multi-vector linear combination (`out = 2.5 * a + 1.5 * b - 0.5 * c`)
- **Arithmetic Precision**: `double` (64-bit IEEE-754 floating point with fast math enabled)
- **Memory Traffic**: 32 bytes/element (3 reads + 1 write)
- **FLOP Intensity**: 5 FLOPs/element (3 multiplications, 1 addition, 1 subtraction)

### Throughput & Speedup Visualizations

```mermaid
xychart-beta
    title "Compute Throughput (GFLOPS) across Vector Sizes"
    x-axis ["1K", "10K", "100K", "1M", "5M"]
    y-axis "GFLOPS (Higher is better)" 0 --> 60
    line [2.17, 0.34, 0.50, 0.49, 0.51]
    line [52.03, 21.06, 16.19, 5.86, 4.39]
    line [43.71, 21.08, 16.82, 5.90, 4.47]
```

> **Chart Legend (Throughput)**: Line 1 = Naive `std::vector` (temporary heap mallocs) | Line 2 = Hand-written Scalar Loop | Line 3 = `vec_math` Expression Templates (zero mallocs)

```mermaid
xychart-beta
    title "Speedup Factor Relative to Naive std::vector (x-fold)"
    x-axis ["1K", "10K", "100K", "1M", "5M"]
    y-axis "Speedup Factor (x)" 0 --> 72
    line [24.03, 61.90, 32.63, 11.95, 8.57]
    line [20.19, 61.98, 33.90, 12.03, 8.74]
```

> **Chart Legend (Speedup vs Naive)**: Line 1 = Hand-written Scalar Loop | Line 2 = `vec_math` Expression Templates (zero mallocs)

### Benchmark Results Across Vector Scales

#### Vector Size: 1K (1000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 2.31 µs | **1.00x** | 5 allocs (39.3 KB) | 2.17 GFLOPS | 13.86 GB/s | `█░░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 0.10 µs | **24.03x** | `0 allocs (0 B)` | 52.03 GFLOPS | 333.02 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 0.11 µs | **20.19x** | `0 allocs (0 B)` | 43.71 GFLOPS | 279.74 GB/s | `█████████████████░░░` |

#### Vector Size: 10K (10000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 146.99 µs | **1.00x** | 5 allocs (390.8 KB) | 0.34 GFLOPS | 2.18 GB/s | `█░░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 2.37 µs | **61.90x** | `0 allocs (0 B)` | 21.06 GFLOPS | 134.77 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 2.37 µs | **61.98x** | `0 allocs (0 B)` | 21.08 GFLOPS | 134.92 GB/s | `████████████████████` |

#### Vector Size: 100K (100000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 1.01 ms | **1.00x** | 5 allocs (3.8 MB) | 0.50 GFLOPS | 3.18 GB/s | `█░░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 30.89 µs | **32.63x** | `0 allocs (0 B)` | 16.19 GFLOPS | 103.59 GB/s | `███████████████████░` |
| **vec_math (Expr Template)** | 29.73 µs | **33.90x** | `0 allocs (0 B)` | 16.82 GFLOPS | 107.64 GB/s | `████████████████████` |

#### Vector Size: 1M (1000000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 10.19 ms | **1.00x** | 5 allocs (38.1 MB) | 0.49 GFLOPS | 3.14 GB/s | `██░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 852.73 µs | **11.95x** | `0 allocs (0 B)` | 5.86 GFLOPS | 37.53 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 847.18 µs | **12.03x** | `0 allocs (0 B)` | 5.90 GFLOPS | 37.77 GB/s | `████████████████████` |

#### Vector Size: 5M (5000000 elements)

| Implementation | Latency / op | Speedup | Heap Allocs | Throughput | Bandwidth | Visual Speedup Bar |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive std::vector** | 48.85 ms | **1.00x** | 5 allocs (190.7 MB) | 0.51 GFLOPS | 3.28 GB/s | `██░░░░░░░░░░░░░░░░░░` |
| **Scalar Loop** | 5.70 ms | **8.57x** | `0 allocs (0 B)` | 4.39 GFLOPS | 28.08 GB/s | `████████████████████` |
| **vec_math (Expr Template)** | 5.59 ms | **8.74x** | `0 allocs (0 B)` | 4.47 GFLOPS | 28.63 GB/s | `████████████████████` |

### Reductions & Extended Math Utilities (1M Elements)

| Operation | Baseline | `vec_math` | Baseline Time | `vec_math` Time | Speedup | Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dot Product (dot)** | std::inner_product | **vec_math::dot (8x unrolled)** | 14.43 ms | **11.45 ms** | **1.26x** | 8.73 GFLOPS |
| **Sum Reduction (sum)** | std::accumulate | **vec_math::sum (8x tree)** | 5.80 ms | **4.55 ms** | **1.28x** | 11.00 GFLOPS |
| **Fused Multiply-Add (fma)** | Scalar a*b + c | **vec_math::fma (Ternary)** | 30.24 ms | **36.42 ms** | **0.83x** | 2.75 GFLOPS |

### `-Ofast` Optimization Analysis

Building with the `-Ofast` flag (or MSVC `/fp:fast /O2 /arch:AVX2`) yields critical microarchitectural enhancements:

1. **Hardware FMA Fusing**: Allows the compiler to fuse multiplication and addition operations into single-cycle hardware `vfmadd213pd` / `vfmadd231pd` instructions, doubling theoretical peak floating-point throughput.
2. **Tree Reassociation**: Enables out-of-order algebraic reassociation across `vec_math`'s 8-way unrolled reduction trees (`sum`, `dot`), breaking serial accumulator dependency chains and saturating CPU execution ports.
3. **Elimination of IEEE-754 Serialization Barriers**: Bypasses costly subnormal handling and strict serialization fences, allowing the fused vector loops to sustain memory bandwidth near theoretical limits.

## SYCL Compatibility

`vec_math` is engineered for full compatibility with modern heterogeneous C++ frameworks, including SYCL compilers (such as Intel oneAPI DPC++ and AdaptiveCpp):

- **Zero SYCL Headers Included**: The library contains strictly zero `#include <sycl/sycl.hpp>` or proprietary vendor header dependencies. Host code compiles cleanly under standard C++20 with no foreign runtime or SDK overhead.
- **TriviallyCopyable Expression Templates**: All expression nodes (`VecRef`, `ScalarVal`, `BinExpr`, `UnaryExpr`, `TernaryExpr`) are standard-layout and satisfy `std::is_trivially_copyable_v<T>`. This allows expressions to be captured by value in SYCL device kernel lambdas (`cgh.parallel_for`) without violating device capture rules.
- **Device-Callable Core Annotations**: All core evaluation operators and expression evaluators are decorated with `VEC_MATH_DEVICE`, `VEC_MATH_HOST_DEVICE`, and `SYCL_EXTERNAL` macros when compiled under SYCL device compilation modes (`__SYCL_DEVICE_ONLY__` or `SYCL_LANGUAGE_VERSION`), enabling direct device execution.


<!-- END_BENCHMARK_SECTION -->
