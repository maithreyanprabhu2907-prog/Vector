# vec_math Documentation

Welcome to the official documentation for **vec_math**, a high-performance, zero-allocation, header-only C++20 vector mathematics library engineered for `std::vector` and heterogeneous compute environments (such as SYCL).

---

## Table of Contents

1. [Architecture & Design Principles](architecture.md)
   - Zero-allocation expression template engine
   - AST fusion and lazy evaluation
   - In-place compound operations and rvalue buffer recycling
   - SIMD auto-vectorization and loop unrolling (8x/4x)
   - Cache locality and memory traffic minimization

2. [API Reference](api_reference.md)
   - Core concepts (`Scalar`, `NumericVector`, `DeviceSpan`, `VecExpression`, `VectorOperand`, `sycl_compatible`)
   - `vec_span<T>` and `device_view<T>` wrappers
   - Overloaded arithmetic operators (`+`, `-`, `*`, `/`)
   - Compound assignment operators (`+=`, `-=`, `*=`, `/=`)
   - Zero-allocation evaluation assignment (`<<=`) and `assign()`
   - Unary operators (`+`, `-`, `abs`) and fused operations (`fma`)
   - Reduction utilities (`sum`, `dot`, `norm`, `norm_sq`)
   - Device evaluation helpers (`device_eval`)
   - Compile-time configuration macros

3. [SYCL & Heterogeneous Acceleration Guide](sycl_guide.md)
   - Architectural overview: Headerless SYCL compatibility
   - TriviallyCopyable expression graph capture
   - Unified Shared Memory (USM) workflows
   - SYCL buffer and accessor integration
   - Step-by-step kernel examples for Intel oneAPI and AdaptiveCpp

4. [Performance & Benchmarking](benchmarks.md)
   - Benchmark methodology and metrics
   - Throughput (GFLOPS) and memory bandwidth (GB/s) analysis
   - Multi-scale comparison against naive `std::vector` and raw scalar loops
   - Reduction benchmark analysis (`dot`, `sum`, `fma`)
   - Auto-vectorization guidelines for MSVC, Clang, and GCC

5. [Integration & Build Guide](integration.md)
   - CMake `FetchContent` integration
   - System installation and `find_package`
   - Single-header drop-in usage
   - Recommended compiler optimization flags (`AVX2`, `FMA`, `Release`)
   - Running the test and benchmark suites

---

## Quick Example

```cpp
#include <vec_math/vector_math.hpp>
#include <vector>
#include <iostream>

int main() {
    using namespace vec_math;

    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> out(4);

    // 1. Zero-malloc fused expression evaluation into pre-allocated memory
    out <<= 2.0 * a + b / 2.0 - 1.0;

    // 2. High-performance compound in-place math
    out += a;
    out *= 0.5;

    // 3. 8-way unrolled reduction operations
    double total = sum(out);
    double similarity = dot(a, b);
    double magnitude = norm(a);

    std::cout << "Sum: " << total << ", Dot: " << similarity << ", Norm: " << magnitude << "\n";
    return 0;
}
```
