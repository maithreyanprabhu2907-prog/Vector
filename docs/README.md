# vec_math Documentation

Welcome to the official documentation for **vec_math**, a high-performance, zero-allocation, header-only C++20 vector mathematics library engineered for `std::vector` with seamless SYCL compiler compatibility.

---

## Table of Contents

1. [Architecture & Design Principles](architecture.md)
   - Zero-allocation expression template engine
   - AST fusion and lazy evaluation
   - In-place compound operations and rvalue buffer recycling
   - SIMD auto-vectorization and loop unrolling (8x/4x)
   - Cache locality and memory traffic minimization
   - SYCL compiler compatibility & TriviallyCopyable expressions

2. [API Reference](api_reference.md)
   - Core concepts (`Scalar`, `NumericVector`, `VecExpression`, `VectorOperand`, `sycl_compatible`)
   - Overloaded arithmetic operators (`+`, `-`, `*`, `/`)
   - Compound assignment operators (`+=`, `-=`, `*=`, `/=`)
   - Zero-allocation evaluation assignment (`<<=`) and `assign()`
   - Unary operators (`+`, `-`, `abs`) and fused operations (`fma`)
   - Reduction utilities (`sum`, `dot`, `norm`, `norm_sq`)
   - Compile-time configuration macros & device annotations

3. [SYCL Compiler Compatibility Guide](sycl_guide.md)
   - Architectural overview: Headerless SYCL compiler compatibility
   - Zero SYCL headers included (no vendor lock-in or build bloat)
   - TriviallyCopyable expression graph capture in device kernels
   - Predefined compiler macro annotations (`SYCL_EXTERNAL`, `VEC_MATH_DEVICE`, `VEC_MATH_HOST_DEVICE`)
   - Compatibility across Intel oneAPI DPC++, AdaptiveCpp, Clang, and standard host compilers

4. [Performance & Benchmarking](benchmarks.md)
   - Benchmark methodology and metrics
   - 3-way performance comparison: Naive `std::vector`, Scalar Loop, and `vec_math`
   - Throughput (GFLOPS) and memory bandwidth (GB/s) across scales (1K to 5M elements)
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
