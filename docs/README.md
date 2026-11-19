# vec_math Documentation

Welcome to the official documentation for **vec_math**, a high-performance, zero-allocation, header-only C++20 vector mathematics library engineered for `std::vector` and `aligned_vector` with pure operator overloading, Discontinuous Galerkin (DG) scientific computing optimizations, and seamless SYCL compiler compatibility.

---

## Table of Contents

1. **[Architecture & Design Principles](architecture.md)**
   - Zero-allocation expression template engine
   - Compile-time AST fusion and lazy evaluation
   - In-place compound operations and rvalue buffer recycling
   - **Discontinuous Galerkin (DG) Solver Microarchitectural Optimizations**:
     * 64-byte cache-line alignment (`aligned_vector<T>`) to eliminate split-line penalties
     * Non-temporal streaming stores (`assign_stream`) bypassing Read-For-Ownership (RFO) cache pollution
     * Dual-FMA execution pipe interleaving and instruction-level parallelism
     * Hybrid single-thread SIMD / multi-thread domain execution model
   - Compiler vectorization pragmas and 8-way unrolled reduction trees
   - SYCL compiler compatibility & TriviallyCopyable expressions

2. **[API Reference](api_reference.md)**
   - Core concepts (`Scalar`, `NumericVector`, `VecExpression`, `VectorOperand`, `sycl_compatible`)
   - 64-byte aligned memory: `aligned_allocator<T, 64>` and `aligned_vector<T>`
   - Pure operator overloading:
     * Binary arithmetic operators (`+`, `-`, `*`, `/`)
     * Pipe inner product / dot product operator (`u | v`)
     * Compound assignment operators & in-place AXPY (`y += alpha * x`, `-=`, `*=`, `/=`)
     * Zero-allocation fused assignment & AXPBY (`y <<= alpha * x + beta * y`)
   - Discontinuous Galerkin (DG) Scientific Math Kit:
     * Element-wise limiters: `clamp(v, lo, hi)`, `min(a, b)`, `max(a, b)`
     * Transcendental functions: `sqrt(v)`, `pow(v, p)`, `exp(v)`, `log(v)`, `sin(v)`, `cos(v)`, `abs(v)`, `fma(a, b, c)`
     * CFL & time-stepping reductions: `min_val(v)`, `max_val(v)`, `norm_inf(v)`, `norm_l1(v)`
     * BLAS-1 reductions: `sum(v)`, `dot(a, b)`, `norm_sq(v)`, `norm(v)`
   - Non-temporal streaming stores: `assign_stream(dst, expr)`
   - Multi-core OpenMP parallel domain helpers: `assign_par(dst, expr)`, `dot_par(u, v)`, `sum_par(v)`
   - Compile-time configuration macros & device annotations

3. **[SYCL Compiler Compatibility Guide](sycl_guide.md)**
   - Architectural philosophy: Headerless SYCL compiler compatibility
   - Strictly zero SYCL headers included (no vendor lock-in, clean standard C++20)
   - TriviallyCopyable expression graph capture in device kernels (`sycl_compatible`)
   - Predefined compiler macro annotations (`SYCL_EXTERNAL`, `VEC_MATH_DEVICE`, `VEC_MATH_HOST_DEVICE`)
   - Interoperability with SYCL buffers, accessors, and device queues
   - Compatibility across Intel oneAPI DPC++, AdaptiveCpp, Clang, and host compilers

4. **[Performance & Benchmarking](benchmarks.md)**
   - Benchmark methodology and hardware specifications (AMD Ryzen 5 5500, Zen 3)
   - Multi-scale performance comparison across scales (1K to 5M elements)
   - Discontinuous Galerkin Runge-Kutta stage update benchmark (AXPBY `y <<= a*x + b*y`)
   - Non-temporal streaming stores benchmark (`assign_stream` delivering up to 42.08 GB/s)
   - Multi-core parallel scaling across CPU cores (`assign_par` achieving 6.10x speedup)
   - Reductions & DG math kit benchmark (`dot`, `sum`, `norm_inf`, `min_val`, `fma`, `clamp`)
   - `-Ofast` microarchitectural optimization analysis

5. **[Integration & Build Guide](integration.md)**
   - CMake `FetchContent` integration
   - System installation and `find_package`
   - Single-header drop-in usage
   - Recommended compiler optimization flags (`AVX2`, `FMA`, `Release`, `-Ofast`)
   - Running the test and benchmark suites

---

## Quick Example

```cpp
#include <vec_math/vector_math.hpp>
#include <iostream>

int main() {
    using namespace vec_math;

    // 1. 64-byte cache-line aligned vectors
    aligned_vector<double> u(1'000'000);
    aligned_vector<double> v(1'000'000);
    aligned_vector<double> out(1'000'000);

    // Fill initial values
    for (size_t i = 0; i < u.size(); ++i) {
        u[i] = static_cast<double>(i) * 0.001;
        v[i] = 1.0;
    }

    // 2. Pure operator overloading: BLAS-1 AXPY (zero heap allocations)
    out += 0.5 * u;

    // 3. High-order Runge-Kutta stage update: AXPBY (zero heap allocations)
    out <<= 0.75 * u + 0.25 * out;

    // 4. Natural pipe syntax for dot / inner product
    double inner_prod = u | v;

    // 5. DG element-wise limiters & CFL reductions
    out <<= clamp(out, 0.0, 10.0);
    double min_step = min_val(out);
    double max_wave = max_val(out);
    double l_inf    = norm_inf(out);

    // 6. Non-temporal streaming store (bypasses RFO cache pollution)
    assign_stream(out, 2.0 * u - 0.5 * v);

    // 7. Multi-core domain parallel evaluation
    assign_par(out, 1.5 * u + 0.5 * v);

    std::cout << "Inner Product: " << inner_prod 
              << ", Min: " << min_step 
              << ", L_inf: " << l_inf << "\n";
    return 0;
}
```
