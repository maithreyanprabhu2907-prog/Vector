# SYCL Compiler Compatibility Guide

`vec_math` is engineered for seamless compatibility with modern SYCL compilers (such as Intel oneAPI DPC++, AdaptiveCpp, and Clang) **without including SYCL headers**.

---

## 1. Architectural Philosophy: Headerless Compatibility

### The Pitfalls of Including SYCL Headers
Traditional heterogeneous libraries often directly include `<sycl/sycl.hpp>`. While convenient for SYCL-only codebases, this introduces critical pain points in production C++ projects:
1. **Compilation Overhead**: `<sycl/sycl.hpp>` can introduce tens of thousands of lines of template code into every translation unit, drastically slowing build times.
2. **Vendor Lock-in & Toolchain Friction**: Pure host translation units fail to compile unless the developer has a complete SYCL toolchain installed (such as Intel oneAPI or AdaptiveCpp).
3. **Macro & Symbol Collisions**: Injecting GPU/device-specific symbols and intrinsics into standard C++ code frequently leads to namespace collisions.

### How `vec_math` Solves This
`vec_math` avoids all SYCL header inclusions. It achieves complete SYCL compatibility through standard C++20 language features and conditionally enabled compiler macros:
- **Zero Header Dependencies**: Strictly zero SYCL headers are included.
- **Pure C++20 Standard Compliance**: Builds cleanly with standard host compilers (MSVC, GCC, Clang) without warnings or errors.
- **Transparent Acceleration**: Automatically participates in device compilation when invoked under a SYCL compiler.

---

## 2. Core Compatibility Guarantees

### A. TriviallyCopyable Expression Templates
The SYCL specification mandates that any object captured by value in a device kernel lambda (`cgh.parallel_for([=](sycl::id<1> idx) { ... })`) must satisfy `std::is_trivially_copyable_v<T>`.

All `vec_math` expression templates and mathematical nodes are standard-layout and TriviallyCopyable:
- `VecRef<T>` (16-byte pointer + size wrapper)
- `ScalarVal<S>` (scalar value wrapper)
- `BinExpr<Op, Lhs, Rhs>` (including `+`, `-`, `*`, `/`, `min`, `max`, `pow`)
- `UnaryExpr<Op, Arg>` (including `+`, `-`, `abs`, `sqrt`, `exp`, `log`, `sin`, `cos`)
- `TernaryExpr<Op, A, B, C>` (including `fma` and `clamp`)

The `sycl_compatible` concept verifies this property at compile time:
```cpp
template<typename T>
concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;

// Verified at compile time
static_assert(vec_math::sycl_compatible<decltype(2.0 * a + b)>);
static_assert(vec_math::sycl_compatible<decltype(vec_math::clamp(a, 0.0, 1.0))>);
static_assert(vec_math::sycl_compatible<decltype(vec_math::sqrt(a))>);
```

### B. Conditional Device Function Annotations
`vec_math` decorates its core mathematical functors, operations, and AST element evaluators with compiler attributes that activate **only** during SYCL device compilation passes:

```cpp
#ifndef SYCL_EXTERNAL
#if defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_LANGUAGE_VERSION)
#if defined(_MSC_VER) && !defined(__clang__)
#define SYCL_EXTERNAL __declspec(sycl_device)
#else
#define SYCL_EXTERNAL __attribute__((sycl_device))
#endif
#else
#define SYCL_EXTERNAL
#endif
#endif

#ifndef VEC_MATH_DEVICE
#if defined(__SYCL_DEVICE_ONLY__)
#define VEC_MATH_DEVICE SYCL_EXTERNAL
#else
#define VEC_MATH_DEVICE
#endif
#endif

#ifndef VEC_MATH_HOST_DEVICE
#if defined(__SYCL_DEVICE_ONLY__) || defined(SYCL_LANGUAGE_VERSION)
#define VEC_MATH_HOST_DEVICE SYCL_EXTERNAL
#else
#define VEC_MATH_HOST_DEVICE
#endif
#endif
```

During host compilation, these macros expand to standard inline attributes (`inline` / `__forceinline`). During device compilation (`__SYCL_DEVICE_ONLY__` or `SYCL_LANGUAGE_VERSION`), they apply the device-callable specifiers required by DPC++ and AdaptiveCpp.

### C. Zero Dynamic Allocation
Device kernels prohibit dynamic heap allocation (`malloc`, `new`, `std::vector::push_back`). All expression nodes are stack-allocated, value-captured, and optimized into registers during kernel generation.

---

## 3. Using `vec_math` in SYCL Projects

Because `vec_math` does not bundle proprietary span/view wrappers or force SYCL types into host code, using it in SYCL projects is straightforward and clean:

### Host-Side Preparation & Vector Math
In SYCL applications, host pre-processing and post-processing often utilize `std::vector` or `vec_math::aligned_vector`:

```cpp
#include <vec_math/vector_math.hpp>
#include <vector>
#include <iostream>

void prepare_data(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) {
    using namespace vec_math;
    // Fused, zero-allocation pre-processing before GPU offload
    out <<= 2.0 * a + b / 2.0;
}
```

### Interoperability with SYCL Queues & Buffers
SYCL pipelines can transfer data directly to and from `std::vector` buffers:

```cpp
// In downstream SYCL application:
// #include <sycl/sycl.hpp> (provided by the application, never by vec_math)
#include <vec_math/vector_math.hpp>
#include <vector>
#include <iostream>

void process_with_sycl(sycl::queue& q, std::vector<float>& host_in, std::vector<float>& host_out) {
    using namespace vec_math;

    // 1. Host-side mathematical preparation with vec_math
    host_in += 10.0f;

    // 2. Offload to SYCL device via buffer
    {
        sycl::buffer<float, 1> buf_in(host_in.data(), sycl::range<1>(host_in.size()));
        sycl::buffer<float, 1> buf_out(host_out.data(), sycl::range<1>(host_out.size()));

        q.submit([&](sycl::handler& cgh) {
            auto acc_in  = buf_in.get_access<sycl::access::mode::read>(cgh);
            auto acc_out = buf_out.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(host_in.size()), [=](sycl::id<1> idx) {
                // Kernel logic executes on accelerator
                acc_out[idx] = acc_in[idx] * 2.0f;
            });
        });
    }

    // 3. Post-processing reduction on host with vec_math
    float total = sum(host_out);
    std::cout << "Total: " << total << "\n";
}
```

### Expression Template Evaluation in Kernels
Because `BinExpr`, `UnaryExpr`, and `TernaryExpr` are `TriviallyCopyable`, scalar values and expression structures can be captured by value in SYCL device lambdas without compiler warnings or runtime faults:

```cpp
// Expression templates can be captured by value in SYCL device lambdas
static_assert(std::is_trivially_copyable_v<vec_math::detail::ScalarVal<double>>);
```

---

## 4. Documentation Cross-References

For deeper technical exploration of other library components:
- **[Architecture & Design Principles](architecture.md)**: Details the zero-allocation AST engine, 64-byte cache alignment, streaming stores, and dual-FMA instruction scheduling.
- **[API Reference](api_reference.md)**: Complete catalog of operators, expression nodes, reductions, and device annotations.
- **[Performance & Benchmarking](benchmarks.md)**: Empirical measurements, throughput charts, and scalability analysis.
- **[Integration & Build Guide](integration.md)**: Build instructions, CMake options, and test workflows.

---

## 5. Compiler Compatibility Matrix

| Compiler | Build Command Example | Target Platforms | Status |
| :--- | :--- | :--- | :--- |
| **Intel oneAPI DPC++** | `icpx -fsycl -std=c++20 main.cpp` | Intel GPUs (Xe/Arc/Max), CPUs, FPGAs, NVIDIA (via Codeplay) | **Fully Supported** |
| **AdaptiveCpp** | `acpp -std=c++20 main.cpp` | NVIDIA CUDA, AMD ROCm, OpenMP CPUs | **Fully Supported** |
| **Clang (LLVM SYCL)** | `clang++ -fsycl -std=c++20 main.cpp` | SPIR-V, Generic SYCL targets | **Fully Supported** |
| **MSVC / GCC / Clang (Host)** | `cmake -B build` | x86_64, ARM64 CPU host applications | **100% Header-Clean** |

---

## 6. Summary of Benefits

- **Zero Vendor Lock-in**: Developers are not tied to Intel, AdaptiveCpp, or any specific accelerator framework.
- **Zero Include Overhead**: Including `<vec_math/vector_math.hpp>` does not pull in `<sycl/sycl.hpp>`.
- **Strictly Safe Device Semantics**: Guaranteed TriviallyCopyable types ensure standard-compliant device kernel capture.
