# SYCL & Heterogeneous Acceleration Guide

`vec_math` is engineered with full **SYCL 2020** device compatibility **without including any SYCL headers**.

---

## 1. Architectural Philosophy: Headerless SYCL Compatibility

### Why Avoid Including SYCL Headers?
Traditional accelerator libraries often directly include `<sycl/sycl.hpp>`. However, doing so causes severe friction in real-world C++ projects:
1. **Compilation Overhead**: `<sycl/sycl.hpp>` can introduce tens of thousands of lines of AST templates, dramatically slowing build times.
2. **Vendor Lock-in**: Host-only targets cannot compile without installing vendor toolchains (such as Intel oneAPI or AdaptiveCpp).
3. **Macro & Symbol Pollution**: Exposing GPU-specific symbols into pure host translation units often creates naming collisions.

### How `vec_math` Achieves Zero-Header Portability
`vec_math` adheres to strict standard C++20 language features while anticipating SYCL compiler behavior:
1. **Device Attribute Conditioning**: Uses predefined compiler macros:
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
   ```
2. **TriviallyCopyable Expression AST**: SYCL specification requires that objects captured by value in device kernel lambdas (`[=](sycl::id<1> idx)`) must satisfy `std::is_trivially_copyable_v<T>`. All `vec_math` expressions (`BinExpr`, `UnaryExpr`, `TernaryExpr`, `ScalarVal`, `vec_span`) model `sycl_compatible`:
   ```cpp
   template<typename T>
   concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;
   ```
3. **Zero Dynamic Allocation**: SYCL device kernels forbid dynamic heap calls (`malloc`, `new`, `std::vector::resize`). All `vec_math` expression trees and spans reside completely on the thread stack and GPU registers.

---

## 2. Memory Models & Spans

Accelerators operate on memory buffers decoupled from host `std::vector`. `vec_math` bridges this with `vec_span<T>` (aliased as `device_view<T>`).

| Memory Source | SYCL API | `vec_math` Wrapper |
| :--- | :--- | :--- |
| **USM Shared** | `sycl::malloc_shared<T>(N, q)` | `make_device_view(ptr, N)` |
| **USM Device** | `sycl::malloc_device<T>(N, q)` | `make_device_view(ptr, N)` |
| **Buffer Accessor** | `acc.get_pointer()` | `make_device_view(acc.get_pointer(), N)` |
| **Host Vector** | `std::vector<T>` | `make_device_view(vec)` |

---

## 3. End-to-End SYCL Examples

### Example 1: Unified Shared Memory (USM) Workflow
```cpp
#include <sycl/sycl.hpp>
#include <vec_math/vector_math.hpp>
#include <iostream>

using namespace vec_math;

void run_usm(sycl::queue& q, std::size_t N) {
    // 1. Allocate USM shared memory (accessible by both host and device)
    double* a_ptr   = sycl::malloc_shared<double>(N, q);
    double* b_ptr   = sycl::malloc_shared<double>(N, q);
    double* out_ptr = sycl::malloc_shared<double>(N, q);

    for (std::size_t i = 0; i < N; ++i) {
        a_ptr[i] = static_cast<double>(i);
        b_ptr[i] = 100.0;
    }

    // 2. Wrap pointers in TriviallyCopyable device views
    device_view<const double> va = make_device_view(a_ptr, N);
    device_view<const double> vb = make_device_view(b_ptr, N);
    device_view<double> vout     = make_device_view(out_ptr, N);

    // 3. Submit parallel work to GPU
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        const std::size_t i = idx[0];

        // Construct expression template in device registers
        auto expr = 2.5 * va + vb / 2.0 - 1.0;

        // Evaluate single element without allocations
        device_eval(vout, expr, i);
    }).wait();

    std::cout << "out[0] = " << out_ptr[0] << ", out[1] = " << out_ptr[1] << "\n";

    sycl::free(a_ptr, q);
    sycl::free(b_ptr, q);
    sycl::free(out_ptr, q);
}
```

---

### Example 2: SYCL Buffer & Accessor Integration
```cpp
#include <sycl/sycl.hpp>
#include <vec_math/vector_math.hpp>
#include <vector>
#include <iostream>

using namespace vec_math;

void run_buffer(sycl::queue& q, std::size_t N) {
    std::vector<float> host_a(N, 2.0f);
    std::vector<float> host_b(N, 5.0f);
    std::vector<float> host_out(N, 0.0f);

    {
        sycl::buffer<float, 1> buf_a(host_a.data(), sycl::range<1>(N));
        sycl::buffer<float, 1> buf_b(host_b.data(), sycl::range<1>(N));
        sycl::buffer<float, 1> buf_out(host_out.data(), sycl::range<1>(N));

        q.submit([&](sycl::handler& cgh) {
            auto acc_a   = buf_a.get_access<sycl::access::mode::read>(cgh);
            auto acc_b   = buf_b.get_access<sycl::access::mode::read>(cgh);
            auto acc_out = buf_out.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
                const std::size_t i = idx[0];

                auto va   = make_device_view(acc_a.get_pointer(), N);
                auto vb   = make_device_view(acc_b.get_pointer(), N);
                auto vout = make_device_view(acc_out.get_pointer(), N);

                // Hardware-fused Multiply-Add on GPU
                auto expr = vec_math::fma(va, vb, 10.0f);
                device_eval(vout, expr, i);
            });
        });
    } // Buffer destruction synchronizes data back to host_out

    std::cout << "Result: " << host_out[0] << "\n";
}
```

---

## 4. Compiler Compatibility Matrix

| Compiler | Target Ecosystem | Status |
| :--- | :--- | :--- |
| **Intel oneAPI DPC++** (`icpx -fsycl`) | Intel GPUs, CPUs, FPGAs, NVIDIA (via Codeplay plugin) | Fully Supported |
| **AdaptiveCpp** (`acpp`) | NVIDIA CUDA, AMD ROCm, OpenMP CPUs | Fully Supported |
| **Clang** (`clang++ -fsycl`) | SPIR-V, Generic SYCL targets | Fully Supported |
| **MSVC / GCC** (Host C++20) | Pure CPU host code (no SYCL toolchain needed) | 100% Header-Clean |
