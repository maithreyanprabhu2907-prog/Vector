# Architecture & Design Principles

`vec_math` is engineered to solve fundamental performance bottlenecks in scientific computing, particularly high-order **Discontinuous Galerkin (DG)** finite element methods:
1. **Elimination of temporary heap allocations** in vector mathematical expressions.
2. **Microarchitectural hardware saturation** via cache alignment, non-temporal streaming stores, and dual-FMA execution port saturation.
3. **Coexistence with domain-decomposition concurrency** through a hybrid single-thread SIMD / multi-thread domain model.

---

## 1. The Problem with Naive Vector Operators

In standard C++, implementing overloaded operators for `std::vector` naively leads to catastrophic performance degradation:

```cpp
// Naive operator implementation
std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
    return res;
}
```

Evaluating an expression such as:
```cpp
out = 2.5 * a + 1.5 * b - 0.5 * c;
```
creates **5 intermediate `std::vector` heap allocations and deallocations** (`malloc` + `free`) per evaluation, along with 5 full memory passes that flood the CPU L1/L2 caches with redundant intermediate buffers.

---

## 2. Zero-Allocation Expression Templates

`vec_math` utilizes **Expression Templates** to represent mathematical expressions as lightweight, compile-time abstract syntax trees (ASTs) rather than immediately evaluating them.

```
                  BinExpr<OpSub>
                  /            \
          BinExpr<OpAdd>    ScalarVal(0.5) * c
          /            \
  ScalarVal(2.5) * a   ScalarVal(1.5) * b
```

### AST Representation
- **`VecRef<T>`**: Non-owning lightweight reference wrapping a pointer to an existing vector's underlying contiguous data (`data()`) and its size. Size: 16 bytes (pointer + size).
- **`ScalarVal<S>`**: Stores a numeric scalar value.
- **`BinExpr<Op, Lhs, Rhs>`**: Encapsulates a binary operation (`OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpMin`, `OpMax`, `OpPow`) and two sub-expressions.
- **`UnaryExpr<Op, Arg>`**: Encapsulates a unary operation (`OpPos`, `OpNeg`, `OpAbs`, `OpSqrt`, `OpExp`, `OpLog`, `OpSin`, `OpCos`).
- **`TernaryExpr<Op, A, B, C>`**: Encapsulates a three-operand operation, such as `OpFma` ($a \times b + c$) and `OpClamp`.

### Lazy Element-Wise Access
Each AST node implements:
```cpp
[[nodiscard]] VEC_MATH_HD_INLINE constexpr value_type operator[](std::size_t i) const noexcept;
```
No computation occurs when the expression is declared. The entire tree is evaluated lazily, element by element, in a single loop pass when assigned to a destination buffer.

---

## 3. Assignment Semantics & Evaluation

`vec_math` provides three distinct evaluation pathways:

### A. Pre-Allocated Zero-Malloc Assignment (`<<=`)
```cpp
out <<= 2.0 * a + b / 2.0;
```
- Validates that `out.capacity() >= src.size()`. If capacity is insufficient, throws `std::length_error` (or aborts under `VEC_MATH_NO_EXCEPTIONS`).
- Resizes `out` without reallocating.
- Writes each computed element directly into `out.data()[i]`.
- **Heap Allocations: 0**.

### B. In-Place Compound Operators (`+=`, `-=`, `*=`, `/=`)
```cpp
out += 2.0 * a; // BLAS-1 AXPY with zero allocations
out *= 0.5;
```
- Operates directly on the destination buffer in place.
- Eliminates any destination memory reallocation.
- **Heap Allocations: 0**.

### C. Rvalue Buffer Recycling
```cpp
auto res = 2.0 * std::move(a) + b;
```
- Reuses the memory buffer of rvalue `std::vector&&` parameters.
- If an operand is a temporary or moved vector, `vec_math` performs in-place mutation on that buffer and moves it to the result.
- **Heap Allocations: 0**.

---

## 4. Discontinuous Galerkin (DG) Solver Microarchitectural Optimizations

Discontinuous Galerkin (DG) methods for hyperbolic conservation laws (e.g. compressible Navier-Stokes, Euler, Maxwell's equations) execute millions of high-order polynomial evaluations per time step. `vec_math` incorporates four specialized microarchitectural optimizations designed specifically for DG solvers:

```
+-------------------------------------------------------------------------------+
|                       DG Solver Hardware Optimizations                        |
+-------------------------------------------------------------------------------+
| 1. 64-Byte Cache Alignment  | 2. Non-Temporal Streaming Stores                |
|    - aligned_vector<T, 64>  |    - assign_stream(dst, expr)                   |
|    - Zero split-line faults |    - Bypasses RFO bus read penalty              |
|    - Aligned AVX2/512 loads |    - 2.47x memory bandwidth on out-of-cache data |
+-----------------------------+-------------------------------------------------+
| 3. Dual-FMA Pipe Saturation | 4. Hybrid Execution Model                       |
|    - 8-way unrolled trees   |    - Element kernels: Single-threaded SIMD      |
|    - Independent accumulators|    - Whole-domain: assign_par / dot_par / sum_par|
|    - Port 0 & Port 1 active |    - Zero OpenMP thread contention              |
+-------------------------------------------------------------------------------+
```

### 1. 64-Byte Cache-Line Alignment (`aligned_vector<T>`)
Modern x86-64 microarchitectures feature 64-byte L1 data cache lines. Standard 256-bit AVX2 registers hold 4 `double` values (32 bytes), while AVX-512 registers hold 8 `double` values (64 bytes).
- **The Split-Line Hazard**: When an unaligned array is loaded, a single SIMD vector instruction often straddles two separate 64-byte cache lines. The CPU memory execution unit must issue two cache accesses, lock cache lines, and perform dynamic alignment splicing, adding 2–4 cycles of latency per vector load.
- **The Solution**: `vec_math::aligned_allocator<T, 64>` and `vec_math::aligned_vector<T>` align memory allocations directly to 64-byte boundaries using `_aligned_malloc` on Windows and `posix_memalign` / `std::aligned_alloc` on POSIX systems.
- **Result**: Compilers generate direct aligned SIMD loads/stores (`vmovapd`, `vmovaps`), completely bypassing compiler alignment peel loops and avoiding split-line cache penalties.

### 2. Streaming Stores Bypassing RFO Cache Pollution (`assign_stream`)
Standard CPU write instructions employ a **Read-For-Ownership (RFO)** cache protocol:
1. When a core writes to a memory location not present in L1/L2/L3 cache, it must first load the entire 64-byte cache line from DRAM into L3/L2/L1 cache.
2. The core modifies the requested bytes in cache.
3. The modified cache line is eventually evicted back to DRAM.

For large DG state vectors that exceed the CPU L3 cache (e.g. 1M – 5M elements, representing 8 MB – 40 MB vs the 16 MB L3 cache on AMD Ryzen 5 5500), RFO causes **double memory bus traffic**: 1 byte read + 1 byte written for every byte stored. Furthermore, it completely evicts hot solver state (polynomial quadrature weights, geometric Jacobians) from cache.

`assign_stream` solves this by utilizing **non-temporal AVX streaming stores**:
- Evaluates the expression directly into destination memory using `_mm256_stream_pd` / `_mm256_stream_ps`.
- Writes bypass L1/L2/L3 cache entirely and accumulate in CPU Write-Combining (WC) store buffers before being written directly to DRAM.
- Emits `_mm_sfence()` to establish memory ordering and visibility.
- **Result**: Eliminates the 50% RFO read penalty on large datasets, delivering up to a **2.47x speedup** (42.08 GB/s streaming vs 17.06 GB/s standard assignment on 1M `double` elements).

### 3. Dual-FMA Pipe Interleaving & Instruction-Level Parallelism
Modern CPU cores (such as AMD Zen 3 / Zen 4 and Intel Golden Cove / Raptor Cove) feature **two 256-bit FMA (Fused Multiply-Add) execution units** capable of retiring two 256-bit FMA instructions per cycle:
- AMD Zen 3: Port 0 and Port 1 both support 256-bit FMA operations, offering a theoretical throughput of two 4-wide `double` FMAs per cycle (8 DP FLOPs/cycle).
- **The Dependency Chain Problem**: If a loop accumulates into a single register (`total += a[i] * b[i]`), each FMA instruction must wait for the previous FMA's 4-cycle execution latency to complete before starting, starving the second execution port and capping throughput at ~25% of theoretical peak.
- **The Solution**: `vec_math` loops are unrolled 8-way and maintain 8 independent accumulator registers:
  ```cpp
  ResType acc0{0}, acc1{0}, acc2{0}, acc3{0};
  ResType acc4{0}, acc5{0}, acc6{0}, acc7{0};
  for (; i < n8; i += 8) {
      acc0 += lhs[i + 0] * rhs[i + 0];
      acc1 += lhs[i + 1] * rhs[i + 1];
      acc2 += lhs[i + 2] * rhs[i + 2];
      acc3 += lhs[i + 3] * rhs[i + 3];
      acc4 += lhs[i + 4] * rhs[i + 4];
      acc5 += lhs[i + 5] * rhs[i + 5];
      acc6 += lhs[i + 6] * rhs[i + 6];
      acc7 += lhs[i + 7] * rhs[i + 7];
  }
  ```
- **Result**: Under `-Ofast` / `/fp:fast`, the compiler pairs independent FMA instructions across Port 0 and Port 1 every cycle, breaking dependency bottlenecks and achieving **43.41 GFLOPS** on a single core.

### 4. Hybrid Single-Thread SIMD / Multi-Thread Domain Execution Model
In practical DG codes, spatial meshes are partitioned into thousands or millions of elements (e.g. order $P=3$ to $P=7$ with 64 to 512 degrees of freedom per element). The outer solver harness already parallelizes over mesh elements using OpenMP (`#pragma omp parallel for`) or MPI domain decomposition.
- **The Nested Contention Trap**: If mathematical operators (`+=`, `<<=`, `*`) automatically spawned internal OpenMP threads, thousands of fine-grained thread fork/join barriers would be triggered per element, inducing catastrophic thread synchronization overhead and cache thrashing.
- **The Solution**:
  1. **Single-Threaded SIMD by Default**: All core operators (`+=`, `-=`, `*=`, `/=`, `<<=`, `|`, `clamp`, `sqrt`, etc.) are strictly single-threaded SIMD. They vectorize within CPU registers without spawning threads, making them ideal for inner element-quadrature loops.
  2. **Explicit Multi-Core Domain Acceleration**: For global mesh-wide state updates and whole-domain reductions, explicit OpenMP-guarded functions are provided:
     - `assign_par(dst, expr)`: Parallel multi-threaded expression assignment.
     - `dot_par(u, v)`: Parallel multi-threaded inner product reduction.
     - `sum_par(v)`: Parallel multi-threaded summation reduction.
- **Result**: Maximum parallelism across all scales: fine-grained element loops run SIMD-vectorized with zero thread overhead, while macroscopic domain updates scale across all 12 hardware threads with up to **6.10x parallel speedup**.

---

## 5. SYCL Compiler Compatibility & Heterogeneous Readiness

`vec_math` is designed for seamless use in projects targeting heterogeneous accelerators (GPUs, FPGAs) with SYCL compilers (such as Intel oneAPI DPC++ and AdaptiveCpp) **without including SYCL headers**:

### Zero Header Inclusions
- The library does not include `<sycl/sycl.hpp>` or any vendor-specific header.
- Pure standard C++20 host applications compile cleanly without requiring SYCL toolchains or runtime libraries.

### TriviallyCopyable Expression Templates
- In SYCL, any object captured by value inside a device kernel lambda (`cgh.parallel_for([=](sycl::id<1> idx) { ... })`) must be TriviallyCopyable (`std::is_trivially_copyable_v<T>`).
- All `vec_math` expression templates (`VecRef`, `ScalarVal`, `BinExpr`, `UnaryExpr`, `TernaryExpr`) are standard layout and TriviallyCopyable.
- The `sycl_compatible` concept verifies this compile-time property:
  ```cpp
  template<typename T>
  concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;
  ```

### Predefined Macro Annotations
- The core math functors and expression node indexing operators are decorated with `VEC_MATH_HD_INLINE` (`VEC_MATH_HOST_DEVICE VEC_MATH_FORCE_INLINE`) and `SYCL_EXTERNAL`.
- These macros detect SYCL device compilation passes (`__SYCL_DEVICE_ONLY__` or `SYCL_LANGUAGE_VERSION`) and apply the appropriate attributes (`__attribute__((sycl_device))` or `__declspec(sycl_device)`). On pure host compilers, they expand to standard inline qualifiers.

### Zero Dynamic Allocation
- SYCL device kernels forbid dynamic allocations (`malloc`, `new`).
- All `vec_math` expression trees reside entirely in CPU/GPU registers and thread stack frames, guaranteeing zero runtime allocations.
