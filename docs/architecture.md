# Architecture & Design Principles

`vec_math` is engineered to solve a fundamental performance bottleneck in C++ numerical computing: **temporary heap allocations in vector arithmetic expressions**.

---

## 1. The Problem with Naive Vector Operators

In standard C++, implementing overloaded operators for `std::vector` naively leads to disastrous allocation overhead:

```cpp
// Naive operator implementation
std::vector<double> operator+(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> res(a.size());
    for (size_t i = 0; i < a.size(); ++i) res[i] = a[i] + b[i];
    return res;
}
```

Evaluating an expression like:
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
- **`BinExpr<Op, Lhs, Rhs>`**: Encapsulates a binary operation (`OpAdd`, `OpSub`, `OpMul`, `OpDiv`) and two sub-expressions.
- **`UnaryExpr<Op, Arg>`**: Encapsulates a unary operation (`OpPos`, `OpNeg`, `OpAbs`).
- **`TernaryExpr<Op, A, B, C>`**: Encapsulates a three-operand operation, such as `OpFma` ($a \times b + c$).

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
out += 2.0 * a;
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

## 4. SIMD Auto-Vectorization & Instruction-Level Parallelism

`vec_math` employs aggressive optimization techniques to ensure modern compilers generate optimal SIMD instructions (AVX2, AVX-512, NEON):

### Compiler Vectorization Pragmas
All loops are annotated with compiler-specific vectorization hints:
- **MSVC**: `__pragma(loop(ivdep))` ignores assumed loop dependencies.
- **Clang**: `#pragma clang loop vectorize(enable)` forces loop vectorization.
- **GCC**: `#pragma GCC ivdep` asserts no loop-carried memory dependencies.
- **OpenMP**: `#pragma omp simd` when built with OpenMP.

### 8x and 4x Loop Unrolling
Evaluation and reduction loops are unrolled 8-way and 4-way:
```cpp
const std::size_t n8 = n & ~std::size_t(7);
VEC_MATH_PRAGMA_SIMD
for (; i < n8; i += 8) {
    dst[i + 0] = static_cast<T>(src[i + 0]);
    dst[i + 1] = static_cast<T>(src[i + 1]);
    dst[i + 2] = static_cast<T>(src[i + 2]);
    dst[i + 3] = static_cast<T>(src[i + 3]);
    dst[i + 4] = static_cast<T>(src[i + 4]);
    dst[i + 5] = static_cast<T>(src[i + 5]);
    dst[i + 6] = static_cast<T>(src[i + 6]);
    dst[i + 7] = static_cast<T>(src[i + 7]);
}
```
Unrolling exposes multiple independent memory operations to CPU execution ports, maximizing instruction-level parallelism (ILP) and saturating memory bus throughput.

### Dependency-Broken Reduction Trees
In reductions such as `vec_math::dot` and `vec_math::sum`, 8 separate accumulator registers are maintained:
```cpp
ResType acc0{0}, acc1{0}, acc2{0}, acc3{0};
ResType acc4{0}, acc5{0}, acc6{0}, acc7{0};
// Loop updates all 8 accumulators independently...
ResType total = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
```
By breaking the single-accumulator serial dependency chain, the CPU can schedule 8 independent FMA/addition instructions per clock cycle. This yields up to a **3x speedup in dot product** and **7x speedup in sum reduction**.

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
