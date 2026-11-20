# API Reference

Complete API reference for the `vec_math` library. All entities reside in namespace `vec_math` or inline namespace `vec_math::ops`.

---

## 1. Concepts & Types

Header: `<vec_math/vector_math.hpp>`

### `Scalar<T>`
```cpp
template<typename T>
concept Scalar = detail::is_valid_numeric_v<T>;
```
Satisfied by any standard arithmetic type (integral or floating point), excluding `bool`.

### `NumericVector<T>`
```cpp
template<typename T>
concept NumericVector = detail::is_std_vector_v<T> && Scalar<typename std::remove_cvref_t<T>::value_type>;
```
Satisfied by `std::vector<T, Alloc>` (including `vec_math::aligned_vector<T>`) where `T` models `Scalar`.

### `VecExpression<T>`
```cpp
template<typename T>
concept VecExpression = detail::is_vec_expr_v<T>;
```
Satisfied by any expression template node inheriting from `detail::ExprBase` (`BinExpr`, `UnaryExpr`, `TernaryExpr`).

### `VectorOperand<T>`
```cpp
template<typename T>
concept VectorOperand = NumericVector<T> || VecExpression<T>;
```
General concept representing any operand participating in vector mathematics.

### `sycl_compatible<T>`
```cpp
template<typename T>
concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;

template<typename T>
inline constexpr bool is_sycl_compatible_v = sycl_compatible<T>;
```
Ensures that a type is TriviallyCopyable and standard layout so it can be safely captured by value in SYCL device kernel lambdas (`cgh.parallel_for`) without violating device capture rules. All `vec_math` expression templates satisfy this concept.

---

## 2. 64-Byte Cache-Line Aligned Memory

Header: `<vec_math/vector_math.hpp>`

Discontinuous Galerkin (DG) solvers and SIMD vector units (AVX2 256-bit, AVX-512) suffer memory access penalties when data structures cross 64-byte cache lines. `vec_math` provides a zero-overhead, 64-byte aligned STL allocator and container alias:

### `aligned_allocator<T, Alignment>`
```cpp
template<typename T, std::size_t Alignment = 64>
struct aligned_allocator;
```
Standard-compliant allocator allocating aligned memory blocks:
- **Windows / MSVC**: Calls `_aligned_malloc(bytes, Alignment)` and `_aligned_free(p)`.
- **POSIX / Linux / macOS**: Calls `posix_memalign` or `std::aligned_alloc(Alignment, rounded_bytes)` and `std::free(p)`.
- Implements full allocator traits including `rebind` for seamless compatibility with STL containers.

### `aligned_vector<T>`
```cpp
template<typename T>
using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;
```
Alias for `std::vector` configured with 64-byte cache alignment.
- Works out-of-the-box with all overloaded operators (`+`, `-`, `*`, `/`, `+=`, `-=`, `*=`, `/=`, `<<=`, `|`).
- Guarantees aligned load/store vectorization without compiler prologue/epilogue peel overhead.

---

## 3. Pure Operator Overloading

Overloaded operators reside in inline namespace `vec_math::ops` and are automatically imported when using namespace `vec_math`.

### Binary Arithmetic Operators (`+`, `-`, `*`, `/`)
Construct a lazy `BinExpr` without performing dynamic memory allocations:

```cpp
// Vector-Vector
template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator+(const Lhs& lhs, const Rhs& rhs);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator-(const Lhs& lhs, const Rhs& rhs);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator*(const Lhs& lhs, const Rhs& rhs);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator/(const Lhs& lhs, const Rhs& rhs);

// Vector-Scalar & Scalar-Vector
template<VectorOperand Lhs, Scalar S>
[[nodiscard]] constexpr auto operator+(const Lhs& lhs, const S& scalar);

template<Scalar S, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator+(const S& scalar, const Rhs& rhs);

template<VectorOperand Lhs, Scalar S>
[[nodiscard]] constexpr auto operator-(const Lhs& lhs, const S& scalar);

template<Scalar S, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator-(const S& scalar, const Rhs& rhs);

template<VectorOperand Lhs, Scalar S>
[[nodiscard]] constexpr auto operator*(const Lhs& lhs, const S& scalar);

template<Scalar S, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator*(const S& scalar, const Rhs& rhs);

template<VectorOperand Lhs, Scalar S>
[[nodiscard]] constexpr auto operator/(const Lhs& lhs, const S& scalar);

template<Scalar S, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator/(const S& scalar, const Rhs& rhs);
```

### Inner Product / Dot Product Operator (`|`)
Natural mathematical pipe syntax for inner product / dot product:
```cpp
template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto operator|(const Lhs& lhs, const Rhs& rhs);
```
Equivalent to `dot(lhs, rhs)`. Evaluates using 8 independent SIMD accumulation registers with zero heap allocations.

```cpp
double inner_prod = u | v;
```

### Compound Assignment Operators & In-Place AXPY (`+=`, `-=`, `*=`, `/=`)
Direct in-place arithmetic without intermediate buffers or reallocations:

```cpp
// Vector-Vector and Vector-Expression (e.g. AXPY)
dst += rhs; // e.g. y += alpha * x (BLAS-1 AXPY with zero allocations)
dst -= rhs;
dst *= rhs;
dst /= rhs;

// Vector-Scalar
dst += scalar;
dst -= scalar;
dst *= scalar;
dst /= scalar;
```

### Zero-Allocation Fused Assignment & AXPBY (`<<=`)
```cpp
template<typename T, typename Alloc, VectorOperand Src>
std::vector<T, Alloc>& operator<<=(std::vector<T, Alloc>& dst, const Src& src);
```
Evaluates arbitrary multi-term expressions directly into `dst` in a single SIMD pass using 8x/4x unrolled loops.
- **AXPBY / Runge-Kutta Stage Update**: `y <<= alpha * x + beta * y;` (replaces procedural BLAS `axpby` with natural syntax).
- **Multi-Term Combinations**: `y <<= a * u + b * v + c * w;`
- **Zero Allocations**: Strictly 0 dynamic heap allocations when `dst.capacity() >= src.size()`.

### Free Function `assign()`
```cpp
template<typename T, typename Alloc, VectorOperand Src>
void assign(std::vector<T, Alloc>& dst, const Src& src);
```
Equivalent to `dst <<= src`.

---

## 4. Discontinuous Galerkin (DG) Scientific Math Kit

Header: `<vec_math/vector_math.hpp>`

The DG scientific computing kit provides element-wise limiters, transcendental functions, and CFL time-stepping reductions essential for high-order finite element and finite volume methods.

### Element-wise Limiters (Positivity & WENO Limiters)
Construct lazy expression nodes evaluated on assignment with zero intermediate allocations:

#### `clamp(v, lo, hi)`
```cpp
template<typename V, typename Lo, typename Hi>
    requires VectorOperand<V> &&
             (VectorOperand<Lo> || Scalar<Lo>) &&
             (VectorOperand<Hi> || Scalar<Hi>)
[[nodiscard]] constexpr auto clamp(const V& v, const Lo& lo, const Hi& hi);
```
Element-wise clamping: restricts each element $v_i$ such that $\text{lo}_i \le v_i \le \text{hi}_i$. Supports scalar bounds, vector bounds, or mixed bounds.

#### `min(lhs, rhs)` and `max(lhs, rhs)`
```cpp
// Vector-Vector, Vector-Scalar, Scalar-Vector
template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto min(const Lhs& lhs, const Rhs& rhs);

template<VectorOperand Lhs, Scalar S>
[[nodiscard]] constexpr auto min(const Lhs& lhs, const S& scalar);

template<Scalar S, VectorOperand Rhs>
[[nodiscard]] constexpr auto min(const S& scalar, const Rhs& rhs);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto max(const Lhs& lhs, const Rhs& rhs);

template<VectorOperand Lhs, Scalar S>
[[nodiscard]] constexpr auto max(const Lhs& lhs, const S& scalar);

template<Scalar S, VectorOperand Rhs>
[[nodiscard]] constexpr auto max(const S& scalar, const Rhs& rhs);
```
Element-wise minimum and maximum expressions.

### Element-wise Transcendental & Scientific Functions
Construct lazy unary and binary expressions evaluating element-wise mathematical functions:

#### `sqrt(arg)`
```cpp
template<VectorOperand Arg>
[[nodiscard]] inline auto sqrt(const Arg& arg);
```
Element-wise square root: $\sqrt{arg_i}$.

#### `exp(arg)`
```cpp
template<VectorOperand Arg>
[[nodiscard]] inline auto exp(const Arg& arg);
```
Element-wise exponential: $e^{arg_i}$.

#### `log(arg)`
```cpp
template<VectorOperand Arg>
[[nodiscard]] inline auto log(const Arg& arg);
```
Element-wise natural logarithm: $\ln(arg_i)$.

#### `sin(arg)` and `cos(arg)`
```cpp
template<VectorOperand Arg>
[[nodiscard]] inline auto sin(const Arg& arg);

template<VectorOperand Arg>
[[nodiscard]] inline auto cos(const Arg& arg);
```
Element-wise trigonometric sine and cosine functions.

#### `pow(base, exp)`
```cpp
template<VectorOperand Base, VectorOperand Exp>
[[nodiscard]] inline auto pow(const Base& base, const Exp& exp);

template<VectorOperand Base, Scalar Exp>
[[nodiscard]] inline auto pow(const Base& base, const Exp& exp);

template<Scalar Base, VectorOperand Exp>
[[nodiscard]] inline auto pow(const Base& base, const Exp& exp);
```
Element-wise power function: $\text{base}_i^{\text{exp}_i}$.

#### `abs(arg)`
```cpp
template<VectorOperand Arg>
[[nodiscard]] constexpr auto abs(const Arg& arg);
```
Element-wise absolute value: $|arg_i|$.

#### `fma(a, b, c)`
```cpp
template<VectorOperand A, VectorOperand B, VectorOperand C>
[[nodiscard]] constexpr auto fma(const A& a, const B& b, const C& c);

template<VectorOperand A, VectorOperand B, Scalar S>
[[nodiscard]] constexpr auto fma(const A& a, const B& b, const S& c);

template<VectorOperand A, Scalar S, VectorOperand C>
[[nodiscard]] constexpr auto fma(const A& a, const S& b, const C& c);
```
Fused multiply-add ternary expression: $a_i \times b_i + c_i$.

### CFL & Time-Stepping Reductions

#### `min_val(v)` and `max_val(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] constexpr auto min_val(const Op& op);

template<VectorOperand Op>
[[nodiscard]] constexpr auto max_val(const Op& op);
```
Finds the minimum and maximum element values in `op`. Essential for determining time-step sizes via the Courant-Friedrichs-Lewy (CFL) stability condition:
$$\Delta t \le \text{CFL} \cdot \min_{i} \left(\frac{\Delta x_i}{\lambda_i}\right)$$

#### `norm_inf(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] inline auto norm_inf(const Op& op);
```
Calculates the $L_\infty$ norm (maximum absolute value): $\|v\|_\infty = \max_i |v_i|$. Used to check residual convergence in iterative solvers.

#### `norm_l1(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] inline auto norm_l1(const Op& op);
```
Calculates the $L_1$ norm (sum of absolute values): $\|v\|_1 = \sum_i |v_i|$.

#### `sum(v)`, `dot(a, b)`, `norm_sq(v)`, `norm(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] constexpr auto sum(const Op& op);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto dot(const Lhs& lhs, const Rhs& rhs);

template<VectorOperand Op>
[[nodiscard]] constexpr auto norm_sq(const Op& op);

template<VectorOperand Op>
[[nodiscard]] inline auto norm(const Op& op);
```
Standard BLAS-1 reductions implemented with 8-way unrolled dependency-broken reduction trees.

---

## 5. Non-Temporal Streaming Stores (`assign_stream`)

Header: `<vec_math/vector_math.hpp>`

```cpp
template<typename T, typename Alloc, VectorOperand Src>
void assign_stream(std::vector<T, Alloc>& dst, const Src& src);
```

When evaluating vector expressions whose memory footprint exceeds the CPU's L3 cache (e.g. $> 16\text{ MB}$), standard cache-line writes issue a Read-For-Ownership (RFO) bus transaction, reading the old cache line from DRAM into cache before writing. This wastes 50% of the memory bus bandwidth.

`assign_stream`:
1. Evaluates expression `src` directly into `dst`.
2. Employs 256-bit AVX non-temporal streaming stores (`_mm256_stream_pd` / `_mm256_stream_ps`) or 128-bit streaming stores (`_mm_stream_pd` / `_mm_stream_ps`).
3. Handles unaligned pointer head/tail peeling gracefully.
4. Executes `_mm_sfence()` at loop completion to guarantee memory visibility.
5. Achieves up to **2.47x memory bandwidth speedup** (up to 42.08 GB/s on 1M `double` elements) on out-of-cache workloads.

---

## 6. Explicit Multi-Core Domain Parallel Helpers (OpenMP)

Header: `<vec_math/vector_math.hpp>`

Default operators (`+=`, `-=`, `<<=`, `|`, etc.) are intentionally single-threaded SIMD so that fine-grained element quadrature loops in DG solvers execute without thread pool contention. For global whole-domain state updates, explicit OpenMP-guarded functions are provided:

### `assign_par(dst, src)`
```cpp
template<typename T, typename Alloc, VectorOperand Src>
void assign_par(std::vector<T, Alloc>& dst, const Src& src);
```
Parallel evaluation across all CPU cores using `#pragma omp parallel for schedule(static)` when `_OPENMP` is enabled. Scales up to **6.10x speedup** on modern multi-core processors.

### `dot_par(u, v)`
```cpp
template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] inline auto dot_par(const Lhs& lhs, const Rhs& rhs);
```
Multi-core parallel inner product using OpenMP parallel reduction.

### `sum_par(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] inline auto sum_par(const Op& op);
```
Multi-core parallel summation reduction using OpenMP parallel reduction.

---

## 7. Configuration Macros & Annotations

| Macro | Description |
| :--- | :--- |
| `VEC_MATH_NO_EXCEPTIONS` | Define before including `vector_math.hpp` to replace exception throwing with `assert` + `std::abort()` (for `-fno-exceptions` and embedded environments). |
| `SYCL_EXTERNAL` | Defined as `__attribute__((sycl_device))` or `__declspec(sycl_device)` when compiled under a SYCL compiler (`__SYCL_DEVICE_ONLY__` or `SYCL_LANGUAGE_VERSION`); expands to empty on host compilers. |
| `VEC_MATH_DEVICE` | Decorator for device-only functions in SYCL compilation passes. |
| `VEC_MATH_HOST_DEVICE` | Decorator for functions callable on both host and device in SYCL compilation passes. |
| `VEC_MATH_FORCE_INLINE` | Compiler-specific forced inline specifier (`__forceinline` on MSVC, `always_inline` on GCC/Clang). |
| `VEC_MATH_HD_INLINE` | Combination of `VEC_MATH_HOST_DEVICE` and `VEC_MATH_FORCE_INLINE`. |
| `VEC_MATH_D_INLINE` | Combination of `VEC_MATH_DEVICE` and `VEC_MATH_FORCE_INLINE`. |
| `VEC_MATH_RESTRICT` | Pointer aliasing restrict modifier (`__restrict` on MSVC, `__restrict__` on GCC/Clang). |
| `VEC_MATH_PRAGMA_SIMD` | Compiler vectorization hint (`__pragma(loop(ivdep))` on MSVC, `#pragma clang loop vectorize(assume_safety)` on Clang, `#pragma GCC ivdep` on GCC, `#pragma omp simd` on OpenMP). |
| `VEC_MATH_HAS_X86_INTRIN` | Defined automatically on x86/x64 architectures when `<immintrin.h>` is available. |
| `VEC_MATH_NO_INTRINSICS` | Define before including `vector_math.hpp` to skip `<immintrin.h>`, reducing translation unit compile times by ~800 ms. |
| `VEC_MATH_HAS_AVX2_FMA` | Defined automatically only when the compiler enables both AVX2 and FMA flags (`-mavx2 -mfma` on GCC/Clang, `/arch:AVX2` on MSVC). |

---

## 8. Discontinuous Galerkin (DG) Math Core & Buffer Evaluation

Header: `<vec_math/vector_math.hpp>`

### Sub-Range Spans (`std::span`)
`std::span<T>` and `std::span<const T>` participate in all vector operations (`+`, `-`, `*`, `/`, `<<=`, `|`) with zero copy and zero heap allocations.
```cpp
std::vector<double> domain(1024, 0.0);
std::span<double> elem_u(domain.data(), 64);
std::span<double> elem_k(k1.data(), 64);

elem_u <<= elem_u + dt * elem_k; // Fused element update, zero copy
```

### Fixed-Size State Vectors (`std::array`)
`std::array<T, N>` (such as 4-variable conserved states: $[\rho, \rho u, \rho v, E]$) operates with zero heap allocations:
```cpp
std::array<double, 4> q = {1.0, 0.0, 0.0, 2.5};
std::array<double, 4> flux = {0.0, 1.0, 0.0, 3.0};
std::array<double, 4> out;
out <<= q + dt * flux; // Evaluated on stack with zero heap malloc
```

### Dense Matrix-Vector Operations (`matvec`)
Optimized for DG polynomial orders ($N_p = 4 \dots 36$) for volume, surface, and mass-inverse operators:
```cpp
// y = A * x (row-major)
matvec(A_matrix, x_vector, y_out, rows, cols);

// y = A^T * x (transposed)
matvec_transposed(A_matrix, x_vector, y_out, rows, cols);
```

### Raw Buffer Write (`eval_into`)
Evaluates any expression directly into raw host or device memory without container allocations:
```cpp
eval_into(raw_ptr, n, 2.0 * a + b);
```

### Deterministic Reductions (`dot_deterministic`, `sum_deterministic`)
Bit-identical strictly sequential reductions for regression runs:
```cpp
double s = sum_deterministic(v);
double d = dot_deterministic(u, v);
```
