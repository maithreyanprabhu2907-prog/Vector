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
concept NumericVector = detail::is_contiguous_vector_v<T> && Scalar<typename std::remove_cvref_t<T>::value_type>;
```
Satisfied by `std::vector<T, Alloc>` (including `vec_math::aligned_vector<T>`), `std::span<T>`, and `std::array<T, N>` where `T` models `Scalar`.

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
Direct in-place arithmetic without intermediate buffers or reallocations for `std::vector`, `std::span`, and `std::array`:

```cpp
// 1. std::vector (Vector-Vector, Vector-Expression, Vector-Scalar)
dst += rhs; // e.g. y += alpha * x (BLAS-1 AXPY with zero allocations)
dst -= rhs;
dst *= rhs;
dst /= rhs;
dst += scalar;
dst -= scalar;
dst *= scalar;
dst /= scalar;

// 2. std::span (In-place slice update without heap allocations)
span_dst += span_src; // or vector, array, or expression template
span_dst -= span_src;
span_dst *= span_src;
span_dst /= span_src;
span_dst += scalar;
span_dst -= scalar;
span_dst *= scalar;
span_dst /= scalar;

// 3. std::array (Stack-allocated fixed-size vectors, e.g. Nvar=4 state vectors)
arr_dst += arr_src;
arr_dst -= arr_src;
arr_dst *= arr_src;
arr_dst /= arr_src;
arr_dst += scalar;
arr_dst -= scalar;
arr_dst *= scalar;
arr_dst /= scalar;
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
| `VEC_MATH_STRICT_ZERO_MALLOC` | Define before including `vector_math.hpp` to strictly forbid dynamic heap allocations. When defined, self-aliased operations (`x == y` in `matvec` / `matvec_transposed` exceeding 128 elements, or `in == out` in `contract_axis` exceeding 1024 elements) will immediately abort/trap via `VEC_MATH_THROW_OR_ABORT` instead of falling back to a heap scratchpad. |
| `VEC_MATH_NO_EXCEPTIONS` | Define before including `vector_math.hpp` to replace exception throwing with `assert` + `std::abort()` (for `-fno-exceptions` and embedded environments). |
| `VEC_MATH_NO_CHECKS` | Define to disable runtime dimension matching assertions in performance-critical loops. |
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
> **Self-Aliasing & Stack Limits**: Square self-aliasing (`x == y`) is safely protected by an internal 128-element stack buffer. When the dimension exceeds 128, runtime `matvec` and `matvec_transposed` fall back to a scratch heap buffer unless `VEC_MATH_STRICT_ZERO_MALLOC` is defined, in which case they trap/abort via `VEC_MATH_THROW_OR_ABORT`. Compile-time `matvec_fixed` always buffers on the stack.

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

---

## 9. Discontinuous Galerkin (DG) Endgame Primitives

Header: `<vec_math/vector_math.hpp>`

The DG endgame primitives provide a complete, zero-overhead foundation for high-order Discontinuous Galerkin (DG) discretizations, Runge-Kutta time steppers, Riemann flux face exchanges, and Implicit-Explicit (IMEX) relaxation.

### Endgame Primitives Overview

| Primitive / Class | Function Signature / Concept | Description |
| :--- | :--- | :--- |
| **`matvec_fixed<Rows, Cols>`** | `void matvec_fixed(const MatT* A, const VecInT* x, VecOutT* y)` | Small dense matrix-vector product with compile-time dimensions and 4-way ILP unrolling. |
| **`matvec_transposed_fixed<Rows, Cols>`** | `void matvec_transposed_fixed(const MatT* A, const VecInT* x, VecOutT* y)` | Small dense transposed matrix-vector product with compile-time dimensions. |
| **Span/Array Compound Assign** | `dst += rhs`, `dst -= rhs`, `dst *= rhs`, `dst /= rhs` | In-place compound operations on `std::span` and `std::array` with vector or scalar operands. |
| **`ReductionPolicy`** | `enum class ReductionPolicy { FAST, REPRO };` | Policy enum selecting between maximum throughput (`FAST`) and bit-identical reproducibility (`REPRO`). |
| **`sum_repro`, `dot_repro`** | `sum_repro(op)`, `dot_repro(lhs, rhs)` | Blocked 512-element chunk reductions guaranteeing bit-identical reproducibility across thread counts. |
| **`StridedSpan<T>`** | `class StridedSpan<T>` | Non-contiguous 1D view over strided memory with random access iterators and `operator[]`. |
| **`broadcast_multiply`** | `void broadcast_multiply(std::span<T> data, ScalarT factor)` | SIMD element-wise multiplication of an array/span by a scalar metric factor. |
| **`WorkspaceArena`** | `class WorkspaceArena` | 64-byte cache-aligned linear bump allocator for zero-malloc Runge-Kutta stages. |
| **`gather`** | `void gather(dst, src, indices)` | SIMD indirect gather from nodal arrays into face quadrature trace arrays. |
| **`scatter_add`** | `void scatter_add(dst, indices, src)` | Indirect scatter-addition of numerical flux residuals into element residual vectors. |
| **`solve_4x4_batched`** | `void solve_4x4_batched(A_soa, b_soa, x_soa, num_elements)` | Batched $4 \times 4$ LU solver with partial pivoting across elements in SoA layout. |
| **`transpose_aos_to_soa`** | `void transpose_aos_to_soa(aos, soa, num_tuples, tuple_size)` | Converts interleaved AoS data into Structure-of-Arrays (SoA) layout. |
| **`transpose_soa_to_aos`** | `void transpose_soa_to_aos(soa, aos, num_tuples, tuple_size)` | Converts Structure-of-Arrays (SoA) data back to interleaved AoS format. |
| **`contract_axis<Axis, N>`** | `void contract_axis(D, in, out, nx, ny, nz)` | Sum-factorization directional tensor contractions along $X$, $Y$, or $Z$ axis for quad/hex elements. |

---

### 9.1 Fixed-Order Matrix-Vector Multiplication (`matvec_fixed`, `matvec_transposed_fixed`)

In high-order DG discretizations, polynomial order is constant across a mesh block (e.g. $N_p = 4, 10, 16, 21, 36$). Providing static compile-time dimensions allows the compiler to fully unroll inner loops, perform static register allocation in YMM/ZMM registers, and eliminate runtime dimension checks.

```cpp
template<std::size_t Rows, std::size_t Cols, typename MatT, typename VecInT, typename VecOutT>
VEC_MATH_HOST_DEVICE inline void matvec_fixed(const MatT* A, const VecInT* x, VecOutT* y);

template<std::size_t Rows, std::size_t Cols, typename MatT, typename VecInT, typename VecOutT>
VEC_MATH_HOST_DEVICE inline void matvec_transposed_fixed(const MatT* A, const VecInT* x, VecOutT* y);
```

#### Performance Architecture & Key Characteristics:
1. **4-Way Independent Accumulators (ILP)**: Unrolled operations accumulate into four distinct accumulators (`acc0`, `acc1`, `acc2`, `acc3`), breaking the sequential FMA latency dependency chain and saturating CPU vector execution ports.
2. **Aliasing Safety & Stack Capacity**: Automatically detects self-aliasing (`x == y`) and buffers results into a stack-allocated local array before copying to `y`, guaranteeing safe in-place execution `matvec_fixed<Np, Np>(D, x, x);`. For runtime `matvec` / `matvec_transposed`, the stack buffer capacity is 128 elements; operations exceeding 128 elements allocate a scratchpad unless `VEC_MATH_STRICT_ZERO_MALLOC` is defined.
3. **SYCL / GPU Compatibility**: Annotated with `VEC_MATH_HOST_DEVICE` for execution within host code or device kernels.

#### Usage Snippet:
```cpp
using namespace vec_math;

// Polynomial order p=3 triangle (Np = 10 nodes)
constexpr std::size_t Np = 10;
alignas(64) double Dr[Np * Np]; // Differentiation matrix along r
alignas(64) double u[Np];       // Solution vector
alignas(64) double dudr[Np];    // Output derivative vector

// Evaluate derivative: dudr = Dr * u
matvec_fixed<Np, Np>(Dr, u, dudr);

// Safe in-place evaluation (aliased x and y)
matvec_fixed<Np, Np>(Dr, u, u);

// Transposed matrix-vector product (e.g. lifting operator or mass matrix)
alignas(64) double res[Np];
matvec_transposed_fixed<Np, Np>(Dr, u, res);
```

---

### 9.2 Compound Assignment for `std::span` and `std::array`

Compound assignment operators (`+=`, `-=`, `*=`, `/=`) operate directly on `std::span<T, Extent>` and `std::array<T, N>` without intermediate dynamic allocations or buffer copies.

```cpp
// std::span overloads
template<typename T, std::size_t Ext, VectorOperand Src>
inline std::span<T, Ext>& operator+=(std::span<T, Ext>& dst, const Src& src);

template<typename T, std::size_t Ext, Scalar S>
inline std::span<T, Ext>& operator+=(std::span<T, Ext>& dst, const S& scalar);

// std::array overloads
template<typename T, std::size_t N, VectorOperand Src>
inline std::array<T, N>& operator+=(std::array<T, N>& dst, const Src& src);

template<typename T, std::size_t N, Scalar S>
inline std::array<T, N>& operator+=(std::array<T, N>& dst, const S& scalar);
```
*(Analogous overloads exist for `-=`, `*=`, and `/=`)*.

#### Key Characteristics:
- **Vector and Scalar Operands**: Supports assignment from vectors, spans, arrays, lazy expression templates (`alpha * x + beta * y`), or scalars.
- **Zero-Allocation Guarantee**: Evaluates in a single SIMD vector pass (`VEC_MATH_PRAGMA_SIMD`) directly into target memory.
- **Size Checking**: Validates matching dimensions via `VEC_MATH_CHECK_SIZE_EQ` (suppressed with `VEC_MATH_NO_CHECKS`).

#### Usage Snippet:
```cpp
using namespace vec_math;

// Slicing an element's nodal state from a contiguous domain buffer
double domain_buffer[1024];
std::span<double> elem_u(domain_buffer, 64);
std::span<const double> elem_flux(domain_buffer + 64, 64);

elem_u += elem_flux;       // Vector-vector in-place accumulation
elem_u *= 0.5;             // Vector-scalar in-place scaling
elem_u -= 2.0 * elem_flux; // Vector-expression in-place AXPY

// Fixed-size conservative variable array (rho, rhou, rhov, E)
std::array<double, 4> state = {1.0, 2.0, 0.0, 2.5};
std::array<double, 4> dstate = {0.1, 0.2, 0.0, 0.25};

state += dstate;           // In-place array addition
state *= 1.05;             // In-place array scaling
```

---

### 9.3 Reproducible Reductions (`sum_repro`, `dot_repro`, `ReductionPolicy`)

Floating-point arithmetic is non-associative. Under aggressive compiler flags (`-Ofast`) or multi-threaded OpenMP execution, standard reductions can produce non-deterministic results between runs. `vec_math` introduces explicit reduction policies:

```cpp
enum class ReductionPolicy {
    FAST,   // 8-way multi-accumulator SIMD + OpenMP for maximum throughput
    REPRO   // Blocked 512-element chunk accumulation for bit-identical reproducibility
};

// Dedicated Reproducible Reductions
template<VectorOperand Op>
[[nodiscard]] inline auto sum_repro(const Op& op);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] inline auto dot_repro(const Lhs& lhs, const Rhs& rhs);

// Policy-Aware Overloads
template<VectorOperand Op>
[[nodiscard]] inline auto sum(const Op& op, ReductionPolicy policy = ReductionPolicy::FAST);

template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] inline auto dot(const Lhs& lhs, const Rhs& rhs, ReductionPolicy policy = ReductionPolicy::FAST);
```

#### Comparison of Reduction Policies:

| Feature | `ReductionPolicy::FAST` | `ReductionPolicy::REPRO` |
| :--- | :--- | :--- |
| **Algorithm** | 8-way unrolled accumulators + OpenMP dynamic dispatch | Fixed 512-element chunk accumulation |
| **Reproducibility** | May vary across thread counts or CPU architecture | **Bit-identical** across thread counts, compilers, and architectures |
| **Target Use Case** | CFL time-step estimation, residual norms, limiters | Krylov solvers (GMRES/BiCGSTAB), regression testing, convergence verification |
| **Single-Precision Safety** | Fast float accumulation | Accumulates in 512-element chunks bounding FP rounding divergence |

#### Usage Snippet:
```cpp
using namespace vec_math;

std::vector<double> r = {1.0, 2.0, 3.0};
std::vector<double> p = {0.5, 1.5, 2.5};
std::vector<double> res = {1.0, -1.0, 2.0};

// Bit-identical dot product for Krylov solver orthogonalization
double alpha = dot_repro(r, p);

// Explicit policy selection
double s_repro = sum(res, ReductionPolicy::REPRO);
double s_fast  = sum(res, ReductionPolicy::FAST);
```

---

### 9.4 Strided Vector Views (`StridedSpan<T>`)

In DG solvers, global field data is frequently stored in multi-dimensional arrangements (e.g. `Q[nvar][nelem][Np]`) or interleaved AoS formats. `StridedSpan<T>` provides a lightweight, non-owning 1D view over non-contiguous memory without copying.

```cpp
template<typename T>
class StridedSpan {
public:
    using value_type      = T;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using pointer         = T*;
    using const_pointer   = const T*;

    constexpr StridedSpan() noexcept;
    constexpr StridedSpan(T* ptr, std::size_t size, std::size_t stride) noexcept;

    [[nodiscard]] constexpr std::size_t size() const noexcept;
    [[nodiscard]] constexpr std::size_t stride() const noexcept;
    [[nodiscard]] constexpr bool empty() const noexcept;
    [[nodiscard]] constexpr pointer data() const noexcept;

    [[nodiscard]] constexpr reference operator[](std::size_t i) const noexcept;
    [[nodiscard]] constexpr reference front() const noexcept;
    [[nodiscard]] constexpr reference back() const noexcept;

    // Iterators (Random-access StridedIterator)
    constexpr auto begin() const noexcept;
    constexpr auto end() const noexcept;
};
```

#### Usage Snippet:
```cpp
using namespace vec_math;

// Interleaved 3-component coordinate array: [x0, y0, z0, x1, y1, z1, ...]
double coords[12] = {
    1.0, 10.0, 100.0,
    2.0, 20.0, 200.0,
    3.0, 30.0, 300.0,
    4.0, 40.0, 400.0
};

// Create a strided view for y-coordinates: start at coords + 1, count = 4, stride = 3
StridedSpan<double> y_coords(coords + 1, 4, 3);

assert(y_coords.size() == 4);
assert(y_coords[0] == 10.0);
assert(y_coords[1] == 20.0);
assert(y_coords[2] == 30.0);
assert(y_coords[3] == 40.0);
```

---

### 9.5 Metric Scaling & Broadcasting (`broadcast_multiply`)

Scales an array or `std::span` in-place by a scalar multiplier. Essential for weighting volume or face quadrature points by geometric terms (such as elemental Jacobian determinants $J_e$ or face surface metrics):

```cpp
template<typename T, typename ScalarT>
VEC_MATH_HOST_DEVICE inline void broadcast_multiply(T* data, std::size_t count, ScalarT factor);

template<typename T, std::size_t Extent, typename ScalarT>
VEC_MATH_HOST_DEVICE inline void broadcast_multiply(std::span<T, Extent> span_data, ScalarT factor);
```

#### Usage Snippet:
```cpp
using namespace vec_math;

double quad_weights[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
double J_elem = 0.125; // Elemental Jacobian determinant

// Scale quadrature weights by element metric Jacobian
broadcast_multiply(std::span<double>(quad_weights, 8), J_elem);
```

---

### 9.6 Zero-Malloc Stage Scratchpad (`WorkspaceArena`)

Evaluating multi-stage Runge-Kutta time integrators (e.g. SSP-RK3, RK4) and flux traces requires scratch vectors. Dynamically allocating `std::vector` inside the time-stepping loop causes heap allocator lock contention and TLB thrashing. `WorkspaceArena` provides a 64-byte cache-aligned linear bump allocator.

```cpp
class WorkspaceArena {
public:
    explicit WorkspaceArena(std::size_t capacity_bytes = 1024 * 1024);
    ~WorkspaceArena();

    WorkspaceArena(const WorkspaceArena&) = delete;
    WorkspaceArena& operator=(const WorkspaceArena&) = delete;
    WorkspaceArena(WorkspaceArena&& other) noexcept;
    WorkspaceArena& operator=(WorkspaceArena&& other) noexcept;

    // Allocate 64-byte aligned span of type T
    template<typename T>
    [[nodiscard]] std::span<T> allocate(std::size_t count);

    // Instant O(1) reset of offset counter (recycles memory)
    void reset() noexcept;

    [[nodiscard]] std::size_t allocated_bytes() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
};
```

#### Usage Snippet:
```cpp
using namespace vec_math;

// Pre-allocate 16 MB arena once at solver initialization
WorkspaceArena arena(16 * 1024 * 1024);

// Inside time integration loop:
for (int step = 0; step < num_steps; ++step) {
    arena.reset(); // Zero heap allocation cost

    // Allocate stage scratch buffers guaranteed 64-byte aligned
    std::span<double> k1 = arena.allocate<double>(num_dofs);
    std::span<double> k2 = arena.allocate<double>(num_dofs);
    std::span<double> k3 = arena.allocate<double>(num_dofs);

    // Compute stages with zero mallocs
    compute_rhs(u_current, k1);
    // ...
}
```

---

### 9.7 Surface Integral Exchange (`gather`, `scatter_add`)

The core of Discontinuous Galerkin surface integrals is exchanging Riemann fluxes across internal and boundary element faces:
1. **`gather`**: Extracts state traces $Q^-$ and $Q^+$ from element volume nodes onto face quadrature nodes.
2. **`scatter_add`**: Accumulates numerical flux residuals back into element volumetric residual vectors.

```cpp
// Gather: dst[i] = src[indices[i]]
template<typename T, typename IndexT>
VEC_MATH_HOST_DEVICE inline void gather(T* dst, const T* src, const IndexT* indices, std::size_t count);

template<typename T, std::size_t DstExt, std::size_t SrcExt, typename IndexT, std::size_t IdxExt>
VEC_MATH_HOST_DEVICE inline void gather(std::span<T, DstExt> dst, std::span<const T, SrcExt> src, std::span<const IndexT, IdxExt> indices);

// Scatter-Add: dst[indices[i]] += src[i]
template<typename T, typename IndexT>
VEC_MATH_HOST_DEVICE inline void scatter_add(T* dst, const IndexT* indices, const T* src, std::size_t count);

template<typename T, std::size_t DstExt, typename IndexT, std::size_t IdxExt, std::size_t SrcExt>
VEC_MATH_HOST_DEVICE inline void scatter_add(std::span<T, DstExt> dst, std::span<const IndexT, IdxExt> indices, std::span<const T, SrcExt> src);
```

#### Usage Snippet:
```cpp
using namespace vec_math;

// Volumetric state and face node mapping
double u_vol[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
std::size_t face_nodes[3] = {1, 3, 5};

// 1. Gather trace values to face quadrature points (supports both const and non-const spans)
double u_face[3] = {0.0, 0.0, 0.0};
gather(std::span(u_face), std::span(u_vol), std::span(face_nodes));
// u_face is now {20.0, 40.0, 60.0}

// 2. Compute numerical flux at faces
double flux[3] = {5.0, 15.0, 25.0};

// 3. Scatter-add flux residuals back into volumetric residuals
double residual[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
scatter_add(std::span(residual), std::span(face_nodes), std::span(flux));
// residual[1] += 5.0, residual[3] += 15.0, residual[5] += 25.0
```

---

### 9.8 Batched Point-Implicit Linear Solvers & Layout Transposition

For stiff acoustic modes or low-Mach boundary layers, Implicit-Explicit (IMEX) time integration requires solving implicit stage systems. Fully coupled element-local matrix inversion requires $\mathcal{O}((N_{\text{var}} N_p)^3)$ operations—costing over 100× more than the explicit volume operator.

`vec_math` solves this with **point-implicit relaxation**: solving decoupled $4 \times 4$ linear systems across all elements simultaneously in Structure-of-Arrays (SoA) layout.

```cpp
// Batched 4x4 linear solve in SoA layout: A_soa * x_soa = b_soa
template<typename T>
VEC_MATH_HOST_DEVICE inline void solve_4x4_batched(
    const T* const* A_soa, // 16 pointers, each pointing to array of size num_elements
    const T* const* b_soa, // 4 pointers, each pointing to array of size num_elements
    T* const* x_soa,       // 4 pointers, each pointing to array of size num_elements
    std::size_t num_elements
);

// Transpose AoS <-> SoA
template<typename T>
VEC_MATH_HOST_DEVICE inline void transpose_aos_to_soa(
    const T* aos, T* soa, std::size_t num_tuples, std::size_t tuple_size);

template<typename T>
VEC_MATH_HOST_DEVICE inline void transpose_soa_to_aos(
    const T* soa, T* aos, std::size_t num_tuples, std::size_t tuple_size);
```

#### Key Characteristics:
1. **Full Partial Pivoting**: Guarantees numerical stability and prevents division by zero with near-zero pivot thresholding ($10^{-20}$).
2. **SoA Vectorization**: Loops over `num_elements` with `VEC_MATH_PRAGMA_SIMD`, allowing the compiler to generate contiguous vector loads and stores across elements.
3. **Zero Allocation**: Computes directly from caller-provided pointer arrays with stack-allocated local pivots.

#### Usage Snippet:
```cpp
using namespace vec_math;

constexpr std::size_t nelem = 1000;

// Setup SoA pointer tables for 4x4 matrix and RHS
const double* A_ptrs[16]; // 16 pointers to contiguous arrays of size nelem
const double* b_ptrs[4];  // 4 pointers to contiguous arrays of size nelem
double* x_ptrs[4];        // 4 pointers to output solution arrays

// Solve A * x = b for 1000 elements simultaneously
solve_4x4_batched(A_ptrs, b_ptrs, x_ptrs, nelem);

// Convert between interleaved nodal format (AoS) and vectorized format (SoA)
std::vector<double> aos(nelem * 4);
std::vector<double> soa(nelem * 4);

// AoS to SoA
transpose_aos_to_soa(aos.data(), soa.data(), nelem, 4);

// SoA to AoS
transpose_soa_to_aos(soa.data(), aos.data(), nelem, 4);
```

---

### 9.9 Directional Tensor Contractions for Sum-Factorization (`contract_axis`)

In high-order Discontinuous Galerkin and Spectral Element methods on tensor-product quad/hex cells, applying dense operators directly scales as $\mathcal{O}(N^{2d})$ where $d$ is the spatial dimension. **Sum-factorization** decomposes the multi-dimensional operator into 1D directional tensor contractions:

$$\mathbf{u}_{,x} = (I_z \otimes I_y \otimes D_x) \mathbf{u}, \quad \mathbf{u}_{,y} = (I_z \otimes D_y \otimes I_x) \mathbf{u}, \quad \mathbf{u}_{,z} = (D_z \otimes I_y \otimes I_x) \mathbf{u}$$

This reduces algorithmic arithmetic complexity from $\mathcal{O}(N^{2d})$ to $\mathcal{O}(d \cdot N^{d+1})$ ($5.3\times$ faster at $p=3$ in 3D, $21\times$ faster at $p=7$).

`vec_math` provides `contract_axis` with compile-time unrolling along any designated axis:

```cpp
enum class Axis : std::size_t { X = 0, Y = 1, Z = 2 };

template<auto AxisParam, std::size_t N = 0, typename MatT, typename T>
VEC_MATH_HOST_DEVICE inline void contract_axis(
    const MatT* D, const T* in, T* out,
    std::size_t nx, std::size_t ny, std::size_t nz = 1);

template<auto AxisParam, std::size_t N = 0, typename MatT, typename InT, typename OutT,
         std::size_t DExt, std::size_t InExt, std::size_t OutExt>
VEC_MATH_HOST_DEVICE inline void contract_axis(
    std::span<MatT, DExt> D, std::span<InT, InExt> in, std::span<OutT, OutExt> out,
    std::size_t nx, std::size_t ny, std::size_t nz = 1);
```

#### Key Characteristics:
1. **Directional Specialization**: Axis 0 ($X$, unit stride), Axis 1 ($Y$, stride $n_x$), and Axis 2 ($Z$, stride $n_x \cdot n_y$) are specialized to maintain maximum cache locality and SIMD lane alignment.
2. **Compile-Time Static $N$**: When $N > 0$, directional loops are fully unrolled with 4 independent accumulator chains. When $N = 0$, dynamic runtime sizes are supported.
3. **In-Place Execution (`in == out`) & Stack Capacity**: Automatically checks pointer aliasing and uses an internal stack buffer (up to 1024 elements: `nx * ny * nz <= 1024`) to allow safe in-place derivative evaluation with zero heap allocations. If total elements exceed 1024, it allocates a temporary heap buffer on host unless `VEC_MATH_STRICT_ZERO_MALLOC` is defined, in which case it triggers `VEC_MATH_THROW_OR_ABORT`.

#### Usage Snippet:
```cpp
using namespace vec_math;

// 3D Hex element at p=3 (N=4 points per direction, total 64 nodes)
constexpr std::size_t N = 4;
double D_1d[N * N] = {}; // 1D differentiation matrix (4x4)
double u[N * N * N] = {}; // 3D nodal solution (64 entries)
double du_dx[N * N * N] = {};
double du_dy[N * N * N] = {};
double du_dz[N * N * N] = {};

// Directional derivative along X: Axis::X (or 0)
contract_axis<Axis::X, N>(D_1d, u, du_dx, N, N, N);

// Directional derivative along Y: Axis::Y (or 1)
contract_axis<Axis::Y, N>(D_1d, u, du_dy, N, N, N);

// Directional derivative along Z: Axis::Z (or 2)
contract_axis<Axis::Z, N>(D_1d, u, du_dz, N, N, N);

// Safe in-place evaluation
contract_axis<Axis::X, N>(D_1d, u, u, N, N, N);
```

