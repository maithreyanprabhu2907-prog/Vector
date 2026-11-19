# API Reference

Complete API reference for the `vec_math` library. All entities reside in namespace `vec_math` or inline namespace `vec_math::ops`.

---

## 1. Concepts

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
Satisfied by `std::vector<T, Alloc>` where `T` models `Scalar`.

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

## 2. Arithmetic Operators

Overloaded operators reside in inline namespace `vec_math::ops` and are automatically available when using namespace `vec_math`.

### Binary Operators (`+`, `-`, `*`, `/`)
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

### Unary Operators
```cpp
template<VectorOperand Arg>
[[nodiscard]] constexpr auto operator-(const Arg& arg);

template<VectorOperand Arg>
[[nodiscard]] constexpr auto operator+(const Arg& arg);

template<VectorOperand Arg>
[[nodiscard]] constexpr auto abs(const Arg& arg);
```

### Fused Multiply-Add (`fma`)
Constructs a ternary expression representing $a \times b + c$:
```cpp
template<VectorOperand A, VectorOperand B, VectorOperand C>
[[nodiscard]] constexpr auto fma(const A& a, const B& b, const C& c);

template<VectorOperand A, VectorOperand B, Scalar S>
[[nodiscard]] constexpr auto fma(const A& a, const B& b, const S& c);

template<VectorOperand A, Scalar S, VectorOperand C>
[[nodiscard]] constexpr auto fma(const A& a, const S& b, const C& c);
```

---

## 3. Evaluation & Assignment

### Pre-Allocated Evaluation (`<<=`)
```cpp
template<typename T, typename Alloc, VectorOperand Src>
std::vector<T, Alloc>& operator<<=(std::vector<T, Alloc>& dst, const Src& src);
```
Evaluates `src` directly into `dst` using 8x/4x unrolled SIMD loops. Requires `dst.capacity() >= src.size()`. **Zero dynamic heap allocations.**

### Free Function `assign()`
```cpp
template<typename T, typename Alloc, VectorOperand Src>
void assign(std::vector<T, Alloc>& dst, const Src& src);
```
Equivalent to `dst <<= src`.

### In-Place Compound Assignment on `std::vector`
```cpp
// Vector-Vector
dst += rhs;
dst -= rhs;
dst *= rhs;
dst /= rhs;

// Vector-Scalar
dst += scalar;
dst -= scalar;
dst *= scalar;
dst /= scalar;
```
Direct in-place arithmetic without intermediate buffers or reallocations.

### Rvalue Buffer Recycling
When operating on temporary or moved `std::vector&&`:
```cpp
auto res = 2.0 * std::move(a) + b;
```
The internal heap buffer of `a` is reused for the result, bypassing new allocations. **Zero dynamic heap allocations.**

---

## 4. Reduction Utilities

Header: `<vec_math/vector_math.hpp>`

### `sum(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] constexpr auto sum(const Op& op);
```
Calculates the sum of all elements using an 8-way dependency-broken reduction tree.

### `dot(a, b)`
```cpp
template<VectorOperand Lhs, VectorOperand Rhs>
[[nodiscard]] constexpr auto dot(const Lhs& lhs, const Rhs& rhs);
```
Calculates the dot product $\sum a_i \cdot b_i$ using 8 independent accumulators for high instruction-level parallelism.

### `norm_sq(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] constexpr auto norm_sq(const Op& op);
```
Calculates the squared Euclidean norm: $\sum v_i^2 = \text{dot}(v, v)$.

### `norm(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] inline auto norm(const Op& op);
```
Calculates the Euclidean L2 norm: $\sqrt{\text{norm\_sq}(v)}$.

---

## 5. Configuration Macros & Annotations

| Macro | Description |
| :--- | :--- |
| `VEC_MATH_NO_EXCEPTIONS` | Define before including `vector_math.hpp` to replace exception throwing with `assert` + `std::abort()` (useful for `-fno-exceptions` and embedded environments). |
| `SYCL_EXTERNAL` | Defined as `__attribute__((sycl_device))` or `__declspec(sycl_device)` when compiled under a SYCL compiler (`__SYCL_DEVICE_ONLY__` or `SYCL_LANGUAGE_VERSION`); expands to empty on host compilers. |
| `VEC_MATH_DEVICE` | Decorator for device-only functions in SYCL compilation passes. |
| `VEC_MATH_HOST_DEVICE` | Decorator for functions callable on both host and device in SYCL compilation passes. |
| `VEC_MATH_FORCE_INLINE` | Compiler-specific forced inline specifier (`__forceinline` on MSVC, `always_inline` on GCC/Clang). |
| `VEC_MATH_HD_INLINE` | Combination of `VEC_MATH_HOST_DEVICE` and `VEC_MATH_FORCE_INLINE`. |
| `VEC_MATH_D_INLINE` | Combination of `VEC_MATH_DEVICE` and `VEC_MATH_FORCE_INLINE`. |
| `VEC_MATH_RESTRICT` | Pointer aliasing restrict modifier (`__restrict` on MSVC, `__restrict__` on GCC/Clang). |
| `VEC_MATH_PRAGMA_SIMD` | Compiler vectorization hint (`__pragma(loop(ivdep))` on MSVC, `#pragma clang loop vectorize(enable)` on Clang, `#pragma GCC ivdep` on GCC, `#pragma omp simd` on OpenMP). |
