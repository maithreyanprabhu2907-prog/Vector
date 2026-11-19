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

### `DeviceSpan<T>`
```cpp
template<typename T>
concept DeviceSpan = detail::is_vec_span_v<T>;
```
Satisfied by `vec_math::vec_span<T>` and `vec_math::device_view<T>`.

### `VecExpression<T>`
```cpp
template<typename T>
concept VecExpression = detail::is_vec_expr_v<T>;
```
Satisfied by any expression template node inheriting from `detail::ExprBase` (`BinExpr`, `UnaryExpr`, `TernaryExpr`).

### `VectorOperand<T>`
```cpp
template<typename T>
concept VectorOperand = NumericVector<T> || DeviceSpan<T> || VecExpression<T>;
```
General concept representing any operand participating in vector mathematics.

### `sycl_compatible<T>`
```cpp
template<typename T>
concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;
```
Ensures that a type can be safely captured by-value inside a SYCL device kernel lambda (`cgh.parallel_for`).

---

## 2. Views & Spans (`vec_span` / `device_view`)

### Class Template `vec_math::vec_span<T>`
A non-owning, TriviallyCopyable view over contiguous memory. Useful for host vectors, raw pointers, SYCL Unified Shared Memory (USM), and SYCL buffer accessors.

```cpp
template<typename T>
class vec_span;

template<typename T>
using device_view = vec_span<T>;
```

#### Member Types
- `value_type`: `std::remove_cv_t<T>`
- `element_type`: `T`
- `pointer`: `T*`
- `const_pointer`: `const T*`
- `reference`: `T&`
- `const_reference`: `const T&`
- `size_type`: `std::size_t`
- `difference_type`: `std::ptrdiff_t`

#### Constructors
```cpp
constexpr vec_span() noexcept;
constexpr vec_span(T* ptr, std::size_t count) noexcept;

template<typename Alloc>
constexpr vec_span(std::vector<T, Alloc>& v) noexcept;

template<typename Alloc>
constexpr vec_span(const std::vector<std::remove_const_t<T>, Alloc>& v) noexcept;
```

#### Member Functions
- `constexpr T* data() const noexcept`: Returns pointer to memory.
- `constexpr std::size_t size() const noexcept`: Returns element count.
- `constexpr bool empty() const noexcept`: Returns `true` if `size() == 0`.
- `constexpr reference operator[](std::size_t i) const noexcept`: Unchecked element access.
- `constexpr pointer begin() const noexcept`: Beginning iterator.
- `constexpr pointer end() const noexcept`: Ending iterator.

#### Compound Assignment Operators on `vec_span<T>`
*(Available when `!std::is_const_v<T>`)*
```cpp
template<VectorOperand Rhs>
constexpr vec_span& operator+=(const Rhs& rhs);

template<Scalar S>
constexpr vec_span& operator+=(const S& scalar) noexcept;

template<VectorOperand Rhs>
constexpr vec_span& operator-=(const Rhs& rhs);

template<Scalar S>
constexpr vec_span& operator-=(const S& scalar) noexcept;

template<VectorOperand Rhs>
constexpr vec_span& operator*=(const Rhs& rhs);

template<Scalar S>
constexpr vec_span& operator*=(const S& scalar) noexcept;

template<VectorOperand Rhs>
constexpr vec_span& operator/=(const Rhs& rhs);

template<Scalar S>
constexpr vec_span& operator/=(const S& scalar) noexcept;

template<VectorOperand Src>
constexpr vec_span& operator<<=(const Src& src);
```

#### Factory Functions
```cpp
template<Scalar T>
[[nodiscard]] constexpr vec_span<T> make_vec_span(T* ptr, std::size_t n) noexcept;

template<Scalar T>
[[nodiscard]] constexpr vec_span<const T> make_vec_span(const T* ptr, std::size_t n) noexcept;

template<Scalar T>
[[nodiscard]] constexpr device_view<T> make_device_view(T* ptr, std::size_t n) noexcept;

template<Scalar T>
[[nodiscard]] constexpr device_view<const T> make_device_view(const T* ptr, std::size_t n) noexcept;
```

---

## 3. Arithmetic Operators

Overloaded operators are in inline namespace `vec_math::ops`.

### Binary Operators (`+`, `-`, `*`, `/`)
Construct a lazy `BinExpr` without performing allocations:
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

// (Similar signatures exist for -, *, /)
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

## 4. Evaluation & Assignment

### Pre-Allocated Evaluation (`<<=`)
```cpp
template<typename T, typename Alloc, VectorOperand Src>
std::vector<T, Alloc>& operator<<=(std::vector<T, Alloc>& dst, const Src& src);
```
Evaluates `src` directly into `dst` using 8x/4x unrolled SIMD loops. Requires `dst.capacity() >= src.size()`. **Zero allocations.**

### Free Function `assign()`
```cpp
template<typename T, typename Alloc, VectorOperand Src>
void assign(std::vector<T, Alloc>& dst, const Src& src);
```
Equivalent to `dst <<= src`.

### In-Place Compound Assignment on `std::vector`
```cpp
dst += rhs;    dst += scalar;
dst -= rhs;    dst -= scalar;
dst *= rhs;    dst *= scalar;
dst /= rhs;    dst /= scalar;
```
Direct in-place arithmetic without intermediate buffers.

### Rvalue Recycling Operators
When operating on temporary or moved `std::vector&&`:
```cpp
auto res = 2.0 * std::move(a) + b;
```
The internal memory buffer of `a` is recycled for the result. **Zero allocations.**

---

## 5. Reduction Utilities

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
Calculates the dot product $\sum a_i \cdot b_i$ using 8 independent accumulators.

### `norm_sq(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] constexpr auto norm_sq(const Op& op);
```
Calculates squared Euclidean norm: `dot(v, v)`.

### `norm(v)`
```cpp
template<VectorOperand Op>
[[nodiscard]] inline auto norm(const Op& op);
```
Calculates Euclidean L2 norm: $\sqrt{\text{dot}(v, v)}$.

---

## 6. Device Evaluation Helpers (SYCL)

```cpp
template<typename Dst, VectorOperand Expr>
VEC_MATH_HD_INLINE constexpr void device_eval(Dst& dst, const Expr& expr);

template<typename Dst, VectorOperand Expr>
VEC_MATH_HD_INLINE constexpr void device_eval(Dst& dst, const Expr& expr, std::size_t i) noexcept;

template<typename T, VectorOperand Expr>
VEC_MATH_HD_INLINE constexpr void device_eval(T* VEC_MATH_RESTRICT dst, const Expr& expr, std::size_t i) noexcept;
```
Evaluates an expression element inside a SYCL work-item without host-device synchronization or dynamic allocations.

---

## 7. Configuration Macros

| Macro | Description |
| :--- | :--- |
| `VEC_MATH_NO_EXCEPTIONS` | Define to replace exceptions with `assert` + `std::abort()` (useful for `-fno-exceptions` and embedded environments). |
| `SYCL_EXTERNAL` | Declared as `__attribute__((sycl_device))` or `__declspec(sycl_device)` when compiling with a SYCL compiler. |
| `VEC_MATH_FORCE_INLINE` | Inlining specifier (`__forceinline` on MSVC, `always_inline` on GCC/Clang). |
| `VEC_MATH_RESTRICT` | Pointer aliasing restrict modifier (`__restrict` / `__restrict__`). |
| `VEC_MATH_PRAGMA_SIMD` | Compiler vectorization hint (`loop(ivdep)`, `clang loop vectorize`, `GCC ivdep`, `omp simd`). |
