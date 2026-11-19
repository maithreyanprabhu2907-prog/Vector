#pragma once

/*
 * Author: Maithreyan(maithreyanprabu2907-prog)
 * Extended & Optimized for Zero-Allocation Vector Math & SYCL Device Compatibility
 *
 * Purpose:
 *   Provide a modern, expressive, intensely optimized vector math library for std::vector
 *   and device-compatible spans/views.
 *   Operator overloading keeps mathematical expressions short, readable, and fully fused.
 *
 * Goals:
 *      - Modern C++20 style, zero-allocation throughout all operations.
 *      - SIMD-vectorized loops with loop unrolling (8x/4x) and restrict pointers.
 *      - TriviallyCopyable expressions and views for seamless SYCL device kernel capture.
 *      - SYCL device annotations without requiring SYCL header dependencies.
 *      - Extended math utilities: dot product, sum, norm, fma, abs.
 */

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// ============================================================================
// SYCL & Device Compatibility Annotations (WITHOUT including SYCL headers)
// ============================================================================

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

// ============================================================================
// Inlining, Restrict, and Vectorization Pragmas
// ============================================================================

#if defined(_MSC_VER)
#define VEC_MATH_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define VEC_MATH_FORCE_INLINE inline __attribute__((always_inline))
#else
#define VEC_MATH_FORCE_INLINE inline
#endif

#define VEC_MATH_HD_INLINE VEC_MATH_HOST_DEVICE VEC_MATH_FORCE_INLINE
#define VEC_MATH_D_INLINE VEC_MATH_DEVICE VEC_MATH_FORCE_INLINE

#if defined(__GNUC__) || defined(__clang__)
#define VEC_MATH_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define VEC_MATH_RESTRICT __restrict
#else
#define VEC_MATH_RESTRICT
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define VEC_MATH_PRAGMA_SIMD __pragma(loop(ivdep))
#elif defined(__clang__)
#define VEC_MATH_PRAGMA_SIMD _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define VEC_MATH_PRAGMA_SIMD _Pragma("GCC ivdep")
#elif defined(_OPENMP)
#define VEC_MATH_PRAGMA_SIMD _Pragma("omp simd")
#else
#define VEC_MATH_PRAGMA_SIMD
#endif

#if defined(__SYCL_DEVICE_ONLY__)
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) ((void)0)
#elif defined(VEC_MATH_NO_EXCEPTIONS)
#include <cstdlib>
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) do { assert(false && (msg)); std::abort(); } while (false)
#else
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) throw exc_type(msg)
#endif

namespace vec_math {

    // Forward declaration of vec_span / device_view
    template<typename T>
    class vec_span;

    template<typename T>
    using device_view = vec_span<T>;

    namespace detail {
        template<typename T>
        struct is_std_vector : std::false_type {};

        template<typename T, typename Alloc>
        struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_vector_v = is_std_vector<std::remove_cvref_t<T>>::value;

        template<typename T>
        struct is_vec_span : std::false_type {};

        template<typename T>
        struct is_vec_span<vec_span<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_vec_span_v = is_vec_span<std::remove_cvref_t<T>>::value;

        template<typename T>
        struct is_valid_numeric : std::bool_constant<
            std::is_arithmetic_v<std::remove_cvref_t<T>> && !std::is_same_v<std::remove_cvref_t<T>, bool>
        > {};

        template<typename T>
        inline constexpr bool is_valid_numeric_v = is_valid_numeric<T>::value;

        struct ExprBase {};

        template<typename T>
        inline constexpr bool is_vec_expr_v = std::is_base_of_v<ExprBase, std::remove_cvref_t<T>>;
    }

    template<typename T>
    concept Scalar = detail::is_valid_numeric_v<T>;

    template<typename T>
    concept NumericVector = detail::is_std_vector_v<T> && Scalar<typename std::remove_cvref_t<T>::value_type>;

    template<typename T>
    concept DeviceSpan = detail::is_vec_span_v<T>;

    template<typename T>
    concept VecExpression = detail::is_vec_expr_v<T>;

    template<typename T>
    concept VectorOperand = NumericVector<T> || DeviceSpan<T> || VecExpression<T>;

    // SYCL Compatibility Concept
    template<typename T>
    concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;

    template<typename T>
    inline constexpr bool is_sycl_compatible_v = sycl_compatible<T>;

    // ============================================================================
    // Device-Compatible Span / View (USM / Device / Host memory wrapper)
    // ============================================================================

    template<typename T>
    class vec_span {
    public:
        using value_type = std::remove_cv_t<T>;
        using element_type = T;
        using pointer = T*;
        using const_pointer = const T*;
        using reference = T&;
        using const_reference = const T&;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = T*;
        using const_iterator = const T*;

    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;

    public:
        constexpr vec_span() noexcept = default;
        constexpr vec_span(const vec_span&) noexcept = default;
        constexpr vec_span(vec_span&&) noexcept = default;
        constexpr vec_span& operator=(const vec_span&) noexcept = default;
        constexpr vec_span& operator=(vec_span&&) noexcept = default;
        ~vec_span() = default;

        VEC_MATH_HD_INLINE constexpr vec_span(T* ptr, std::size_t count) noexcept
            : data_(ptr), size_(count) {}

        template<typename Alloc> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE constexpr vec_span(std::vector<value_type, Alloc>& v) noexcept
            : data_(v.data()), size_(v.size()) {}

        template<typename Alloc>
        VEC_MATH_HD_INLINE constexpr vec_span(const std::vector<value_type, Alloc>& v) noexcept
            : data_(v.data()), size_(v.size()) {}

        template<typename U> requires (std::is_const_v<T> && std::is_same_v<std::remove_const_t<T>, U>)
        VEC_MATH_HD_INLINE constexpr vec_span(const vec_span<U>& other) noexcept
            : data_(other.data()), size_(other.size()) {}

        [[nodiscard]] VEC_MATH_HD_INLINE constexpr T* data() const noexcept { return data_; }
        [[nodiscard]] VEC_MATH_HD_INLINE constexpr std::size_t size() const noexcept { return size_; }
        [[nodiscard]] VEC_MATH_HD_INLINE constexpr bool empty() const noexcept { return size_ == 0; }

        [[nodiscard]] VEC_MATH_HD_INLINE constexpr reference operator[](std::size_t i) const noexcept {
            return data_[i];
        }

        [[nodiscard]] VEC_MATH_HD_INLINE constexpr iterator begin() const noexcept { return data_; }
        [[nodiscard]] VEC_MATH_HD_INLINE constexpr iterator end() const noexcept { return data_ + size_; }
        [[nodiscard]] VEC_MATH_HD_INLINE constexpr const_iterator cbegin() const noexcept { return data_; }
        [[nodiscard]] VEC_MATH_HD_INLINE constexpr const_iterator cend() const noexcept { return data_ + size_; }
    };

    template<Scalar T>
    [[nodiscard]] VEC_MATH_HD_INLINE constexpr vec_span<T> make_vec_span(T* ptr, std::size_t count) noexcept {
        return vec_span<T>(ptr, count);
    }

    template<Scalar T>
    [[nodiscard]] VEC_MATH_HD_INLINE constexpr device_view<T> make_device_view(T* ptr, std::size_t count) noexcept {
        return device_view<T>(ptr, count);
    }

    namespace detail {
        // Functors
        struct OpAdd {
            template<typename L, typename R>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const L &l, const R &r) const noexcept { return l + r; }
        };

        struct OpSub {
            template<typename L, typename R>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const L &l, const R &r) const noexcept { return l - r; }
        };

        struct OpMul {
            template<typename L, typename R>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const L &l, const R &r) const noexcept { return l * r; }
        };

        struct OpDiv {
            template<typename L, typename R>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const L &l, const R &r) const noexcept { return l / r; }
        };

        struct OpNeg {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const T &x) const noexcept { return -x; }
        };

        struct OpPos {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const T &x) const noexcept { return +x; }
        };

        struct OpAbs {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const T &x) const noexcept {
                if constexpr (std::is_signed_v<T> || std::is_floating_point_v<T>) {
                    return x < static_cast<T>(0) ? -x : x;
                } else {
                    return x;
                }
            }
        };

        struct OpFma {
            template<typename A, typename B, typename C>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const A &a, const B &b, const C &c) const noexcept {
                return a * b + c;
            }
        };

        // TriviallyCopyable View of vector operands
        template<NumericVector Vec>
        struct VecRef {
            using value_type = typename std::remove_cvref_t<Vec>::value_type;
            const value_type* VEC_MATH_RESTRICT ptr_ = nullptr;
            std::size_t size_ = 0;

            constexpr VecRef() noexcept = default;
            constexpr VecRef(const VecRef&) noexcept = default;
            constexpr VecRef(VecRef&&) noexcept = default;
            constexpr VecRef& operator=(const VecRef&) noexcept = default;
            constexpr VecRef& operator=(VecRef&&) noexcept = default;
            ~VecRef() = default;

            explicit constexpr VecRef(const Vec &v) noexcept
                : ptr_(v.data()), size_(v.size()) {}

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr std::size_t size() const noexcept { return size_; }
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr value_type operator[](std::size_t i) const noexcept { return ptr_[i]; }
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr const value_type* data() const noexcept { return ptr_; }
        };

        template<Scalar S>
        struct ScalarVal {
            using value_type = S;
            S val;

            constexpr ScalarVal() noexcept : val(0) {}
            constexpr ScalarVal(const ScalarVal&) noexcept = default;
            constexpr ScalarVal(ScalarVal&&) noexcept = default;
            constexpr ScalarVal& operator=(const ScalarVal&) noexcept = default;
            constexpr ScalarVal& operator=(ScalarVal&&) noexcept = default;
            ~ScalarVal() = default;

            explicit constexpr ScalarVal(S s) noexcept : val(s) {}

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr S operator[](std::size_t) const noexcept { return val; }
        };

        template<typename T>
        struct OperandStorage;

        template<NumericVector Vec>
        struct OperandStorage<Vec> {
            using type = VecRef<std::remove_cvref_t<Vec>>;
        };

        template<DeviceSpan Span>
        struct OperandStorage<Span> {
            using type = std::remove_cvref_t<Span>;
        };

        template<Scalar S>
        struct OperandStorage<S> {
            using type = ScalarVal<std::remove_cvref_t<S>>;
        };

        template<VecExpression Expr>
        struct OperandStorage<Expr> {
            using type = std::remove_cvref_t<Expr>;
        };

        template<typename T>
        using operand_storage_t = typename OperandStorage<T>::type;

        template<typename T>
        [[nodiscard]] constexpr auto make_operand(const T &x) noexcept {
            if constexpr (detail::is_vec_span_v<T>) {
                return x;
            } else {
                return operand_storage_t<T>(x);
            }
        }

        template<typename T>
        inline constexpr bool is_scalar_val_v = false;
        template<typename S>
        inline constexpr bool is_scalar_val_v<ScalarVal<S>> = true;

        // Binary Expression (TriviallyCopyable when Lhs and Rhs are TriviallyCopyable)
        template<typename Op, typename Lhs, typename Rhs>
        struct BinExpr : ExprBase {
            using value_type = std::common_type_t<typename Lhs::value_type, typename Rhs::value_type>;

            [[no_unique_address]] Op op;
            Lhs lhs;
            Rhs rhs;

            constexpr BinExpr(Op o, Lhs l, Rhs r) : op(o), lhs(std::move(l)), rhs(std::move(r)) {
                if constexpr (!is_scalar_val_v<Lhs> && !is_scalar_val_v<Rhs>) {
                    if (lhs.size() != rhs.size()) [[unlikely]] {
                        VEC_MATH_THROW_OR_ABORT(std::invalid_argument,
                                                "vec_math: vector size mismatch in binary expression");
                    }
                }
            }

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr std::size_t size() const noexcept {
                if constexpr (is_scalar_val_v<Lhs>) { return rhs.size(); } else { return lhs.size(); }
            }

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr value_type operator[](std::size_t i) const noexcept {
                return static_cast<value_type>(op(lhs[i], rhs[i]));
            }

            template<typename Alloc = std::allocator<value_type>>
            [[nodiscard]] std::vector<value_type, Alloc> eval(const Alloc &alloc = Alloc{}) const {
                std::vector<value_type, Alloc> result(size(), alloc);
                const std::size_t n = result.size();
                value_type * VEC_MATH_RESTRICT dst = result.data();

                std::size_t i = 0;
                const std::size_t n8 = n & ~std::size_t(7);
                VEC_MATH_PRAGMA_SIMD
                for (; i < n8; i += 8) {
                    dst[i + 0] = (*this)[i + 0];
                    dst[i + 1] = (*this)[i + 1];
                    dst[i + 2] = (*this)[i + 2];
                    dst[i + 3] = (*this)[i + 3];
                    dst[i + 4] = (*this)[i + 4];
                    dst[i + 5] = (*this)[i + 5];
                    dst[i + 6] = (*this)[i + 6];
                    dst[i + 7] = (*this)[i + 7];
                }
                const std::size_t n4 = n & ~std::size_t(3);
                for (; i < n4; i += 4) {
                    dst[i + 0] = (*this)[i + 0];
                    dst[i + 1] = (*this)[i + 1];
                    dst[i + 2] = (*this)[i + 2];
                    dst[i + 3] = (*this)[i + 3];
                }
                for (; i < n; ++i) { dst[i] = (*this)[i]; }
                return result;
            }

            operator std::vector<value_type>() const { return eval(); }
        };

        // Unary Expression (TriviallyCopyable when Operand is TriviallyCopyable)
        template<typename Op, typename Operand>
        struct UnaryExpr : ExprBase {
            using value_type = decltype(std::declval<Op>()(std::declval<typename Operand::value_type>()));

            [[no_unique_address]] Op op;
            Operand operand;

            constexpr UnaryExpr(Op o, Operand arg) : op(o), operand(std::move(arg)) {}

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr std::size_t size() const noexcept { return operand.size(); }

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr value_type operator[](std::size_t i) const noexcept {
                return static_cast<value_type>(op(operand[i]));
            }

            template<typename Alloc = std::allocator<value_type>>
            [[nodiscard]] std::vector<value_type, Alloc> eval(const Alloc &alloc = Alloc{}) const {
                std::vector<value_type, Alloc> result(size(), alloc);
                const std::size_t n = result.size();
                value_type * VEC_MATH_RESTRICT dst = result.data();

                std::size_t i = 0;
                const std::size_t n8 = n & ~std::size_t(7);
                VEC_MATH_PRAGMA_SIMD
                for (; i < n8; i += 8) {
                    dst[i + 0] = (*this)[i + 0];
                    dst[i + 1] = (*this)[i + 1];
                    dst[i + 2] = (*this)[i + 2];
                    dst[i + 3] = (*this)[i + 3];
                    dst[i + 4] = (*this)[i + 4];
                    dst[i + 5] = (*this)[i + 5];
                    dst[i + 6] = (*this)[i + 6];
                    dst[i + 7] = (*this)[i + 7];
                }
                const std::size_t n4 = n & ~std::size_t(3);
                for (; i < n4; i += 4) {
                    dst[i + 0] = (*this)[i + 0];
                    dst[i + 1] = (*this)[i + 1];
                    dst[i + 2] = (*this)[i + 2];
                    dst[i + 3] = (*this)[i + 3];
                }
                for (; i < n; ++i) { dst[i] = (*this)[i]; }
                return result;
            }

            operator std::vector<value_type>() const { return eval(); }
        };

        // Ternary Expression (TriviallyCopyable for FMA, etc.)
        template<typename Op, typename A, typename B, typename C>
        struct TernaryExpr : ExprBase {
            using value_type = decltype(std::declval<Op>()(
                std::declval<typename A::value_type>(),
                std::declval<typename B::value_type>(),
                std::declval<typename C::value_type>()
            ));

            [[no_unique_address]] Op op;
            A a;
            B b;
            C c;

            constexpr TernaryExpr(Op o, A arg_a, B arg_b, C arg_c)
                : op(o), a(std::move(arg_a)), b(std::move(arg_b)), c(std::move(arg_c)) {
                std::size_t expected_size = 0;
                bool has_size = false;
                auto check_size = [&](const auto &operand) {
                    using OpType = std::remove_cvref_t<decltype(operand)>;
                    if constexpr (!is_scalar_val_v<OpType>) {
                        if (!has_size) {
                            expected_size = operand.size();
                            has_size = true;
                        } else if (operand.size() != expected_size) {
                            VEC_MATH_THROW_OR_ABORT(std::invalid_argument,
                                                    "vec_math: vector size mismatch in ternary expression");
                        }
                    }
                };
                check_size(a);
                check_size(b);
                check_size(c);
            }

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr std::size_t size() const noexcept {
                if constexpr (!is_scalar_val_v<A>) { return a.size(); }
                else if constexpr (!is_scalar_val_v<B>) { return b.size(); }
                else { return c.size(); }
            }

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr value_type operator[](std::size_t i) const noexcept {
                return static_cast<value_type>(op(a[i], b[i], c[i]));
            }

            template<typename Alloc = std::allocator<value_type>>
            [[nodiscard]] std::vector<value_type, Alloc> eval(const Alloc &alloc = Alloc{}) const {
                std::vector<value_type, Alloc> result(size(), alloc);
                const std::size_t n = result.size();
                value_type * VEC_MATH_RESTRICT dst = result.data();

                std::size_t i = 0;
                const std::size_t n8 = n & ~std::size_t(7);
                VEC_MATH_PRAGMA_SIMD
                for (; i < n8; i += 8) {
                    dst[i + 0] = (*this)[i + 0];
                    dst[i + 1] = (*this)[i + 1];
                    dst[i + 2] = (*this)[i + 2];
                    dst[i + 3] = (*this)[i + 3];
                    dst[i + 4] = (*this)[i + 4];
                    dst[i + 5] = (*this)[i + 5];
                    dst[i + 6] = (*this)[i + 6];
                    dst[i + 7] = (*this)[i + 7];
                }
                const std::size_t n4 = n & ~std::size_t(3);
                for (; i < n4; i += 4) {
                    dst[i + 0] = (*this)[i + 0];
                    dst[i + 1] = (*this)[i + 1];
                    dst[i + 2] = (*this)[i + 2];
                    dst[i + 3] = (*this)[i + 3];
                }
                for (; i < n; ++i) { dst[i] = (*this)[i]; }
                return result;
            }

            operator std::vector<value_type>() const { return eval(); }
        };

        template<typename Op, typename L, typename R>
        [[nodiscard]] constexpr auto make_bin_expr(Op op, const L &l, const R &r) {
            return BinExpr<Op, operand_storage_t<L>, operand_storage_t<R>>(op, make_operand(l), make_operand(r));
        }

        template<typename Op, typename Arg>
        [[nodiscard]] constexpr auto make_unary_expr(Op op, const Arg &arg) {
            return UnaryExpr<Op, operand_storage_t<Arg>>(op, make_operand(arg));
        }

        // ============================================================================
        // Optimized Compound Application Kernels (8x/4x unrolled with SIMD pragmas)
        // ============================================================================

        template<typename T, typename Alloc, Scalar S, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_scalar_inplace(std::vector<T, Alloc> &dst, const S &scalar, CompoundFn fn) noexcept {
            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            const T s = static_cast<T>(scalar);

            std::size_t i = 0;
            const std::size_t n8 = n & ~std::size_t(7);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n8; i += 8) {
                fn(p[i + 0], s);
                fn(p[i + 1], s);
                fn(p[i + 2], s);
                fn(p[i + 3], s);
                fn(p[i + 4], s);
                fn(p[i + 5], s);
                fn(p[i + 6], s);
                fn(p[i + 7], s);
            }
            const std::size_t n4 = n & ~std::size_t(3);
            for (; i < n4; i += 4) {
                fn(p[i + 0], s);
                fn(p[i + 1], s);
                fn(p[i + 2], s);
                fn(p[i + 3], s);
            }
            for (; i < n; ++i) { fn(p[i], s); }
        }

        template<typename T, typename Alloc, VectorOperand Src, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_vec_inplace(std::vector<T, Alloc> &dst, const Src &src, CompoundFn fn) {
            if (dst.size() != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in compound assignment");
            }

            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            auto storage = make_operand(src);

            std::size_t i = 0;
            const std::size_t n8 = n & ~std::size_t(7);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n8; i += 8) {
                fn(p[i + 0], static_cast<T>(storage[i + 0]));
                fn(p[i + 1], static_cast<T>(storage[i + 1]));
                fn(p[i + 2], static_cast<T>(storage[i + 2]));
                fn(p[i + 3], static_cast<T>(storage[i + 3]));
                fn(p[i + 4], static_cast<T>(storage[i + 4]));
                fn(p[i + 5], static_cast<T>(storage[i + 5]));
                fn(p[i + 6], static_cast<T>(storage[i + 6]));
                fn(p[i + 7], static_cast<T>(storage[i + 7]));
            }
            const std::size_t n4 = n & ~std::size_t(3);
            for (; i < n4; i += 4) {
                fn(p[i + 0], static_cast<T>(storage[i + 0]));
                fn(p[i + 1], static_cast<T>(storage[i + 1]));
                fn(p[i + 2], static_cast<T>(storage[i + 2]));
                fn(p[i + 3], static_cast<T>(storage[i + 3]));
            }
            for (; i < n; ++i) { fn(p[i], static_cast<T>(storage[i])); }
        }

        template<typename T, typename Alloc, VectorOperand Src>
        VEC_MATH_FORCE_INLINE void assign_zero_malloc(std::vector<T, Alloc> &dst, const Src &src) {
            const std::size_t n = src.size();
            if (dst.capacity() < n) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::length_error,
                                        "vec_math: destination capacity < source size (zero-malloc invariant violated)");
            }

            dst.resize(n);

            T * VEC_MATH_RESTRICT p = dst.data();
            auto storage = make_operand(src);

            std::size_t i = 0;
            const std::size_t n8 = n & ~std::size_t(7);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n8; i += 8) {
                p[i + 0] = static_cast<T>(storage[i + 0]);
                p[i + 1] = static_cast<T>(storage[i + 1]);
                p[i + 2] = static_cast<T>(storage[i + 2]);
                p[i + 3] = static_cast<T>(storage[i + 3]);
                p[i + 4] = static_cast<T>(storage[i + 4]);
                p[i + 5] = static_cast<T>(storage[i + 5]);
                p[i + 6] = static_cast<T>(storage[i + 6]);
                p[i + 7] = static_cast<T>(storage[i + 7]);
            }
            const std::size_t n4 = n & ~std::size_t(3);
            for (; i < n4; i += 4) {
                p[i + 0] = static_cast<T>(storage[i + 0]);
                p[i + 1] = static_cast<T>(storage[i + 1]);
                p[i + 2] = static_cast<T>(storage[i + 2]);
                p[i + 3] = static_cast<T>(storage[i + 3]);
            }
            for (; i < n; ++i) { p[i] = static_cast<T>(storage[i]); }
        }

        template<Scalar T, VectorOperand Src>
        VEC_MATH_HD_INLINE void assign_span(vec_span<T> dst, const Src &src) {
            if (dst.size() != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument,
                                        "vec_math: span size mismatch in assignment");
            }
            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            auto storage = make_operand(src);

            std::size_t i = 0;
            const std::size_t n8 = n & ~std::size_t(7);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n8; i += 8) {
                p[i + 0] = static_cast<T>(storage[i + 0]);
                p[i + 1] = static_cast<T>(storage[i + 1]);
                p[i + 2] = static_cast<T>(storage[i + 2]);
                p[i + 3] = static_cast<T>(storage[i + 3]);
                p[i + 4] = static_cast<T>(storage[i + 4]);
                p[i + 5] = static_cast<T>(storage[i + 5]);
                p[i + 6] = static_cast<T>(storage[i + 6]);
                p[i + 7] = static_cast<T>(storage[i + 7]);
            }
            const std::size_t n4 = n & ~std::size_t(3);
            for (; i < n4; i += 4) {
                p[i + 0] = static_cast<T>(storage[i + 0]);
                p[i + 1] = static_cast<T>(storage[i + 1]);
                p[i + 2] = static_cast<T>(storage[i + 2]);
                p[i + 3] = static_cast<T>(storage[i + 3]);
            }
            for (; i < n; ++i) { p[i] = static_cast<T>(storage[i]); }
        }

        template<Scalar T, VectorOperand Src, typename CompoundFn>
        VEC_MATH_HD_INLINE void apply_span_inplace(vec_span<T> dst, const Src &src, CompoundFn fn) {
            if (dst.size() != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument,
                                        "vec_math: span size mismatch in compound assignment");
            }
            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            auto storage = make_operand(src);

            std::size_t i = 0;
            const std::size_t n8 = n & ~std::size_t(7);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n8; i += 8) {
                fn(p[i + 0], static_cast<T>(storage[i + 0]));
                fn(p[i + 1], static_cast<T>(storage[i + 1]));
                fn(p[i + 2], static_cast<T>(storage[i + 2]));
                fn(p[i + 3], static_cast<T>(storage[i + 3]));
                fn(p[i + 4], static_cast<T>(storage[i + 4]));
                fn(p[i + 5], static_cast<T>(storage[i + 5]));
                fn(p[i + 6], static_cast<T>(storage[i + 6]));
                fn(p[i + 7], static_cast<T>(storage[i + 7]));
            }
            const std::size_t n4 = n & ~std::size_t(3);
            for (; i < n4; i += 4) {
                fn(p[i + 0], static_cast<T>(storage[i + 0]));
                fn(p[i + 1], static_cast<T>(storage[i + 1]));
                fn(p[i + 2], static_cast<T>(storage[i + 2]));
                fn(p[i + 3], static_cast<T>(storage[i + 3]));
            }
            for (; i < n; ++i) { fn(p[i], static_cast<T>(storage[i])); }
        }

        template<Scalar T, Scalar S, typename CompoundFn>
        VEC_MATH_HD_INLINE void apply_span_scalar_inplace(vec_span<T> dst, const S &scalar, CompoundFn fn) noexcept {
            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            const T s = static_cast<T>(scalar);

            std::size_t i = 0;
            const std::size_t n8 = n & ~std::size_t(7);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n8; i += 8) {
                fn(p[i + 0], s);
                fn(p[i + 1], s);
                fn(p[i + 2], s);
                fn(p[i + 3], s);
                fn(p[i + 4], s);
                fn(p[i + 5], s);
                fn(p[i + 6], s);
                fn(p[i + 7], s);
            }
            const std::size_t n4 = n & ~std::size_t(3);
            for (; i < n4; i += 4) {
                fn(p[i + 0], s);
                fn(p[i + 1], s);
                fn(p[i + 2], s);
                fn(p[i + 3], s);
            }
            for (; i < n; ++i) { fn(p[i], s); }
        }
    }

    // ============================================================================
    // Operators
    // ============================================================================

    inline namespace ops {
        // Zero-malloc assign for std::vector
        template<typename T, typename Alloc, VectorOperand Src>
        inline std::vector<T, Alloc> &operator<<=(std::vector<T, Alloc> &dst, const Src &src) {
            detail::assign_zero_malloc(dst, src);
            return dst;
        }

        // Zero-malloc assign for vec_span
        template<Scalar T, VectorOperand Src> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator<<=(vec_span<T> &dst, const Src &src) {
            detail::assign_span(dst, src);
            return dst;
        }

        template<Scalar T, VectorOperand Src> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> operator<<=(vec_span<T> &&dst, const Src &src) {
            detail::assign_span(dst, src);
            return dst;
        }

        // Compound assignment operators for std::vector
        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator+=(std::vector<T, Alloc> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a += b; });
            return lhs;
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator+=(std::vector<T, Alloc> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a += s; });
            return lhs;
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator-=(std::vector<T, Alloc> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a -= b; });
            return lhs;
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator-=(std::vector<T, Alloc> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a -= s; });
            return lhs;
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator*=(std::vector<T, Alloc> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a *= b; });
            return lhs;
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator*=(std::vector<T, Alloc> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a *= s; });
            return lhs;
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator/=(std::vector<T, Alloc> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a /= b; });
            return lhs;
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::vector<T, Alloc> &operator/=(std::vector<T, Alloc> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a /= s; });
            return lhs;
        }

        // Compound assignment operators for vec_span
        template<Scalar T, VectorOperand Rhs> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator+=(vec_span<T> &lhs, const Rhs &rhs) {
            detail::apply_span_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a += b; });
            return lhs;
        }

        template<Scalar T, Scalar S> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator+=(vec_span<T> &lhs, const S &scalar) noexcept {
            detail::apply_span_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a += s; });
            return lhs;
        }

        template<Scalar T, VectorOperand Rhs> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator-=(vec_span<T> &lhs, const Rhs &rhs) {
            detail::apply_span_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a -= b; });
            return lhs;
        }

        template<Scalar T, Scalar S> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator-=(vec_span<T> &lhs, const S &scalar) noexcept {
            detail::apply_span_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a -= s; });
            return lhs;
        }

        template<Scalar T, VectorOperand Rhs> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator*=(vec_span<T> &lhs, const Rhs &rhs) {
            detail::apply_span_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a *= b; });
            return lhs;
        }

        template<Scalar T, Scalar S> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator*=(vec_span<T> &lhs, const S &scalar) noexcept {
            detail::apply_span_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a *= s; });
            return lhs;
        }

        template<Scalar T, VectorOperand Rhs> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator/=(vec_span<T> &lhs, const Rhs &rhs) {
            detail::apply_span_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a /= b; });
            return lhs;
        }

        template<Scalar T, Scalar S> requires (!std::is_const_v<T>)
        VEC_MATH_HD_INLINE vec_span<T> &operator/=(vec_span<T> &lhs, const S &scalar) noexcept {
            detail::apply_span_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a /= s; });
            return lhs;
        }

        // Unary arithmetic operators
        template<VectorOperand Arg>
        [[nodiscard]] constexpr auto operator-(const Arg &arg) { return detail::make_unary_expr(detail::OpNeg{}, arg); }

        template<VectorOperand Arg>
        [[nodiscard]] constexpr auto operator+(const Arg &arg) { return detail::make_unary_expr(detail::OpPos{}, arg); }

        // Binary +
        template<VectorOperand Lhs, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator+(const Lhs &lhs, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpAdd{}, lhs, rhs);
        }

        template<VectorOperand Lhs, Scalar S>
        [[nodiscard]] constexpr auto operator+(const Lhs &lhs, const S &scalar) {
            return detail::make_bin_expr(detail::OpAdd{}, lhs, scalar);
        }

        template<Scalar S, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator+(const S &scalar, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpAdd{}, scalar, rhs);
        }

        // Binary -
        template<VectorOperand Lhs, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator-(const Lhs &lhs, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpSub{}, lhs, rhs);
        }

        template<VectorOperand Lhs, Scalar S>
        [[nodiscard]] constexpr auto operator-(const Lhs &lhs, const S &scalar) {
            return detail::make_bin_expr(detail::OpSub{}, lhs, scalar);
        }

        template<Scalar S, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator-(const S &scalar, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpSub{}, scalar, rhs);
        }

        // Binary *
        template<VectorOperand Lhs, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator*(const Lhs &lhs, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpMul{}, lhs, rhs);
        }

        template<VectorOperand Lhs, Scalar S>
        [[nodiscard]] constexpr auto operator*(const Lhs &lhs, const S &scalar) {
            return detail::make_bin_expr(detail::OpMul{}, lhs, scalar);
        }

        template<Scalar S, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator*(const S &scalar, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpMul{}, scalar, rhs);
        }

        // Binary /
        template<VectorOperand Lhs, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator/(const Lhs &lhs, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpDiv{}, lhs, rhs);
        }

        template<VectorOperand Lhs, Scalar S>
        [[nodiscard]] constexpr auto operator/(const Lhs &lhs, const S &scalar) {
            return detail::make_bin_expr(detail::OpDiv{}, lhs, scalar);
        }

        template<Scalar S, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator/(const S &scalar, const Rhs &rhs) {
            return detail::make_bin_expr(detail::OpDiv{}, scalar, rhs);
        }

        // Rvalue vector optimizations
        template<typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(std::vector<T, Alloc> &&v) noexcept {
            const std::size_t n = v.size();
            T * VEC_MATH_RESTRICT p = v.data();
            std::size_t i = 0;
            const std::size_t n4 = n & ~std::size_t(3);
            VEC_MATH_PRAGMA_SIMD
            for (; i < n4; i += 4) {
                p[i + 0] = -p[i + 0];
                p[i + 1] = -p[i + 1];
                p[i + 2] = -p[i + 2];
                p[i + 3] = -p[i + 3];
            }
            for (; i < n; ++i) { p[i] = -p[i]; }
            return std::move(v);
        }

        template<typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator+(std::vector<T, Alloc> &&v) noexcept {
            return std::move(v);
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator+(std::vector<T, Alloc> &&lhs, const Rhs &rhs) {
            lhs += rhs;
            return std::move(lhs);
        }

        template<VectorOperand Lhs, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator+(const Lhs &lhs, std::vector<T, Alloc> &&rhs) {
            rhs += lhs;
            return std::move(rhs);
        }

        template<typename T, typename Alloc, typename Alloc2> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator+(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
            lhs += rhs;
            return std::move(lhs);
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator+(std::vector<T, Alloc> &&lhs, const S &scalar) noexcept {
            lhs += scalar;
            return std::move(lhs);
        }

        template<Scalar S, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator+(const S &scalar, std::vector<T, Alloc> &&rhs) noexcept {
            rhs += scalar;
            return std::move(rhs);
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(std::vector<T, Alloc> &&lhs, const Rhs &rhs) {
            lhs -= rhs;
            return std::move(lhs);
        }

        template<VectorOperand Lhs, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(const Lhs &lhs, std::vector<T, Alloc> &&rhs) {
            detail::apply_vec_inplace(rhs, lhs, [](T &r, const T &l) noexcept { r = l - r; });
            return std::move(rhs);
        }

        template<typename T, typename Alloc, typename Alloc2> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
            lhs -= rhs;
            return std::move(lhs);
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(std::vector<T, Alloc> &&lhs, const S &scalar) noexcept {
            lhs -= scalar;
            return std::move(lhs);
        }

        template<Scalar S, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(const S &scalar, std::vector<T, Alloc> &&rhs) noexcept {
            detail::apply_scalar_inplace(rhs, scalar, [](T &r, const T &s) noexcept { r = s - r; });
            return std::move(rhs);
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator*(std::vector<T, Alloc> &&lhs, const Rhs &rhs) {
            lhs *= rhs;
            return std::move(lhs);
        }

        template<VectorOperand Lhs, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator*(const Lhs &lhs, std::vector<T, Alloc> &&rhs) {
            rhs *= lhs;
            return std::move(rhs);
        }

        template<typename T, typename Alloc, typename Alloc2> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator*(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
            lhs *= rhs;
            return std::move(lhs);
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator*(std::vector<T, Alloc> &&lhs, const S &scalar) noexcept {
            lhs *= scalar;
            return std::move(lhs);
        }

        template<Scalar S, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator*(const S &scalar, std::vector<T, Alloc> &&rhs) noexcept {
            rhs *= scalar;
            return std::move(rhs);
        }

        template<typename T, typename Alloc, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator/(std::vector<T, Alloc> &&lhs, const Rhs &rhs) {
            lhs /= rhs;
            return std::move(lhs);
        }

        template<VectorOperand Lhs, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator/(const Lhs &lhs, std::vector<T, Alloc> &&rhs) {
            detail::apply_vec_inplace(rhs, lhs, [](T &r, const T &l) noexcept { r = l / r; });
            return std::move(rhs);
        }

        template<typename T, typename Alloc, typename Alloc2> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator/(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
            lhs /= rhs;
            return std::move(lhs);
        }

        template<typename T, typename Alloc, Scalar S> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator/(std::vector<T, Alloc> &&lhs, const S &scalar) noexcept {
            lhs /= scalar;
            return std::move(lhs);
        }

        template<Scalar S, typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator/(const S &scalar, std::vector<T, Alloc> &&rhs) noexcept {
            detail::apply_scalar_inplace(rhs, scalar, [](T &r, const T &s) noexcept { r = s / r; });
            return std::move(rhs);
        }
    }

    template<typename T, typename Alloc, VectorOperand Src>
    inline void assign(std::vector<T, Alloc> &dst, const Src &src) {
        detail::assign_zero_malloc(dst, src);
    }

    template<Scalar T, VectorOperand Src> requires (!std::is_const_v<T>)
    VEC_MATH_HD_INLINE void assign(vec_span<T> dst, const Src &src) {
        detail::assign_span(dst, src);
    }

    // ============================================================================
    // Extended Math Utilities (dot, sum, norm, norm_sq, fma, abs)
    // ============================================================================

    // abs: element-wise absolute value expression
    template<VectorOperand Arg>
    [[nodiscard]] constexpr auto abs(const Arg &arg) {
        return detail::make_unary_expr(detail::OpAbs{}, arg);
    }

    // fma: fused multiply-add ternary expression a * b + c
    template<VectorOperand A, VectorOperand B, VectorOperand C>
    [[nodiscard]] constexpr auto fma(const A &a, const B &b, const C &c) {
        return detail::TernaryExpr<detail::OpFma,
            detail::operand_storage_t<A>,
            detail::operand_storage_t<B>,
            detail::operand_storage_t<C>>(
                detail::OpFma{},
                detail::make_operand(a),
                detail::make_operand(b),
                detail::make_operand(c)
        );
    }

    template<VectorOperand A, Scalar B, VectorOperand C>
    [[nodiscard]] constexpr auto fma(const A &a, const B &b, const C &c) {
        return detail::TernaryExpr<detail::OpFma,
            detail::operand_storage_t<A>,
            detail::operand_storage_t<B>,
            detail::operand_storage_t<C>>(
                detail::OpFma{},
                detail::make_operand(a),
                detail::make_operand(b),
                detail::make_operand(c)
        );
    }

    template<Scalar A, VectorOperand B, VectorOperand C>
    [[nodiscard]] constexpr auto fma(const A &a, const B &b, const C &c) {
        return detail::TernaryExpr<detail::OpFma,
            detail::operand_storage_t<A>,
            detail::operand_storage_t<B>,
            detail::operand_storage_t<C>>(
                detail::OpFma{},
                detail::make_operand(a),
                detail::make_operand(b),
                detail::make_operand(c)
        );
    }

    template<VectorOperand A, VectorOperand B, Scalar C>
    [[nodiscard]] constexpr auto fma(const A &a, const B &b, const C &c) {
        return detail::TernaryExpr<detail::OpFma,
            detail::operand_storage_t<A>,
            detail::operand_storage_t<B>,
            detail::operand_storage_t<C>>(
                detail::OpFma{},
                detail::make_operand(a),
                detail::make_operand(b),
                detail::make_operand(c)
        );
    }

    // sum: 8x unrolled reduction with 8 accumulators for maximum IPC
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto sum(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) { return value_type{0}; }

        value_type acc0{0};
        value_type acc1{0};
        value_type acc2{0};
        value_type acc3{0};
        value_type acc4{0};
        value_type acc5{0};
        value_type acc6{0};
        value_type acc7{0};

        std::size_t i = 0;
        const std::size_t n8 = n & ~std::size_t(7);
        VEC_MATH_PRAGMA_SIMD
        for (; i < n8; i += 8) {
            acc0 = acc0 + static_cast<value_type>(op[i + 0]);
            acc1 = acc1 + static_cast<value_type>(op[i + 1]);
            acc2 = acc2 + static_cast<value_type>(op[i + 2]);
            acc3 = acc3 + static_cast<value_type>(op[i + 3]);
            acc4 = acc4 + static_cast<value_type>(op[i + 4]);
            acc5 = acc5 + static_cast<value_type>(op[i + 5]);
            acc6 = acc6 + static_cast<value_type>(op[i + 6]);
            acc7 = acc7 + static_cast<value_type>(op[i + 7]);
        }
        const std::size_t n4 = n & ~std::size_t(3);
        for (; i < n4; i += 4) {
            acc0 = acc0 + static_cast<value_type>(op[i + 0]);
            acc1 = acc1 + static_cast<value_type>(op[i + 1]);
            acc2 = acc2 + static_cast<value_type>(op[i + 2]);
            acc3 = acc3 + static_cast<value_type>(op[i + 3]);
        }
        value_type total = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
        for (; i < n; ++i) {
            total = total + static_cast<value_type>(op[i]);
        }
        return total;
    }

    // dot: 8x unrolled dot product with 8 accumulators for maximum IPC
    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto dot(const Lhs &lhs_in, const Rhs &rhs_in) {
        if (lhs_in.size() != rhs_in.size()) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in dot product");
        }
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using ResType = std::common_type_t<
            typename detail::operand_storage_t<Lhs>::value_type,
            typename detail::operand_storage_t<Rhs>::value_type
        >;
        const std::size_t n = lhs_in.size();
        if (n == 0) { return ResType{0}; }

        ResType acc0{0};
        ResType acc1{0};
        ResType acc2{0};
        ResType acc3{0};
        ResType acc4{0};
        ResType acc5{0};
        ResType acc6{0};
        ResType acc7{0};

        std::size_t i = 0;
        const std::size_t n8 = n & ~std::size_t(7);
        VEC_MATH_PRAGMA_SIMD
        for (; i < n8; i += 8) {
            acc0 = acc0 + static_cast<ResType>(lhs_op[i + 0]) * static_cast<ResType>(rhs_op[i + 0]);
            acc1 = acc1 + static_cast<ResType>(lhs_op[i + 1]) * static_cast<ResType>(rhs_op[i + 1]);
            acc2 = acc2 + static_cast<ResType>(lhs_op[i + 2]) * static_cast<ResType>(rhs_op[i + 2]);
            acc3 = acc3 + static_cast<ResType>(lhs_op[i + 3]) * static_cast<ResType>(rhs_op[i + 3]);
            acc4 = acc4 + static_cast<ResType>(lhs_op[i + 4]) * static_cast<ResType>(rhs_op[i + 4]);
            acc5 = acc5 + static_cast<ResType>(lhs_op[i + 5]) * static_cast<ResType>(rhs_op[i + 5]);
            acc6 = acc6 + static_cast<ResType>(lhs_op[i + 6]) * static_cast<ResType>(rhs_op[i + 6]);
            acc7 = acc7 + static_cast<ResType>(lhs_op[i + 7]) * static_cast<ResType>(rhs_op[i + 7]);
        }
        const std::size_t n4 = n & ~std::size_t(3);
        for (; i < n4; i += 4) {
            acc0 = acc0 + static_cast<ResType>(lhs_op[i + 0]) * static_cast<ResType>(rhs_op[i + 0]);
            acc1 = acc1 + static_cast<ResType>(lhs_op[i + 1]) * static_cast<ResType>(rhs_op[i + 1]);
            acc2 = acc2 + static_cast<ResType>(lhs_op[i + 2]) * static_cast<ResType>(rhs_op[i + 2]);
            acc3 = acc3 + static_cast<ResType>(lhs_op[i + 3]) * static_cast<ResType>(rhs_op[i + 3]);
        }
        ResType total = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
        for (; i < n; ++i) {
            total = total + static_cast<ResType>(lhs_op[i]) * static_cast<ResType>(rhs_op[i]);
        }
        return total;
    }

    // norm_sq: squared L2 norm
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto norm_sq(const Op &op) {
        return dot(op, op);
    }

    // norm: Euclidean L2 norm
    template<VectorOperand Op>
    [[nodiscard]] inline auto norm(const Op &op) {
        using Val = typename detail::operand_storage_t<Op>::value_type;
        using ResType = std::conditional_t<std::is_floating_point_v<Val>, Val, double>;
        const auto d = dot(op, op);
        return std::sqrt(static_cast<ResType>(d));
    }

    // ============================================================================
    // SYCL Device Evaluation Helpers (zero-allocation kernel execution)
    // ============================================================================

    template<typename Dst, VectorOperand Expr>
    VEC_MATH_HD_INLINE constexpr void device_eval(Dst &dst, const Expr &expr) {
        const std::size_t n = expr.size();
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<typename Dst::value_type>(expr[i]);
        }
    }

    template<typename Dst, VectorOperand Expr>
    VEC_MATH_HD_INLINE constexpr void device_eval(Dst &dst, const Expr &expr, std::size_t i) noexcept {
        dst[i] = static_cast<typename Dst::value_type>(expr[i]);
    }

    template<typename T, VectorOperand Expr>
    VEC_MATH_HD_INLINE constexpr void device_eval(T * VEC_MATH_RESTRICT dst, const Expr &expr, std::size_t i) noexcept {
        dst[i] = static_cast<T>(expr[i]);
    }
}
