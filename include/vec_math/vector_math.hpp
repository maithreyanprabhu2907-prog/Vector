#pragma once

/*
 * Author: Maithreyan(maithreyanprabu2907-prog)
 * Extended & Optimized for Zero-Allocation Vector Math & SYCL Device Compatibility
 *
 * Purpose:
 *   Provide a modern, expressive, intensely optimized vector math library for std::vector.
 *   Operator overloading keeps mathematical expressions short, readable, and fully fused.
 *
 * Goals:
 *      - Modern C++20 style, zero-allocation throughout all operations.
 *      - SIMD-vectorized loops with loop unrolling (8x/4x) and restrict pointers.
 *      - TriviallyCopyable expression templates for seamless SYCL device kernel capture.
 *      - SYCL device annotations without requiring SYCL header dependencies.
 *      - Extended math utilities: dot product, sum, norm, norm_sq, fma, abs.
 */

#include <cassert>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <immintrin.h>
#define VEC_MATH_HAS_X86_INTRIN 1
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

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
#define VEC_MATH_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define VEC_MATH_FORCE_INLINE inline __attribute__((always_inline))
#define VEC_MATH_NOINLINE __attribute__((noinline))
#else
#define VEC_MATH_FORCE_INLINE inline
#define VEC_MATH_NOINLINE
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

    namespace detail {
        template<typename T>
        struct is_std_vector : std::false_type {};

        template<typename T, typename Alloc>
        struct is_std_vector<std::vector<T, Alloc>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_vector_v = is_std_vector<std::remove_cvref_t<T>>::value;

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
    concept VecExpression = detail::is_vec_expr_v<T>;

    template<typename T>
    concept VectorOperand = NumericVector<T> || VecExpression<T>;

    // SYCL Compatibility Concept (verifies TriviallyCopyable for device kernel capture)
    template<typename T>
    concept sycl_compatible = std::is_trivially_copyable_v<std::remove_cvref_t<T>>;

    template<typename T>
    inline constexpr bool is_sycl_compatible_v = sycl_compatible<T>;

    // ============================================================================
    // 64-Byte Cache-Line Aligned Memory
    // ============================================================================

    template<typename T, std::size_t Alignment = 64>
    struct aligned_allocator {
        using value_type = T;
        static constexpr std::size_t alignment = Alignment;

        template<typename U>
        struct rebind {
            using other = aligned_allocator<U, Alignment>;
        };

        constexpr aligned_allocator() noexcept = default;

        template<typename U>
        constexpr aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

        [[nodiscard]] T* allocate(std::size_t n) {
            if (n == 0) return nullptr;
            if (n > std::size_t(-1) / sizeof(T)) {
                throw std::bad_array_new_length{};
            }
            const std::size_t bytes = n * sizeof(T);
#if defined(_MSC_VER) || defined(__MINGW32__)
            void* ptr = _aligned_malloc(bytes, Alignment);
            if (!ptr) throw std::bad_alloc{};
            return static_cast<T*>(ptr);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
            void* ptr = nullptr;
            constexpr std::size_t align = Alignment < sizeof(void*) ? sizeof(void*) : Alignment;
            if (posix_memalign(&ptr, align, bytes) != 0 || !ptr) {
                throw std::bad_alloc{};
            }
            return static_cast<T*>(ptr);
#else
            const std::size_t rounded_bytes = ((bytes + Alignment - 1) / Alignment) * Alignment;
            void* ptr = std::aligned_alloc(Alignment, rounded_bytes);
            if (!ptr) throw std::bad_alloc{};
            return static_cast<T*>(ptr);
#endif
        }

        void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
            if (!p) return;
#if defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(p);
#else
            std::free(p);
#endif
        }

        template<typename U, std::size_t A2>
        constexpr bool operator==(const aligned_allocator<U, A2>&) const noexcept {
            return Alignment == A2;
        }

        template<typename U, std::size_t A2>
        constexpr bool operator!=(const aligned_allocator<U, A2>&) const noexcept {
            return Alignment != A2;
        }
    };

    template<typename T>
    using aligned_vector = std::vector<T, aligned_allocator<T, 64>>;

    namespace detail {
        // Arithmetic Functors
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

        struct OpMin {
            template<typename L, typename R>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const L &l, const R &r) const noexcept {
                return l < r ? l : r;
            }
        };

        struct OpMax {
            template<typename L, typename R>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const L &l, const R &r) const noexcept {
                return l > r ? l : r;
            }
        };

        struct OpClamp {
            template<typename V, typename Lo, typename Hi>
            [[nodiscard]] VEC_MATH_HD_INLINE constexpr auto operator()(const V &v, const Lo &lo, const Hi &hi) const noexcept {
                return v < lo ? lo : (v > hi ? hi : v);
            }
        };

        struct OpSqrt {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE auto operator()(const T &x) const noexcept {
                using std::sqrt;
                return sqrt(x);
            }
        };

        struct OpExp {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE auto operator()(const T &x) const noexcept {
                using std::exp;
                return exp(x);
            }
        };

        struct OpLog {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE auto operator()(const T &x) const noexcept {
                using std::log;
                return log(x);
            }
        };

        struct OpSin {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE auto operator()(const T &x) const noexcept {
                using std::sin;
                return sin(x);
            }
        };

        struct OpCos {
            template<typename T>
            [[nodiscard]] VEC_MATH_HD_INLINE auto operator()(const T &x) const noexcept {
                using std::cos;
                return cos(x);
            }
        };

        struct OpPow {
            template<typename B, typename E>
            [[nodiscard]] VEC_MATH_HD_INLINE auto operator()(const B &base, const E &exp) const noexcept {
                using std::pow;
                return pow(base, exp);
            }
        };

        // TriviallyCopyable View of std::vector operands for expression evaluation
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

            VEC_MATH_HD_INLINE constexpr VecRef(const value_type* ptr, std::size_t count) noexcept
                : ptr_(ptr), size_(count) {}

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
            return operand_storage_t<T>(x);
        }

        template<typename T>
        inline constexpr bool is_scalar_val_v = false;
        template<typename S>
        inline constexpr bool is_scalar_val_v<ScalarVal<S>> = true;

        // Binary Expression (TriviallyCopyable when Lhs and Rhs are TriviallyCopyable)
        template<typename Op, typename Lhs, typename Rhs>
        struct BinExpr : ExprBase {
            using value_type = std::remove_cvref_t<decltype(std::declval<Op>()(
                std::declval<typename Lhs::value_type>(),
                std::declval<typename Rhs::value_type>()
            ))>;

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

            constexpr BinExpr(const BinExpr&) noexcept = default;
            constexpr BinExpr(BinExpr&&) noexcept = default;
            constexpr BinExpr& operator=(const BinExpr&) noexcept = default;
            constexpr BinExpr& operator=(BinExpr&&) noexcept = default;
            ~BinExpr() = default;

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

                VEC_MATH_PRAGMA_SIMD
                for (std::size_t i = 0; i < n; ++i) { dst[i] = (*this)[i]; }
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

            constexpr UnaryExpr(const UnaryExpr&) noexcept = default;
            constexpr UnaryExpr(UnaryExpr&&) noexcept = default;
            constexpr UnaryExpr& operator=(const UnaryExpr&) noexcept = default;
            constexpr UnaryExpr& operator=(UnaryExpr&&) noexcept = default;
            ~UnaryExpr() = default;

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr std::size_t size() const noexcept { return operand.size(); }

            [[nodiscard]] VEC_MATH_HD_INLINE constexpr value_type operator[](std::size_t i) const noexcept {
                return static_cast<value_type>(op(operand[i]));
            }

            template<typename Alloc = std::allocator<value_type>>
            [[nodiscard]] std::vector<value_type, Alloc> eval(const Alloc &alloc = Alloc{}) const {
                std::vector<value_type, Alloc> result(size(), alloc);
                const std::size_t n = result.size();
                value_type * VEC_MATH_RESTRICT dst = result.data();

                VEC_MATH_PRAGMA_SIMD
                for (std::size_t i = 0; i < n; ++i) { dst[i] = (*this)[i]; }
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

            constexpr TernaryExpr(const TernaryExpr&) noexcept = default;
            constexpr TernaryExpr(TernaryExpr&&) noexcept = default;
            constexpr TernaryExpr& operator=(const TernaryExpr&) noexcept = default;
            constexpr TernaryExpr& operator=(TernaryExpr&&) noexcept = default;
            ~TernaryExpr() = default;

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

                VEC_MATH_PRAGMA_SIMD
                for (std::size_t i = 0; i < n; ++i) { dst[i] = (*this)[i]; }
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

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { fn(p[i], s); }
        }

        template<typename T, typename Alloc, VectorOperand Src, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_vec_inplace(std::vector<T, Alloc> &dst, const Src &src, CompoundFn fn) {
            if (dst.size() != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in compound assignment");
            }

            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            auto storage = make_operand(src);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { fn(p[i], static_cast<T>(storage[i])); }
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

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { p[i] = static_cast<T>(storage[i]); }
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
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { p[i] = -p[i]; }
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

    namespace detail {
    }

    // sum: reduction with SIMD auto-vectorization
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto sum(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) { return value_type{0}; }

        value_type total{0};
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            total = total + static_cast<value_type>(op[i]);
        }
        return total;
    }

    // dot: dot product with SIMD auto-vectorization and FMA
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

        ResType total{0};
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
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
    // Operator Overloading for Dot / Inner Product
    // ============================================================================

    inline namespace ops {
        // Inner product / dot product operator: u | v
        template<VectorOperand Lhs, VectorOperand Rhs>
        [[nodiscard]] constexpr auto operator|(const Lhs &lhs, const Rhs &rhs) {
            return dot(lhs, rhs);
        }
    }

    // ============================================================================
    // Non-Temporal Streaming Stores
    // ============================================================================

    template<typename T, typename Alloc, VectorOperand Src>
    inline void assign_stream(std::vector<T, Alloc> &dst, const Src &src) {
        const std::size_t n = src.size();
        if (dst.capacity() < n) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::length_error,
                                    "vec_math: destination capacity < source size in assign_stream");
        }
        dst.resize(n);
        if (n == 0) return;

        T * VEC_MATH_RESTRICT p = dst.data();
        auto storage = detail::make_operand(src);

#if defined(VEC_MATH_HAS_X86_INTRIN)
        if constexpr (std::is_same_v<T, double>) {
            std::size_t i = 0;
#if defined(__AVX__)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 31) != 0) {
                p[i] = static_cast<double>(storage[i]);
                ++i;
            }
            for (; i + 4 <= n; i += 4) {
                __m256d val = _mm256_setr_pd(
                    static_cast<double>(storage[i]),
                    static_cast<double>(storage[i + 1]),
                    static_cast<double>(storage[i + 2]),
                    static_cast<double>(storage[i + 3])
                );
                _mm256_stream_pd(p + i, val);
            }
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 15) != 0) {
                p[i] = static_cast<double>(storage[i]);
                ++i;
            }
            for (; i + 2 <= n; i += 2) {
                __m128d val = _mm_setr_pd(
                    static_cast<double>(storage[i]),
                    static_cast<double>(storage[i + 1])
                );
                _mm_stream_pd(p + i, val);
            }
#endif
            for (; i < n; ++i) {
                p[i] = static_cast<double>(storage[i]);
            }
            _mm_sfence();
        } else if constexpr (std::is_same_v<T, float>) {
            std::size_t i = 0;
#if defined(__AVX__)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 31) != 0) {
                p[i] = static_cast<float>(storage[i]);
                ++i;
            }
            for (; i + 8 <= n; i += 8) {
                __m256 val = _mm256_setr_ps(
                    static_cast<float>(storage[i]),
                    static_cast<float>(storage[i + 1]),
                    static_cast<float>(storage[i + 2]),
                    static_cast<float>(storage[i + 3]),
                    static_cast<float>(storage[i + 4]),
                    static_cast<float>(storage[i + 5]),
                    static_cast<float>(storage[i + 6]),
                    static_cast<float>(storage[i + 7])
                );
                _mm256_stream_ps(p + i, val);
            }
#endif
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 15) != 0) {
                p[i] = static_cast<float>(storage[i]);
                ++i;
            }
            for (; i + 4 <= n; i += 4) {
                __m128 val = _mm_setr_ps(
                    static_cast<float>(storage[i]),
                    static_cast<float>(storage[i + 1]),
                    static_cast<float>(storage[i + 2]),
                    static_cast<float>(storage[i + 3])
                );
                _mm_stream_ps(p + i, val);
            }
#endif
            for (; i < n; ++i) {
                p[i] = static_cast<float>(storage[i]);
            }
            _mm_sfence();
        } else
#endif
        {
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = static_cast<T>(storage[i]);
            }
        }
    }

    // ============================================================================
    // Explicit Parallel Helpers (OpenMP-guarded)
    // ============================================================================

    template<typename T, typename Alloc, VectorOperand Src>
    inline void assign_par(std::vector<T, Alloc> &dst, const Src &src) {
        const std::size_t n = src.size();
        if (dst.capacity() < n) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::length_error,
                                    "vec_math: destination capacity < source size in assign_par");
        }
        dst.resize(n);
        if (n == 0) return;

        T * VEC_MATH_RESTRICT p = dst.data();
        auto storage = detail::make_operand(src);

#if defined(_OPENMP)
        const std::int64_t n_signed = static_cast<std::int64_t>(n);
        #pragma omp parallel for schedule(static)
        for (std::int64_t i = 0; i < n_signed; ++i) {
            p[i] = static_cast<T>(storage[static_cast<std::size_t>(i)]);
        }
#else
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            p[i] = static_cast<T>(storage[i]);
        }
#endif
    }

    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] inline auto dot_par(const Lhs &lhs_in, const Rhs &rhs_in) {
        if (lhs_in.size() != rhs_in.size()) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in dot_par");
        }
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using ResType = std::common_type_t<
            typename detail::operand_storage_t<Lhs>::value_type,
            typename detail::operand_storage_t<Rhs>::value_type
        >;
        const std::size_t n = lhs_in.size();
        if (n == 0) { return ResType{0}; }

#if defined(_OPENMP)
        ResType total{0};
        const std::int64_t n_signed = static_cast<std::int64_t>(n);
        #pragma omp parallel for schedule(static) reduction(+:total)
        for (std::int64_t i = 0; i < n_signed; ++i) {
            total += static_cast<ResType>(lhs_op[static_cast<std::size_t>(i)]) *
                     static_cast<ResType>(rhs_op[static_cast<std::size_t>(i)]);
        }
        return total;
#else
        return dot(lhs_in, rhs_in);
#endif
    }

    template<VectorOperand Op>
    [[nodiscard]] inline auto sum_par(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) { return value_type{0}; }

#if defined(_OPENMP)
        value_type total{0};
        const std::int64_t n_signed = static_cast<std::int64_t>(n);
        #pragma omp parallel for schedule(static) reduction(+:total)
        for (std::int64_t i = 0; i < n_signed; ++i) {
            total += static_cast<value_type>(op[static_cast<std::size_t>(i)]);
        }
        return total;
#else
        return sum(op_in);
#endif
    }

    // ============================================================================
    // CFL & Time-stepping Reductions
    // ============================================================================

    template<VectorOperand Op>
    [[nodiscard]] constexpr auto min_val(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: min_val called on empty operand");
        }
        value_type res = static_cast<value_type>(op[0]);
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 1; i < n; ++i) {
            value_type val = static_cast<value_type>(op[i]);
            if (val < res) { res = val; }
        }
        return res;
    }

    template<VectorOperand Op>
    [[nodiscard]] constexpr auto max_val(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: max_val called on empty operand");
        }
        value_type res = static_cast<value_type>(op[0]);
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 1; i < n; ++i) {
            value_type val = static_cast<value_type>(op[i]);
            if (val > res) { res = val; }
        }
        return res;
    }

    template<VectorOperand Op>
    [[nodiscard]] inline auto norm_inf(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using Val = typename detail::operand_storage_t<Op>::value_type;
        using ResType = std::conditional_t<std::is_floating_point_v<Val>, Val, double>;
        const std::size_t n = op.size();
        if (n == 0) { return ResType{0}; }
        ResType res{0};
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            ResType val = std::abs(static_cast<ResType>(op[i]));
            if (val > res) { res = val; }
        }
        return res;
    }

    template<VectorOperand Op>
    [[nodiscard]] inline auto norm_l1(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using Val = typename detail::operand_storage_t<Op>::value_type;
        using ResType = std::conditional_t<std::is_floating_point_v<Val>, Val, double>;
        const std::size_t n = op.size();
        if (n == 0) { return ResType{0}; }
        ResType total{0};
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            total += std::abs(static_cast<ResType>(op[i]));
        }
        return total;
    }

    // ============================================================================
    // Element-wise Limiters (Positivity & WENO Limiters)
    // ============================================================================

    template<typename V, typename Lo, typename Hi>
        requires VectorOperand<V> &&
                 (VectorOperand<Lo> || Scalar<Lo>) &&
                 (VectorOperand<Hi> || Scalar<Hi>)
    [[nodiscard]] constexpr auto clamp(const V &v, const Lo &lo, const Hi &hi) {
        return detail::TernaryExpr<detail::OpClamp,
            detail::operand_storage_t<V>,
            detail::operand_storage_t<Lo>,
            detail::operand_storage_t<Hi>>(
                detail::OpClamp{},
                detail::make_operand(v),
                detail::make_operand(lo),
                detail::make_operand(hi)
        );
    }

    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto min(const Lhs &lhs, const Rhs &rhs) {
        return detail::make_bin_expr(detail::OpMin{}, lhs, rhs);
    }

    template<VectorOperand Lhs, Scalar S>
    [[nodiscard]] constexpr auto min(const Lhs &lhs, const S &scalar) {
        return detail::make_bin_expr(detail::OpMin{}, lhs, scalar);
    }

    template<Scalar S, VectorOperand Rhs>
    [[nodiscard]] constexpr auto min(const S &scalar, const Rhs &rhs) {
        return detail::make_bin_expr(detail::OpMin{}, scalar, rhs);
    }

    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto max(const Lhs &lhs, const Rhs &rhs) {
        return detail::make_bin_expr(detail::OpMax{}, lhs, rhs);
    }

    template<VectorOperand Lhs, Scalar S>
    [[nodiscard]] constexpr auto max(const Lhs &lhs, const S &scalar) {
        return detail::make_bin_expr(detail::OpMax{}, lhs, scalar);
    }

    template<Scalar S, VectorOperand Rhs>
    [[nodiscard]] constexpr auto max(const S &scalar, const Rhs &rhs) {
        return detail::make_bin_expr(detail::OpMax{}, scalar, rhs);
    }

    // ============================================================================
    // Element-wise Scientific Functions
    // ============================================================================

    template<VectorOperand Arg>
    [[nodiscard]] inline auto sqrt(const Arg &arg) {
        return detail::make_unary_expr(detail::OpSqrt{}, arg);
    }

    template<VectorOperand Arg>
    [[nodiscard]] inline auto exp(const Arg &arg) {
        return detail::make_unary_expr(detail::OpExp{}, arg);
    }

    template<VectorOperand Arg>
    [[nodiscard]] inline auto log(const Arg &arg) {
        return detail::make_unary_expr(detail::OpLog{}, arg);
    }

    template<VectorOperand Arg>
    [[nodiscard]] inline auto sin(const Arg &arg) {
        return detail::make_unary_expr(detail::OpSin{}, arg);
    }

    template<VectorOperand Arg>
    [[nodiscard]] inline auto cos(const Arg &arg) {
        return detail::make_unary_expr(detail::OpCos{}, arg);
    }

    template<VectorOperand Base, VectorOperand Exp>
    [[nodiscard]] inline auto pow(const Base &base, const Exp &exp) {
        return detail::make_bin_expr(detail::OpPow{}, base, exp);
    }

    template<VectorOperand Base, Scalar Exp>
    [[nodiscard]] inline auto pow(const Base &base, const Exp &exp) {
        return detail::make_bin_expr(detail::OpPow{}, base, exp);
    }

    template<Scalar Base, VectorOperand Exp>
    [[nodiscard]] inline auto pow(const Base &base, const Exp &exp) {
        return detail::make_bin_expr(detail::OpPow{}, base, exp);
    }
}
