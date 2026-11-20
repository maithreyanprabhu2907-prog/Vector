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

#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdlib.h>
#include <new>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#endif

#if !defined(VEC_MATH_NO_INTRINSICS) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#include <immintrin.h>
#define VEC_MATH_HAS_X86_INTRIN 1
#endif

#if !defined(__SYCL_DEVICE_ONLY__) && !defined(VEC_MATH_NO_INTRINSICS)
  #if defined(__GNUC__) || defined(__clang__)
    #if defined(__AVX2__) && defined(__FMA__)
      #define VEC_MATH_HAS_AVX2_FMA 1
    #endif
  #elif defined(_MSC_VER)
    #if defined(__AVX2__)
      #define VEC_MATH_HAS_AVX2_FMA 1
    #endif
  #endif
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
        struct is_std_span : std::false_type {};

        template<typename T, std::size_t Extent>
        struct is_std_span<std::span<T, Extent>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_span_v = is_std_span<std::remove_cvref_t<T>>::value;

        template<typename T>
        struct is_std_array : std::false_type {};

        template<typename T, std::size_t N>
        struct is_std_array<std::array<T, N>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_std_array_v = is_std_array<std::remove_cvref_t<T>>::value;

        template<typename T>
        inline constexpr bool is_contiguous_vector_v = is_std_vector_v<T> || is_std_span_v<T> || is_std_array_v<T>;

        template<typename T>
        struct is_complex_val : std::false_type {};

        template<typename T>
        struct is_complex_val<std::complex<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_complex_v = is_complex_val<std::remove_cvref_t<T>>::value;

        template<typename T>
        struct real_of {
            using type = std::remove_cvref_t<T>;
        };

        template<typename T>
        struct real_of<std::complex<T>> {
            using type = std::remove_cvref_t<T>;
        };

        template<typename T>
        using real_of_t = typename real_of<T>::type;

        template<typename T>
        VEC_MATH_HD_INLINE constexpr auto mag2(const T& v) noexcept {
            if constexpr (is_complex_v<T>) {
                const auto re = v.real();
                const auto im = v.imag();
                return re * re + im * im;
            } else {
                return v * v;
            }
        }

        template<typename T>
        VEC_MATH_HD_INLINE constexpr auto conj_helper(const T& v) {
            if constexpr (is_complex_v<T>) {
                return std::conj(v);
            } else {
                return v;
            }
        }

        template<typename T>
        struct is_valid_numeric : std::bool_constant<
            (std::is_arithmetic_v<std::remove_cvref_t<T>> || is_complex_v<T>) &&
            !std::is_same_v<std::remove_cvref_t<T>, bool>
        > {};

        template<typename T>
        inline constexpr bool is_valid_numeric_v = is_valid_numeric<T>::value;

        struct ExprBase {};

        template<typename T>
        inline constexpr bool is_vec_expr_v = std::is_base_of_v<ExprBase, std::remove_cvref_t<T>>;
    }

    template<typename T>
    using real_of = detail::real_of<T>;

    template<typename T>
    using real_of_t = typename detail::real_of<T>::type;

    template<typename T>
    concept Scalar = detail::is_valid_numeric_v<T>;

    template<typename T>
    concept NumericVector = detail::is_contiguous_vector_v<T> && Scalar<typename std::remove_cvref_t<T>::value_type>;

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
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
            void* ptr = _aligned_malloc(bytes, Alignment);
            if (!ptr) throw std::bad_alloc{};
            return static_cast<T*>(ptr);
#else
            void* ptr = nullptr;
            constexpr std::size_t align = Alignment < sizeof(void*) ? sizeof(void*) : Alignment;
            if (posix_memalign(&ptr, align, bytes) != 0 || !ptr) {
                throw std::bad_alloc{};
            }
            return static_cast<T*>(ptr);
#endif
        }

        void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
            if (!p) return;
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
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
            T * p = dst.data();
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
            T * p = dst.data();
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

            T * p = dst.data();
            auto storage = make_operand(src);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { p[i] = static_cast<T>(storage[i]); }
        }

        template<typename T, std::size_t Extent, VectorOperand Src>
        VEC_MATH_FORCE_INLINE void assign_zero_malloc(std::span<T, Extent> dst, const Src &src) {
            if (dst.size() != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: span size mismatch in assignment");
            }
            T * p = dst.data();
            auto storage = make_operand(src);
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < dst.size(); ++i) { p[i] = static_cast<T>(storage[i]); }
        }

        template<typename T, std::size_t N, VectorOperand Src>
        VEC_MATH_FORCE_INLINE void assign_zero_malloc(std::array<T, N> &dst, const Src &src) {
            if (N != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: array size mismatch in assignment");
            }
            T * p = dst.data();
            auto storage = make_operand(src);
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < N; ++i) { p[i] = static_cast<T>(storage[i]); }
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

        // Zero-malloc assign for std::span
        template<typename T, std::size_t Extent, VectorOperand Src>
        inline std::span<T, Extent> operator<<=(std::span<T, Extent> dst, const Src &src) {
            detail::assign_zero_malloc(dst, src);
            return dst;
        }

        // Zero-malloc assign for std::array
        template<typename T, std::size_t N, VectorOperand Src>
        inline std::array<T, N> &operator<<=(std::array<T, N> &dst, const Src &src) {
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

    // sum_deterministic: bit-identical strictly sequential reduction for regression tests
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto sum_deterministic(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        value_type total{0};
        for (std::size_t i = 0; i < n; ++i) {
            total = total + static_cast<value_type>(op[i]);
        }
        return total;
    }

    // sum: reduction with 8-way multi-accumulator dependency breaking
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto sum(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) { return value_type{0}; }

        value_type a0{0}, a1{0}, a2{0}, a3{0}, a4{0}, a5{0}, a6{0}, a7{0};
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            a0 += static_cast<value_type>(op[i + 0]);
            a1 += static_cast<value_type>(op[i + 1]);
            a2 += static_cast<value_type>(op[i + 2]);
            a3 += static_cast<value_type>(op[i + 3]);
            a4 += static_cast<value_type>(op[i + 4]);
            a5 += static_cast<value_type>(op[i + 5]);
            a6 += static_cast<value_type>(op[i + 6]);
            a7 += static_cast<value_type>(op[i + 7]);
        }
        for (; i < n; ++i) {
            a0 += static_cast<value_type>(op[i]);
        }
        return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    }

    // dot_deterministic: bit-identical strictly sequential inner product for regression tests
    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto dot_deterministic(const Lhs &lhs_in, const Rhs &rhs_in) {
        if (lhs_in.size() != rhs_in.size()) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in dot_deterministic");
        }
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using ResType = std::common_type_t<
            typename detail::operand_storage_t<Lhs>::value_type,
            typename detail::operand_storage_t<Rhs>::value_type
        >;
        const std::size_t n = lhs_in.size();
        ResType total{0};
        for (std::size_t i = 0; i < n; ++i) {
            total = total + static_cast<ResType>(lhs_op[i]) * static_cast<ResType>(rhs_op[i]);
        }
        return total;
    }

    // dot: dot product with SIMD intrinsics and 8-way multi-accumulator dependency breaking
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

#if defined(VEC_MATH_HAS_AVX2_FMA)
        if constexpr (requires { { lhs_op.data() }; { rhs_op.data() }; } && std::is_same_v<ResType, double>) {
            const double* VEC_MATH_RESTRICT p1 = lhs_op.data();
            const double* VEC_MATH_RESTRICT p2 = rhs_op.data();
            __m256d acc0 = _mm256_setzero_pd();
            __m256d acc1 = _mm256_setzero_pd();
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                __m256d l0 = _mm256_loadu_pd(p1 + i);
                __m256d r0 = _mm256_loadu_pd(p2 + i);
                acc0 = _mm256_fmadd_pd(l0, r0, acc0);

                __m256d l1 = _mm256_loadu_pd(p1 + i + 4);
                __m256d r1 = _mm256_loadu_pd(p2 + i + 4);
                acc1 = _mm256_fmadd_pd(l1, r1, acc1);
            }
            acc0 = _mm256_add_pd(acc0, acc1);
            __m128d lo = _mm256_castpd256_pd128(acc0);
            __m128d hi = _mm256_extractf128_pd(acc0, 1);
            __m128d sum128 = _mm_add_pd(lo, hi);
            __m128d high_half = _mm_unpackhi_pd(sum128, sum128);
            __m128d scalar_sum = _mm_add_sd(sum128, high_half);
            double total = _mm_cvtsd_f64(scalar_sum);

            for (; i < n; ++i) {
                total += p1[i] * p2[i];
            }
            return total;
        } else if constexpr (requires { { lhs_op.data() }; { rhs_op.data() }; } && std::is_same_v<ResType, float>) {
            const float* VEC_MATH_RESTRICT p1 = lhs_op.data();
            const float* VEC_MATH_RESTRICT p2 = rhs_op.data();
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            std::size_t i = 0;
            for (; i + 16 <= n; i += 16) {
                __m256 l0 = _mm256_loadu_ps(p1 + i);
                __m256 r0 = _mm256_loadu_ps(p2 + i);
                acc0 = _mm256_fmadd_ps(l0, r0, acc0);

                __m256 l1 = _mm256_loadu_ps(p1 + i + 8);
                __m256 r1 = _mm256_loadu_ps(p2 + i + 8);
                acc1 = _mm256_fmadd_ps(l1, r1, acc1);
            }
            acc0 = _mm256_add_ps(acc0, acc1);
            __m128 lo = _mm256_castps256_ps128(acc0);
            __m128 hi = _mm256_extractf128_ps(acc0, 1);
            __m128 sum128 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehl_ps(sum128, sum128);
            __m128 sum64 = _mm_add_ps(sum128, shuf);
            __m128 shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
            __m128 scalar_sum = _mm_add_ss(sum64, shuf2);
            float total = _mm_cvtss_f32(scalar_sum);

            for (; i < n; ++i) {
                total += p1[i] * p2[i];
            }
            return total;
        } else
#endif
        {
            ResType a0{0}, a1{0}, a2{0}, a3{0}, a4{0}, a5{0}, a6{0}, a7{0};
            std::size_t i = 0;
            for (; i + 8 <= n; i += 8) {
                a0 += static_cast<ResType>(lhs_op[i + 0]) * static_cast<ResType>(rhs_op[i + 0]);
                a1 += static_cast<ResType>(lhs_op[i + 1]) * static_cast<ResType>(rhs_op[i + 1]);
                a2 += static_cast<ResType>(lhs_op[i + 2]) * static_cast<ResType>(rhs_op[i + 2]);
                a3 += static_cast<ResType>(lhs_op[i + 3]) * static_cast<ResType>(rhs_op[i + 3]);
                a4 += static_cast<ResType>(lhs_op[i + 4]) * static_cast<ResType>(rhs_op[i + 4]);
                a5 += static_cast<ResType>(lhs_op[i + 5]) * static_cast<ResType>(rhs_op[i + 5]);
                a6 += static_cast<ResType>(lhs_op[i + 6]) * static_cast<ResType>(rhs_op[i + 6]);
                a7 += static_cast<ResType>(lhs_op[i + 7]) * static_cast<ResType>(rhs_op[i + 7]);
            }
            for (; i < n; ++i) {
                a0 += static_cast<ResType>(lhs_op[i]) * static_cast<ResType>(rhs_op[i]);
            }
            return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
        }
    }

    // norm_sq: squared L2 norm (single-pass, zero redundant evaluation)
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto norm_sq(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using Val = typename detail::operand_storage_t<Op>::value_type;
        using RealVal = typename real_of<Val>::type;
        using ResType = std::conditional_t<std::is_floating_point_v<RealVal>, RealVal, double>;
        const std::size_t n = op.size();
        if (n == 0) { return ResType{0}; }

        ResType a0{0}, a1{0}, a2{0}, a3{0}, a4{0}, a5{0}, a6{0}, a7{0};
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            a0 += static_cast<ResType>(detail::mag2(op[i + 0]));
            a1 += static_cast<ResType>(detail::mag2(op[i + 1]));
            a2 += static_cast<ResType>(detail::mag2(op[i + 2]));
            a3 += static_cast<ResType>(detail::mag2(op[i + 3]));
            a4 += static_cast<ResType>(detail::mag2(op[i + 4]));
            a5 += static_cast<ResType>(detail::mag2(op[i + 5]));
            a6 += static_cast<ResType>(detail::mag2(op[i + 6]));
            a7 += static_cast<ResType>(detail::mag2(op[i + 7]));
        }
        for (; i < n; ++i) {
            a0 += static_cast<ResType>(detail::mag2(op[i]));
        }
        return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
    }

    // norm: Euclidean L2 norm
    template<VectorOperand Op>
    [[nodiscard]] inline auto norm(const Op &op) {
        using Val = typename detail::operand_storage_t<Op>::value_type;
        using RealVal = typename real_of<Val>::type;
        using ResType = std::conditional_t<std::is_floating_point_v<RealVal>, RealVal, double>;
        return std::sqrt(static_cast<ResType>(norm_sq(op)));
    }

    // dotc: conjugate inner product for complex vectors (std::conj on lhs)
    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto dotc(const Lhs &lhs_in, const Rhs &rhs_in) {
        if (lhs_in.size() != rhs_in.size()) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in dotc");
        }
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using LhsVal = typename detail::operand_storage_t<Lhs>::value_type;
        using RhsVal = typename detail::operand_storage_t<Rhs>::value_type;
        using ResType = std::common_type_t<LhsVal, RhsVal>;
        const std::size_t n = lhs_in.size();
        if (n == 0) { return ResType{0}; }

        ResType a0{0}, a1{0}, a2{0}, a3{0}, a4{0}, a5{0}, a6{0}, a7{0};
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8) {
            a0 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 0])) * static_cast<ResType>(rhs_op[i + 0]);
            a1 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 1])) * static_cast<ResType>(rhs_op[i + 1]);
            a2 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 2])) * static_cast<ResType>(rhs_op[i + 2]);
            a3 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 3])) * static_cast<ResType>(rhs_op[i + 3]);
            a4 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 4])) * static_cast<ResType>(rhs_op[i + 4]);
            a5 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 5])) * static_cast<ResType>(rhs_op[i + 5]);
            a6 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 6])) * static_cast<ResType>(rhs_op[i + 6]);
            a7 += static_cast<ResType>(detail::conj_helper(lhs_op[i + 7])) * static_cast<ResType>(rhs_op[i + 7]);
        }
        for (; i < n; ++i) {
            a0 += static_cast<ResType>(detail::conj_helper(lhs_op[i])) * static_cast<ResType>(rhs_op[i]);
        }
        return ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
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
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
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
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
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
    // Direct Device / Buffer Evaluation & Matrix-Vector Operations
    // ============================================================================

    template<typename T, VectorOperand Src>
    VEC_MATH_HD_INLINE void eval_into(T* dst, std::size_t n, const Src& src) {
        if (n != src.size()) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: destination size mismatch in eval_into");
        }
        auto storage = detail::make_operand(src);
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<T>(storage[i]);
        }
    }

    template<typename MatT, typename VecInT, typename VecOutT>
    VEC_MATH_HOST_DEVICE inline void matvec(const MatT* A, const VecInT* x, VecOutT* y, std::size_t rows, std::size_t cols) {
        if (rows == 0 || cols == 0) return;
        const bool aliased = (static_cast<const void*>(x) == static_cast<const void*>(y));
        constexpr std::size_t STACK_BUF_CAP = 128;
        if (aliased) {
            if (rows <= STACK_BUF_CAP) {
                VecOutT local_buf[STACK_BUF_CAP];
                matvec(A, x, local_buf, rows, cols);
                for (std::size_t r = 0; r < rows; ++r) {
                    y[r] = local_buf[r];
                }
                return;
            }
#if !defined(__SYCL_DEVICE_ONLY__)
            else {
                std::vector<VecOutT> local_buf(rows);
                matvec(A, x, local_buf.data(), rows, cols);
                for (std::size_t r = 0; r < rows; ++r) {
                    y[r] = local_buf[r];
                }
                return;
            }
#endif
        }

        for (std::size_t r = 0; r < rows; ++r) {
            VecOutT sum{0};
            const MatT* row = A + r * cols;
            if (cols <= 36) {
                std::size_t c = 0;
                for (; c + 4 <= cols; c += 4) {
                    sum = sum + static_cast<VecOutT>(row[c + 0]) * static_cast<VecOutT>(x[c + 0])
                              + static_cast<VecOutT>(row[c + 1]) * static_cast<VecOutT>(x[c + 1])
                              + static_cast<VecOutT>(row[c + 2]) * static_cast<VecOutT>(x[c + 2])
                              + static_cast<VecOutT>(row[c + 3]) * static_cast<VecOutT>(x[c + 3]);
                }
                for (; c < cols; ++c) {
                    sum = sum + static_cast<VecOutT>(row[c]) * static_cast<VecOutT>(x[c]);
                }
            } else {
                VEC_MATH_PRAGMA_SIMD
                for (std::size_t c = 0; c < cols; ++c) {
                    sum = sum + static_cast<VecOutT>(row[c]) * static_cast<VecOutT>(x[c]);
                }
            }
            y[r] = sum;
        }
    }

    template<typename MatT, typename VecInT, typename VecOutT>
    VEC_MATH_HOST_DEVICE inline void matvec_transposed(const MatT* A, const VecInT* x, VecOutT* y, std::size_t rows, std::size_t cols) {
        if (rows == 0 || cols == 0) return;
        const bool aliased = (static_cast<const void*>(x) == static_cast<const void*>(y));
        constexpr std::size_t STACK_BUF_CAP = 128;
        if (aliased) {
            if (cols <= STACK_BUF_CAP) {
                VecOutT local_buf[STACK_BUF_CAP];
                matvec_transposed(A, x, local_buf, rows, cols);
                for (std::size_t c = 0; c < cols; ++c) {
                    y[c] = local_buf[c];
                }
                return;
            }
#if !defined(__SYCL_DEVICE_ONLY__)
            else {
                std::vector<VecOutT> local_buf(cols);
                matvec_transposed(A, x, local_buf.data(), rows, cols);
                for (std::size_t c = 0; c < cols; ++c) {
                    y[c] = local_buf[c];
                }
                return;
            }
#endif
        }

        for (std::size_t c = 0; c < cols; ++c) {
            y[c] = VecOutT{0};
        }
        for (std::size_t r = 0; r < rows; ++r) {
            const MatT* row = A + r * cols;
            const auto xr = static_cast<VecOutT>(x[r]);
            if (cols <= 36) {
                std::size_t c = 0;
                for (; c + 4 <= cols; c += 4) {
                    y[c + 0] = y[c + 0] + static_cast<VecOutT>(row[c + 0]) * xr;
                    y[c + 1] = y[c + 1] + static_cast<VecOutT>(row[c + 1]) * xr;
                    y[c + 2] = y[c + 2] + static_cast<VecOutT>(row[c + 2]) * xr;
                    y[c + 3] = y[c + 3] + static_cast<VecOutT>(row[c + 3]) * xr;
                }
                for (; c < cols; ++c) {
                    y[c] = y[c] + static_cast<VecOutT>(row[c]) * xr;
                }
            } else {
                VEC_MATH_PRAGMA_SIMD
                for (std::size_t c = 0; c < cols; ++c) {
                    y[c] = y[c] + static_cast<VecOutT>(row[c]) * xr;
                }
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
