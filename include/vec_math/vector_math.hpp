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
 *      - Modern C++26 style, zero-allocation throughout all operations.
 *      - Portable C++26 std::simd vectorized kernels (AVX2, AVX-512, ARM NEON/SVE).
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
#if !defined(__SYCL_DEVICE_ONLY__) && __has_include(<simd>)
#include <simd>
#if defined(__glibcxx_simd) || defined(__cpp_lib_simd)
#define VEC_MATH_HAS_STD_SIMD 1
#endif
#endif
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

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define VEC_MATH_PRAGMA_SIMD __pragma(loop(ivdep))
#elif defined(__clang__)
#define VEC_MATH_PRAGMA_SIMD _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__)
#define VEC_MATH_PRAGMA_SIMD _Pragma("GCC ivdep")
#elif defined(_OPENMP)
#define VEC_MATH_PRAGMA_SIMD _Pragma("omp simd")
#else
#define VEC_MATH_PRAGMA_SIMD
#endif

#if defined(__SYCL_DEVICE_ONLY__)
#include <cassert>
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) ((void)0)
#define VEC_MATH_THROW_BAD_ALLOC(msg) ((void)0)
#define VEC_MATH_DEVICE_ASSERT(cond, msg) assert((cond) && (msg))
#elif defined(VEC_MATH_NO_EXCEPTIONS)
#include <cassert>
#include <cstdlib>
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) do { assert(false && (msg)); std::abort(); } while (false)
#define VEC_MATH_THROW_BAD_ALLOC(msg) do { assert(false && (msg)); std::abort(); } while (false)
#define VEC_MATH_DEVICE_ASSERT(cond, msg) do { if (!(cond)) { assert(false && (msg)); std::abort(); } } while (false)
#else
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) throw exc_type(msg)
#define VEC_MATH_THROW_BAD_ALLOC(msg) throw std::bad_alloc()
#define VEC_MATH_DEVICE_ASSERT(cond, msg) do { if (!(cond)) { throw std::invalid_argument(msg); } } while (false)
#endif

#if defined(VEC_MATH_NO_CHECKS)
#define VEC_MATH_CHECK_SIZE_EQ(sz1, sz2, msg) ((void)0)
#else
#define VEC_MATH_CHECK_SIZE_EQ(sz1, sz2, msg) \
    do { \
        if ((sz1) != (sz2)) [[unlikely]] { \
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, msg); \
        } \
    } while (false)
#endif

namespace vec_math {

    template<typename T>
    class StridedSpan;

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
        struct is_strided_span : std::false_type {};

        template<typename T>
        struct is_strided_span<StridedSpan<T>> : std::true_type {};

        template<typename T>
        inline constexpr bool is_strided_span_v = is_strided_span<std::remove_cvref_t<T>>::value;

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
    concept NumericVector = (detail::is_contiguous_vector_v<T> || detail::is_strided_span_v<T>) && Scalar<typename std::remove_cvref_t<T>::value_type>;

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

    namespace detail {
        inline void* aligned_alloc_helper(std::size_t align, std::size_t bytes) {
            if (bytes == 0) return nullptr;
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
            void* ptr = _aligned_malloc(bytes, align);
            if (!ptr) throw std::bad_alloc{};
            return ptr;
#else
            void* ptr = nullptr;
            std::size_t a = align < sizeof(void*) ? sizeof(void*) : align;
            if (posix_memalign(&ptr, a, bytes) != 0 || !ptr) {
                throw std::bad_alloc{};
            }
            return ptr;
#endif
        }

        inline void aligned_free_helper(void* ptr) noexcept {
            if (!ptr) return;
#if defined(_WIN32) || defined(_MSC_VER) || defined(__MINGW32__)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    }

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
            return static_cast<T*>(detail::aligned_alloc_helper(Alignment, n * sizeof(T)));
        }

        void deallocate(T* p, [[maybe_unused]] std::size_t n) noexcept {
            detail::aligned_free_helper(p);
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
            const value_type* ptr_ = nullptr;
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
            using type = std::conditional_t<
                is_strided_span_v<Vec>,
                std::remove_cvref_t<Vec>,
                VecRef<std::remove_cvref_t<Vec>>
            >;
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

            Op op;
            Lhs lhs;
            Rhs rhs;

            constexpr BinExpr(Op o, const Lhs& l, const Rhs& r) : op(o), lhs(l), rhs(r) {
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

            Op op;
            Operand operand;

            constexpr UnaryExpr(Op o, const Operand& arg) : op(o), operand(arg) {}

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

            Op op;
            A a;
            B b;
            C c;

            constexpr TernaryExpr(Op o, const A& arg_a, const B& arg_b, const C& arg_c)
                : op(o), a(arg_a), b(arg_b), c(arg_c) {
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

        inline bool memory_ranges_overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
            if (!a || !b || a_bytes == 0 || b_bytes == 0) return false;
            const auto* a_beg = static_cast<const char*>(a);
            const auto* a_end = a_beg + a_bytes;
            const auto* b_beg = static_cast<const char*>(b);
            const auto* b_end = b_beg + b_bytes;
            return (a_beg < b_end) && (b_beg < a_end);
        }

        template<typename S>
        VEC_MATH_HD_INLINE constexpr bool has_partial_overlap(const ScalarVal<S>&, const char*, const char*) noexcept {
            return false;
        }

        template<typename Vec>
        VEC_MATH_HD_INLINE bool has_partial_overlap(const VecRef<Vec>& ref, const char* dst_begin, const char* dst_end) noexcept {
            if (!ref.data()) return false;
            const char* p_begin = reinterpret_cast<const char*>(ref.data());
            const char* p_end = p_begin + ref.size() * sizeof(typename VecRef<Vec>::value_type);
            return (p_begin < dst_end) && (dst_begin < p_end) && (p_begin != dst_begin);
        }

        template<typename T>
        VEC_MATH_HD_INLINE bool has_partial_overlap(const StridedSpan<T>& ref, const char* dst_begin, const char* dst_end) noexcept {
            if (!ref.data() || ref.size() == 0) return false;
            const char* p_begin = reinterpret_cast<const char*>(ref.data());
            const char* p_end = p_begin + ref.size() * ref.stride() * sizeof(T);
            return (p_begin < dst_end) && (dst_begin < p_end) && (p_begin != dst_begin);
        }

        template<typename Op, typename L, typename R>
        VEC_MATH_HD_INLINE bool has_partial_overlap(const BinExpr<Op, L, R>& expr, const char* dst_begin, const char* dst_end) noexcept {
            return has_partial_overlap(expr.lhs, dst_begin, dst_end) || has_partial_overlap(expr.rhs, dst_begin, dst_end);
        }

        template<typename Op, typename Arg>
        VEC_MATH_HD_INLINE bool has_partial_overlap(const UnaryExpr<Op, Arg>& expr, const char* dst_begin, const char* dst_end) noexcept {
            return has_partial_overlap(expr.operand, dst_begin, dst_end);
        }

        template<typename Op, typename A, typename B, typename C>
        VEC_MATH_HD_INLINE bool has_partial_overlap(const TernaryExpr<Op, A, B, C>& expr, const char* dst_begin, const char* dst_end) noexcept {
            return has_partial_overlap(expr.a, dst_begin, dst_end) ||
                   has_partial_overlap(expr.b, dst_begin, dst_end) ||
                   has_partial_overlap(expr.c, dst_begin, dst_end);
        }
    }

    template<typename T, VectorOperand Src>
    VEC_MATH_HD_INLINE void eval_into(T* dst, std::size_t n, const Src& src) {
        if (n != src.size()) [[unlikely]] {
            VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: destination size mismatch in eval_into");
        }
        if (n == 0) return;
        auto storage = detail::make_operand(src);
        const char* dst_begin = reinterpret_cast<const char*>(dst);
        const char* dst_end = dst_begin + n * sizeof(T);
        if (detail::has_partial_overlap(storage, dst_begin, dst_end)) {
            constexpr std::size_t STACK_BUF_CAP = 1024;
            if (n <= STACK_BUF_CAP) {
                T local_buf[STACK_BUF_CAP];
                for (std::size_t i = 0; i < n; ++i) {
                    local_buf[i] = static_cast<T>(storage[i]);
                }
                for (std::size_t i = 0; i < n; ++i) {
                    dst[i] = local_buf[i];
                }
                return;
            }
#if defined(VEC_MATH_STRICT_ZERO_MALLOC)
            else {
                VEC_MATH_THROW_BAD_ALLOC("vec_math: eval_into partial overlap exceeds stack buffer capacity under VEC_MATH_STRICT_ZERO_MALLOC");
            }
#elif !defined(__SYCL_DEVICE_ONLY__)
            else {
                std::vector<T> local_buf(n);
                for (std::size_t i = 0; i < n; ++i) {
                    local_buf[i] = static_cast<T>(storage[i]);
                }
                for (std::size_t i = 0; i < n; ++i) {
                    dst[i] = local_buf[i];
                }
                return;
            }
#endif
        }

        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = static_cast<T>(storage[i]);
        }
    }

    namespace detail {
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
            VEC_MATH_CHECK_SIZE_EQ(dst.size(), src.size(), "vec_math: vector size mismatch in compound assignment");
            const std::size_t n = dst.size();
            T * p = dst.data();
            auto storage = make_operand(src);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { fn(p[i], static_cast<T>(storage[i])); }
        }

        template<typename T, std::size_t Extent, Scalar S, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_scalar_inplace(std::span<T, Extent> dst, const S &scalar, CompoundFn fn) noexcept {
            const std::size_t n = dst.size();
            T * p = dst.data();
            const T s = static_cast<T>(scalar);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { fn(p[i], s); }
        }

        template<typename T, std::size_t Extent, VectorOperand Src, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_vec_inplace(std::span<T, Extent> dst, const Src &src, CompoundFn fn) {
            VEC_MATH_CHECK_SIZE_EQ(dst.size(), src.size(), "vec_math: span size mismatch in compound assignment");
            const std::size_t n = dst.size();
            T * p = dst.data();
            auto storage = make_operand(src);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) { fn(p[i], static_cast<T>(storage[i])); }
        }

        template<typename T, std::size_t N, Scalar S, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_scalar_inplace(std::array<T, N> &dst, const S &scalar, CompoundFn fn) noexcept {
            T * p = dst.data();
            const T s = static_cast<T>(scalar);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < N; ++i) { fn(p[i], s); }
        }

        template<typename T, std::size_t N, VectorOperand Src, typename CompoundFn>
        VEC_MATH_FORCE_INLINE void apply_vec_inplace(std::array<T, N> &dst, const Src &src, CompoundFn fn) {
            VEC_MATH_CHECK_SIZE_EQ(N, src.size(), "vec_math: array size mismatch in compound assignment");
            T * p = dst.data();
            auto storage = make_operand(src);

            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < N; ++i) { fn(p[i], static_cast<T>(storage[i])); }
        }

        template<typename T, typename Alloc, VectorOperand Src>
        VEC_MATH_FORCE_INLINE void assign_zero_malloc(std::vector<T, Alloc> &dst, const Src &src) {
            const std::size_t n = src.size();
            if (dst.capacity() < n) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::length_error,
                                        "vec_math: destination capacity < source size (zero-malloc invariant violated)");
            }
            dst.resize(n);
            eval_into(dst.data(), n, src);
        }

        template<typename T, std::size_t Extent, VectorOperand Src>
        VEC_MATH_FORCE_INLINE void assign_zero_malloc(std::span<T, Extent> dst, const Src &src) {
            VEC_MATH_CHECK_SIZE_EQ(dst.size(), src.size(), "vec_math: span size mismatch in assignment");
            eval_into(dst.data(), dst.size(), src);
        }

        template<typename T, std::size_t N, VectorOperand Src>
        VEC_MATH_FORCE_INLINE void assign_zero_malloc(std::array<T, N> &dst, const Src &src) {
            VEC_MATH_CHECK_SIZE_EQ(N, src.size(), "vec_math: array size mismatch in assignment");
            eval_into(dst.data(), N, src);
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

        // Compound assignment operators for std::span
        template<typename T, std::size_t Extent, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator+=(std::span<T, Extent> lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a += b; });
            return lhs;
        }

        template<typename T, std::size_t Extent, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator+=(std::span<T, Extent> lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a += s; });
            return lhs;
        }

        template<typename T, std::size_t Extent, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator-=(std::span<T, Extent> lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a -= b; });
            return lhs;
        }

        template<typename T, std::size_t Extent, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator-=(std::span<T, Extent> lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a -= s; });
            return lhs;
        }

        template<typename T, std::size_t Extent, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator*=(std::span<T, Extent> lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a *= b; });
            return lhs;
        }

        template<typename T, std::size_t Extent, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator*=(std::span<T, Extent> lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a *= s; });
            return lhs;
        }

        template<typename T, std::size_t Extent, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator/=(std::span<T, Extent> lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a /= b; });
            return lhs;
        }

        template<typename T, std::size_t Extent, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::span<T, Extent> operator/=(std::span<T, Extent> lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a /= s; });
            return lhs;
        }

        // Compound assignment operators for std::array
        template<typename T, std::size_t N, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator+=(std::array<T, N> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a += b; });
            return lhs;
        }

        template<typename T, std::size_t N, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator+=(std::array<T, N> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a += s; });
            return lhs;
        }

        template<typename T, std::size_t N, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator-=(std::array<T, N> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a -= b; });
            return lhs;
        }

        template<typename T, std::size_t N, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator-=(std::array<T, N> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a -= s; });
            return lhs;
        }

        template<typename T, std::size_t N, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator*=(std::array<T, N> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a *= b; });
            return lhs;
        }

        template<typename T, std::size_t N, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator*=(std::array<T, N> &lhs, const S &scalar) noexcept {
            detail::apply_scalar_inplace(lhs, scalar, [](T &a, const T &s) noexcept { a *= s; });
            return lhs;
        }

        template<typename T, std::size_t N, VectorOperand Rhs> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator/=(std::array<T, N> &lhs, const Rhs &rhs) {
            detail::apply_vec_inplace(lhs, rhs, [](T &a, const T &b) noexcept { a /= b; });
            return lhs;
        }

        template<typename T, std::size_t N, Scalar S> requires detail::is_valid_numeric_v<T>
        inline std::array<T, N> &operator/=(std::array<T, N> &lhs, const S &scalar) noexcept {
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

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math", "no-associative-math")
#endif

    namespace detail {
        template<typename ValueT, typename Accessor>
        constexpr ValueT pairwise_tree_sum(Accessor&& acc, std::size_t start, std::size_t count) {
            if (count == 0) return ValueT{0};
            if (count == 1) return static_cast<ValueT>(acc(start));
            if (count == 2) return static_cast<ValueT>(acc(start)) + static_cast<ValueT>(acc(start + 1));
            const std::size_t mid = count / 2;
            return pairwise_tree_sum<ValueT>(acc, start, mid) +
                   pairwise_tree_sum<ValueT>(acc, start + mid, count - mid);
        }

        template<typename ValueT, typename BlockFunc>
        constexpr ValueT pairwise_tree_blocks(BlockFunc&& block_fn, std::size_t b_start, std::size_t b_count) {
            if (b_count == 0) return ValueT{0};
            if (b_count == 1) return block_fn(b_start);
            if (b_count == 2) return block_fn(b_start) + block_fn(b_start + 1);
            const std::size_t mid = b_count / 2;
            return pairwise_tree_blocks<ValueT>(block_fn, b_start, mid) +
                   pairwise_tree_blocks<ValueT>(block_fn, b_start + mid, b_count - mid);
        }
    }

    enum class ReductionPolicy {
        FAST,
        REPRO
    };

    // sum_repro: Blocked tree reduction guaranteeing bit-identical results across thread counts and bounded FP32 error
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto sum_repro(const Op &op_in) {
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;
        const std::size_t n = op.size();
        if (n == 0) return value_type{0};

        constexpr std::size_t BLOCK_SIZE = 512;
        const std::size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        auto block_fn = [&](std::size_t b) -> value_type {
            const std::size_t start = b * BLOCK_SIZE;
            const std::size_t count = (start + BLOCK_SIZE < n) ? BLOCK_SIZE : (n - start);
            return detail::pairwise_tree_sum<value_type>([&](std::size_t idx) {
                return op[idx];
            }, start, count);
        };

        return detail::pairwise_tree_blocks<value_type>(block_fn, 0, num_blocks);
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

    // dot_repro: Blocked tree inner product guaranteeing bit-identical results across thread counts and bounded FP32 error
    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto dot_repro(const Lhs &lhs_in, const Rhs &rhs_in) {
        VEC_MATH_CHECK_SIZE_EQ(lhs_in.size(), rhs_in.size(), "vec_math: vector size mismatch in dot_repro");
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using ResType = std::common_type_t<
            typename detail::operand_storage_t<Lhs>::value_type,
            typename detail::operand_storage_t<Rhs>::value_type
        >;
        const std::size_t n = lhs_in.size();
        if (n == 0) return ResType{0};

        constexpr std::size_t BLOCK_SIZE = 512;
        const std::size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        auto block_fn = [&](std::size_t b) -> ResType {
            const std::size_t start = b * BLOCK_SIZE;
            const std::size_t count = (start + BLOCK_SIZE < n) ? BLOCK_SIZE : (n - start);
            return detail::pairwise_tree_sum<ResType>([&](std::size_t idx) {
                return static_cast<ResType>(lhs_op[idx]) * static_cast<ResType>(rhs_op[idx]);
            }, start, count);
        };

        return detail::pairwise_tree_blocks<ResType>(block_fn, 0, num_blocks);
    }

    // dot_deterministic: bit-identical strictly sequential inner product for regression tests
    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto dot_deterministic(const Lhs &lhs_in, const Rhs &rhs_in) {
        VEC_MATH_CHECK_SIZE_EQ(lhs_in.size(), rhs_in.size(), "vec_math: vector size mismatch in dot_deterministic");
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

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

    // sum: reduction with portable C++26 std::simd and dual-accumulator dependency breaking
    template<VectorOperand Op>
    [[nodiscard]] constexpr auto sum(const Op &op_in, ReductionPolicy policy = ReductionPolicy::FAST) {
        if (policy == ReductionPolicy::REPRO) {
            return sum_repro(op_in);
        }
        auto op = detail::make_operand(op_in);
        using value_type = std::remove_const_t<typename detail::operand_storage_t<Op>::value_type>;
        const std::size_t n = op.size();
        if (n == 0) { return value_type{0}; }

#if defined(VEC_MATH_HAS_STD_SIMD)
        if !consteval {
            if constexpr (!detail::is_strided_span_v<decltype(op)> && requires { { op.data() }; } &&
                          (std::is_same_v<value_type, double> || std::is_same_v<value_type, float>)) {
                using V = std::simd::vec<value_type>;
                constexpr std::size_t W = V::size();
                const value_type* VEC_MATH_RESTRICT p = op.data();
                V acc0(value_type{0});
                V acc1(value_type{0});
                std::size_t i = 0;
                for (; i + 2 * W <= n; i += 2 * W) {
                    acc0 += std::simd::unchecked_load<V>(std::span<const value_type>(p + i, W));
                    acc1 += std::simd::unchecked_load<V>(std::span<const value_type>(p + i + W, W));
                }
                for (; i + W <= n; i += W) {
                    acc0 += std::simd::unchecked_load<V>(std::span<const value_type>(p + i, W));
                }
                value_type total = std::simd::reduce(acc0 + acc1);
                for (; i < n; ++i) {
                    total += p[i];
                }
                return total;
            }
        }
#endif
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

    // dot: dot product with portable C++26 std::simd and dual-accumulator dependency breaking
    template<VectorOperand Lhs, VectorOperand Rhs>
    [[nodiscard]] constexpr auto dot(const Lhs &lhs_in, const Rhs &rhs_in, ReductionPolicy policy = ReductionPolicy::FAST) {
        if (policy == ReductionPolicy::REPRO) {
            return dot_repro(lhs_in, rhs_in);
        }
        VEC_MATH_CHECK_SIZE_EQ(lhs_in.size(), rhs_in.size(), "vec_math: vector size mismatch in dot product");
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using LhsVal = std::remove_const_t<typename detail::operand_storage_t<Lhs>::value_type>;
        using RhsVal = std::remove_const_t<typename detail::operand_storage_t<Rhs>::value_type>;
        using ResType = std::common_type_t<LhsVal, RhsVal>;
        const std::size_t n = lhs_in.size();
        if (n == 0) { return ResType{0}; }

#if defined(VEC_MATH_HAS_STD_SIMD)
        if !consteval {
            if constexpr (!detail::is_strided_span_v<decltype(lhs_op)> && !detail::is_strided_span_v<decltype(rhs_op)> &&
                          requires { { lhs_op.data() }; { rhs_op.data() }; } &&
                          std::is_same_v<LhsVal, ResType> && std::is_same_v<RhsVal, ResType> &&
                          (std::is_same_v<ResType, double> || std::is_same_v<ResType, float>)) {
                using V = std::simd::vec<ResType>;
                constexpr std::size_t W = V::size();
                const ResType* VEC_MATH_RESTRICT p1 = lhs_op.data();
                const ResType* VEC_MATH_RESTRICT p2 = rhs_op.data();
                V acc0(ResType{0});
                V acc1(ResType{0});
                std::size_t i = 0;
                for (; i + 2 * W <= n; i += 2 * W) {
                    const V l0 = std::simd::unchecked_load<V>(std::span<const ResType>(p1 + i, W));
                    const V r0 = std::simd::unchecked_load<V>(std::span<const ResType>(p2 + i, W));
                    acc0 += l0 * r0;
                    const V l1 = std::simd::unchecked_load<V>(std::span<const ResType>(p1 + i + W, W));
                    const V r1 = std::simd::unchecked_load<V>(std::span<const ResType>(p2 + i + W, W));
                    acc1 += l1 * r1;
                }
                for (; i + W <= n; i += W) {
                    const V l0 = std::simd::unchecked_load<V>(std::span<const ResType>(p1 + i, W));
                    const V r0 = std::simd::unchecked_load<V>(std::span<const ResType>(p2 + i, W));
                    acc0 += l0 * r0;
                }
                ResType total = std::simd::reduce(acc0 + acc1);
                for (; i < n; ++i) {
                    total += p1[i] * p2[i];
                }
                return total;
            }
        }
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
            if constexpr (!detail::is_strided_span_v<decltype(storage)> && requires { { storage.data() }; } && std::is_same_v<typename decltype(storage)::value_type, double>) {
                const double* VEC_MATH_RESTRICT sp = storage.data();
                for (; i + 4 <= n; i += 4) {
                    _mm256_stream_pd(p + i, _mm256_loadu_pd(sp + i));
                }
            } else {
                for (; i + 4 <= n; i += 4) {
                    __m256d val = _mm256_setr_pd(
                        static_cast<double>(storage[i]),
                        static_cast<double>(storage[i + 1]),
                        static_cast<double>(storage[i + 2]),
                        static_cast<double>(storage[i + 3])
                    );
                    _mm256_stream_pd(p + i, val);
                }
            }
            for (; i < n; ++i) {
                p[i] = static_cast<double>(storage[i]);
            }
            _mm_sfence();
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 15) != 0) {
                p[i] = static_cast<double>(storage[i]);
                ++i;
            }
            if constexpr (!detail::is_strided_span_v<decltype(storage)> && requires { { storage.data() }; } && std::is_same_v<typename decltype(storage)::value_type, double>) {
                const double* VEC_MATH_RESTRICT sp = storage.data();
                for (; i + 2 <= n; i += 2) {
                    _mm_stream_pd(p + i, _mm_loadu_pd(sp + i));
                }
            } else {
                for (; i + 2 <= n; i += 2) {
                    __m128d val = _mm_setr_pd(
                        static_cast<double>(storage[i]),
                        static_cast<double>(storage[i + 1])
                    );
                    _mm_stream_pd(p + i, val);
                }
            }
            for (; i < n; ++i) {
                p[i] = static_cast<double>(storage[i]);
            }
            _mm_sfence();
#else
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t j = 0; j < n; ++j) {
                p[j] = static_cast<double>(storage[j]);
            }
#endif
        } else if constexpr (std::is_same_v<T, float>) {
            std::size_t i = 0;
#if defined(__AVX__)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 31) != 0) {
                p[i] = static_cast<float>(storage[i]);
                ++i;
            }
            if constexpr (!detail::is_strided_span_v<decltype(storage)> && requires { { storage.data() }; } && std::is_same_v<typename decltype(storage)::value_type, float>) {
                const float* VEC_MATH_RESTRICT sp = storage.data();
                for (; i + 8 <= n; i += 8) {
                    _mm256_stream_ps(p + i, _mm256_loadu_ps(sp + i));
                }
            } else {
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
            }
            for (; i < n; ++i) {
                p[i] = static_cast<float>(storage[i]);
            }
            _mm_sfence();
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
            while (i < n && (reinterpret_cast<std::uintptr_t>(p + i) & 15) != 0) {
                p[i] = static_cast<float>(storage[i]);
                ++i;
            }
            if constexpr (!detail::is_strided_span_v<decltype(storage)> && requires { { storage.data() }; } && std::is_same_v<typename decltype(storage)::value_type, float>) {
                const float* VEC_MATH_RESTRICT sp = storage.data();
                for (; i + 4 <= n; i += 4) {
                    _mm_stream_ps(p + i, _mm_loadu_ps(sp + i));
                }
            } else {
                for (; i + 4 <= n; i += 4) {
                    __m128 val = _mm_setr_ps(
                        static_cast<float>(storage[i]),
                        static_cast<float>(storage[i + 1]),
                        static_cast<float>(storage[i + 2]),
                        static_cast<float>(storage[i + 3])
                    );
                    _mm_stream_ps(p + i, val);
                }
            }
            for (; i < n; ++i) {
                p[i] = static_cast<float>(storage[i]);
            }
            _mm_sfence();
#else
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t j = 0; j < n; ++j) {
                p[j] = static_cast<float>(storage[j]);
            }
#endif
        } else
#endif
        {
#if defined(VEC_MATH_HAS_STD_SIMD)
            if constexpr (!detail::is_strided_span_v<decltype(storage)> && requires { { storage.data() }; } &&
                          std::is_same_v<std::remove_const_t<typename decltype(storage)::value_type>, T> &&
                          (std::is_same_v<T, double> || std::is_same_v<T, float>)) {
                using V = std::simd::vec<T>;
                constexpr std::size_t W = V::size();
                const T* VEC_MATH_RESTRICT sp = storage.data();
                std::size_t i = 0;
                for (; i + W <= n; i += W) {
                    const V v = std::simd::unchecked_load<V>(std::span<const T>(sp + i, W));
                    std::simd::unchecked_store(v, std::span<T>(p + i, W));
                }
                for (; i < n; ++i) {
                    p[i] = sp[i];
                }
                return;
            }
#endif
            VEC_MATH_PRAGMA_SIMD
            for (std::size_t i = 0; i < n; ++i) {
                p[i] = static_cast<T>(storage[i]);
            }
        }
    }

    // ============================================================================
    // Direct Device / Buffer Evaluation & Matrix-Vector Operations
    // ============================================================================

    namespace detail {
        template<typename MatT, typename VecInT, typename VecOutT>
        VEC_MATH_HOST_DEVICE inline void matvec_noalias(
            const MatT* VEC_MATH_RESTRICT A,
            const VecInT* VEC_MATH_RESTRICT x,
            VecOutT* VEC_MATH_RESTRICT y,
            std::size_t rows,
            std::size_t cols
        ) {
            for (std::size_t r = 0; r < rows; ++r) {
                const MatT* VEC_MATH_RESTRICT row = A + r * cols;
#if defined(VEC_MATH_HAS_STD_SIMD)
                if constexpr (std::is_same_v<MatT, VecOutT> && std::is_same_v<VecInT, VecOutT> &&
                              (std::is_same_v<VecOutT, double> || std::is_same_v<VecOutT, float>)) {
                    using V = std::simd::vec<VecOutT>;
                    constexpr std::size_t W = V::size();
                    V acc0(VecOutT{0});
                    V acc1(VecOutT{0});
                    std::size_t c = 0;
                    for (; c + 2 * W <= cols; c += 2 * W) {
                        acc0 += std::simd::unchecked_load<V>(std::span<const VecOutT>(row + c, W)) *
                                std::simd::unchecked_load<V>(std::span<const VecOutT>(x + c, W));
                        acc1 += std::simd::unchecked_load<V>(std::span<const VecOutT>(row + c + W, W)) *
                                std::simd::unchecked_load<V>(std::span<const VecOutT>(x + c + W, W));
                    }
                    for (; c + W <= cols; c += W) {
                        acc0 += std::simd::unchecked_load<V>(std::span<const VecOutT>(row + c, W)) *
                                std::simd::unchecked_load<V>(std::span<const VecOutT>(x + c, W));
                    }
                    VecOutT sum = std::simd::reduce(acc0 + acc1);
                    for (; c < cols; ++c) {
                        sum += row[c] * x[c];
                    }
                    y[r] = sum;
                    continue;
                }
#endif
                std::size_t c = 0;
                VecOutT acc0{0}, acc1{0}, acc2{0}, acc3{0};
                for (; c + 4 <= cols; c += 4) {
                    acc0 = acc0 + static_cast<VecOutT>(row[c + 0]) * static_cast<VecOutT>(x[c + 0]);
                    acc1 = acc1 + static_cast<VecOutT>(row[c + 1]) * static_cast<VecOutT>(x[c + 1]);
                    acc2 = acc2 + static_cast<VecOutT>(row[c + 2]) * static_cast<VecOutT>(x[c + 2]);
                    acc3 = acc3 + static_cast<VecOutT>(row[c + 3]) * static_cast<VecOutT>(x[c + 3]);
                }
                VecOutT sum = (acc0 + acc1) + (acc2 + acc3);
                for (; c < cols; ++c) {
                    sum = sum + static_cast<VecOutT>(row[c]) * static_cast<VecOutT>(x[c]);
                }
                y[r] = sum;
            }
        }

        template<typename MatT, typename VecInT, typename VecOutT>
        VEC_MATH_HOST_DEVICE inline void matvec_transposed_noalias(
            const MatT* VEC_MATH_RESTRICT A,
            const VecInT* VEC_MATH_RESTRICT x,
            VecOutT* VEC_MATH_RESTRICT y,
            std::size_t rows,
            std::size_t cols
        ) {
            for (std::size_t c = 0; c < cols; ++c) {
                y[c] = VecOutT{0};
            }
            for (std::size_t r = 0; r < rows; ++r) {
                const MatT* VEC_MATH_RESTRICT row = A + r * cols;
                const auto xr = static_cast<VecOutT>(x[r]);
#if defined(VEC_MATH_HAS_STD_SIMD)
                if constexpr (std::is_same_v<MatT, VecOutT> &&
                              (std::is_same_v<VecOutT, double> || std::is_same_v<VecOutT, float>)) {
                    using V = std::simd::vec<VecOutT>;
                    constexpr std::size_t W = V::size();
                    const V vxr(xr);
                    std::size_t c = 0;
                    for (; c + W <= cols; c += W) {
                        const V yv = std::simd::unchecked_load<V>(std::span<const VecOutT>(y + c, W));
                        const V rv = std::simd::unchecked_load<V>(std::span<const VecOutT>(row + c, W));
                        std::simd::unchecked_store(yv + rv * vxr, std::span<VecOutT>(y + c, W));
                    }
                    for (; c < cols; ++c) {
                        y[c] += row[c] * xr;
                    }
                    continue;
                }
#endif
                VEC_MATH_PRAGMA_SIMD
                for (std::size_t c = 0; c < cols; ++c) {
                    y[c] = y[c] + static_cast<VecOutT>(row[c]) * xr;
                }
            }
        }
    }

    template<typename MatT, typename VecInT, typename VecOutT>
    VEC_MATH_HOST_DEVICE inline void matvec(const MatT* A, const VecInT* x, VecOutT* y, std::size_t rows, std::size_t cols) {
        if (rows == 0 || cols == 0) return;
        const bool aliased = detail::memory_ranges_overlap(x, cols * sizeof(VecInT), y, rows * sizeof(VecOutT));
        constexpr std::size_t STACK_BUF_CAP = 128;
        if (aliased) {
            if (rows <= STACK_BUF_CAP) {
                VecOutT local_buf[STACK_BUF_CAP];
                detail::matvec_noalias(A, x, local_buf, rows, cols);
                for (std::size_t r = 0; r < rows; ++r) {
                    y[r] = local_buf[r];
                }
                return;
            }
#if defined(VEC_MATH_STRICT_ZERO_MALLOC)
            else {
                VEC_MATH_THROW_BAD_ALLOC("vec_math: matvec aliasing exceeds stack buffer capacity under VEC_MATH_STRICT_ZERO_MALLOC");
            }
#elif !defined(__SYCL_DEVICE_ONLY__)
            else {
                std::vector<VecOutT> local_buf(rows);
                detail::matvec_noalias(A, x, local_buf.data(), rows, cols);
                for (std::size_t r = 0; r < rows; ++r) {
                    y[r] = local_buf[r];
                }
                return;
            }
#endif
        }
        detail::matvec_noalias(A, x, y, rows, cols);
    }

    template<typename MatT, typename VecInT, typename VecOutT>
    VEC_MATH_HOST_DEVICE inline void matvec_transposed(const MatT* A, const VecInT* x, VecOutT* y, std::size_t rows, std::size_t cols) {
        if (rows == 0 || cols == 0) return;
        const bool aliased = detail::memory_ranges_overlap(x, rows * sizeof(VecInT), y, cols * sizeof(VecOutT));
        constexpr std::size_t STACK_BUF_CAP = 128;
        if (aliased) {
            if (cols <= STACK_BUF_CAP) {
                VecOutT local_buf[STACK_BUF_CAP];
                detail::matvec_transposed_noalias(A, x, local_buf, rows, cols);
                for (std::size_t c = 0; c < cols; ++c) {
                    y[c] = local_buf[c];
                }
                return;
            }
#if defined(VEC_MATH_STRICT_ZERO_MALLOC)
            else {
                VEC_MATH_THROW_BAD_ALLOC("vec_math: matvec_transposed aliasing exceeds stack buffer capacity under VEC_MATH_STRICT_ZERO_MALLOC");
            }
#elif !defined(__SYCL_DEVICE_ONLY__)
            else {
                std::vector<VecOutT> local_buf(cols);
                detail::matvec_transposed_noalias(A, x, local_buf.data(), rows, cols);
                for (std::size_t c = 0; c < cols; ++c) {
                    y[c] = local_buf[c];
                }
                return;
            }
#endif
        }
        detail::matvec_transposed_noalias(A, x, y, rows, cols);
    }

    // Compile-time fixed Np small-matrix vector multiplication
    template<std::size_t Rows, std::size_t Cols, typename MatT, typename VecInT, typename VecOutT>
    VEC_MATH_HOST_DEVICE inline void matvec_fixed(const MatT* A, const VecInT* x, VecOutT* y) {
        if constexpr (Rows == 0 || Cols == 0) {
            return;
        } else {
            const bool aliased = detail::memory_ranges_overlap(x, Cols * sizeof(VecInT), y, Rows * sizeof(VecOutT));
            if (aliased) {
                VecOutT local_buf[Rows];
                detail::matvec_noalias(A, x, local_buf, Rows, Cols);
                for (std::size_t r = 0; r < Rows; ++r) {
                    y[r] = local_buf[r];
                }
                return;
            }
            detail::matvec_noalias(A, x, y, Rows, Cols);
        }
    }

    template<std::size_t Rows, std::size_t Cols, typename MatT, typename VecInT, typename VecOutT>
    VEC_MATH_HOST_DEVICE inline void matvec_transposed_fixed(const MatT* A, const VecInT* x, VecOutT* y) {
        if constexpr (Rows == 0 || Cols == 0) {
            return;
        } else {
            const bool aliased = detail::memory_ranges_overlap(x, Rows * sizeof(VecInT), y, Cols * sizeof(VecOutT));
            if (aliased) {
                VecOutT local_buf[Cols];
                detail::matvec_transposed_noalias(A, x, local_buf, Rows, Cols);
                for (std::size_t c = 0; c < Cols; ++c) {
                    y[c] = local_buf[c];
                }
                return;
            }
            detail::matvec_transposed_noalias(A, x, y, Rows, Cols);
        }
    }

    // ============================================================================
    // DG Primitives: StridedSpan, WorkspaceArena, Gather/Scatter, Batched 4x4
    // ============================================================================

    template<typename T>
    class StridedSpan {
    public:
        using value_type = T;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = const T&;
        using pointer = T*;
        using const_pointer = const T*;

        constexpr StridedSpan() noexcept : ptr_(nullptr), size_(0), stride_(1) {}
        constexpr StridedSpan(T* ptr, std::size_t size, std::size_t stride) noexcept
            : ptr_(ptr), size_(size), stride_(stride) {}

        [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
        [[nodiscard]] constexpr std::size_t stride() const noexcept { return stride_; }
        [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
        [[nodiscard]] constexpr pointer data() const noexcept { return ptr_; }

        [[nodiscard]] constexpr reference operator[](std::size_t i) const noexcept {
            return ptr_[i * stride_];
        }

    private:
        T* ptr_;
        std::size_t size_;
        std::size_t stride_;
    };

    template<typename T, typename ScalarT>
    VEC_MATH_HOST_DEVICE inline void broadcast_multiply(T* data, std::size_t count, ScalarT factor) {
        const auto f = static_cast<T>(factor);
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < count; ++i) {
            data[i] = data[i] * f;
        }
    }

    template<typename T, std::size_t Extent, typename ScalarT>
    VEC_MATH_HOST_DEVICE inline void broadcast_multiply(std::span<T, Extent> span_data, ScalarT factor) {
        broadcast_multiply(span_data.data(), span_data.size(), factor);
    }

    class WorkspaceArena {
    public:
        explicit WorkspaceArena(std::size_t capacity_bytes = 1024 * 1024)
            : capacity_(capacity_bytes), offset_(0) {
            buffer_ = static_cast<char*>(detail::aligned_alloc_helper(64, capacity_bytes));
        }

        ~WorkspaceArena() {
            if (buffer_) {
                detail::aligned_free_helper(buffer_);
            }
        }

        WorkspaceArena(const WorkspaceArena&) = delete;
        WorkspaceArena& operator=(const WorkspaceArena&) = delete;

        WorkspaceArena(WorkspaceArena&& other) noexcept
            : buffer_(other.buffer_), capacity_(other.capacity_), offset_(other.offset_) {
            other.buffer_ = nullptr;
            other.capacity_ = 0;
            other.offset_ = 0;
        }

        WorkspaceArena& operator=(WorkspaceArena&& other) noexcept {
            if (this != &other) {
                if (buffer_) detail::aligned_free_helper(buffer_);
                buffer_ = other.buffer_;
                capacity_ = other.capacity_;
                offset_ = other.offset_;
                other.buffer_ = nullptr;
                other.capacity_ = 0;
                other.offset_ = 0;
            }
            return *this;
        }

        template<typename T>
        [[nodiscard]] std::span<T> allocate(std::size_t count) {
            constexpr std::size_t alignment = 64;
            std::size_t current_addr = reinterpret_cast<std::size_t>(buffer_ + offset_);
            std::size_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
            std::size_t new_offset = (aligned_addr - reinterpret_cast<std::size_t>(buffer_)) + count * sizeof(T);
            if (new_offset > capacity_) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::length_error, "vec_math: WorkspaceArena capacity exceeded");
            }
            offset_ = new_offset;
            return std::span<T>(reinterpret_cast<T*>(aligned_addr), count);
        }

        void reset() noexcept {
            offset_ = 0;
        }

        [[nodiscard]] std::size_t allocated_bytes() const noexcept { return offset_; }
        [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    private:
        char* buffer_;
        std::size_t capacity_;
        std::size_t offset_;
    };

    template<typename T, typename IndexT>
    VEC_MATH_HOST_DEVICE inline void gather(T* dst, const T* src, const IndexT* indices, std::size_t count) {
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = src[indices[i]];
        }
    }

    template<typename T, std::size_t DstExt, typename SrcT, std::size_t SrcExt, typename IndexT, std::size_t IdxExt>
    VEC_MATH_HOST_DEVICE inline void gather(std::span<T, DstExt> dst, std::span<SrcT, SrcExt> src, std::span<IndexT, IdxExt> indices) {
        static_assert(std::is_convertible_v<SrcT, T>, "vec_math: source element type must be convertible to destination type in gather");
        VEC_MATH_CHECK_SIZE_EQ(dst.size(), indices.size(), "vec_math: size mismatch in gather");
        for (std::size_t i = 0; i < indices.size(); ++i) {
            dst[i] = static_cast<T>(src[indices[i]]);
        }
    }

    template<typename T, typename IndexT>
    VEC_MATH_HOST_DEVICE inline void scatter_add(T* dst, const IndexT* indices, const T* src, std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            dst[indices[i]] += src[i];
        }
    }

    template<typename T, std::size_t DstExt, typename IndexT, std::size_t IdxExt, typename SrcT, std::size_t SrcExt>
    VEC_MATH_HOST_DEVICE inline void scatter_add(std::span<T, DstExt> dst, std::span<IndexT, IdxExt> indices, std::span<SrcT, SrcExt> src) {
        static_assert(std::is_convertible_v<SrcT, T>, "vec_math: source element type must be convertible to destination type in scatter_add");
        VEC_MATH_CHECK_SIZE_EQ(indices.size(), src.size(), "vec_math: size mismatch in scatter_add");
        for (std::size_t i = 0; i < src.size(); ++i) {
            dst[indices[i]] += static_cast<T>(src[i]);
        }
    }

    // Point-implicit NxN linear solve in SoA layout across elements: A_soa x_soa = b_soa
    template<std::size_t N, typename T>
    VEC_MATH_HOST_DEVICE inline void solve_nxn_batched(
        const T* const* A_soa, // N*N pointers, each pointing to array of size num_elements
        const T* const* b_soa, // N pointers, each pointing to array of size num_elements
        T* const* x_soa,       // N pointers, each pointing to array of size num_elements
        std::size_t num_elements
    ) {
        if constexpr (N == 0) {
            return;
        } else {
            if (num_elements == 0) return;

            int singular_flag = 0;
            std::size_t e = 0;

#if defined(VEC_MATH_HAS_STD_SIMD)
            if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
                using V = std::simd::vec<T>;
                using M = typename V::mask_type;
                constexpr std::size_t W = V::size();
                M any_singular(false);

                for (; e + W <= num_elements; e += W) {
                    V a[N][N];
                    V rhs[N];
                    for (std::size_t r = 0; r < N; ++r) {
                        for (std::size_t c = 0; c < N; ++c) {
                            a[r][c] = std::simd::unchecked_load<V>(
                                std::span<const T>(A_soa[r * N + c] + e, W));
                        }
                        rhs[r] = std::simd::unchecked_load<V>(
                            std::span<const T>(b_soa[r] + e, W));
                    }

                    for (std::size_t i = 0; i < N; ++i) {
                        for (std::size_t k = i + 1; k < N; ++k) {
                            const V abs_k = std::simd::select(a[k][i] < V(T{0}), -a[k][i], a[k][i]);
                            const V abs_i = std::simd::select(a[i][i] < V(T{0}), -a[i][i], a[i][i]);
                            const M do_swap = abs_k > abs_i;
                            for (std::size_t c = i; c < N; ++c) {
                                const V ai = a[i][c];
                                const V ak = a[k][c];
                                a[i][c] = std::simd::select(do_swap, ak, ai);
                                a[k][c] = std::simd::select(do_swap, ai, ak);
                            }
                            const V bi = rhs[i];
                            const V bk = rhs[k];
                            rhs[i] = std::simd::select(do_swap, bk, bi);
                            rhs[k] = std::simd::select(do_swap, bi, bk);
                        }

                        const V diag = a[i][i];
                        const V abs_diag = std::simd::select(diag < V(T{0}), -diag, diag);
                        const M is_sing = abs_diag <= V(T{1e-20});
                        any_singular = any_singular || is_sing;
                        const V safe_diag = std::simd::select(is_sing, V(T{1}), diag);
                        const V inv_diag = V(T{1}) / safe_diag;
                        a[i][i] = safe_diag;

                        for (std::size_t k = i + 1; k < N; ++k) {
                            const V factor = a[k][i] * inv_diag;
                            for (std::size_t c = i + 1; c < N; ++c) {
                                a[k][c] -= factor * a[i][c];
                            }
                            rhs[k] -= factor * rhs[i];
                        }
                    }

                    V sol[N];
                    for (std::size_t idx = N; idx > 0; --idx) {
                        const std::size_t r = idx - 1;
                        V acc = rhs[r];
                        for (std::size_t c = r + 1; c < N; ++c) {
                            acc -= a[r][c] * sol[c];
                        }
                        sol[r] = acc / a[r][r];
                    }

                    for (std::size_t r = 0; r < N; ++r) {
                        std::simd::unchecked_store(sol[r], std::span<T>(x_soa[r] + e, W));
                    }
                }

                if (std::simd::any_of(any_singular)) {
                    singular_flag = 1;
                }
            }
#endif

            VEC_MATH_PRAGMA_SIMD
            for (; e < num_elements; ++e) {
                T a[N][N];
                T rhs[N];
                for (std::size_t r = 0; r < N; ++r) {
                    for (std::size_t c = 0; c < N; ++c) {
                        a[r][c] = A_soa[r * N + c][e];
                    }
                    rhs[r] = b_soa[r][e];
                }

                for (std::size_t i = 0; i < N; ++i) {
                    for (std::size_t k = i + 1; k < N; ++k) {
                        const bool do_swap = std::abs(a[k][i]) > std::abs(a[i][i]);
                        for (std::size_t c = i; c < N; ++c) {
                            const T ai = a[i][c];
                            const T ak = a[k][c];
                            a[i][c] = do_swap ? ak : ai;
                            a[k][c] = do_swap ? ai : ak;
                        }
                        const T bi = rhs[i];
                        const T bk = rhs[k];
                        rhs[i] = do_swap ? bk : bi;
                        rhs[k] = do_swap ? bi : bk;
                    }

                    const T diag = a[i][i];
                    const bool is_sing = std::abs(diag) <= T{1e-20};
                    if (is_sing) { singular_flag = 1; }
                    const T safe_diag = is_sing ? T{1} : diag;
                    const T inv_diag = T{1} / safe_diag;
                    a[i][i] = safe_diag;

                    for (std::size_t k = i + 1; k < N; ++k) {
                        const T factor = a[k][i] * inv_diag;
                        for (std::size_t c = i + 1; c < N; ++c) {
                            a[k][c] -= factor * a[i][c];
                        }
                        rhs[k] -= factor * rhs[i];
                    }
                }

                T sol[N];
                for (std::size_t idx = N; idx > 0; --idx) {
                    const std::size_t r = idx - 1;
                    T acc = rhs[r];
                    for (std::size_t c = r + 1; c < N; ++c) {
                        acc -= a[r][c] * sol[c];
                    }
                    sol[r] = acc / a[r][r];
                }

                for (std::size_t r = 0; r < N; ++r) {
                    x_soa[r][e] = sol[r];
                }
            }

            if (singular_flag) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::runtime_error, "vec_math: singular matrix in solve_nxn_batched");
            }
        }
    }

    // Point-implicit 4x4 linear solve in SoA layout across elements: A_soa x_soa = b_soa
    template<typename T>
    VEC_MATH_HOST_DEVICE inline void solve_4x4_batched(
        const T* const* A_soa, // 16 pointers, each pointing to array of size num_elements
        const T* const* b_soa, // 4 pointers, each pointing to array of size num_elements
        T* const* x_soa,       // 4 pointers, each pointing to array of size num_elements
        std::size_t num_elements
    ) {
        solve_nxn_batched<4, T>(A_soa, b_soa, x_soa, num_elements);
    }

    template<typename T>
    VEC_MATH_HOST_DEVICE inline void transpose_aos_to_soa(
        const T* aos,
        T* soa,
        std::size_t num_tuples,
        std::size_t tuple_size
    ) {
        for (std::size_t i = 0; i < num_tuples; ++i) {
            for (std::size_t v = 0; v < tuple_size; ++v) {
                soa[v * num_tuples + i] = aos[i * tuple_size + v];
            }
        }
    }

    template<typename T>
    VEC_MATH_HOST_DEVICE inline void transpose_soa_to_aos(
        const T* soa,
        T* aos,
        std::size_t num_tuples,
        std::size_t tuple_size
    ) {
        for (std::size_t v = 0; v < tuple_size; ++v) {
            for (std::size_t i = 0; i < num_tuples; ++i) {
                aos[i * tuple_size + v] = soa[v * num_tuples + i];
            }
        }
    }

    // ============================================================================
    // Stage S10: Sum-Factorization / Tensor Contractions
    // ============================================================================

    enum class Axis : std::size_t {
        X = 0,
        Y = 1,
        Z = 2
    };

    namespace detail {
        template<auto AxisParam, std::size_t N = 0, typename MatT, typename T>
        VEC_MATH_HOST_DEVICE inline void contract_axis_noalias(
            const MatT* VEC_MATH_RESTRICT D,
            const T* VEC_MATH_RESTRICT in,
            T* VEC_MATH_RESTRICT out,
            [[maybe_unused]] std::size_t nx,
            [[maybe_unused]] std::size_t ny,
            [[maybe_unused]] std::size_t nz
        ) {
            constexpr std::size_t axis_idx = static_cast<std::size_t>(AxisParam);
            if constexpr (axis_idx == 0) {
                const std::size_t n_pencils = ny * nz;
                const std::size_t dim = (N > 0) ? N : nx;
                for (std::size_t p = 0; p < n_pencils; ++p) {
                    const T* VEC_MATH_RESTRICT in_ptr = in + p * dim;
                    T* VEC_MATH_RESTRICT out_ptr = out + p * dim;
                    for (std::size_t i = 0; i < dim; ++i) {
                        const MatT* VEC_MATH_RESTRICT row = D + i * dim;
                        T acc0{0}, acc1{0}, acc2{0}, acc3{0};
                        std::size_t j = 0;
                        for (; j + 4 <= dim; j += 4) {
                            acc0 += static_cast<T>(row[j + 0]) * in_ptr[j + 0];
                            acc1 += static_cast<T>(row[j + 1]) * in_ptr[j + 1];
                            acc2 += static_cast<T>(row[j + 2]) * in_ptr[j + 2];
                            acc3 += static_cast<T>(row[j + 3]) * in_ptr[j + 3];
                        }
                        T sum = (acc0 + acc1) + (acc2 + acc3);
                        for (; j < dim; ++j) {
                            sum += static_cast<T>(row[j]) * in_ptr[j];
                        }
                        out_ptr[i] = sum;
                    }
                }
            } else if constexpr (axis_idx == 1) {
                const std::size_t dim_y = (N > 0) ? N : ny;
                for (std::size_t k = 0; k < nz; ++k) {
                    const T* VEC_MATH_RESTRICT in_k = in + k * (nx * dim_y);
                    T* VEC_MATH_RESTRICT out_k = out + k * (nx * dim_y);
                    for (std::size_t j = 0; j < dim_y; ++j) {
                        T* VEC_MATH_RESTRICT out_row = out_k + j * nx;
                        const MatT* VEC_MATH_RESTRICT row = D + j * dim_y;
                        const T d0 = static_cast<T>(row[0]);
                        const T* VEC_MATH_RESTRICT in_0 = in_k;
                        VEC_MATH_PRAGMA_SIMD
                        for (std::size_t i = 0; i < nx; ++i) {
                            out_row[i] = d0 * in_0[i];
                        }
                        for (std::size_t q = 1; q < dim_y; ++q) {
                            const T dq = static_cast<T>(row[q]);
                            const T* VEC_MATH_RESTRICT in_q = in_k + q * nx;
                            VEC_MATH_PRAGMA_SIMD
                            for (std::size_t i = 0; i < nx; ++i) {
                                out_row[i] += dq * in_q[i];
                            }
                        }
                    }
                }
            } else if constexpr (axis_idx == 2) {
                const std::size_t nxy = nx * ny;
                const std::size_t dim_z = (N > 0) ? N : nz;
                for (std::size_t k = 0; k < dim_z; ++k) {
                    T* VEC_MATH_RESTRICT out_plane = out + k * nxy;
                    const MatT* VEC_MATH_RESTRICT row = D + k * dim_z;
                    const T d0 = static_cast<T>(row[0]);
                    VEC_MATH_PRAGMA_SIMD
                    for (std::size_t p = 0; p < nxy; ++p) {
                        out_plane[p] = d0 * in[p];
                    }
                    for (std::size_t r = 1; r < dim_z; ++r) {
                        const T dr = static_cast<T>(row[r]);
                        const T* VEC_MATH_RESTRICT in_r = in + r * nxy;
                        VEC_MATH_PRAGMA_SIMD
                        for (std::size_t p = 0; p < nxy; ++p) {
                            out_plane[p] += dr * in_r[p];
                        }
                    }
                }
            }
        }
    }

    template<auto AxisParam, std::size_t N = 0, typename MatT, typename T>
    VEC_MATH_HOST_DEVICE inline void contract_axis(
        const MatT* D,
        const T* in,
        T* out,
        [[maybe_unused]] std::size_t nx,
        [[maybe_unused]] std::size_t ny,
        [[maybe_unused]] std::size_t nz = 1
    ) {
        constexpr std::size_t axis_idx = static_cast<std::size_t>(AxisParam);
        static_assert(axis_idx <= 2, "vec_math: contract_axis Axis must be 0 (X), 1 (Y), or 2 (Z)");

        if (nx == 0 || ny == 0 || nz == 0) return;

        VEC_MATH_DEVICE_ASSERT(D != nullptr, "vec_math: null matrix pointer in contract_axis");
        VEC_MATH_DEVICE_ASSERT(in != nullptr, "vec_math: null input pointer in contract_axis");
        VEC_MATH_DEVICE_ASSERT(out != nullptr, "vec_math: null output pointer in contract_axis");

        if constexpr (axis_idx == 0 && N > 0) {
            VEC_MATH_DEVICE_ASSERT(nx == N, "vec_math: nx must match N for Axis 0 contraction");
        }
        if constexpr (axis_idx == 1 && N > 0) {
            VEC_MATH_DEVICE_ASSERT(ny == N, "vec_math: ny must match N for Axis 1 contraction");
        }
        if constexpr (axis_idx == 2 && N > 0) {
            VEC_MATH_DEVICE_ASSERT(nz == N, "vec_math: nz must match N for Axis 2 contraction");
        }

        const std::size_t total = nx * ny * nz;
        const bool aliased = detail::memory_ranges_overlap(in, total * sizeof(T), out, total * sizeof(T));
        if (aliased) {
            constexpr std::size_t STACK_TOTAL_CAP = 1024;
            if (total <= STACK_TOTAL_CAP) {
                T local_buf[STACK_TOTAL_CAP];
                detail::contract_axis_noalias<AxisParam, N>(D, in, local_buf, nx, ny, nz);
                for (std::size_t idx = 0; idx < total; ++idx) {
                    out[idx] = local_buf[idx];
                }
                return;
            }
#if defined(VEC_MATH_STRICT_ZERO_MALLOC)
            else {
                VEC_MATH_THROW_BAD_ALLOC("vec_math: contract_axis aliased buffer exceeds stack capacity under VEC_MATH_STRICT_ZERO_MALLOC");
            }
#elif !defined(__SYCL_DEVICE_ONLY__)
            else {
                std::vector<T> local_buf(total);
                detail::contract_axis_noalias<AxisParam, N>(D, in, local_buf.data(), nx, ny, nz);
                for (std::size_t idx = 0; idx < total; ++idx) {
                    out[idx] = local_buf[idx];
                }
                return;
            }
#endif
        }

        detail::contract_axis_noalias<AxisParam, N>(D, in, out, nx, ny, nz);
    }

    template<auto AxisParam, std::size_t N = 0, typename MatT, typename InT, typename OutT, std::size_t DExt, std::size_t InExt, std::size_t OutExt>
    VEC_MATH_HOST_DEVICE inline void contract_axis(
        std::span<MatT, DExt> D,
        std::span<InT, InExt> in,
        std::span<OutT, OutExt> out,
        std::size_t nx,
        std::size_t ny,
        std::size_t nz = 1
    ) {
        contract_axis<AxisParam, N>(D.data(), in.data(), out.data(), nx, ny, nz);
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
#if defined(_OPENMP)
        const std::size_t n = lhs_in.size();
        if (n < 16384) {
            return dot(lhs_in, rhs_in);
        }
        auto lhs_op = detail::make_operand(lhs_in);
        auto rhs_op = detail::make_operand(rhs_in);
        using ResType = std::common_type_t<
            typename detail::operand_storage_t<Lhs>::value_type,
            typename detail::operand_storage_t<Rhs>::value_type
        >;

        // Eliminate false sharing on 64-byte cache lines by isolating thread accumulators
        constexpr int MAX_OMP_THREADS = 128;
        struct alignas(64) PaddedAcc {
            ResType value{0};
        };
        alignas(64) PaddedAcc partial_sums[MAX_OMP_THREADS];

        int actual_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            if (tid == 0) {
                actual_threads = nthreads;
            }
            if (tid < MAX_OMP_THREADS) {
                std::size_t chunk = (n + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
                std::size_t start = static_cast<std::size_t>(tid) * chunk;
                std::size_t end = (start + chunk < n) ? (start + chunk) : n;
                if (start < end) {
                    if constexpr (!detail::is_strided_span_v<decltype(lhs_op)> && !detail::is_strided_span_v<decltype(rhs_op)> &&
                                  requires { { lhs_op.data() }; { rhs_op.data() }; }) {
                        partial_sums[tid].value = static_cast<ResType>(dot(
                            std::span(lhs_op.data() + start, end - start),
                            std::span(rhs_op.data() + start, end - start)
                        ));
                    } else {
                        ResType local_acc0{0}, local_acc1{0}, local_acc2{0}, local_acc3{0};
                        std::size_t i = start;
                        for (; i + 4 <= end; i += 4) {
                            local_acc0 += static_cast<ResType>(lhs_op[i + 0]) * static_cast<ResType>(rhs_op[i + 0]);
                            local_acc1 += static_cast<ResType>(lhs_op[i + 1]) * static_cast<ResType>(rhs_op[i + 1]);
                            local_acc2 += static_cast<ResType>(lhs_op[i + 2]) * static_cast<ResType>(rhs_op[i + 2]);
                            local_acc3 += static_cast<ResType>(lhs_op[i + 3]) * static_cast<ResType>(rhs_op[i + 3]);
                        }
                        ResType local_sum = (local_acc0 + local_acc1) + (local_acc2 + local_acc3);
                        for (; i < end; ++i) {
                            local_sum += static_cast<ResType>(lhs_op[i]) * static_cast<ResType>(rhs_op[i]);
                        }
                        partial_sums[tid].value = local_sum;
                    }
                }
            }
        }

        if (actual_threads > MAX_OMP_THREADS) actual_threads = MAX_OMP_THREADS;
        ResType total{0};
        for (int t = 0; t < actual_threads; ++t) {
            total += partial_sums[t].value;
        }
        return total;
#else
        return dot(lhs_in, rhs_in);
#endif
    }

    template<VectorOperand Op>
    [[nodiscard]] inline auto sum_par(const Op &op_in) {
#if defined(_OPENMP)
        const std::size_t n = op_in.size();
        if (n < 16384) {
            return sum(op_in);
        }
        auto op = detail::make_operand(op_in);
        using value_type = typename detail::operand_storage_t<Op>::value_type;

        // Eliminate false sharing on 64-byte cache lines by isolating thread accumulators
        constexpr int MAX_OMP_THREADS = 128;
        struct alignas(64) PaddedAcc {
            value_type value{0};
        };
        alignas(64) PaddedAcc partial_sums[MAX_OMP_THREADS];

        int actual_threads = 1;
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            if (tid == 0) {
                actual_threads = nthreads;
            }
            if (tid < MAX_OMP_THREADS) {
                std::size_t chunk = (n + static_cast<std::size_t>(nthreads) - 1) / static_cast<std::size_t>(nthreads);
                std::size_t start = static_cast<std::size_t>(tid) * chunk;
                std::size_t end = (start + chunk < n) ? (start + chunk) : n;
                if (start < end) {
                    if constexpr (!detail::is_strided_span_v<decltype(op)> && requires { { op.data() }; }) {
                        partial_sums[tid].value = static_cast<value_type>(sum(
                            std::span(op.data() + start, end - start)
                        ));
                    } else {
                        value_type local_acc0{0}, local_acc1{0}, local_acc2{0}, local_acc3{0};
                        std::size_t i = start;
                        for (; i + 4 <= end; i += 4) {
                            local_acc0 += static_cast<value_type>(op[i + 0]);
                            local_acc1 += static_cast<value_type>(op[i + 1]);
                            local_acc2 += static_cast<value_type>(op[i + 2]);
                            local_acc3 += static_cast<value_type>(op[i + 3]);
                        }
                        value_type local_sum = (local_acc0 + local_acc1) + (local_acc2 + local_acc3);
                        for (; i < end; ++i) {
                            local_sum += static_cast<value_type>(op[i]);
                        }
                        partial_sums[tid].value = local_sum;
                    }
                }
            }
        }

        if (actual_threads > MAX_OMP_THREADS) actual_threads = MAX_OMP_THREADS;
        value_type total{0};
        for (int t = 0; t < actual_threads; ++t) {
            total += partial_sums[t].value;
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
        using Val = std::remove_const_t<typename detail::operand_storage_t<Op>::value_type>;
        using ResType = std::conditional_t<std::is_floating_point_v<Val>, Val, double>;
        const std::size_t n = op.size();
        if (n == 0) { return ResType{0}; }
        ResType res{0};
        VEC_MATH_PRAGMA_SIMD
        for (std::size_t i = 0; i < n; ++i) {
            ResType val = std::abs(static_cast<ResType>(op[i]));
            if (std::isnan(val) || val > res) { res = val; }
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

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

