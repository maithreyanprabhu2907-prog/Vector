/*
* Author: Maithreyan(maithreyanprabu2907-prog)
 *
 * Purpose:
 *   Provide a modern, expressive vector math library for std::vector.
 *   Operator overloading keeps mathematical expressions short and readable.
 *
 * Goals:
 *      -Implement simple vector math in modern C++ style
 *      -Use operator overloading to make mathematical code more readable
 *      -Offer a fast, simple and flexible interface for vector math operations
 *      -**Ultimately Reduce Heap allocations** Focusing on reducing malloc and free overheads.
 *      -Target C++20 and later **Soon switching to C++26 after it becomes stable**
 *
 * Features:
 *      -Gives math functions for {
 *          + , - ,  *, /  (For both vector,vector and scalar,vector operations)
 *          +=, -=, *=, /= (For vector,vector operations)
 *      }
 */

#include <cassert>
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>


#if defined(__GNUC__) || defined(__clang__)
#define VEC_MATH_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define VEC_MATH_RESTRICT __restrict
#else
#define VEC_MATH_RESTRICT
#endif


#ifdef VEC_MATH_NO_EXCEPTIONS
#include <cstdlib>
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) do { assert(false && (msg)); std::abort(); } while (false)
#else
#define VEC_MATH_THROW_OR_ABORT(exc_type, msg) throw exc_type(msg)
#endif


namespace vec_math {
    namespace detail {
        template<typename T>
        struct is_std_vector : std::false_type {
        };


        template<typename T, typename Alloc>
        struct is_std_vector<std::vector<T, Alloc> > : std::true_type {
        };


        template<typename T>
        inline constexpr bool is_std_vector_v = is_std_vector<std::remove_cvref_t<T> >::value;


        template<typename T>
        struct is_valid_numeric : std::bool_constant<
                    std::is_arithmetic_v<std::remove_cvref_t<T> > && !std::is_same_v<std::remove_cvref_t<T>, bool>
                > {
        };


        template<typename T>
        inline constexpr bool is_valid_numeric_v = is_valid_numeric<T>::value;


        struct ExprBase {
        };


        template<typename T>
        inline constexpr bool is_vec_expr_v = std::is_base_of_v<ExprBase, std::remove_cvref_t<T> >;
    }


    template<typename T> concept Scalar = detail::is_valid_numeric_v<T>;


    template<typename T> concept NumericVector =
            detail::is_std_vector_v<T> && Scalar<typename std::remove_cvref_t<T>::value_type>;


    template<typename T> concept VecExpression = detail::is_vec_expr_v<T>;


    template<typename T> concept VectorOperand = NumericVector<T> || VecExpression<T>;


    namespace detail {
        struct OpAdd {
            template<typename L, typename R>
            [[nodiscard]] constexpr auto operator()(const L &l, const R &r) const noexcept { return l + r; }
        };


        struct OpSub {
            template<typename L, typename R>
            [[nodiscard]] constexpr auto operator()(const L &l, const R &r) const noexcept { return l - r; }
        };


        struct OpMul {
            template<typename L, typename R>
            [[nodiscard]] constexpr auto operator()(const L &l, const R &r) const noexcept { return l * r; }
        };


        struct OpDiv {
            template<typename L, typename R>
            [[nodiscard]] constexpr auto operator()(const L &l, const R &r) const noexcept { return l / r; }
        };


        struct OpNeg {
            template<typename T>
            [[nodiscard]] constexpr auto operator()(const T &x) const noexcept { return -x; }
        };


        struct OpPos {
            template<typename T>
            [[nodiscard]] constexpr auto operator()(const T &x) const noexcept { return +x; }
        };


        template<NumericVector Vec>
        struct VecRef {
            using value_type = typename Vec::value_type;
            const Vec *ptr;


            explicit constexpr VecRef(const Vec &v) noexcept : ptr(&v) {
            }


            [[nodiscard]] constexpr std::size_t size() const noexcept { return ptr->size(); }


            [[nodiscard]] constexpr value_type operator[](std::size_t i) const noexcept { return ptr->data()[i]; }
        };


        template<Scalar S>
        struct ScalarVal {
            using value_type = S;
            S val;


            explicit constexpr ScalarVal(S s) noexcept : val(s) {
            }


            [[nodiscard]] constexpr S operator[](std::size_t) const noexcept { return val; }
        };


        template<typename T>
        struct OperandStorage;

        template<NumericVector Vec>
        struct OperandStorage<Vec> {
            using type = VecRef<std::remove_cvref_t<Vec> >;
        };

        template<Scalar S>
        struct OperandStorage<S> {
            using type = ScalarVal<std::remove_cvref_t<S> >;
        };

        template<VecExpression Expr>
        struct OperandStorage<Expr> {
            using type = std::remove_cvref_t<Expr>;
        };

        template<typename T>
        using operand_storage_t = typename OperandStorage<T>::type;


        template<typename T>
        [[nodiscard]] constexpr auto make_operand(const T &x) noexcept { return operand_storage_t<T>(x); }


        template<typename T>
        inline constexpr bool is_scalar_val_v = false;
        template<typename S>
        inline constexpr bool is_scalar_val_v<ScalarVal<S> > = true;


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


            [[nodiscard]] constexpr std::size_t size() const noexcept {
                if constexpr (is_scalar_val_v<Lhs>) { return rhs.size(); } else { return lhs.size(); }
            }


            [[nodiscard]] constexpr value_type operator[](std::size_t i) const noexcept {
                return static_cast<value_type>(op(lhs[i], rhs[i]));
            }


            template<typename Alloc = std::allocator<value_type> >
            [[nodiscard]] std::vector<value_type, Alloc> eval(const Alloc &alloc = Alloc{}) const {
                std::vector<value_type, Alloc> result(size(), alloc);
                const std::size_t n = result.size();


                value_type * VEC_MATH_RESTRICT dst = result.data();
                for (std::size_t i = 0; i < n; ++i) { dst[i] = (*this)[i]; }
                return result;
            }


            operator std::vector<value_type>() const { return eval(); }
        };


        template<typename Op, typename Operand>
        struct UnaryExpr : ExprBase {
            using value_type = decltype(std::declval<Op>()(std::declval<typename Operand::value_type>()));

            [[no_unique_address]] Op op;
            Operand operand;


            constexpr UnaryExpr(Op o, Operand arg) : op(o), operand(std::move(arg)) {
            }


            [[nodiscard]] constexpr std::size_t size() const noexcept { return operand.size(); }


            [[nodiscard]] constexpr value_type operator[](std::size_t i) const noexcept { return op(operand[i]); }


            template<typename Alloc = std::allocator<value_type> >
            [[nodiscard]] std::vector<value_type, Alloc> eval(const Alloc &alloc = Alloc{}) const {
                std::vector<value_type, Alloc> result(size(), alloc);
                const std::size_t n = result.size();
                value_type * VEC_MATH_RESTRICT dst = result.data();
                for (std::size_t i = 0; i < n; ++i) { dst[i] = (*this)[i]; }
                return result;
            }


            operator std::vector<value_type>() const { return eval(); }
        };


        template<typename Op, typename L, typename R>
        [[nodiscard]] constexpr auto make_bin_expr(Op op, const L &l, const R &r) {
            return BinExpr<Op, operand_storage_t<L>, operand_storage_t<R> >(op, make_operand(l), make_operand(r));
        }


        template<typename Op, typename Arg>
        [[nodiscard]] constexpr auto make_unary_expr(Op op, const Arg &arg) {
            return UnaryExpr<Op, operand_storage_t<Arg> >(op, make_operand(arg));
        }


        template<typename T, typename Alloc, Scalar S, typename CompoundFn>
        inline void apply_scalar_inplace(std::vector<T, Alloc> &dst, const S &scalar, CompoundFn fn) noexcept {
            const std::size_t n = dst.size();
            T * VEC_MATH_RESTRICT p = dst.data();
            const T s = static_cast<T>(scalar);
            for (std::size_t i = 0; i < n; ++i) { fn(p[i], s); }
        }


        template<typename T, typename Alloc, VectorOperand Src, typename CompoundFn>
        inline void apply_vec_inplace(std::vector<T, Alloc> &dst, const Src &src, CompoundFn fn) {
            if (dst.size() != src.size()) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::invalid_argument, "vec_math: vector size mismatch in compound assignment");
            }


            const std::size_t n = dst.size();
            T *p = dst.data();


            for (std::size_t i = 0; i < n; ++i) { fn(p[i], static_cast<T>(src[i])); }
        }


        template<typename T, typename Alloc, VectorOperand Src>
        inline void assign_zero_malloc(std::vector<T, Alloc> &dst, const Src &src) {
            const std::size_t n = src.size();
            if (dst.capacity() < n) [[unlikely]] {
                VEC_MATH_THROW_OR_ABORT(std::length_error,
                                        "vec_math: destination capacity < source size (zero-malloc invariant violated)");
            }


            dst.resize(n);


            T *p = dst.data();
            for (std::size_t i = 0; i < n; ++i) { p[i] = static_cast<T>(src[i]); }
        }
    }


    inline namespace ops {
        template<typename T, typename Alloc, VectorOperand Src>
        inline std::vector<T, Alloc> &operator<<=(std::vector<T, Alloc> &dst, const Src &src) {
            detail::assign_zero_malloc(dst, src);
            return dst;
        }


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


        template<VectorOperand Arg>
        [[nodiscard]] constexpr auto operator-(const Arg &arg) { return detail::make_unary_expr(detail::OpNeg{}, arg); }


        template<VectorOperand Arg>
        [[nodiscard]] constexpr auto operator+(const Arg &arg) { return detail::make_unary_expr(detail::OpPos{}, arg); }


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

        template<typename T, typename Alloc> requires detail::is_valid_numeric_v<T>
        [[nodiscard]] inline std::vector<T, Alloc> operator-(std::vector<T, Alloc> &&v) noexcept {
            const std::size_t n = v.size();
            T * VEC_MATH_RESTRICT p = v.data();
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
        [[nodiscard]] inline std::vector<T, Alloc> operator
        +(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
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
        [[nodiscard]] inline std::vector<T, Alloc> operator
        -(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
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
        [[nodiscard]] inline std::vector<T, Alloc> operator
        *(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
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
        [[nodiscard]] inline std::vector<T, Alloc> operator
        /(std::vector<T, Alloc> &&lhs, std::vector<T, Alloc2> &&rhs) {
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
}
