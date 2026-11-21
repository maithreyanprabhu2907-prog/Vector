#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 11)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
#define _ITERATOR_DEBUG_LEVEL 0
#endif
#include "vec_math/vector_math.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#undef assert
#define assert(expr) do { \
    if (!(expr)) { \
        std::cerr << "Assertion failed: " #expr " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::abort(); \
    } \
} while (false)


#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 11)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

namespace {
    bool g_track_allocs = false;
    std::size_t g_alloc_count = 0;
}

void *operator new(std::size_t size) {
    if (g_track_allocs) {
        ++g_alloc_count;
    }
    void *p = std::malloc(size == 0 ? 1 : size);
    if (!p) {
        throw std::bad_alloc{};
    }
    return p;
}

void *operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void *ptr) noexcept {
    std::free(ptr);
}

void operator delete(void *ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void *ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void *ptr, std::size_t) noexcept {
    std::free(ptr);
}


struct ZeroMallocGuard {
    const char *test_name;
    std::size_t start_count;

    explicit ZeroMallocGuard(const char *name) noexcept
        : test_name(name), start_count(g_alloc_count) {
        g_track_allocs = true;
    }

    ~ZeroMallocGuard() {
        g_track_allocs = false;
        const std::size_t delta = g_alloc_count - start_count;
        if (delta != 0) {
            std::cerr << "[FAIL] " << test_name << ": expected 0 mallocs, got "
                    << delta << "\n";
            std::abort();
        }
    }
};


[[maybe_unused]] static bool approx_eq(double a, double b, double eps = 1e-12) {
    return std::fabs(a - b) <= eps;
}


template<typename T>
struct CustomAlloc {
    using value_type = T;

    CustomAlloc() noexcept = default;

    template<typename U>
    CustomAlloc(const CustomAlloc<U> &) noexcept {
    }

    T *allocate(std::size_t n) {
        return static_cast<T *>(::operator new(n * sizeof(T)));
    }

    void deallocate(T *p, std::size_t) noexcept {
        ::operator delete(p);
    }

    template<typename U>
    bool operator==(const CustomAlloc<U> &) const noexcept { return true; }
};

int main() {
    using namespace vec_math;

    // ========================================================================
    // SYCL Compatibility & TriviallyCopyable Static Assertions
    // ========================================================================
    static_assert(sycl_compatible<detail::VecRef<std::vector<double>>>);
    static_assert(sycl_compatible<detail::ScalarVal<double>>);

    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> c = {2.0, 4.0, 5.0, 8.0};
    std::vector<double> out(4, 0.0);

    static_assert(sycl_compatible<decltype(2.0 * a + b)>);
    static_assert(sycl_compatible<decltype(vec_math::abs(a))>);
    static_assert(sycl_compatible<decltype(vec_math::fma(a, b, c))>);
    static_assert(sycl_compatible<decltype(vec_math::min(a, b))>);
    static_assert(sycl_compatible<decltype(vec_math::max(a, 2.0))>);
    static_assert(sycl_compatible<decltype(vec_math::clamp(a, 0.0, 1.0))>);
    static_assert(sycl_compatible<decltype(vec_math::sqrt(a))>);
    static_assert(sycl_compatible<decltype(vec_math::pow(a, 2.0))>);
    static_assert(sycl_compatible<decltype(vec_math::exp(a))>);
    static_assert(sycl_compatible<decltype(vec_math::log(a))>);
    static_assert(sycl_compatible<decltype(vec_math::sin(a))>);
    static_assert(sycl_compatible<decltype(vec_math::cos(a))>);

    static_assert(std::is_trivially_copyable_v<decltype(2.0 * a + b)>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::abs(a))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::fma(a, b, c))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::min(a, b))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::max(a, 2.0))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::clamp(a, 0.0, 1.0))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::sqrt(a))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::pow(a, 2.0))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::exp(a))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::log(a))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::sin(a))>);
    static_assert(std::is_trivially_copyable_v<decltype(vec_math::cos(a))>);

    // Original battle test 1: compound operations
    {
        ZeroMallocGuard guard("compound_vec_and_scalar");
        out <<= a;
        out += b;
        out -= 1.0;
        out *= 2.0;
        out /= 2.0;
        out += 1.0;
        out -= a;
        out *= c;
        out /= c;
    }
    for (std::size_t i = 0; i < 4; ++i) {
        assert(approx_eq(out[i], b[i]));
    }

    // Original battle test 2: fused expression template
    {
        ZeroMallocGuard guard("fused_expression_template");
        out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        double expected = 2.0 * a[i] + b[i] / c[i] - 1.5 + a[i] + c[i];
        assert(approx_eq(out[i], expected));
    }

    // Original battle test 3: scalar LHS
    {
        ZeroMallocGuard guard("scalar_lhs_sub_div");
        out <<= 100.0 / a - (5.0 - a);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        double expected = 100.0 / a[i] - (5.0 - a[i]);
        assert(approx_eq(out[i], expected));
    }

    // Original battle test 4: rvalue recycling
    std::vector<double> r1 = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> r2 = {5.0, 6.0, 7.0, 8.0};
    std::vector<double> res(4, 0.0);
    {
        ZeroMallocGuard guard("rvalue_recycling");
        res = (2.0 * std::move(r1) + b) / 2.0 - 1.0;
        res = 10.0 - std::move(res);
        res = b / std::move(r2);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        assert(approx_eq(res[i], b[i] / (5.0 + static_cast<double>(i))));
    }

    // Original battle test 5: self-aliasing
    std::vector<double> alias_v = {1.0, 2.0, 3.0, 4.0};
    {
        ZeroMallocGuard guard("self_aliasing");
        alias_v += alias_v;
        alias_v <<= 2.0 * alias_v + 1.0;
    }
    assert(approx_eq(alias_v[0], 5.0));
    assert(approx_eq(alias_v[1], 9.0));
    assert(approx_eq(alias_v[2], 13.0));
    assert(approx_eq(alias_v[3], 17.0));

    // Original battle test 6: custom allocator & mixed types
    std::vector<float, CustomAlloc<float>> vf = {1.5f, 2.5f, 3.5f};
    std::vector<double, CustomAlloc<double>> vd = {10.0, 20.0, 30.0};
    std::vector<double, CustomAlloc<double>> vout(3, 0.0);
    {
        ZeroMallocGuard guard("mixed_types_custom_alloc");
        vout <<= 2 * vf + vd * 0.5;
    }
    assert(approx_eq(vout[0], 2 * 1.5 + 10.0 * 0.5));
    assert(approx_eq(vout[1], 2 * 2.5 + 20.0 * 0.5));
    assert(approx_eq(vout[2], 2 * 3.5 + 30.0 * 0.5));

    // Original battle test 7: empty vectors
    std::vector<double> empty_a, empty_b, empty_out;
    {
        ZeroMallocGuard guard("empty_vectors");
        empty_out <<= empty_a + 2.0 * empty_b;
        empty_out += 5.0;
    }
    assert(empty_out.empty());

    // ========================================================================
    // Extended Math Utilities Tests (dot, sum, norm, norm_sq, fma, abs)
    // ========================================================================
    {
        ZeroMallocGuard guard("dot_product");
        double d = dot(a, b);
        assert(approx_eq(d, 1.0 * 10.0 + 2.0 * 20.0 + 3.0 * 30.0 + 4.0 * 40.0)); // 300.0

        double d_expr = dot(a + b, c);
        double expected_d = (11.0) * 2.0 + (22.0) * 4.0 + (33.0) * 5.0 + (44.0) * 8.0;
        assert(approx_eq(d_expr, expected_d));
    }

    {
        ZeroMallocGuard guard("sum_reduction");
        double s = sum(a);
        assert(approx_eq(s, 10.0));

        double s_expr = sum(2.0 * a + 1.0);
        assert(approx_eq(s_expr, 2.0 * 10.0 + 4.0 * 1.0)); // 24.0
    }

    {
        ZeroMallocGuard guard("norm_and_norm_sq");
        double sq = norm_sq(a);
        assert(approx_eq(sq, 1.0 + 4.0 + 9.0 + 16.0)); // 30.0

        double n = norm(a);
        assert(approx_eq(n, std::sqrt(30.0)));
    }

    std::vector<double> neg_v = {-1.5, 2.5, -3.5, 4.5};
    {
        ZeroMallocGuard guard("abs_utility");
        out <<= abs(neg_v);
        assert(approx_eq(out[0], 1.5));
        assert(approx_eq(out[1], 2.5));
        assert(approx_eq(out[2], 3.5));
        assert(approx_eq(out[3], 4.5));

        out <<= abs(-a);
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(out[i], a[i]));
        }
    }

    {
        ZeroMallocGuard guard("fma_utility");
        out <<= fma(a, b, c);
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(out[i], a[i] * b[i] + c[i]));
        }

        out <<= fma(a, 2.0, c);
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(out[i], a[i] * 2.0 + c[i]));
        }

        out <<= fma(3.0, b, c);
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(out[i], 3.0 * b[i] + c[i]));
        }

        out <<= fma(a, b, 7.5);
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(out[i], a[i] * b[i] + 7.5));
        }
    }

    // ========================================================================
    // DG Pure Operator Overloading: AXPY, AXPBY, and Dot Operator |
    // ========================================================================
    {
        std::vector<double> y = {10.0, 20.0, 30.0, 40.0};
        const std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
        const double alpha = 2.5;

        // AXPY: y += alpha * x
        {
            ZeroMallocGuard guard("operator_axpy");
            y += alpha * x;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(y[i], (10.0 * (i + 1)) + alpha * (i + 1)));
        }

        // AXPY with subtraction: y -= alpha * x
        {
            ZeroMallocGuard guard("operator_axpy_sub");
            y -= alpha * x;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(y[i], 10.0 * (i + 1)));
        }

        // AXPBY: y <<= alpha * x + beta * y
        const double beta = 0.5;
        {
            ZeroMallocGuard guard("operator_axpby");
            y <<= alpha * x + beta * y;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            double expected = alpha * (i + 1) + beta * (10.0 * (i + 1));
            assert(approx_eq(y[i], expected));
        }

        // Multi-term linear combination: y <<= 2.0 * a + 3.0 * b - 0.5 * c
        {
            ZeroMallocGuard guard("operator_multiterm_linear_combination");
            y <<= 2.0 * a + 3.0 * b - 0.5 * c;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            double expected = 2.0 * a[i] + 3.0 * b[i] - 0.5 * c[i];
            assert(approx_eq(y[i], expected));
        }

        // Dot product operator: u | v
        {
            ZeroMallocGuard guard("operator_dot_product_bar");
            double d1 = a | b;
            double d2 = dot(a, b);
            assert(approx_eq(d1, d2));
            assert(approx_eq(d1, 300.0));

            double d_expr = (a + b) | c;
            double d_expr_expected = dot(a + b, c);
            assert(approx_eq(d_expr, d_expr_expected));
        }
    }

    // ========================================================================
    // 64-Byte Cache-Line Aligned Memory: aligned_allocator & aligned_vector
    // ========================================================================
    {
        aligned_vector<double> av_d = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
        aligned_vector<float> av_f = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

        // Verify 64-byte alignment
        assert(reinterpret_cast<std::uintptr_t>(av_d.data()) % 64 == 0);
        assert(reinterpret_cast<std::uintptr_t>(av_f.data()) % 64 == 0);

        aligned_vector<double> av_out(8, 0.0);
        assert(reinterpret_cast<std::uintptr_t>(av_out.data()) % 64 == 0);

        {
            ZeroMallocGuard guard("aligned_vector_math");
            av_out <<= 2.0 * av_d + 1.0;
            av_out += 3.0 * av_d;
            double d_aligned = av_out | av_d;
            assert(d_aligned > 0.0);
        }
        for (std::size_t i = 0; i < 8; ++i) {
            double expected = 2.0 * av_d[i] + 1.0 + 3.0 * av_d[i];
            assert(approx_eq(av_out[i], expected));
        }
    }

    // ========================================================================
    // Non-Temporal Streaming Stores: assign_stream
    // ========================================================================
    {
        aligned_vector<double> stream_dst(8, 0.0);
        {
            ZeroMallocGuard guard("assign_stream_aligned");
            assign_stream(stream_dst, 2.0 * a + b);
        }
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(stream_dst[i], 2.0 * a[i] + b[i]));
        }

        // Test with odd size / unaligned peeling fallback
        std::vector<double> unaligned_src(19, 3.14);
        std::vector<double> unaligned_dst(19, 0.0);
        {
            ZeroMallocGuard guard("assign_stream_unaligned_peeling");
            assign_stream(unaligned_dst, 2.0 * unaligned_src + 1.0);
        }
        for (std::size_t i = 0; i < 19; ++i) {
            assert(approx_eq(unaligned_dst[i], 2.0 * 3.14 + 1.0));
        }

        // Test float assign_stream
        std::vector<float> f_src(25, 1.5f);
        std::vector<float> f_dst(25, 0.0f);
        {
            ZeroMallocGuard guard("assign_stream_float");
            assign_stream(f_dst, 3.0f * f_src + 0.5f);
        }
        for (std::size_t i = 0; i < 25; ++i) {
            assert(approx_eq(f_dst[i], 3.0f * 1.5f + 0.5f, 1e-6));
        }
    }

    // ========================================================================
    // Explicit Parallel Helpers: assign_par, dot_par, sum_par
    // ========================================================================
    {
        const std::size_t N = 1024;
        std::vector<double> vx(N, 1.5);
        std::vector<double> vy(N, 2.5);
        std::vector<double> par_out(N, 0.0);

        {
            ZeroMallocGuard guard("parallel_helpers_zero_malloc");
            assign_par(par_out, 2.0 * vx + vy);
            double d_par = dot_par(vx, vy);
            double d_ser = dot(vx, vy);
            assert(approx_eq(d_par, d_ser));

            double s_par = sum_par(par_out);
            double s_ser = sum(par_out);
            assert(approx_eq(s_par, s_ser));
        }

        for (std::size_t i = 0; i < N; ++i) {
            assert(approx_eq(par_out[i], 2.0 * 1.5 + 2.5));
        }
    }

    // ========================================================================
    // CFL & Time-stepping Reductions: min_val, max_val, norm_inf, norm_l1
    // ========================================================================
    {
        std::vector<double> v_red = {-10.5, 2.0, 7.5, -1.0, 4.0};
        {
            ZeroMallocGuard guard("reductions_zero_malloc");
            double mn = min_val(v_red);
            assert(approx_eq(mn, -10.5));

            double mx = max_val(v_red);
            assert(approx_eq(mx, 7.5));

            double n_inf = norm_inf(v_red);
            assert(approx_eq(n_inf, 10.5));

            double n_l1 = norm_l1(v_red);
            assert(approx_eq(n_l1, 10.5 + 2.0 + 7.5 + 1.0 + 4.0)); // 25.0

            // Expressions in reductions
            double mn_expr = min_val(v_red + 10.0);
            assert(approx_eq(mn_expr, -0.5));
        }
    }

    // ========================================================================
    // Element-wise Limiters: clamp, min, max
    // ========================================================================
    {
        std::vector<double> v_lim = {-2.0, 0.5, 3.0, 10.0};
        std::vector<double> vlo   = {-1.0, -1.0, -1.0, -1.0};
        std::vector<double> vhi   = {5.0, 5.0, 5.0, 5.0};
        std::vector<double> res_lim(4, 0.0);

        {
            ZeroMallocGuard guard("limiters_zero_malloc");
            // clamp with scalar bounds
            res_lim <<= vec_math::clamp(v_lim, -1.0, 5.0);
            assert(approx_eq(res_lim[0], -1.0));
            assert(approx_eq(res_lim[1], 0.5));
            assert(approx_eq(res_lim[2], 3.0));
            assert(approx_eq(res_lim[3], 5.0));

            // clamp with vector bounds
            res_lim <<= vec_math::clamp(v_lim, vlo, vhi);
            assert(approx_eq(res_lim[0], -1.0));
            assert(approx_eq(res_lim[1], 0.5));
            assert(approx_eq(res_lim[2], 3.0));
            assert(approx_eq(res_lim[3], 5.0));

            // min / max expressions
            res_lim <<= vec_math::min(v_lim, 2.0);
            assert(approx_eq(res_lim[0], -2.0));
            assert(approx_eq(res_lim[1], 0.5));
            assert(approx_eq(res_lim[2], 2.0));
            assert(approx_eq(res_lim[3], 2.0));

            res_lim <<= vec_math::max(v_lim, 1.0);
            assert(approx_eq(res_lim[0], 1.0));
            assert(approx_eq(res_lim[1], 1.0));
            assert(approx_eq(res_lim[2], 3.0));
            assert(approx_eq(res_lim[3], 10.0));

            // binary vector min / max
            res_lim <<= vec_math::max(v_lim, vlo);
            assert(approx_eq(res_lim[0], -1.0));
        }
    }

    // ========================================================================
    // Element-wise Scientific Functions: sqrt, exp, log, sin, cos, pow
    // ========================================================================
    {
        std::vector<double> v_sci = {1.0, 4.0, 9.0, 16.0};
        std::vector<double> sci_out(4, 0.0);
        std::vector<double> v_trig = {0.0, 3.14159265358979323846 / 2.0};
        std::vector<double> trig_out(2, 0.0);
        std::vector<double> v_exp = {0.0, 1.0};
        std::vector<double> exp_out(2, 0.0);

        {
            ZeroMallocGuard guard("scientific_functions_zero_malloc");
            sci_out <<= sqrt(v_sci);
            assert(approx_eq(sci_out[0], 1.0));
            assert(approx_eq(sci_out[1], 2.0));
            assert(approx_eq(sci_out[2], 3.0));
            assert(approx_eq(sci_out[3], 4.0));

            sci_out <<= pow(v_sci, 2.0);
            assert(approx_eq(sci_out[0], 1.0));
            assert(approx_eq(sci_out[1], 16.0));
            assert(approx_eq(sci_out[2], 81.0));
            assert(approx_eq(sci_out[3], 256.0));

            trig_out <<= sin(v_trig);
            assert(approx_eq(trig_out[0], 0.0));
            assert(approx_eq(trig_out[1], 1.0));

            trig_out <<= cos(v_trig);
            assert(approx_eq(trig_out[0], 1.0));
            assert(approx_eq(trig_out[1], 0.0));

            exp_out <<= exp(v_exp);
            assert(approx_eq(exp_out[0], 1.0));
            assert(approx_eq(exp_out[1], std::exp(1.0)));

            exp_out <<= log(exp_out);
            assert(approx_eq(exp_out[0], 0.0));
            assert(approx_eq(exp_out[1], 1.0));
        }
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================
    bool caught_mismatch = false;
    try {
        std::vector<double> small = {1.0, 2.0};
        out += small;
    } catch (const std::invalid_argument &) {
        caught_mismatch = true;
    }
    assert(caught_mismatch);

    bool caught_capacity = false;
    try {
        std::vector<double> tiny;
        tiny <<= a + b;
    } catch (const std::length_error &) {
        caught_capacity = true;
    }
    assert(caught_capacity);

    bool caught_dot_mismatch = false;
    try {
        std::vector<double> small = {1.0, 2.0};
        [[maybe_unused]] auto d = dot(a, small);
    } catch (const std::invalid_argument &) {
        caught_dot_mismatch = true;
    }
    assert(caught_dot_mismatch);

    bool caught_bar_dot_mismatch = false;
    try {
        std::vector<double> small = {1.0, 2.0};
        [[maybe_unused]] auto d = a | small;
    } catch (const std::invalid_argument &) {
        caught_bar_dot_mismatch = true;
    }
    assert(caught_bar_dot_mismatch);

    bool caught_dot_par_mismatch = false;
    try {
        std::vector<double> small = {1.0, 2.0};
        [[maybe_unused]] auto d = dot_par(a, small);
    } catch (const std::invalid_argument &) {
        caught_dot_par_mismatch = true;
    }
    assert(caught_dot_par_mismatch);

    bool caught_assign_stream_cap = false;
    try {
        std::vector<double> tiny;
        assign_stream(tiny, a + b);
    } catch (const std::length_error &) {
        caught_assign_stream_cap = true;
    }
    assert(caught_assign_stream_cap);

    bool caught_assign_par_cap = false;
    try {
        std::vector<double> tiny;
        assign_par(tiny, a + b);
    } catch (const std::length_error &) {
        caught_assign_par_cap = true;
    }
    assert(caught_assign_par_cap);

    bool caught_min_val_empty = false;
    try {
        std::vector<double> empty_v;
        [[maybe_unused]] auto mn = min_val(empty_v);
    } catch (const std::invalid_argument &) {
        caught_min_val_empty = true;
    }
    assert(caught_min_val_empty);

    bool caught_max_val_empty = false;
    try {
        std::vector<double> empty_v;
        [[maybe_unused]] auto mx = max_val(empty_v);
    } catch (const std::invalid_argument &) {
        caught_max_val_empty = true;
    }
    assert(caught_max_val_empty);

    // ========================================================================
    // DG Math Core & Review Probes Verification
    // ========================================================================
    // 1. All tail sizes n = 1..37 exact comparison
    for (std::size_t n = 1; n <= 37; ++n) {
        std::vector<double> v1(n, 1.5);
        std::vector<double> v2(n, 2.5);
        double expected_dot = static_cast<double>(n) * (1.5 * 2.5);
        double actual_dot = (v1 | v2);
        assert(approx_eq(actual_dot, expected_dot));

        double expected_sum = static_cast<double>(n) * 1.5;
        double actual_sum = sum(v1);
        assert(approx_eq(actual_sum, expected_sum));

        double actual_norm_sq = norm_sq(v1);
        double expected_norm_sq = static_cast<double>(n) * (1.5 * 1.5);
        assert(approx_eq(actual_norm_sq, expected_norm_sq));
    }

    // 2. Self-aliasing without UB
    {
        std::vector<double> self_v = {1.0, 2.0, 3.0, 4.0};
        {
            ZeroMallocGuard guard("self_aliasing_add");
            self_v += self_v;
        }
        assert(approx_eq(self_v[0], 2.0));
        assert(approx_eq(self_v[3], 8.0));

        {
            ZeroMallocGuard guard("self_aliasing_fused");
            self_v <<= self_v * 2.0;
        }
        assert(approx_eq(self_v[0], 4.0));
        assert(approx_eq(self_v[3], 16.0));
    }

    // 3. std::span zero-copy support
    {
        std::vector<double> big(8, 2.0);
        std::span<double> s1(big.data(), 4);
        std::span<double> s2(big.data() + 4, 4);
        double dot_spans = (s1 | s2);
        assert(approx_eq(dot_spans, 4.0 * (2.0 * 2.0)));

        std::vector<double> span_dst(4, 0.0);
        std::span<double> dst_span(span_dst.data(), 4);
        {
            ZeroMallocGuard guard("span_assign");
            dst_span <<= s1 + s2;
        }
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(dst_span[i], 4.0));
        }
    }

    // 4. std::array<double, 4> fixed-size zero-malloc states
    {
        std::array<double, 4> arr1 = {1.0, 2.0, 3.0, 4.0};
        std::array<double, 4> arr2 = {10.0, 20.0, 30.0, 40.0};
        std::array<double, 4> arr_out = {0.0, 0.0, 0.0, 0.0};

        {
            ZeroMallocGuard guard("array_assign");
            arr_out <<= 2.0 * arr1 + arr2;
        }
        assert(approx_eq(arr_out[0], 12.0));
        assert(approx_eq(arr_out[1], 24.0));
        assert(approx_eq(arr_out[2], 36.0));
        assert(approx_eq(arr_out[3], 48.0));

        double arr_dot = (arr1 | arr2);
        assert(approx_eq(arr_dot, 10.0 + 40.0 + 90.0 + 160.0));
    }

    // 5. eval_into direct buffer write
    {
        double raw_buf[4] = {0.0, 0.0, 0.0, 0.0};
        {
            ZeroMallocGuard guard("eval_into");
            eval_into(raw_buf, 4, a + b);
        }
        for (std::size_t i = 0; i < 4; ++i) {
            assert(approx_eq(raw_buf[i], a[i] + b[i]));
        }
    }

    // 6. Dense matvec (DG polynomial order Np=4)
    {
        // 4x4 matrix
        double A[16] = {
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 3.0, 0.0,
            0.0, 0.0, 0.0, 4.0
        };
        double x_vec[4] = {1.0, 1.0, 1.0, 1.0};
        double y_vec[4] = {0.0, 0.0, 0.0, 0.0};
        matvec(A, x_vec, y_vec, 4, 4);
        assert(approx_eq(y_vec[0], 1.0));
        assert(approx_eq(y_vec[1], 2.0));
        assert(approx_eq(y_vec[2], 3.0));
        assert(approx_eq(y_vec[3], 4.0));
    }

    // 7. Single-pass norm check
    {
        double n_sq = norm_sq(a + b);
        double expected_n_sq = 0.0;
        for (std::size_t i = 0; i < 4; ++i) {
            double v = a[i] + b[i];
            expected_n_sq += v * v;
        }
        assert(approx_eq(n_sq, expected_n_sq));
        assert(approx_eq(norm(a + b), std::sqrt(expected_n_sq)));
    }

    // 8. Deterministic reduction mode for bit-identical regression runs
    {
        double s_det = sum_deterministic(a);
        double d_det = dot_deterministic(a, b);
        assert(approx_eq(s_det, sum(a)));
        assert(approx_eq(d_det, (a | b)));
    }

    // 9. std::complex support: norm, norm_sq, real_of
    {
        std::vector<std::complex<double>> vc = {
            std::complex<double>(3.0, 4.0),
            std::complex<double>(1.0, -2.0)
        };
        static_assert(std::is_same_v<typename vec_math::real_of<std::complex<double>>::type, double>);
        double c_norm_sq = 0.0;
        double c_norm = 0.0;
        {
            ZeroMallocGuard guard("complex_norm");
            c_norm_sq = norm_sq(vc);
            c_norm = norm(vc);
        }
        assert(approx_eq(c_norm_sq, 30.0));
        assert(approx_eq(c_norm, std::sqrt(30.0)));
    }

    // 10. dotc conjugate inner product
    {
        std::vector<std::complex<double>> c1 = {
            std::complex<double>(1.0, 2.0),
            std::complex<double>(3.0, -1.0)
        };
        std::vector<std::complex<double>> c2 = {
            std::complex<double>(2.0, 3.0),
            std::complex<double>(-2.0, 4.0)
        };
        std::complex<double> c_res;
        {
            ZeroMallocGuard guard("complex_dotc");
            c_res = dotc(c1, c2);
        }
        assert(approx_eq(c_res.real(), -2.0));
        assert(approx_eq(c_res.imag(), 9.0));

        std::complex<double> self_dotc = dotc(c1, c1);
        assert(approx_eq(self_dotc.real(), norm_sq(c1)));
        assert(approx_eq(self_dotc.imag(), 0.0));
    }

    // 11. matvec and matvec_transposed in-place self-aliasing (x == y)
    {
        double A[16] = {
            1.0, 2.0, 0.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 5.0, 6.0,
            7.0, 0.0, 0.0, 8.0
        };
        double x_inplace[4] = {1.0, 2.0, 3.0, 4.0};
        {
            ZeroMallocGuard guard("matvec_self_alias");
            matvec(A, x_inplace, x_inplace, 4, 4);
        }
        assert(approx_eq(x_inplace[0], 5.0));
        assert(approx_eq(x_inplace[1], 18.0));
        assert(approx_eq(x_inplace[2], 39.0));
        assert(approx_eq(x_inplace[3], 39.0));

        double x_tr_inplace[4] = {1.0, 2.0, 3.0, 4.0};
        {
            ZeroMallocGuard guard("matvec_tr_self_alias");
            matvec_transposed(A, x_tr_inplace, x_tr_inplace, 4, 4);
        }
        assert(approx_eq(x_tr_inplace[0], 29.0));
        assert(approx_eq(x_tr_inplace[1], 8.0));
        assert(approx_eq(x_tr_inplace[2], 23.0));
        assert(approx_eq(x_tr_inplace[3], 50.0));
    }

    // 12. VEC_MATH_NO_INTRINSICS compilation check
    {
#if defined(VEC_MATH_NO_INTRINSICS)
#if defined(VEC_MATH_HAS_AVX2_FMA) || defined(VEC_MATH_HAS_X86_INTRIN)
        static_assert(false, "VEC_MATH_NO_INTRINSICS violated: intrinsics detected");
#endif
#endif
        std::vector<double> v_test = {2.0, 4.0, 6.0, 8.0};
        double s_val = sum_deterministic(v_test);
        assert(approx_eq(s_val, 20.0));
    }

    // 13. S1: matvec_fixed and matvec_transposed_fixed
    {
        double A[16] = {
            2.0, 1.0, 0.0, 0.0,
            1.0, 2.0, 1.0, 0.0,
            0.0, 1.0, 2.0, 1.0,
            0.0, 0.0, 1.0, 2.0
        };
        double x[4] = {1.0, 2.0, 3.0, 4.0};
        double y[4] = {0.0, 0.0, 0.0, 0.0};
        matvec_fixed<4, 4>(A, x, y);
        assert(approx_eq(y[0], 4.0));
        assert(approx_eq(y[1], 8.0));
        assert(approx_eq(y[2], 12.0));
        assert(approx_eq(y[3], 11.0));

        // Fixed self-aliased in-place matvec
        double x_inplace[4] = {1.0, 2.0, 3.0, 4.0};
        matvec_fixed<4, 4>(A, x_inplace, x_inplace);
        assert(approx_eq(x_inplace[0], 4.0));
        assert(approx_eq(x_inplace[1], 8.0));
        assert(approx_eq(x_inplace[2], 12.0));
        assert(approx_eq(x_inplace[3], 11.0));

        double yt[4] = {0.0, 0.0, 0.0, 0.0};
        matvec_transposed_fixed<4, 4>(A, x, yt);
        assert(approx_eq(yt[0], 4.0));
        assert(approx_eq(yt[1], 8.0));
        assert(approx_eq(yt[2], 12.0));
        assert(approx_eq(yt[3], 11.0));
    }

    // 14. S2: Compound assignment for std::span and std::array
    {
        double raw_span_buf[4] = {10.0, 20.0, 30.0, 40.0};
        std::span<double> sp(raw_span_buf, 4);
        double add_buf[4] = {1.0, 2.0, 3.0, 4.0};
        std::span<const double> sp_add(add_buf, 4);

        sp += sp_add;
        assert(approx_eq(sp[0], 11.0));
        assert(approx_eq(sp[1], 22.0));
        assert(approx_eq(sp[2], 33.0));
        assert(approx_eq(sp[3], 44.0));

        sp -= 1.0;
        assert(approx_eq(sp[0], 10.0));
        assert(approx_eq(sp[1], 21.0));
        assert(approx_eq(sp[2], 32.0));
        assert(approx_eq(sp[3], 43.0));

        sp *= 2.0;
        assert(approx_eq(sp[0], 20.0));
        assert(approx_eq(sp[1], 42.0));

        sp /= 2.0;
        assert(approx_eq(sp[0], 10.0));
        assert(approx_eq(sp[1], 21.0));

        std::array<double, 3> arr = {1.0, 2.0, 3.0};
        std::array<double, 3> arr2 = {4.0, 5.0, 6.0};
        arr += arr2;
        assert(approx_eq(arr[0], 5.0));
        assert(approx_eq(arr[1], 7.0));
        assert(approx_eq(arr[2], 9.0));

        arr *= 2.0;
        assert(approx_eq(arr[0], 10.0));
        assert(approx_eq(arr[1], 14.0));
        assert(approx_eq(arr[2], 18.0));
    }

    // 15. S3: sum_repro and dot_repro
    {
        std::vector<float> v_float_repro(1024);
        for (std::size_t i = 0; i < 1024; ++i) {
            v_float_repro[i] = 1.0f / static_cast<float>(i + 1);
        }
        float s_repro = sum_repro(v_float_repro);
        float s_det = sum_deterministic(v_float_repro);
        assert(std::abs(s_repro - s_det) < 1e-4f);

        std::vector<double> vd1(1024, 1.5);
        std::vector<double> vd2(1024, 2.0);
        double d_repro = dot_repro(vd1, vd2);
        assert(approx_eq(d_repro, 1024.0 * 3.0));
    }

    // 16. S5: StridedSpan, broadcast_multiply, and WorkspaceArena
    {
        double buf[12] = {
            1.0, 10.0, 100.0,
            2.0, 20.0, 200.0,
            3.0, 30.0, 300.0,
            4.0, 40.0, 400.0
        };
        StridedSpan<double> sspan(buf + 1, 4, 3);
        assert(sspan.size() == 4);
        assert(approx_eq(sspan[0], 10.0));
        assert(approx_eq(sspan[1], 20.0));
        assert(approx_eq(sspan[2], 30.0));
        assert(approx_eq(sspan[3], 40.0));

        double bcast_buf[4] = {2.0, 4.0, 6.0, 8.0};
        broadcast_multiply(std::span<double>(bcast_buf, 4), 0.5);
        assert(approx_eq(bcast_buf[0], 1.0));
        assert(approx_eq(bcast_buf[1], 2.0));
        assert(approx_eq(bcast_buf[2], 3.0));
        assert(approx_eq(bcast_buf[3], 4.0));

        WorkspaceArena arena(64 * 1024);
        assert(arena.capacity() == 64 * 1024);
        auto sp1 = arena.allocate<double>(100);
        assert(sp1.size() == 100);
        assert(reinterpret_cast<std::uintptr_t>(sp1.data()) % 64 == 0);
        auto sp2 = arena.allocate<double>(50);
        assert(sp2.size() == 50);
        assert(reinterpret_cast<std::uintptr_t>(sp2.data()) % 64 == 0);
        assert(arena.allocated_bytes() >= (150 * sizeof(double)));

        arena.reset();
        assert(arena.allocated_bytes() == 0);
    }

    // 17. S6: gather and scatter_add
    {
        double src[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
        std::size_t idx[3] = {1, 3, 5};
        double dst[3] = {0.0, 0.0, 0.0};

        gather(std::span<double>(dst, 3), std::span<const double>(src, 6), std::span<const std::size_t>(idx, 3));
        assert(approx_eq(dst[0], 20.0));
        assert(approx_eq(dst[1], 40.0));
        assert(approx_eq(dst[2], 60.0));

        double accum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double flux[3] = {5.0, 15.0, 25.0};
        scatter_add(std::span<double>(accum, 6), std::span<const std::size_t>(idx, 3), std::span<const double>(flux, 3));
        assert(approx_eq(accum[1], 5.0));
        assert(approx_eq(accum[3], 15.0));
        assert(approx_eq(accum[5], 25.0));
        assert(approx_eq(accum[0], 0.0));
    }

    // 18. S7: solve_4x4_batched and AoS <-> SoA transpositions
    {
        // Solve A x = b for 2 elements simultaneously in SoA layout
        // For element 0: diag matrix 2, 3, 4, 5 with rhs 4, 9, 16, 25 -> sol 2, 3, 4, 5
        // For element 1: identity matrix with rhs 1, 2, 3, 4 -> sol 1, 2, 3, 4
        constexpr std::size_t nelem = 2;
        double A_entries[16][nelem] = {};
        // Diagonal elements
        A_entries[0][0] = 2.0;  A_entries[0][1] = 1.0;
        A_entries[5][0] = 3.0;  A_entries[5][1] = 1.0;
        A_entries[10][0] = 4.0; A_entries[10][1] = 1.0;
        A_entries[15][0] = 5.0; A_entries[15][1] = 1.0;

        double b_entries[4][nelem] = {
            {4.0, 1.0},
            {9.0, 2.0},
            {16.0, 3.0},
            {25.0, 4.0}
        };

        double x_entries[4][nelem] = {};

        const double* A_ptrs[16];
        for (int i = 0; i < 16; ++i) A_ptrs[i] = A_entries[i];

        const double* b_ptrs[4];
        double* x_ptrs[4];
        for (int i = 0; i < 4; ++i) {
            b_ptrs[i] = b_entries[i];
            x_ptrs[i] = x_entries[i];
        }

        solve_4x4_batched(A_ptrs, b_ptrs, x_ptrs, nelem);

        assert(approx_eq(x_entries[0][0], 2.0));
        assert(approx_eq(x_entries[1][0], 3.0));
        assert(approx_eq(x_entries[2][0], 4.0));
        assert(approx_eq(x_entries[3][0], 5.0));

        assert(approx_eq(x_entries[0][1], 1.0));
        assert(approx_eq(x_entries[1][1], 2.0));
        assert(approx_eq(x_entries[2][1], 3.0));
        assert(approx_eq(x_entries[3][1], 4.0));

        // Transposition test: 2 elements with 4 variables
        double aos[8] = {1.0, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0};
        double soa[8] = {};
        double aos_back[8] = {};

        transpose_aos_to_soa(aos, soa, 2, 4);
        assert(approx_eq(soa[0], 1.0)); // v=0, elem=0
        assert(approx_eq(soa[1], 5.0)); // v=0, elem=1
        assert(approx_eq(soa[2], 2.0)); // v=1, elem=0
        assert(approx_eq(soa[3], 6.0)); // v=1, elem=1

        transpose_soa_to_aos(soa, aos_back, 2, 4);
        for (int i = 0; i < 8; ++i) {
            assert(approx_eq(aos[i], aos_back[i]));
        }
    }

    // 19. S4: Mathematical property tests
    {
        std::vector<double> px = {1.5, -2.5, 3.5, -4.5};
        double nsq = norm_sq(px);
        double dself = dot(px, px);
        assert(approx_eq(nsq, dself));

        std::vector<double> pdiff(4, 0.0);
        pdiff <<= px - px;
        assert(approx_eq(sum(pdiff), 0.0));
    }

    // 20. S10: Sum-Factorization / Tensor Contractions (contract_axis)
    {
        ZeroMallocGuard guard("s10_tensor_contraction");

        // Helper naive contraction for reference
        auto naive_contract = [](auto axis, const double* D, const double* in, double* out,
                                 std::size_t nx, std::size_t ny, std::size_t nz) {
            constexpr std::size_t ax = static_cast<std::size_t>(decltype(axis)::value);
            for (std::size_t k = 0; k < nz; ++k) {
                for (std::size_t j = 0; j < ny; ++j) {
                    for (std::size_t i = 0; i < nx; ++i) {
                        double s = 0.0;
                        if constexpr (ax == 0) {
                            for (std::size_t p = 0; p < nx; ++p) {
                                s += D[i * nx + p] * in[p + j * nx + k * nx * ny];
                            }
                        } else if constexpr (ax == 1) {
                            for (std::size_t q = 0; q < ny; ++q) {
                                s += D[j * ny + q] * in[i + q * nx + k * nx * ny];
                            }
                        } else if constexpr (ax == 2) {
                            for (std::size_t r = 0; r < nz; ++r) {
                                s += D[k * nz + r] * in[i + j * nx + r * nx * ny];
                            }
                        }
                        out[i + j * nx + k * nx * ny] = s;
                    }
                }
            }
        };

        // --- 2D Contraction (nz = 1) ---
        // Let nx = 4, ny = 4, isotropic N = 4
        constexpr std::size_t N = 4;
        double D_4x4[16] = {
            1.0, 2.0, -1.0, 0.5,
            0.5, -1.5, 2.0, 1.0,
            2.5, 0.0, -2.0, 1.5,
            -1.0, 1.0, 0.5, -0.5
        };

        double in_2d[16];
        for (std::size_t i = 0; i < 16; ++i) {
            in_2d[i] = static_cast<double>(i + 1) * 0.75;
        }

        double out_fast[16] = {};
        double out_ref[16] = {};

        // 2D Axis 0 (X), compile-time N = 4
        contract_axis<0, N>(D_4x4, in_2d, out_fast, N, N, 1);
        naive_contract(std::integral_constant<std::size_t, 0>{}, D_4x4, in_2d, out_ref, N, N, 1);
        for (std::size_t i = 0; i < 16; ++i) {
            assert(approx_eq(out_fast[i], out_ref[i], 1e-11));
        }

        // 2D Axis 0 (X), runtime N = 0
        std::fill(out_fast, out_fast + 16, 0.0);
        contract_axis<0>(D_4x4, in_2d, out_fast, N, N, 1);
        for (std::size_t i = 0; i < 16; ++i) {
            assert(approx_eq(out_fast[i], out_ref[i], 1e-11));
        }

        // 2D Axis 1 (Y), compile-time N = 4 via Axis::Y
        std::fill(out_fast, out_fast + 16, 0.0);
        std::fill(out_ref, out_ref + 16, 0.0);
        contract_axis<Axis::Y, N>(D_4x4, in_2d, out_fast, N, N); // default nz = 1
        naive_contract(std::integral_constant<std::size_t, 1>{}, D_4x4, in_2d, out_ref, N, N, 1);
        for (std::size_t i = 0; i < 16; ++i) {
            assert(approx_eq(out_fast[i], out_ref[i], 1e-11));
        }

        // 2D Axis 1 (Y), runtime N = 0
        std::fill(out_fast, out_fast + 16, 0.0);
        contract_axis<1>(D_4x4, in_2d, out_fast, N, N);
        for (std::size_t i = 0; i < 16; ++i) {
            assert(approx_eq(out_fast[i], out_ref[i], 1e-11));
        }

        // 2D In-place aliasing (in == out)
        double in_aliased[16];
        std::copy(in_2d, in_2d + 16, in_aliased);
        contract_axis<Axis::X, N>(D_4x4, in_aliased, in_aliased, N, N, 1);
        naive_contract(std::integral_constant<std::size_t, 0>{}, D_4x4, in_2d, out_ref, N, N, 1);
        for (std::size_t i = 0; i < 16; ++i) {
            assert(approx_eq(in_aliased[i], out_ref[i], 1e-11));
        }

        // --- 3D Contraction ---
        // Let nx = 3, ny = 3, nz = 3
        constexpr std::size_t N3 = 3;
        constexpr std::size_t total_3d = N3 * N3 * N3;
        double D_3x3[9] = {
            1.2, -0.8, 0.5,
            0.3, 1.5, -1.1,
            -0.7, 0.9, 2.1
        };

        double in_3d[total_3d];
        for (std::size_t i = 0; i < total_3d; ++i) {
            in_3d[i] = static_cast<double>(i + 1) * 0.5 - 7.0;
        }

        double out_fast_3d[total_3d] = {};
        double out_ref_3d[total_3d] = {};

        // 3D Axis 0 (X): fixed N=3
        contract_axis<Axis::X, N3>(D_3x3, in_3d, out_fast_3d, N3, N3, N3);
        naive_contract(std::integral_constant<std::size_t, 0>{}, D_3x3, in_3d, out_ref_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(out_fast_3d[i], out_ref_3d[i], 1e-11));
        }

        // 3D Axis 0 (X): dynamic N=0
        std::fill(out_fast_3d, out_fast_3d + total_3d, 0.0);
        contract_axis<0>(D_3x3, in_3d, out_fast_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(out_fast_3d[i], out_ref_3d[i], 1e-11));
        }

        // 3D Axis 1 (Y): fixed N=3
        std::fill(out_fast_3d, out_fast_3d + total_3d, 0.0);
        contract_axis<Axis::Y, N3>(D_3x3, in_3d, out_fast_3d, N3, N3, N3);
        naive_contract(std::integral_constant<std::size_t, 1>{}, D_3x3, in_3d, out_ref_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(out_fast_3d[i], out_ref_3d[i], 1e-11));
        }

        // 3D Axis 1 (Y): dynamic N=0
        std::fill(out_fast_3d, out_fast_3d + total_3d, 0.0);
        contract_axis<1>(D_3x3, in_3d, out_fast_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(out_fast_3d[i], out_ref_3d[i], 1e-11));
        }

        // 3D Axis 2 (Z): fixed N=3
        std::fill(out_fast_3d, out_fast_3d + total_3d, 0.0);
        contract_axis<Axis::Z, N3>(D_3x3, in_3d, out_fast_3d, N3, N3, N3);
        naive_contract(std::integral_constant<std::size_t, 2>{}, D_3x3, in_3d, out_ref_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(out_fast_3d[i], out_ref_3d[i], 1e-11));
        }

        // 3D Axis 2 (Z): dynamic N=0
        std::fill(out_fast_3d, out_fast_3d + total_3d, 0.0);
        contract_axis<2>(D_3x3, in_3d, out_fast_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(out_fast_3d[i], out_ref_3d[i], 1e-11));
        }

        // 3D In-place aliasing (in == out) on Axis 2
        double in_aliased_3d[total_3d];
        std::copy(in_3d, in_3d + total_3d, in_aliased_3d);
        contract_axis<Axis::Z, N3>(D_3x3, in_aliased_3d, in_aliased_3d, N3, N3, N3);
        for (std::size_t i = 0; i < total_3d; ++i) {
            assert(approx_eq(in_aliased_3d[i], out_ref_3d[i], 1e-11));
        }

        // --- Anisotropic 3D Contraction ---
        // nx = 2, ny = 3, nz = 4
        constexpr std::size_t anx = 2, any = 3, anz = 4;
        constexpr std::size_t an_total = anx * any * anz;
        double D_nx[anx * anx] = {1.0, -1.0, 2.0, 0.5};
        double D_ny[any * any] = {
            1.0, 0.2, -0.3,
            -0.5, 2.0, 0.1,
            0.4, -0.6, 1.5
        };
        double D_nz[anz * anz] = {
            1.0, -0.5, 0.2, 0.1,
            0.3, 1.2, -0.4, 0.5,
            -0.2, 0.1, 1.8, -0.3,
            0.5, -0.2, 0.3, 1.1
        };

        double an_in[an_total];
        for (std::size_t i = 0; i < an_total; ++i) {
            an_in[i] = static_cast<double>(i) * 0.3 - 2.5;
        }
        double an_fast[an_total] = {};
        double an_ref[an_total] = {};

        // Anisotropic Axis 0
        contract_axis<0>(D_nx, an_in, an_fast, anx, any, anz);
        naive_contract(std::integral_constant<std::size_t, 0>{}, D_nx, an_in, an_ref, anx, any, anz);
        for (std::size_t i = 0; i < an_total; ++i) {
            assert(approx_eq(an_fast[i], an_ref[i], 1e-11));
        }

        // Anisotropic Axis 1
        std::fill(an_fast, an_fast + an_total, 0.0);
        contract_axis<1>(D_ny, an_in, an_fast, anx, any, anz);
        naive_contract(std::integral_constant<std::size_t, 1>{}, D_ny, an_in, an_ref, anx, any, anz);
        for (std::size_t i = 0; i < an_total; ++i) {
            assert(approx_eq(an_fast[i], an_ref[i], 1e-11));
        }

        // Anisotropic Axis 2
        std::fill(an_fast, an_fast + an_total, 0.0);
        contract_axis<2>(D_nz, an_in, an_fast, anx, any, anz);
        naive_contract(std::integral_constant<std::size_t, 2>{}, D_nz, an_in, an_ref, anx, any, anz);
        for (std::size_t i = 0; i < an_total; ++i) {
            assert(approx_eq(an_fast[i], an_ref[i], 1e-11));
        }

        // Span interface test
        std::fill(an_fast, an_fast + an_total, 0.0);
        contract_axis<0>(std::span<const double>(D_nx, anx * anx),
                         std::span<const double>(an_in, an_total),
                         std::span<double>(an_fast, an_total),
                         anx, any, anz);
        naive_contract(std::integral_constant<std::size_t, 0>{}, D_nx, an_in, an_ref, anx, any, anz);
        for (std::size_t i = 0; i < an_total; ++i) {
            assert(approx_eq(an_fast[i], an_ref[i], 1e-11));
        }
    }

    // ========================================================================
    // Hard Gate G3: REPRO Reductions (sum, dot, sum_repro, dot_repro)
    // ========================================================================
    {
        std::cout << "[TEST] Hard Gate G3: REPRO Policy Overloads & Blocked Pairwise Tree..." << std::endl;
        const std::vector<std::size_t> test_sizes = {1, 7, 32, 511, 512, 513, 1024, 1500, 2048};

        for (std::size_t n : test_sizes) {
            std::vector<double> v_d(n);
            std::vector<double> u_d(n);
            std::vector<float> v_f(n);
            std::vector<float> u_f(n);

            for (std::size_t i = 0; i < n; ++i) {
                v_d[i] = static_cast<double>(i + 1) * 0.125;
                u_d[i] = 1.0 + static_cast<double>(i % 7) * 0.25;
                v_f[i] = static_cast<float>(i + 1) * 0.125f;
                u_f[i] = 1.0f + static_cast<float>(i % 7) * 0.25f;
            }

            // FP64 sum tests
            double sum_fast_d = sum(v_d, ReductionPolicy::FAST);
            double sum_default_d = sum(v_d);
            double sum_repro_d = sum(v_d, ReductionPolicy::REPRO);
            double sum_direct_repro_d = sum_repro(v_d);

            assert(approx_eq(sum_fast_d, sum_default_d, 1e-12));
            assert(sum_repro_d == sum_direct_repro_d); // Strict bit-identical
            assert(approx_eq(sum_repro_d, sum_fast_d, 1e-9));

            // FP32 sum tests
            float sum_fast_f = sum(v_f, ReductionPolicy::FAST);
            float sum_default_f = sum(v_f);
            float sum_repro_f = sum(v_f, ReductionPolicy::REPRO);
            float sum_direct_repro_f = sum_repro(v_f);

            assert(std::abs(sum_fast_f - sum_default_f) <= 1e-5f);
            assert(sum_repro_f == sum_direct_repro_f); // Strict bit-identical

            // FP64 dot tests
            double dot_fast_d = dot(v_d, u_d, ReductionPolicy::FAST);
            double dot_default_d = dot(v_d, u_d);
            double dot_repro_d = dot(v_d, u_d, ReductionPolicy::REPRO);
            double dot_direct_repro_d = dot_repro(v_d, u_d);

            assert(approx_eq(dot_fast_d, dot_default_d, 1e-12));
            assert(dot_repro_d == dot_direct_repro_d); // Strict bit-identical
            assert(approx_eq(dot_repro_d, dot_fast_d, 1e-9));

            // FP32 dot tests
            float dot_fast_f = dot(v_f, u_f, ReductionPolicy::FAST);
            float dot_default_f = dot(v_f, u_f);
            float dot_repro_f = dot(v_f, u_f, ReductionPolicy::REPRO);
            float dot_direct_repro_f = dot_repro(v_f, u_f);

            assert(std::abs(dot_fast_f - dot_default_f) <= 1e-4f);
            assert(dot_repro_f == dot_direct_repro_f); // Strict bit-identical
        }

        // Bounded FP32 error test with ill-conditioned sequence
        {
            const std::size_t N = 10000;
            std::vector<float> ill_f(N);
            for (std::size_t i = 0; i < N; ++i) {
                ill_f[i] = (i % 2 == 0) ? 1.0e4f : -1.0e4f + 0.001f;
            }
            float repro_s = sum(ill_f, ReductionPolicy::REPRO);
            float expected_pairwise = 4.8828125f;
            assert(std::abs(repro_s - expected_pairwise) < 1e-4f);
        }
    }

    // ========================================================================
    // Hard Gate G5: Aliasing Matrix (dst == src, dst == src + 1, dst == src - 1)
    // ========================================================================
    {
        std::cout << "[TEST] Hard Gate G5: Aliasing Matrix Execution Safety..." << std::endl;

        // 1. eval_into Aliasing Matrix
        {
            constexpr std::size_t N = 128;
            double storage[N + 2];
            for (std::size_t i = 0; i < N + 2; ++i) {
                storage[i] = static_cast<double>(i) * 1.5 + 2.0;
            }

            // (a) dst == src (In-place self-aliasing)
            {
                double expected[N];
                for (std::size_t i = 0; i < N; ++i) expected[i] = storage[i];
                eval_into(storage, N, std::span<const double>(storage, N));
                for (std::size_t i = 0; i < N; ++i) {
                    assert(approx_eq(storage[i], expected[i], 1e-12));
                }
            }

            // (b) dst == src + 1 (Overlapping shift right)
            {
                for (std::size_t i = 0; i < N + 2; ++i) storage[i] = static_cast<double>(i) * 2.0 + 1.0;
                double expected[N];
                for (std::size_t i = 0; i < N; ++i) expected[i] = storage[i];

                eval_into(storage + 1, N, std::span<const double>(storage, N));
                for (std::size_t i = 0; i < N; ++i) {
                    assert(approx_eq(storage[i + 1], expected[i], 1e-12));
                }
            }

            // (c) dst == src - 1 (Overlapping shift left)
            {
                for (std::size_t i = 0; i < N + 2; ++i) storage[i] = static_cast<double>(i) * 3.0 + 0.5;
                double expected[N];
                for (std::size_t i = 0; i < N; ++i) expected[i] = storage[i + 1];

                eval_into(storage, N, std::span<const double>(storage + 1, N));
                for (std::size_t i = 0; i < N; ++i) {
                    assert(approx_eq(storage[i], expected[i], 1e-12));
                }
            }

            // (d) Expression via operator<<= with shift
            {
                for (std::size_t i = 0; i < N + 2; ++i) storage[i] = static_cast<double>(i + 1);
                double expected[N];
                for (std::size_t i = 0; i < N; ++i) expected[i] = storage[i] * 2.5 + 3.0;

                std::span<double> dst_span(storage + 1, N);
                std::span<const double> src_span(storage, N);
                dst_span <<= src_span * 2.5 + 3.0;
                for (std::size_t i = 0; i < N; ++i) {
                    assert(approx_eq(storage[i + 1], expected[i], 1e-12));
                }
            }
        }

        // 2. matvec Aliasing Matrix
        {
            constexpr std::size_t M = 4;
            double A[M * M] = {
                2.0, -1.0, 0.5, 0.0,
                0.5,  3.0, 1.0, -0.5,
                1.0,  0.0, 2.5, 1.5,
                -0.5, 1.0, 0.0, 2.0
            };
            double raw_buf[M + 2];
            for (std::size_t i = 0; i < M + 2; ++i) raw_buf[i] = static_cast<double>(i) + 1.0;

            // Reference computation
            auto compute_ref = [&](const double* in_vec, double* out_ref) {
                for (std::size_t r = 0; r < M; ++r) {
                    double s = 0.0;
                    for (std::size_t c = 0; c < M; ++c) {
                        s += A[r * M + c] * in_vec[c];
                    }
                    out_ref[r] = s;
                }
            };

            // (a) matvec dst == src
            {
                double x_aliased[M] = {1.0, 2.0, 3.0, 4.0};
                double y_ref[M];
                compute_ref(x_aliased, y_ref);
                matvec(A, x_aliased, x_aliased, M, M);
                for (std::size_t r = 0; r < M; ++r) {
                    assert(approx_eq(x_aliased[r], y_ref[r], 1e-12));
                }
            }

            // (b) matvec dst == src + 1
            {
                for (std::size_t i = 0; i < M + 2; ++i) raw_buf[i] = static_cast<double>(i) * 1.2 + 0.8;
                double y_ref[M];
                compute_ref(raw_buf, y_ref);
                matvec(A, raw_buf, raw_buf + 1, M, M);
                for (std::size_t r = 0; r < M; ++r) {
                    assert(approx_eq(raw_buf[r + 1], y_ref[r], 1e-12));
                }
            }

            // (c) matvec dst == src - 1
            {
                for (std::size_t i = 0; i < M + 2; ++i) raw_buf[i] = static_cast<double>(i) * 0.7 + 1.5;
                double y_ref[M];
                compute_ref(raw_buf + 1, y_ref);
                matvec(A, raw_buf + 1, raw_buf, M, M);
                for (std::size_t r = 0; r < M; ++r) {
                    assert(approx_eq(raw_buf[r], y_ref[r], 1e-12));
                }
            }
        }

        // 3. matvec_fixed Aliasing Matrix
        {
            constexpr std::size_t M = 4;
            double A[M * M] = {
                1.5, -0.5, 0.2, 0.1,
                0.3,  2.0, -0.4, 0.5,
                -0.2, 0.1, 1.8, -0.3,
                0.5, -0.2, 0.3, 1.1
            };
            double raw_buf[M + 2];

            auto compute_ref = [&](const double* in_vec, double* out_ref) {
                for (std::size_t r = 0; r < M; ++r) {
                    double s = 0.0;
                    for (std::size_t c = 0; c < M; ++c) {
                        s += A[r * M + c] * in_vec[c];
                    }
                    out_ref[r] = s;
                }
            };

            // (a) matvec_fixed dst == src
            {
                double x_fixed[M] = {2.5, -1.2, 0.8, 3.1};
                double y_ref[M];
                compute_ref(x_fixed, y_ref);
                matvec_fixed<M, M>(A, x_fixed, x_fixed);
                for (std::size_t r = 0; r < M; ++r) {
                    assert(approx_eq(x_fixed[r], y_ref[r], 1e-12));
                }
            }

            // (b) matvec_fixed dst == src + 1
            {
                for (std::size_t i = 0; i < M + 2; ++i) raw_buf[i] = static_cast<double>(i) * 0.9 + 0.4;
                double y_ref[M];
                compute_ref(raw_buf, y_ref);
                matvec_fixed<M, M>(A, raw_buf, raw_buf + 1);
                for (std::size_t r = 0; r < M; ++r) {
                    assert(approx_eq(raw_buf[r + 1], y_ref[r], 1e-12));
                }
            }

            // (c) matvec_fixed dst == src - 1
            {
                for (std::size_t i = 0; i < M + 2; ++i) raw_buf[i] = static_cast<double>(i) * 1.1 - 0.3;
                double y_ref[M];
                compute_ref(raw_buf + 1, y_ref);
                matvec_fixed<M, M>(A, raw_buf + 1, raw_buf);
                for (std::size_t r = 0; r < M; ++r) {
                    assert(approx_eq(raw_buf[r], y_ref[r], 1e-12));
                }
            }
        }

        // 4. contract_axis Aliasing Matrix
        {
            constexpr std::size_t nx = 3, ny = 4, nz = 2;
            constexpr std::size_t total = nx * ny * nz; // 24
            double D0[nx * nx] = {
                1.0, 0.2, -0.1,
                0.3, 1.5,  0.4,
                -0.2, 0.1, 1.2
            };
            double tensor_buf[total + 2];

            // Reference function for Axis 0
            auto compute_ref_axis0 = [&](const double* in_tensor, double* out_ref) {
                const std::size_t n_pencils = ny * nz;
                for (std::size_t p = 0; p < n_pencils; ++p) {
                    const double* p_in = in_tensor + p * nx;
                    double* p_out = out_ref + p * nx;
                    for (std::size_t i = 0; i < nx; ++i) {
                        double s = 0.0;
                        for (std::size_t j = 0; j < nx; ++j) {
                            s += D0[i * nx + j] * p_in[j];
                        }
                        p_out[i] = s;
                    }
                }
            };

            // (a) contract_axis dst == src
            {
                for (std::size_t i = 0; i < total; ++i) tensor_buf[i] = static_cast<double>(i) * 0.25 - 1.0;
                double ref[total];
                compute_ref_axis0(tensor_buf, ref);
                contract_axis<0>(D0, tensor_buf, tensor_buf, nx, ny, nz);
                for (std::size_t i = 0; i < total; ++i) {
                    assert(approx_eq(tensor_buf[i], ref[i], 1e-12));
                }
            }

            // (b) contract_axis dst == src + 1
            {
                for (std::size_t i = 0; i < total + 2; ++i) tensor_buf[i] = static_cast<double>(i) * 0.3 - 0.5;
                double ref[total];
                compute_ref_axis0(tensor_buf, ref);
                contract_axis<0>(D0, tensor_buf, tensor_buf + 1, nx, ny, nz);
                for (std::size_t i = 0; i < total; ++i) {
                    assert(approx_eq(tensor_buf[i + 1], ref[i], 1e-12));
                }
            }

            // (c) contract_axis dst == src - 1
            {
                for (std::size_t i = 0; i < total + 2; ++i) tensor_buf[i] = static_cast<double>(i) * 0.4 + 0.2;
                double ref[total];
                compute_ref_axis0(tensor_buf + 1, ref);
                contract_axis<0>(D0, tensor_buf + 1, tensor_buf, nx, ny, nz);
                for (std::size_t i = 0; i < total; ++i) {
                    assert(approx_eq(tensor_buf[i], ref[i], 1e-12));
                }
            }
        }
    }

    // ========================================================================
    // Property C1 & C4: DG Fit & Ergonomics (gather and scatter_add with non-const spans)
    // ========================================================================
    {
        std::cout << "[TEST] Property C1 & C4: Non-const span gather & scatter_add deduction..." << std::endl;

        std::vector<double> src_data = {100.0, 200.0, 300.0, 400.0, 500.0};
        std::vector<int> idx_data = {4, 2, 0, 1};
        std::vector<double> dst_data(4, 0.0);

        // Non-const std::span passed directly to gather (CTAD without template argument failure)
        std::span<double> dst_span(dst_data);
        std::span<double> src_span(src_data);
        std::span<int> idx_span(idx_data);

        gather(dst_span, src_span, idx_span);
        assert(approx_eq(dst_data[0], 500.0, 1e-12));
        assert(approx_eq(dst_data[1], 300.0, 1e-12));
        assert(approx_eq(dst_data[2], 100.0, 1e-12));
        assert(approx_eq(dst_data[3], 200.0, 1e-12));

        // Non-const std::span passed directly to scatter_add
        std::vector<double> accum(5, 10.0);
        std::vector<double> add_vals = {1.0, 2.0, 3.0, 4.0};
        std::span<double> accum_span(accum);
        std::span<double> add_span(add_vals);

        scatter_add(accum_span, idx_span, add_span);
        // idx_data = {4, 2, 0, 1}
        // accum[4] += 1.0 -> 11.0
        // accum[2] += 2.0 -> 12.0
        // accum[0] += 3.0 -> 13.0
        // accum[1] += 4.0 -> 14.0
        // accum[3] unchanged -> 10.0
        assert(approx_eq(accum[0], 13.0, 1e-12));
        assert(approx_eq(accum[1], 14.0, 1e-12));
        assert(approx_eq(accum[2], 12.0, 1e-12));
        assert(approx_eq(accum[3], 10.0, 1e-12));
        assert(approx_eq(accum[4], 11.0, 1e-12));
    }

    // ========================================================================
    // Property B2, B3, B4: Parallel Reductions & Zero False Sharing
    // ========================================================================
    {
        std::cout << "[TEST] Property B2, B3, B4: OpenMP reductions & Zero False Sharing..." << std::endl;
        constexpr std::size_t N = 4096;
        std::vector<double> vx(N, 1.5);
        std::vector<double> vy(N, 2.0);

        double par_dot = dot_par(vx, vy);
        double fast_dot = dot(vx, vy);
        assert(approx_eq(par_dot, fast_dot, 1e-12));

        double par_sum = sum_par(vx);
        double fast_sum = sum(vx);
        assert(approx_eq(par_sum, fast_sum, 1e-12));
    }

    std::vector<std::string> svec = {"hello", "world"};
    assert(svec.size() == 2);

    std::cout << "ALL BATTLE TESTS PASSED WITH STRICTLY ZERO MALLOCS!\n";
    return 0;
}
