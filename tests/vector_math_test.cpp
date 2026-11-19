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
            res_lim <<= clamp(v_lim, -1.0, 5.0);
            assert(approx_eq(res_lim[0], -1.0));
            assert(approx_eq(res_lim[1], 0.5));
            assert(approx_eq(res_lim[2], 3.0));
            assert(approx_eq(res_lim[3], 5.0));

            // clamp with vector bounds
            res_lim <<= clamp(v_lim, vlo, vhi);
            assert(approx_eq(res_lim[0], -1.0));
            assert(approx_eq(res_lim[3], 5.0));

            // min / max expressions
            res_lim <<= min(v_lim, 2.0);
            assert(approx_eq(res_lim[0], -2.0));
            assert(approx_eq(res_lim[1], 0.5));
            assert(approx_eq(res_lim[2], 2.0));
            assert(approx_eq(res_lim[3], 2.0));

            res_lim <<= max(v_lim, 1.0);
            assert(approx_eq(res_lim[0], 1.0));
            assert(approx_eq(res_lim[1], 1.0));
            assert(approx_eq(res_lim[2], 3.0));
            assert(approx_eq(res_lim[3], 10.0));

            // binary vector min / max
            res_lim <<= max(v_lim, vlo);
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

    std::vector<std::string> svec = {"hello", "world"};
    assert(svec.size() == 2);

    std::cout << "ALL BATTLE TESTS PASSED WITH STRICTLY ZERO MALLOCS!\n";
    return 0;
}
