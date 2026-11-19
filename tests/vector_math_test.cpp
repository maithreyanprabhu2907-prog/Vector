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
    static_assert(sycl_compatible<vec_span<float>>);
    static_assert(sycl_compatible<vec_span<const double>>);
    static_assert(sycl_compatible<device_view<int>>);
    static_assert(sycl_compatible<device_view<const float>>);

    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> c = {2.0, 4.0, 5.0, 8.0};
    std::vector<double> out(4, 0.0);

    static_assert(sycl_compatible<decltype(2.0 * a + b)>);
    static_assert(sycl_compatible<decltype(vec_math::abs(a))>);
    static_assert(sycl_compatible<decltype(vec_math::fma(a, b, c))>);

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
    // Device-Compatible Span / View & SYCL Evaluation Tests
    // ========================================================================
    {
        double raw_a[4] = {1.0, 2.0, 3.0, 4.0};
        double raw_b[4] = {10.0, 20.0, 30.0, 40.0};
        double raw_out[4] = {0.0, 0.0, 0.0, 0.0};

        vec_span<const double> span_a(raw_a, 4);
        vec_span<const double> span_b(raw_b, 4);
        device_view<double> span_out(raw_out, 4);

        {
            ZeroMallocGuard guard("span_expressions_and_compound");
            span_out <<= 2.0 * span_a + span_b;
            span_out += span_a;
            span_out -= 1.0;
            span_out *= 2.0;
            span_out /= 2.0;
        }

        for (std::size_t i = 0; i < 4; ++i) {
            double expected = (2.0 * raw_a[i] + raw_b[i]) + raw_a[i] - 1.0;
            assert(approx_eq(span_out[i], expected));
        }

        // device_eval full & element-wise
        {
            ZeroMallocGuard guard("device_eval_tests");
            device_eval(span_out, span_a + span_b);
            for (std::size_t i = 0; i < 4; ++i) {
                assert(approx_eq(span_out[i], raw_a[i] + raw_b[i]));
            }

            for (std::size_t i = 0; i < 4; ++i) {
                device_eval(raw_out, fma(span_a, span_b, 5.0), i);
            }
            for (std::size_t i = 0; i < 4; ++i) {
                assert(approx_eq(raw_out[i], raw_a[i] * raw_b[i] + 5.0));
            }
        }

        // Span reduction
        {
            ZeroMallocGuard guard("span_reductions");
            double s_span = sum(span_a);
            assert(approx_eq(s_span, 10.0));

            double d_span = dot(span_a, span_b);
            assert(approx_eq(d_span, 300.0));
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

    std::vector<std::string> svec = {"hello", "world"};
    assert(svec.size() == 2);

    std::cout << "ALL BATTLE TESTS PASSED WITH STRICTLY ZERO MALLOCS!\n";
    return 0;
}
