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


static bool approx_eq(double a, double b, double eps = 1e-12) {
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


    std::vector<double> a = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> b = {10.0, 20.0, 30.0, 40.0};
    std::vector<double> c = {2.0, 4.0, 5.0, 8.0};
    std::vector<double> out(4, 0.0);


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


    {
        ZeroMallocGuard guard("fused_expression_template");

        out <<= 2.0 * a + b / c - 1.5 + (+a) - (-c);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        double expected = 2.0 * a[i] + b[i] / c[i] - 1.5 + a[i] + c[i];
        assert(approx_eq(out[i], expected));
    }


    {
        ZeroMallocGuard guard("scalar_lhs_sub_div");
        out <<= 100.0 / a - (5.0 - a);
    }
    for (std::size_t i = 0; i < 4; ++i) {
        double expected = 100.0 / a[i] - (5.0 - a[i]);
        assert(approx_eq(out[i], expected));
    }


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


    std::vector<float, CustomAlloc<float> > vf = {1.5f, 2.5f, 3.5f};
    std::vector<double, CustomAlloc<double> > vd = {10.0, 20.0, 30.0};
    std::vector<double, CustomAlloc<double> > vout(3, 0.0);
    {
        ZeroMallocGuard guard("mixed_types_custom_alloc");
        vout <<= 2 * vf + vd * 0.5;
    }
    assert(approx_eq(vout[0], 2 * 1.5 + 10.0 * 0.5));
    assert(approx_eq(vout[1], 2 * 2.5 + 20.0 * 0.5));
    assert(approx_eq(vout[2], 2 * 3.5 + 30.0 * 0.5));


    std::vector<double> empty_a, empty_b, empty_out;
    {
        ZeroMallocGuard guard("empty_vectors");
        empty_out <<= empty_a + 2.0 * empty_b;
        empty_out += 5.0;
    }
    assert(empty_out.empty());


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


    std::vector<std::string> svec = {"hello", "world"};
    assert(svec.size() == 2);

    std::cout << "ALL BATTLE TESTS PASSED WITH STRICTLY ZERO MALLOCS!\n";
    return 0;
}
