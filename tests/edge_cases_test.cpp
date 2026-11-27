#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 11)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

#if defined(_MSC_VER) && defined(_DEBUG)
#define _ITERATOR_DEBUG_LEVEL 0
#endif

#include <keisan/keisan.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#undef assert
#define TEST_ASSERT(expr)                                                                 \
    do {                                                                                  \
        if (!(expr)) {                                                                    \
            std::cerr << "[FAIL] " << #expr << " at " << __FILE__ << ":" << __LINE__      \
                      << "\n";                                                            \
            std::abort();                                                                 \
        }                                                                                 \
    } while (false)

namespace {

bool g_track_allocs = false;
std::size_t g_alloc_count = 0;

struct ZeroMallocGuard {
    const char* name;
    std::size_t start;

    explicit ZeroMallocGuard(const char* n) noexcept : name(n), start(g_alloc_count) {
        g_track_allocs = true;
    }

    ~ZeroMallocGuard() {
        g_track_allocs = false;
        const std::size_t delta = g_alloc_count - start;
        if (delta != 0) {
            std::cerr << "[FAIL] ZeroMallocGuard(" << name << "): expected 0 mallocs, got "
                      << delta << "\n";
            std::abort();
        }
    }
};

} // namespace

void* operator new(std::size_t size) {
    if (g_track_allocs) {
        ++g_alloc_count;
    }
    void* p = std::malloc(size == 0 ? 1 : size);
    if (!p) {
        throw std::bad_alloc{};
    }
    return p;
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* ptr) noexcept {
    std::free(ptr);
}

void operator delete(void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr) noexcept {
    std::free(ptr);
}

void operator delete[](void* ptr, std::size_t) noexcept {
    std::free(ptr);
}

namespace {

template <typename T, typename U>
bool approx_eq(T a, U b, double eps = 1e-12) {
    return std::abs(static_cast<double>(a) - static_cast<double>(b)) <= eps;
}

template <typename T>
bool bitwise_eq(T a, T b) {
    static_assert(std::is_trivially_copyable_v<T>);
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// ============================================================================
// 1. Concepts, Type Traits, and Constexpr Evaluation
// ============================================================================

constexpr double test_constexpr_ops() {
    using namespace keisan;
    constexpr std::array<double, 4> a{1.0, 2.0, 3.0, 4.0};
    constexpr std::array<double, 4> b{5.0, 6.0, 7.0, 8.0};
    constexpr std::array<double, 4> c{-1.0, 0.5, -2.0, 3.0};

    auto expr = 2.0 * a + b - c;
    static_assert(sycl_compatible<decltype(expr)>);
    static_assert(is_sycl_compatible_v<decltype(expr)>);

    const double s = sum(expr);
    const double d = a | b;
    const double d2 = dot(a, b);
    const double nsq = norm_sq(a);
    const double mn = min_val(c);
    const double mx = max_val(c);
    const double s_abs = sum(abs(c));
    const double s_fma = sum(fma(a, 2.0, b));
    const double s_clamp = sum(clamp(c, 0.0, 2.0));
    const double s_minmax = sum(min(a, 2.5)) + sum(max(a, 2.5));
    const double s_repro = sum(a, ReductionPolicy::REPRO);
    const double d_repro = dot(a, b, ReductionPolicy::REPRO);

    return s + d + d2 + nsq + mn + mx + s_abs + s_fma + s_clamp + s_minmax + s_repro + d_repro;
}

constexpr int test_constexpr_integral() {
    using namespace keisan;
    constexpr std::array<int, 5> a{1, 2, 3, 4, 5};
    constexpr std::array<int, 5> b{2, 2, 2, 2, 2};
    return sum(3 * a - b) + dot(a, b);
}

void test_complex_runtime() {
    using namespace keisan;
    std::vector<std::complex<double>> zd{{1.0, 2.0}, {3.0, 4.0}};
    std::vector<std::complex<double>> wd{{1.0, 0.0}, {0.0, 1.0}};
    std::vector<std::complex<double>> outd(2);
    {
        ZeroMallocGuard guard("complex_double_ops");
        outd <<= zd + wd;
        const auto dc = dotc(zd, wd);
        TEST_ASSERT(approx_eq(dc.real(), 5.0));
        TEST_ASSERT(approx_eq(dc.imag(), 1.0));
        TEST_ASSERT(approx_eq(norm_sq(zd), 30.0));
        TEST_ASSERT(approx_eq(norm(zd), std::sqrt(30.0)));
    }
    TEST_ASSERT(approx_eq(outd[0].real(), 2.0) && approx_eq(outd[0].imag(), 2.0));

    std::vector<std::complex<float>> zf{{1.0f, 2.0f}, {3.0f, 4.0f}};
    std::vector<std::complex<float>> wf{{1.0f, 0.0f}, {0.0f, 1.0f}};
    {
        ZeroMallocGuard guard("complex_float_ops");
        const auto dc = dotc(zf, wf);
        TEST_ASSERT(approx_eq(dc.real(), 5.0f, 1e-5));
        TEST_ASSERT(approx_eq(dc.imag(), 1.0f, 1e-5));
        TEST_ASSERT(approx_eq(norm_sq(zf), 30.0f, 1e-5));
    }
}

void test_concepts_and_constexpr() {
    using namespace keisan;

    // Positive concept checks
    static_assert(Scalar<float>);
    static_assert(Scalar<double>);
    static_assert(Scalar<int>);
    static_assert(Scalar<std::int64_t>);
    static_assert(Scalar<std::complex<float>>);
    static_assert(Scalar<std::complex<double>>);

    // Negative concept checks
    static_assert(!Scalar<bool>);
    static_assert(!Scalar<std::string>);
    static_assert(!NumericVector<std::vector<bool>>);
    static_assert(!NumericVector<std::string>);
    static_assert(!VecExpression<std::vector<double>>);

    // NumericVector & VectorOperand checks
    static_assert(NumericVector<std::vector<double>>);
    static_assert(NumericVector<aligned_vector<float>>);
    static_assert(NumericVector<std::span<double>>);
    static_assert(NumericVector<std::span<const float>>);
    static_assert(NumericVector<std::array<int, 4>>);
    static_assert(NumericVector<StridedSpan<double>>);
    static_assert(NumericVector<StridedSpan<const double>>);

    static_assert(std::is_same_v<real_of_t<std::complex<double>>, double>);
    static_assert(std::is_same_v<real_of_t<std::complex<float>>, float>);
    static_assert(std::is_same_v<real_of_t<double>, double>);

    // Constexpr evaluation checks
    constexpr double ce_res = test_constexpr_ops();
    // 381.0 - (linf=3.0 + l1=6.5) = 371.5
    static_assert(ce_res == 371.5);

    constexpr int ce_int = test_constexpr_integral();
    // sum(3*a - b) = 3*15 - 10 = 35; dot(a, b) = 30 -> 65
    static_assert(ce_int == 65);
}

// ============================================================================
// 2. Empty (Size-0) and Tail Sizes (1, 3, 7, 15, 31, 37, 64, 1023)
// ============================================================================

template <typename T>
void test_sizes_for_type() {
    using namespace keisan;
    const std::vector<std::size_t> sizes = {0, 1, 3, 7, 15, 31, 37, 64, 511, 512, 513, 1023};

    for (std::size_t n : sizes) {
        std::vector<T> a(n), b(n), out(n, T{0});
        aligned_vector<T> aa(n), ab(n), aout(n, T{0});

        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<T>(i % 13) + T{1};
            b[i] = static_cast<T>((i * 3) % 11) + T{2};
            aa[i] = a[i];
            ab[i] = b[i];
        }

        // Expression evaluation & compound ops (ZeroMallocGuard)
        {
            ZeroMallocGuard guard("sizes_expr_eval");
            out <<= T{2} * a + b;
            aout <<= T{2} * aa + ab;
            assign(out, T{2} * a + b);
            assign_stream(aout, T{2} * aa + ab);
            assign_par(out, T{2} * a + b);
        }

        for (std::size_t i = 0; i < n; ++i) {
            const T expected = T{2} * a[i] + b[i];
            TEST_ASSERT(approx_eq(out[i], expected, 1e-5));
            TEST_ASSERT(approx_eq(aout[i], expected, 1e-5));
        }

        // Reference reductions
        double ref_sum = 0.0;
        double ref_dot = 0.0;
        double ref_l1 = 0.0;
        double ref_linf = 0.0;
        T ref_min = n > 0 ? a[0] : std::numeric_limits<T>::max();
        T ref_max = n > 0 ? a[0] : std::numeric_limits<T>::lowest();
        for (std::size_t i = 0; i < n; ++i) {
            ref_sum += static_cast<double>(a[i]);
            ref_dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            ref_l1 += std::abs(static_cast<double>(a[i]));
            ref_linf = std::max(ref_linf, std::abs(static_cast<double>(a[i])));
            ref_min = std::min(ref_min, a[i]);
            ref_max = std::max(ref_max, a[i]);
        }

        {
            ZeroMallocGuard guard("sizes_reductions");
            TEST_ASSERT(approx_eq(sum(a), ref_sum, 1e-3));
            TEST_ASSERT(approx_eq(sum(a, ReductionPolicy::REPRO), ref_sum, 1e-3));
            TEST_ASSERT(approx_eq(sum_deterministic(a), ref_sum, 1e-3));
            TEST_ASSERT(approx_eq(sum_par(a), ref_sum, 1e-3));

            TEST_ASSERT(approx_eq(dot(a, b), ref_dot, 1e-2));
            TEST_ASSERT(approx_eq(dot(a, b, ReductionPolicy::REPRO), ref_dot, 1e-2));
            TEST_ASSERT(approx_eq(dot_deterministic(a, b), ref_dot, 1e-2));
            TEST_ASSERT(approx_eq(dot_par(a, b), ref_dot, 1e-2));

            TEST_ASSERT(approx_eq(norm_sq(a), static_cast<double>(dot(a, a)), 1e-2));
            TEST_ASSERT(approx_eq(norm(a), std::sqrt(static_cast<double>(dot(a, a))), 1e-3));
            TEST_ASSERT(approx_eq(norm_l1(a), ref_l1, 1e-3));
            TEST_ASSERT(approx_eq(norm_inf(a), ref_linf, 1e-5));

            if (n > 0) {
                TEST_ASSERT(approx_eq(min_val(a), ref_min, 1e-5));
                TEST_ASSERT(approx_eq(max_val(a), ref_max, 1e-5));
            }
        }
#if !defined(KEISAN_NO_CHECKS) && !defined(KEISAN_NO_EXCEPTIONS)
        if (n == 0) {
            bool threw_min = false;
            try { (void)min_val(a); } catch (const std::invalid_argument&) { threw_min = true; }
            TEST_ASSERT(threw_min);

            bool threw_max = false;
            try { (void)max_val(a); } catch (const std::invalid_argument&) { threw_max = true; }
            TEST_ASSERT(threw_max);
        }
#endif
    }
}

void test_integral_types() {
    using namespace keisan;
    const std::vector<std::size_t> sizes = {0, 1, 7, 15, 37, 64};
    for (std::size_t n : sizes) {
        std::vector<int> a(n), b(n), out(n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<int>(i) - 10;
            b[i] = static_cast<int>(i % 5) + 1;
        }
        {
            ZeroMallocGuard guard("integral_ops");
            out <<= 3 * a + b;
            out += 2;
            out -= b;
        }
        for (std::size_t i = 0; i < n; ++i) {
            TEST_ASSERT(out[i] == 3 * a[i] + 2);
        }
        int expected_sum = 0;
        int expected_dot = 0;
        for (std::size_t i = 0; i < n; ++i) {
            expected_sum += a[i];
            expected_dot += a[i] * b[i];
        }
        TEST_ASSERT(sum(a) == expected_sum);
        TEST_ASSERT(dot(a, b) == expected_dot);
    }
}

// ============================================================================
// 3. StridedSpan, std::array, std::span Interop & Aliasing
// ============================================================================

void test_strided_span_and_views() {
    using namespace keisan;

    // Empty StridedSpan
    {
        StridedSpan<double> empty_ss(nullptr, 0, 3);
        TEST_ASSERT(empty_ss.empty());
        TEST_ASSERT(empty_ss.size() == 0);
        TEST_ASSERT(sum(empty_ss) == 0.0);
        TEST_ASSERT(dot(empty_ss, empty_ss) == 0.0);
    }

    // Interleaved AoS data: [x, y, z]
    const std::size_t n = 37;
    std::vector<double> aos(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        aos[i * 3 + 0] = static_cast<double>(i) + 1.0;
        aos[i * 3 + 1] = static_cast<double>(i) * 2.0 - 5.0;
        aos[i * 3 + 2] = 0.5;
    }

    StridedSpan<double> x_view(aos.data(), n, 3);
    StridedSpan<double> y_view(aos.data() + 1, n, 3);
    StridedSpan<const double> x_cview(aos.data(), n, 3);
    TEST_ASSERT(approx_eq(x_cview[0], 1.0));
    std::vector<double> dense(n, 3.0);
    std::vector<double> out(n, 0.0);

    {
        ZeroMallocGuard guard("strided_span_expressions");
        out <<= 2.0 * x_view + y_view - dense;
        const double s = sum(x_view);
        const double sr = sum(x_view, ReductionPolicy::REPRO);
        const double d = dot(x_view, y_view);
        const double dr = dot(x_view, y_view, ReductionPolicy::REPRO);
        const double linf = norm_inf(y_view);
        TEST_ASSERT(approx_eq(s, sr, 1e-12));
        TEST_ASSERT(approx_eq(d, dr, 1e-10));
        TEST_ASSERT(linf > 0.0);
    }

    for (std::size_t i = 0; i < n; ++i) {
        const double expected = 2.0 * aos[i * 3] + aos[i * 3 + 1] - 3.0;
        TEST_ASSERT(approx_eq(out[i], expected));
    }

    // std::array & std::span compound ops + self-aliasing
    {
        std::array<double, 4> arr{1.0, 2.0, 3.0, 4.0};
        std::array<double, 4> rhs{0.5, 1.5, 2.5, 3.5};
        std::vector<double> sp_buf{10.0, 20.0, 30.0, 40.0};
        std::span<double> sp(sp_buf);

        ZeroMallocGuard guard("array_span_compound");
        arr <<= 2.0 * arr + rhs;
        arr += arr;
        arr -= 1.0;
        arr *= 0.5;
        arr /= 1.0;

        sp <<= arr + 1.0;
        sp += arr;
        sp -= 2.0;
        sp *= 2.0;
        sp /= 4.0;

        for (std::size_t i = 0; i < 4; ++i) {
            const double orig = static_cast<double>(i + 1);
            const double expected_arr = (2.0 * (2.0 * orig + rhs[i]) - 1.0) * 0.5;
            TEST_ASSERT(approx_eq(arr[i], expected_arr));
            TEST_ASSERT(approx_eq(sp[i], (expected_arr * 2.0 - 1.0) * 0.5));
        }
    }
}

// ============================================================================
// 4. DG Kernels: Edge Cases, Tail Sizes, In-Place, & Error Handling
// ============================================================================

void test_dg_kernels_edge_cases() {
    using namespace keisan;

    // matvec_fixed & matvec_transposed_fixed: aliasing
    {
        constexpr std::size_t M = 4;
        double A[M * M] = {
            1.0, 2.0, 0.0, -1.0,
            0.5, 1.5, 2.0,  0.0,
            -1.0, 0.0, 3.0, 1.0,
            2.0, -0.5, 1.0, 4.0
        };
        double buf[M + 2] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
        double ref[M];
        for (std::size_t c = 0; c < M; ++c) {
            double s = 0.0;
            for (std::size_t r = 0; r < M; ++r) s += A[r * M + c] * buf[r];
            ref[c] = s;
        }
        // dst == src + 1 for matvec_transposed_fixed
        matvec_transposed_fixed<M, M>(A, buf, buf + 1);
        for (std::size_t c = 0; c < M; ++c) {
            TEST_ASSERT(approx_eq(buf[c + 1], ref[c]));
        }
    }

    // solve_4x4_batched: size 0, tail sizes (1, 3, 7, 15, 31, 37, 64), float & double, in-place (x == b)
    for (std::size_t E : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{7},
                          std::size_t{15}, std::size_t{31}, std::size_t{37}, std::size_t{64}}) {
        std::vector<double> A16(16 * E), b(4 * E);
        std::vector<const double*> Ap(16), bp(4);
        std::vector<double*> xp_inplace(4);
        for (std::size_t i = 0; i < 16; ++i) Ap[i] = A16.data() + i * E;
        for (std::size_t i = 0; i < 4; ++i) {
            bp[i] = b.data() + i * E;
            xp_inplace[i] = b.data() + i * E; // in-place x == b
        }

        for (std::size_t e = 0; e < E; ++e) {
            // Diagonally dominant 4x4 system with known solution x = [1, 2, 3, 4]
            const double x_true[4] = {1.0, 2.0, 3.0, 4.0};
            for (int r = 0; r < 4; ++r) {
                double rhs = 0.0;
                for (int c = 0; c < 4; ++c) {
                    const double val = (r == c ? 10.0 : 0.25) + 0.01 * static_cast<double>(e % 7 + r + c);
                    A16[(r * 4 + c) * E + e] = val;
                    rhs += val * x_true[c];
                }
                b[r * E + e] = rhs;
            }
        }

        solve_4x4_batched(Ap.data(), bp.data(), xp_inplace.data(), E);
        for (std::size_t e = 0; e < E; ++e) {
            for (int r = 0; r < 4; ++r) {
                TEST_ASSERT(approx_eq(b[r * E + e], static_cast<double>(r + 1), 1e-11));
            }
        }
    }

    // solve_nxn_batched<N>: N in {2, 3, 4, 5, 6}, double & float, tail sizes, pivoting, in-place, singular
    auto run_solve_nxn = []<std::size_t N, typename T>() {
        const T tol = std::is_same_v<T, float> ? T{1e-4} : T{1e-10};
        for (std::size_t E : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{7},
                              std::size_t{15}, std::size_t{31}, std::size_t{37}, std::size_t{64}}) {
            std::vector<T> A(N * N * E), b(N * E), x(N * E, T{0});
            std::vector<const T*> Ap(N * N), bp(N);
            std::vector<T*> xp(N), xp_inplace(N);
            for (std::size_t i = 0; i < N * N; ++i) Ap[i] = A.data() + i * E;
            for (std::size_t i = 0; i < N; ++i) {
                bp[i] = b.data() + i * E;
                xp[i] = x.data() + i * E;
                xp_inplace[i] = b.data() + i * E;
            }

            for (std::size_t e = 0; e < E; ++e) {
                for (std::size_t r = 0; r < N; ++r) {
                    T rhs{0};
                    for (std::size_t c = 0; c < N; ++c) {
                        // Shift pivot on odd elements to exercise branchless partial pivoting
                        const std::size_t rr = (e % 2 == 1 && N >= 2) ? ((r + 1) % N) : r;
                        const T val = (rr == c ? T{10} : T{0.25}) +
                                      T{0.01} * static_cast<T>(e % 7 + r + c);
                        A[(r * N + c) * E + e] = val;
                        rhs += val * static_cast<T>(c + 1);
                    }
                    b[r * E + e] = rhs;
                }
            }

            solve_nxn_batched<N>(Ap.data(), bp.data(), xp.data(), E);
            for (std::size_t e = 0; e < E; ++e) {
                for (std::size_t r = 0; r < N; ++r) {
                    TEST_ASSERT(approx_eq(x[r * E + e], static_cast<T>(r + 1), tol));
                }
            }

            // In-place (x == b)
            solve_nxn_batched<N>(Ap.data(), bp.data(), xp_inplace.data(), E);
            for (std::size_t e = 0; e < E; ++e) {
                for (std::size_t r = 0; r < N; ++r) {
                    TEST_ASSERT(approx_eq(b[r * E + e], static_cast<T>(r + 1), tol));
                }
            }
        }

#if !defined(KEISAN_NO_EXCEPTIONS)
        // Singular matrix check (both SIMD body at e=0 and tail at e=E-1)
        for (std::size_t E : {std::size_t{1}, std::size_t{9}}) {
            for (std::size_t bad_e : {std::size_t{0}, E - 1}) {
                std::vector<T> A(N * N * E, T{0}), b(N * E, T{1}), x(N * E, T{0});
                std::vector<const T*> Ap(N * N), bp(N);
                std::vector<T*> xp(N);
                for (std::size_t i = 0; i < N * N; ++i) Ap[i] = A.data() + i * E;
                for (std::size_t i = 0; i < N; ++i) {
                    bp[i] = b.data() + i * E;
                    xp[i] = x.data() + i * E;
                }
                for (std::size_t e = 0; e < E; ++e) {
                    for (std::size_t r = 0; r < N; ++r) {
                        A[(r * N + r) * E + e] = (e == bad_e && r == N - 1) ? T{0} : T{2};
                    }
                }
                bool threw = false;
                try {
                    solve_nxn_batched<N>(Ap.data(), bp.data(), xp.data(), E);
                } catch (const std::runtime_error&) {
                    threw = true;
                }
                TEST_ASSERT(threw);
            }
        }
#endif
    };

    run_solve_nxn.template operator()<2, double>();
    run_solve_nxn.template operator()<3, double>();
    run_solve_nxn.template operator()<4, double>();
    run_solve_nxn.template operator()<5, double>();
    run_solve_nxn.template operator()<6, double>();
    run_solve_nxn.template operator()<2, float>();
    run_solve_nxn.template operator()<3, float>();
    run_solve_nxn.template operator()<4, float>();
    run_solve_nxn.template operator()<5, float>();
    run_solve_nxn.template operator()<6, float>();

    // Scale-invariant singularity check for float:
    // 1) Small-scale well-conditioned (1e-8 * I) must succeed.
    // 2) O(1) numerically singular (pivot 1e-9 after elimination) must throw.
    {
        constexpr std::size_t E = 9; // exercises both SIMD body and scalar tail
        std::vector<float> A(4 * E, 0.0f), b(2 * E, 0.0f), x(2 * E, 0.0f);
        std::vector<const float*> Ap = {A.data(), A.data() + E, A.data() + 2 * E, A.data() + 3 * E};
        std::vector<const float*> bp = {b.data(), b.data() + E};
        std::vector<float*> xp = {x.data(), x.data() + E};
        for (std::size_t e = 0; e < E; ++e) {
            A[0 * E + e] = 1e-8f;
            A[3 * E + e] = 1e-8f;
            b[0 * E + e] = 2e-8f;
            b[1 * E + e] = 3e-8f;
        }
        solve_nxn_batched<2>(Ap.data(), bp.data(), xp.data(), E);
        for (std::size_t e = 0; e < E; ++e) {
            TEST_ASSERT(approx_eq(x[0 * E + e], 2.0f, 1e-5f));
            TEST_ASSERT(approx_eq(x[1 * E + e], 3.0f, 1e-5f));
        }

#if !defined(KEISAN_NO_EXCEPTIONS)
        for (std::size_t bad_e : {std::size_t{0}, E - 1}) {
            std::fill(A.begin(), A.end(), 0.0f);
            std::fill(b.begin(), b.end(), 1.0f);
            for (std::size_t e = 0; e < E; ++e) {
                A[0 * E + e] = 1.0f;
                A[1 * E + e] = 1.0f;
                A[2 * E + e] = 1.0f;
                A[3 * E + e] = (e == bad_e) ? (1.0f + 2e-7f) : 2.0f;
            }
            bool threw = false;
            try {
                solve_nxn_batched<2>(Ap.data(), bp.data(), xp.data(), E);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            TEST_ASSERT(threw);
        }
#endif
    }

    // Latent bug regression checks: matvec_fixed<0,0> and sum(StridedSpan<const T>)
    {
        double dummy = 42.0;
        matvec_fixed<0, 0>(&dummy, &dummy, &dummy);
        matvec_transposed_fixed<0, 0>(&dummy, &dummy, &dummy);
        TEST_ASSERT(dummy == 42.0);

        const double arr[4] = {1.0, 2.0, 3.0, 4.0};
        StridedSpan<const double> cspan(arr, 4, 1);
        TEST_ASSERT(approx_eq(sum(cspan), 10.0));
    }

    // transpose_aos_to_soa / transpose_soa_to_aos with tail sizes
    for (std::size_t nelem : {std::size_t{0}, std::size_t{1}, std::size_t{7}, std::size_t{37}}) {
        for (std::size_t nvars : {std::size_t{1}, std::size_t{3}, std::size_t{4}, std::size_t{7}}) {
            std::vector<double> aos(nelem * nvars), soa(nelem * nvars), back(nelem * nvars);
            for (std::size_t i = 0; i < aos.size(); ++i) aos[i] = static_cast<double>(i) * 0.5 + 1.0;
            transpose_aos_to_soa(aos.data(), soa.data(), nelem, nvars);
            transpose_soa_to_aos(soa.data(), back.data(), nelem, nvars);
            TEST_ASSERT(aos == back);
        }
    }

    // gather / scatter_add / broadcast_multiply size-0 & tail sizes
    {
        double* null_d = nullptr;
        int* null_i = nullptr;
        gather(null_d, null_d, null_i, 0);
        scatter_add(null_d, null_i, null_d, 0);
        broadcast_multiply(null_d, 0, 2.0);
    }
}

// ============================================================================
// 5. NaN / Inf Propagation & Error Handling
// ============================================================================

void test_nan_inf_and_errors() {
    using namespace keisan;

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    std::vector<double> v_nan{1.0, qnan, 3.0, 4.0};
    std::vector<double> v_ones{1.0, 1.0, 1.0, 1.0};
    TEST_ASSERT(std::isnan(sum(v_nan)));
    TEST_ASSERT(std::isnan(sum_repro(v_nan)));
    TEST_ASSERT(std::isnan(dot(v_nan, v_ones)));
    TEST_ASSERT(std::isnan(dot_repro(v_nan, v_ones)));
    TEST_ASSERT(std::isnan(norm_inf(v_nan)));
    TEST_ASSERT(std::isnan(norm_l1(v_nan)));

    std::vector<double> v_inf{1.0, inf, 3.0, -2.0};
    TEST_ASSERT(std::isinf(sum(v_inf)));
    TEST_ASSERT(std::isinf(dot(v_inf, v_ones)));
    TEST_ASSERT(std::isinf(norm_inf(v_inf)));
    TEST_ASSERT(std::isinf(norm_l1(v_inf)));

#if !defined(KEISAN_NO_CHECKS) && !defined(KEISAN_NO_EXCEPTIONS)
    // Size mismatch exceptions
    std::vector<double> a(4, 1.0), b(5, 1.0);
    auto expect_throw = [](auto&& fn) {
        bool threw = false;
        try {
            fn();
        } catch (const std::exception&) {
            threw = true;
        }
        TEST_ASSERT(threw);
    };
    expect_throw([&] { a <<= b; });
    expect_throw([&] { a += b; });
    expect_throw([&] { (void)dot(a, b); });
    expect_throw([&] { (void)dot_repro(a, b); });
    expect_throw([&] { (void)dot_par(a, b); });
    expect_throw([&] { assign_stream(a, b); });
    expect_throw([&] { assign_par(a, b); });
#endif
}

// ============================================================================
// 6. OpenMP Determinism & Reproducibility Across Thread Counts
// ============================================================================

void test_openmp_determinism() {
    using namespace keisan;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::uniform_real_distribution<float> fdist(-100.0f, 100.0f);

    const std::size_t n = 100003;
    std::vector<double> ad(n), bd(n);
    std::vector<float> af(n), bf(n);
    for (std::size_t i = 0; i < n; ++i) {
        ad[i] = dist(rng);
        bd[i] = dist(rng);
        af[i] = fdist(rng);
        bf[i] = fdist(rng);
    }

    const double ref_sd = sum_repro(ad);
    const double ref_dd = dot_repro(ad, bd);
    const float ref_sf = sum_repro(af);
    const float ref_df = dot_repro(af, bf);

#if defined(_OPENMP)
    const int prev_threads = omp_get_max_threads();
    for (int t : {1, 2, 4}) {
        omp_set_num_threads(t);
        TEST_ASSERT(bitwise_eq(sum(ad, ReductionPolicy::REPRO), ref_sd));
        TEST_ASSERT(bitwise_eq(dot(ad, bd, ReductionPolicy::REPRO), ref_dd));
        TEST_ASSERT(bitwise_eq(sum(af, ReductionPolicy::REPRO), ref_sf));
        TEST_ASSERT(bitwise_eq(dot(af, bf, ReductionPolicy::REPRO), ref_df));

        // Parallel fast reductions match serial within floating-point tolerance
        TEST_ASSERT(approx_eq(sum_par(ad), ref_sd, 1e-8));
        TEST_ASSERT(approx_eq(dot_par(ad, bd), ref_dd, 1e-6));
    }
    omp_set_num_threads(prev_threads);
#else
    TEST_ASSERT(bitwise_eq(sum(ad, ReductionPolicy::REPRO), ref_sd));
    TEST_ASSERT(bitwise_eq(dot(ad, bd, ReductionPolicy::REPRO), ref_dd));
    TEST_ASSERT(bitwise_eq(sum(af, ReductionPolicy::REPRO), ref_sf));
    TEST_ASSERT(bitwise_eq(dot(af, bf, ReductionPolicy::REPRO), ref_df));
#endif
}

} // namespace

int main() {
#if defined(_OPENMP)
    // Warm up OpenMP runtime thread pool before ZeroMallocGuard checks.
    {
        using namespace keisan;
        std::vector<double> w_in(64, 1.0), w_out(64, 0.0);
        assign_par(w_out, 2.0 * w_in);
    }
#endif
    test_concepts_and_constexpr();
    test_complex_runtime();
    test_sizes_for_type<double>();
    test_sizes_for_type<float>();
    test_integral_types();
    test_strided_span_and_views();
    test_dg_kernels_edge_cases();
    test_nan_inf_and_errors();
    test_openmp_determinism();

    std::cout << "ALL EDGE CASE, CONSTEXPR, AND DETERMINISM TESTS PASSED!\n";
    return 0;
}
