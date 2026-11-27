// benchmarks/regression_bench.cpp — Standardized anti-hoisted benchmark & regression suite
// Usage: regression_bench [--quick]

#include <keisan/keisan.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(_MSC_VER)
#define BENCH_NOINLINE __declspec(noinline)
#else
#define BENCH_NOINLINE __attribute__((noinline))
#endif

using namespace keisan;

namespace {

bool g_quick = false;
int g_impossible = 0;
double g_peak_gflops = 0.0;
double g_peak_copy_gbps = 0.0;

inline void escape_memory(void* p) {
#if defined(_MSC_VER) && !defined(__clang__)
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)p;
#else
    asm volatile("" : : "r"(p) : "memory");
#endif
}

// Mutates an element by 1.0 (> 0.5 ULP for |x| < 2^52) so the compiler cannot
// hoist loop bodies across repetitions.
bool mutation_is_effective(double* p) {
    const double before = *p;
    *p += 1.0;
    const bool changed = (*p != before);
    *p = before;
    return changed;
}

struct Timing {
    double min_ms;
    double max_ms;
};

template <typename F>
Timing time_kernel(F&& f, int reps, int trials = 5) {
    for (int i = 0; i < 2; ++i) {
        f();
    }
    Timing t{1e300, 0.0};
    for (int k = 0; k < trials; ++k) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            f();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        t.min_ms = std::min(t.min_ms, ms);
        t.max_ms = std::max(t.max_ms, ms);
    }
    return t;
}

void report(const char* key, double value, const Timing& t, const char* note, const char* unit,
            double ceiling = 0.0) {
    const double spread = (t.min_ms > 0.0) ? (t.max_ms / t.min_ms) : 0.0;
    const bool impossible = (ceiling > 0.0 && value > ceiling * 10.0);
    if (impossible) {
        ++g_impossible;
    }
    std::printf("%s=%.3f  (%s, %s)%s\n", key, value, note, unit,
                impossible ? "  IMPOSSIBLE" : "");
    std::printf("%s_SPREAD=%.2f%s\n", key, spread, spread > 1.5 ? "  UNSTABLE" : "");
}

// -----------------------------------------------------------------------------
// Calibrations
// -----------------------------------------------------------------------------

BENCH_NOINLINE void k_peak_flops(int iters) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256d a0 = _mm256_set1_pd(1.0), a1 = _mm256_set1_pd(1.1);
    __m256d a2 = _mm256_set1_pd(1.2), a3 = _mm256_set1_pd(1.3);
    __m256d a4 = _mm256_set1_pd(1.4), a5 = _mm256_set1_pd(1.5);
    __m256d a6 = _mm256_set1_pd(1.6), a7 = _mm256_set1_pd(1.7);
    const __m256d b = _mm256_set1_pd(1.0000001), c = _mm256_set1_pd(1e-12);
    for (int i = 0; i < iters; ++i) {
        a0 = _mm256_fmadd_pd(a0, b, c);
        a1 = _mm256_fmadd_pd(a1, b, c);
        a2 = _mm256_fmadd_pd(a2, b, c);
        a3 = _mm256_fmadd_pd(a3, b, c);
        a4 = _mm256_fmadd_pd(a4, b, c);
        a5 = _mm256_fmadd_pd(a5, b, c);
        a6 = _mm256_fmadd_pd(a6, b, c);
        a7 = _mm256_fmadd_pd(a7, b, c);
    }
    const __m256d s = _mm256_add_pd(
        _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)),
        _mm256_add_pd(_mm256_add_pd(a4, a5), _mm256_add_pd(a6, a7)));
    double out[4];
    _mm256_storeu_pd(out, s);
    escape_memory(out);
#else
    double a0 = 1.0, a1 = 1.1, a2 = 1.2, a3 = 1.3, a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
    const double b = 1.0000001, c = 1e-12;
    for (int i = 0; i < iters; ++i) {
        a0 = a0 * b + c;
        a1 = a1 * b + c;
        a2 = a2 * b + c;
        a3 = a3 * b + c;
        a4 = a4 * b + c;
        a5 = a5 * b + c;
        a6 = a6 * b + c;
        a7 = a7 * b + c;
    }
    double out[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    escape_memory(out);
#endif
}

BENCH_NOINLINE void k_copy(double* a, double* b, std::size_t n) {
    a[0] += 1.0;
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = a[i];
    }
    escape_memory(b);
}

// -----------------------------------------------------------------------------
// Vector / Streaming Kernels
// -----------------------------------------------------------------------------

BENCH_NOINLINE void k_triad(aligned_vector<double>& a, const aligned_vector<double>& b,
                            aligned_vector<double>& c) {
    a[0] += 1.0;
    c <<= a + 2.0 * b;
    escape_memory(c.data());
}

BENCH_NOINLINE void k_triad_stream(aligned_vector<double>& a, const aligned_vector<double>& b,
                                   aligned_vector<double>& c) {
    a[0] += 1.0;
    assign_stream(c, a + 2.0 * b);
    escape_memory(c.data());
}

BENCH_NOINLINE void k_axpy(const aligned_vector<double>& a, aligned_vector<double>& b) {
    b[0] += 1.0;
    b += 3.0 * a;
    escape_memory(b.data());
}

// -----------------------------------------------------------------------------
// Reduction Kernels
// -----------------------------------------------------------------------------

BENCH_NOINLINE double k_dot_serial(double* a, const double* b, std::size_t n) {
    a[0] += 1.0;
    return dot(std::span<const double>(a, n), std::span<const double>(b, n));
}

BENCH_NOINLINE double k_dot_par(double* a, const double* b, std::size_t n) {
    a[0] += 1.0;
    return dot_par(std::span<const double>(a, n), std::span<const double>(b, n));
}

BENCH_NOINLINE double k_dot_repro(double* a, const double* b, std::size_t n) {
    a[0] += 1.0;
    return dot_repro(std::span<const double>(a, n), std::span<const double>(b, n));
}

BENCH_NOINLINE double k_dot_det(double* a, const double* b, std::size_t n) {
    a[0] += 1.0;
    return dot_deterministic(std::span<const double>(a, n), std::span<const double>(b, n));
}

// -----------------------------------------------------------------------------
// DG Kernels
// -----------------------------------------------------------------------------

void ref_matvec(const double* A, const double* x, double* y, std::size_t R, std::size_t C) {
    for (std::size_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < C; ++c) {
            s += A[r * C + c] * x[c];
        }
        y[r] = s;
    }
}

BENCH_NOINLINE void k_matvec_rt(double* A, const double* x, double* y, std::size_t Np,
                                std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        A[0] += 1.0;
        matvec(A, x, y, Np, Np);
    }
    escape_memory(y);
}

template <std::size_t M>
BENCH_NOINLINE void k_matvec_fixed(double* A, const double* x, double* y, std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        A[0] += 1.0;
        matvec_fixed<M, M>(A, x, y);
    }
    escape_memory(y);
}

template <std::size_t N>
BENCH_NOINLINE void k_contract_sf(const double* D, double* u, double* o, std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        u[0] += 1.0;
        contract_axis<Axis::X, N>(D, u, o, N, N, 1);
        contract_axis<Axis::Y, N>(D, o, u, N, N, 1);
    }
    escape_memory(u);
}

BENCH_NOINLINE void k_solve4(const double* const* Ap, const double* const* bp, double* const* xp,
                             std::size_t E) {
    solve_4x4_batched<double>(Ap, bp, xp, E);
    escape_memory(xp[0]);
}

template <std::size_t N>
BENCH_NOINLINE void k_solve_nxn(const double* const* Ap, const double* const* bp, double* const* xp,
                                std::size_t E) {
    solve_nxn_batched<N, double>(Ap, bp, xp, E);
    escape_memory(xp[0]);
}

// -----------------------------------------------------------------------------
// Sections
// -----------------------------------------------------------------------------

void section_peaks(std::size_t n) {
    const int iters = g_quick ? 2000000 : 20000000;
    const Timing tf = time_kernel([&] { k_peak_flops(iters); }, 1, 3);
#if defined(__AVX2__) && defined(__FMA__)
    const double flops = 2.0 * 4.0 * 8.0 * static_cast<double>(iters);
    g_peak_gflops = flops / (tf.min_ms * 1e6);
    std::printf("PEAK_GFLOPS=%.2f  (AVX2 FMA, 8x4 lanes)\n", g_peak_gflops);
#else
    const double flops = 2.0 * 8.0 * static_cast<double>(iters);
    g_peak_gflops = flops / (tf.min_ms * 1e6);
    std::printf("PEAK_GFLOPS=%.2f  (scalar FMA)\n", g_peak_gflops);
#endif

    std::vector<double> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = std::sin(0.001 * static_cast<double>(i));
    }
    const Timing tc = time_kernel([&] { k_copy(a.data(), b.data(), n); }, 1, 5);
    g_peak_copy_gbps =
        (2.0 * static_cast<double>(n) * sizeof(double)) / 1e9 / (tc.min_ms * 1e-3);
    std::printf("PEAK_COPY_GBPS=%.2f  (STREAM copy, spread %.2f)\n", g_peak_copy_gbps,
                tc.max_ms / tc.min_ms);
}

void section_stream(std::size_t n) {
    aligned_vector<double> a(n), b(n), c(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = std::sin(0.001 * static_cast<double>(i));
        b[i] = std::cos(0.002 * static_cast<double>(i));
    }
    const double bytes3 = 3.0 * static_cast<double>(n) * sizeof(double);

    Timing t = time_kernel([&] { k_triad(a, b, c); }, 1);
    report("STREAM_TRIAD_GBPS", bytes3 / 1e9 / (t.min_ms * 1e-3), t, "2 read + 1 write", "GB/s",
           g_peak_copy_gbps);

    t = time_kernel([&] { k_triad_stream(a, b, c); }, 1);
    report("STREAM_NON_TEMPORAL_GBPS", bytes3 / 1e9 / (t.min_ms * 1e-3), t,
           "assign_stream", "GB/s", g_peak_copy_gbps);

    t = time_kernel([&] { k_axpy(a, b); }, 1);
    report("STREAM_AXPY_GBPS", bytes3 / 1e9 / (t.min_ms * 1e-3), t, "2 read + 1 write", "GB/s",
           g_peak_copy_gbps);
}

void section_dot(std::size_t n) {
    aligned_vector<double> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = std::sin(0.001 * static_cast<double>(i));
        b[i] = std::cos(0.002 * static_cast<double>(i));
    }
    const double bytes = 2.0 * static_cast<double>(n) * sizeof(double);
    volatile double sink = 0.0;

    Timing t = time_kernel([&] { sink = sink + k_dot_serial(a.data(), b.data(), n); }, 1);
    report("DOT_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), t, "1 thread", "GB/s", g_peak_copy_gbps);

#if defined(_OPENMP)
    t = time_kernel([&] { sink = sink + k_dot_par(a.data(), b.data(), n); }, 3);
    report("DOT_PAR_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), t, "omp", "GB/s", g_peak_copy_gbps);
#else
    std::printf("DOT_PAR_GBPS=SKIPPED  (not built with OpenMP)\n");
#endif

    t = time_kernel([&] { sink = sink + k_dot_repro(a.data(), b.data(), n); }, 1);
    report("DOT_REPRO_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), t, "bit-reproducible", "GB/s",
           g_peak_copy_gbps);

    t = time_kernel([&] { sink = sink + k_dot_det(a.data(), b.data(), n); }, 1);
    report("DOT_DET_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), t, "deterministic", "GB/s",
           g_peak_copy_gbps);
    (void)sink;
}

template <std::size_t Np>
void section_matvec_one() {
    const std::size_t E = std::max<std::size_t>(20000, 4000000 / (Np * Np));
    std::vector<double> A(Np * Np), x(Np), y(Np), ref(Np);
    for (std::size_t i = 0; i < Np * Np; ++i) {
        A[i] = std::sin(0.001 * static_cast<double>(i) + 0.3);
    }
    for (std::size_t i = 0; i < Np; ++i) {
        x[i] = 1.0 + 0.01 * static_cast<double>(i);
    }

    // Correctness cross-check for both runtime and fixed matvec before timing mutates A
    ref_matvec(A.data(), x.data(), ref.data(), Np, Np);
    matvec(A.data(), x.data(), y.data(), Np, Np);
    double md = 0.0;
    for (std::size_t i = 0; i < Np; ++i) {
        md = std::max(md, std::abs(y[i] - ref[i]));
    }
    if (md > 1e-10) {
        std::printf("  !! matvec Np=%zu maxdiff %.3e — NOT timed\n", Np, md);
        ++g_impossible;
        return;
    }

    matvec_fixed<Np, Np>(A.data(), x.data(), y.data());
    md = 0.0;
    for (std::size_t i = 0; i < Np; ++i) {
        md = std::max(md, std::abs(y[i] - ref[i]));
    }
    if (md > 1e-10) {
        std::printf("  !! matvec_fixed Np=%zu maxdiff %.3e — NOT timed\n", Np, md);
        ++g_impossible;
        return;
    }

    const double flops = 2.0 * static_cast<double>(Np) * static_cast<double>(Np) * static_cast<double>(E);
    const int reps = g_quick ? 1 : 3;
    char key[64];

    Timing t = time_kernel([&] { k_matvec_rt(A.data(), x.data(), y.data(), Np, E); }, reps);
    std::snprintf(key, sizeof(key), "MATVEC_NP%zu_GFLOPS", Np);
    report(key, flops / (t.min_ms * 1e6), t, "runtime cols", "GFLOPS", g_peak_gflops);

    t = time_kernel([&] { k_matvec_fixed<Np>(A.data(), x.data(), y.data(), E); }, reps);
    std::snprintf(key, sizeof(key), "MATVEC_FIXED_NP%zu_GFLOPS", Np);
    report(key, flops / (t.min_ms * 1e6), t, "compile-time cols", "GFLOPS", g_peak_gflops);
}

template <std::size_t N>
void section_contract_one(std::size_t E) {
    constexpr std::size_t Np = N * N;
    std::vector<double> D(N * N), u(Np), o(Np), M(Np * Np), od(Np);
    for (std::size_t i = 0; i < N * N; ++i) D[i] = std::sin(0.5 * static_cast<double>(i));
    for (std::size_t i = 0; i < Np; ++i) u[i] = std::cos(0.1 * static_cast<double>(i));
    for (std::size_t i = 0; i < Np * Np; ++i) M[i] = std::sin(0.0007 * static_cast<double>(i));

    const double sf_flops = 2.0 * 2.0 * static_cast<double>(N) * static_cast<double>(Np) * static_cast<double>(E);
    const double dense_flops = 2.0 * static_cast<double>(Np) * static_cast<double>(Np) * static_cast<double>(E);

    const Timing ts = time_kernel([&] { k_contract_sf<N>(D.data(), u.data(), o.data(), E); }, 1);
    const Timing td = time_kernel([&] { k_matvec_rt(M.data(), u.data(), od.data(), Np, E); }, 1);

    char key[64];
    std::snprintf(key, sizeof(key), "SF_N%zu_GFLOPS", N);
    report(key, sf_flops / (ts.min_ms * 1e6), ts, "sum-factorized", "GFLOPS", g_peak_gflops);
    std::snprintf(key, sizeof(key), "SF_N%zu_DENSE_GFLOPS", N);
    report(key, dense_flops / (td.min_ms * 1e6), td, "dense matvec", "GFLOPS", g_peak_gflops);
    std::printf("SF_N%zu_SPEEDUP=%.3f  (dense / sum-factorized)\n", N, td.min_ms / ts.min_ms);
}

void section_solve4() {
    const std::size_t E = g_quick ? 50000 : 200000;
    std::vector<double> A16(16 * E), b(4 * E), x(4 * E, 0.0);
    for (std::size_t i = 0; i < 16 * E; ++i) {
        A16[i] = std::sin(0.0007 * static_cast<double>(i));
    }
    for (std::size_t e = 0; e < E; ++e) {
        for (int r = 0; r < 4; ++r) {
            A16[(r * 4 + r) * E + e] += 9.0;
        }
    }
    for (std::size_t i = 0; i < 4 * E; ++i) {
        b[i] = 1.0;
    }

    std::vector<const double*> Ap(16), bp(4);
    std::vector<double*> xp(4);
    for (int i = 0; i < 16; ++i) Ap[i] = A16.data() + i * E;
    for (int i = 0; i < 4; ++i) {
        bp[i] = b.data() + i * E;
        xp[i] = x.data() + i * E;
    }

    const Timing t = time_kernel([&] { k_solve4(Ap.data(), bp.data(), xp.data(), E); }, 3);
    std::printf("SOLVE4_MELEM_S=%.3f  (%.2f GFLOPS, %.3g systems)\n",
                static_cast<double>(E) / (t.min_ms * 1e3),
                (30.0 * static_cast<double>(E)) / (t.min_ms * 1e6), static_cast<double>(E));
    std::printf("SOLVE4_SPREAD=%.2f\n", t.max_ms / t.min_ms);
}

template <std::size_t N>
void section_solve_nxn() {
    const std::size_t E = g_quick ? 50000 : 200000;
    std::vector<double> A(N * N * E), b(N * E), x(N * E, 0.0);
    for (std::size_t i = 0; i < N * N * E; ++i) {
        A[i] = std::sin(0.0007 * static_cast<double>(i));
    }
    for (std::size_t e = 0; e < E; ++e) {
        for (std::size_t r = 0; r < N; ++r) {
            A[(r * N + r) * E + e] += 12.0;
        }
    }
    for (std::size_t i = 0; i < N * E; ++i) {
        b[i] = 1.0;
    }

    std::vector<const double*> Ap(N * N), bp(N);
    std::vector<double*> xp(N);
    for (std::size_t i = 0; i < N * N; ++i) Ap[i] = A.data() + i * E;
    for (std::size_t i = 0; i < N; ++i) {
        bp[i] = b.data() + i * E;
        xp[i] = x.data() + i * E;
    }

    const Timing t = time_kernel([&] { k_solve_nxn<N>(Ap.data(), bp.data(), xp.data(), E); }, 3);
    std::printf("SOLVE%zu_MELEM_S=%.3f  (%.3g systems)\n",
                N, static_cast<double>(E) / (t.min_ms * 1e3), static_cast<double>(E));
    std::printf("SOLVE%zu_SPREAD=%.2f\n", N, t.max_ms / t.min_ms);
}

} // namespace

int main(int argc, char** argv) {
    g_quick = (argc > 1 && std::string(argv[1]) == "--quick");

    std::printf("# keisan regression benchmark%s\n", g_quick ? " (quick)" : "");
#if defined(_OPENMP)
    std::printf("THREADS=%d\n", omp_get_max_threads());
#else
    std::printf("THREADS=1\n");
#endif

    {
        double probe_vals[3] = {1.0001, 1.0, 12345.678};
        bool ok = true;
        for (double& v : probe_vals) {
            ok = ok && mutation_is_effective(&v);
        }
        std::printf("MUTATION=%s\n", ok ? "OK" : "NOOP");
        if (!ok) {
            return 2;
        }
    }

    const std::size_t n = g_quick ? (1u << 20) : (1u << 22);
    section_peaks(n);
    section_stream(n);
    section_dot(n);

    section_matvec_one<10>();
    section_matvec_one<20>();
    section_matvec_one<36>();
    section_matvec_one<64>();

    section_contract_one<4>(g_quick ? 50000u : 300000u);
    section_contract_one<6>(g_quick ? 20000u : 100000u);
    section_solve4();
    section_solve_nxn<3>();
    section_solve_nxn<6>();

    std::printf("IMPOSSIBLE_COUNT=%d\n", g_impossible);
    std::printf("# done\n");
    return g_impossible == 0 ? 0 : 1;
}
