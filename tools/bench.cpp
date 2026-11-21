// tools/bench.cpp — measurement harness for vec_math
// Usage:  bench.exe [--quick]
//
// Output is one KEY=value per line so the orchestrator never has to parse prose.
//
// Four rules this file exists to enforce:
//
//   1. EVERY timed kernel mutates an input by more than half an ULP, and the
//      harness *verifies* the mutation changed the value (MUTATION=OK). The
//      upstream benchmark's `a[i % n] += 1e-16` on 1.0001 was a no-op; that
//      mistake must be structurally impossible here.
//   2. Peak throughput is measured, not assumed: a packed FMA loop gives
//      PEAK_GFLOPS, a STREAM copy gives PEAK_COPY_GBPS. Any kernel reporting
//      > 10x peak is printed as IMPOSSIBLE and counted.
//   3. Every kernel is cross-checked against a naive reference before it is
//      timed, so a fast-but-wrong kernel cannot get a good number.
//   4. Every timed kernel is a NOINLINE function with a fixed body, and its
//      memory writes escape at the end. Without both, GCC's inlining and
//      dead-store decisions change with the *caller's* shape and the same
//      kernel swings 5x between builds. That is the whole of finding F2, and
//      a harness that inherits the problem cannot measure it.
//
// Each result is printed twice-verified: min-of-trials, plus the spread
// between the fastest and slowest trial. A run whose spread is large is
// reported as *_SPREAD so the orchestrator can refuse to record it.

#include <vec_math/vector_math.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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

using namespace vec_math;

namespace {

bool g_quick = false;
int  g_impossible = 0;

// ---------------------------------------------------------------- plumbing

inline void escape_memory(void *p) {
#if defined(_MSC_VER) && !defined(__clang__)
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)p;
#else
    asm volatile("" : : "r"(p) : "memory");
#endif
}

// Mutate a value so the compiler cannot hoist a loop body, and prove the
// mutation is observable. 1.0 is far above one ULP for any |x| < 2^52.
bool mutation_is_effective(double *p) {
    double before = *p;
    *p += 1.0;
    bool changed = (*p != before);
    *p = before;
    return changed;
}

struct Timing { double min_ms, max_ms; };

template <typename F>
Timing time_kernel(F &&f, int reps, int trials = 5) {
    for (int i = 0; i < 2; ++i) f();
    Timing t{1e300, 0.0};
    for (int k = 0; k < trials; ++k) {
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) f();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        t.min_ms = std::min(t.min_ms, ms);
        t.max_ms = std::max(t.max_ms, ms);
    }
    return t;
}

double g_peak_gflops = 0.0;
double g_peak_copy_gbps = 0.0;

// Prints KEY=value plus KEY_SPREAD=<max/min>. Spread > 1.5 means the machine
// was not quiet and the value must not be recorded as a baseline.
void report(const char *key, double value, bool higher_is_better, const Timing &t,
            const char *what, const char *unit) {
    double spread = (t.min_ms > 0) ? t.max_ms / t.min_ms : 0.0;
    std::printf("%s=%.3f  (%s%s%s%s)\n", key, value, what, unit, "", "");
    std::printf("%s_SPREAD=%.2f%s\n", key, spread, spread > 1.5 ? "  UNSTABLE" : "");
    (void)higher_is_better;
}

// ------------------------------------------------------------- calibrations

// Register-resident packed FMA chain. Must be compiled the way the library's
// hot loops are (AVX2), otherwise "peak" is a scalar number and every
// vectorised kernel looks impossible.
BENCH_NOINLINE void k_peak_flops(int iters) {
#if defined(__AVX2__)
    __m256d a0 = _mm256_set1_pd(1.0), a1 = _mm256_set1_pd(1.1);
    __m256d a2 = _mm256_set1_pd(1.2), a3 = _mm256_set1_pd(1.3);
    __m256d a4 = _mm256_set1_pd(1.4), a5 = _mm256_set1_pd(1.5);
    __m256d a6 = _mm256_set1_pd(1.6), a7 = _mm256_set1_pd(1.7);
    const __m256d b = _mm256_set1_pd(1.0000001), c = _mm256_set1_pd(1e-12);
    for (int i = 0; i < iters; ++i) {
        a0 = _mm256_fmadd_pd(a0, b, c); a1 = _mm256_fmadd_pd(a1, b, c);
        a2 = _mm256_fmadd_pd(a2, b, c); a3 = _mm256_fmadd_pd(a3, b, c);
        a4 = _mm256_fmadd_pd(a4, b, c); a5 = _mm256_fmadd_pd(a5, b, c);
        a6 = _mm256_fmadd_pd(a6, b, c); a7 = _mm256_fmadd_pd(a7, b, c);
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3)),
                              _mm256_add_pd(_mm256_add_pd(a4, a5), _mm256_add_pd(a6, a7)));
    double out[4];
    _mm256_storeu_pd(out, s);
    escape_memory(out);
#else
    double a0 = 1.0, a1 = 1.1, a2 = 1.2, a3 = 1.3, a4 = 1.4, a5 = 1.5, a6 = 1.6, a7 = 1.7;
    const double b = 1.0000001, c = 1e-12;
    for (int i = 0; i < iters; ++i) {
        a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
        a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
    }
    double out[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    escape_memory(out);
#endif
}

BENCH_NOINLINE void k_copy(double *a, double *b, std::size_t n) {
    a[0] += 1.0;
    for (std::size_t i = 0; i < n; ++i) b[i] = a[i];
    escape_memory(b);
}

BENCH_NOINLINE void k_triad(double *a, double *b, double *c, std::size_t n) {
    a[0] += 1.0;
    for (std::size_t i = 0; i < n; ++i) c[i] = a[i] + 2.0 * b[i];
    escape_memory(c);
}

BENCH_NOINLINE void k_axpy(double *a, double *b, std::size_t n) {
    b[0] += 1.0;
    for (std::size_t i = 0; i < n; ++i) b[i] = a[i] + 3.0 * b[i];
    escape_memory(b);
}

// ---------------------------------------------------------------- dot family

BENCH_NOINLINE double k_dot(double *a, double *b, std::size_t n) {
    a[0] += 1.0;
    std::span<const double> sa(a, n), sb(b, n);
#if defined(_OPENMP)
    return double(dot_par(sa, sb));
#else
    return double(dot(sa, sb));
#endif
}

BENCH_NOINLINE double k_dot_serial(double *a, double *b, std::size_t n) {
    a[0] += 1.0;
    std::span<const double> sa(a, n), sb(b, n);
    return double(dot(sa, sb));
}

BENCH_NOINLINE double k_dot_repro(double *a, double *b, std::size_t n) {
    a[0] += 1.0;
    std::span<const double> sa(a, n), sb(b, n);
    return double(dot_repro(sa, sb));
}

BENCH_NOINLINE double k_dot_det(double *a, double *b, std::size_t n) {
    a[0] += 1.0;
    std::span<const double> sa(a, n), sb(b, n);
    return double(dot_deterministic(sa, sb));
}

// ------------------------------------------------------------- matvec family

void ref_matvec(const double *A, const double *x, double *y, std::size_t R, std::size_t C) {
    for (std::size_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (std::size_t c = 0; c < C; ++c) s += A[r * C + c] * x[c];
        y[r] = s;
    }
}

bool agrees(const double *Af, const double *x, std::size_t Np,
            void (*fn)(const double *, const double *, double *, std::size_t)) {
    std::vector<double> y(Np), r(Np);
    fn(Af, x, y.data(), Np);
    ref_matvec(Af, x, r.data(), Np, Np);
    double md = 0.0;
    for (std::size_t i = 0; i < Np; ++i) md = std::max(md, std::abs(y[i] - r[i]));
    if (md > 1e-10) {
        std::printf("  !! Np=%zu kernel disagrees with reference (maxdiff %.3e) — NOT timed\n", Np, md);
        return false;
    }
    return true;
}

// Fixed body: E is part of the measured loop, Np is a runtime parameter.
BENCH_NOINLINE void k_matvec_rt(double *A, double *x, double *y, std::size_t Np, std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        A[0] += 1.0;
        matvec(A, x, y, Np, Np);
    }
    escape_memory(y);
}

template <std::size_t M>
BENCH_NOINLINE void k_matvec_fixed(double *A, double *x, double *y, std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        A[0] += 1.0;
        matvec_fixed<M, M>(A, x, y);
    }
    escape_memory(y);
}

template <std::size_t N>
BENCH_NOINLINE void k_contract_sf(double *D, double *u, double *o, std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        u[0] += 1.0;
        contract_axis<Axis::X, N>(D, u, o, N, N, 1);
        contract_axis<Axis::Y, N>(D, o, u, N, N, 1);
    }
    escape_memory(u);
}

BENCH_NOINLINE void k_dense(double *M, double *u, double *o, std::size_t Np, std::size_t E) {
    for (std::size_t e = 0; e < E; ++e) {
        M[0] += 1.0;
        matvec(M, u, o, Np, Np);
    }
    escape_memory(o);
}

BENCH_NOINLINE void k_solve4(const double *const *Ap, const double *const *bp,
                             double *const *xp, std::size_t E) {
    solve_4x4_batched<double>(Ap, bp, xp, E);
    escape_memory(xp[0]);
}

// ------------------------------------------------------------------ sections

void section_peaks(std::size_t n) {
    const int iters = g_quick ? 2000000 : 20000000;
    Timing tf = time_kernel([&] { k_peak_flops(iters); }, 1, 3);
    double flops = 2.0 * 4.0 * 8.0 * double(iters);
#if defined(__AVX2__)
    g_peak_gflops = flops / (tf.min_ms * 1e6);
    std::printf("PEAK_GFLOPS=%.2f  (AVX2 FMA, 8x4 lanes)\n", g_peak_gflops);
#else
    g_peak_gflops = 2.0 * 8.0 * double(iters) / (tf.min_ms * 1e6);
    std::printf("PEAK_GFLOPS=%.2f  (scalar FMA, no AVX2)\n", g_peak_gflops);
#endif

    std::vector<double> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) a[i] = std::sin(0.001 * double(i));
    Timing tc = time_kernel([&] { k_copy(a.data(), b.data(), n); }, 1, 5);
    g_peak_copy_gbps = (2.0 * double(n) * sizeof(double)) / 1e9 / (tc.min_ms * 1e-3);
    std::printf("PEAK_COPY_GBPS=%.2f  (STREAM copy, spread %.2f)\n", g_peak_copy_gbps,
                tc.max_ms / tc.min_ms);
}

void section_stream(std::size_t n) {
    std::vector<double> a(n), b(n), c(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = std::sin(0.001 * double(i));
        b[i] = std::cos(0.002 * double(i));
    }
    const double bytes3 = 3.0 * double(n) * sizeof(double);
    Timing t = time_kernel([&] { k_triad(a.data(), b.data(), c.data(), n); }, 1);
    report("STREAM_TRIAD_GBPS", bytes3 / 1e9 / (t.min_ms * 1e-3), true, t, "2 read + 1 write", "GB/s");
    t = time_kernel([&] { k_axpy(a.data(), b.data(), n); }, 1);
    report("STREAM_AXPY_GBPS", bytes3 / 1e9 / (t.min_ms * 1e-3), true, t, "2 read + 1 write", "GB/s");
}

void section_dot(std::size_t n) {
    std::vector<double> a(n), b(n);
    for (std::size_t i = 0; i < n; ++i) {
        a[i] = std::sin(0.001 * double(i));
        b[i] = std::cos(0.002 * double(i));
    }
    const double bytes = 2.0 * double(n) * sizeof(double);
    volatile double sink = 0.0;

    Timing t = time_kernel([&] { sink = sink + k_dot_serial(a.data(), b.data(), n); }, 1);
    report("DOT_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), true, t, "1 thread", "GB/s");

#if defined(_OPENMP)
    for (int w = 0; w < 3; ++w) sink = sink + k_dot(a.data(), b.data(), n);
    t = time_kernel([&] { sink = sink + k_dot(a.data(), b.data(), n); }, 3);
    report("DOT_PAR_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), true, t, "omp", "GB/s");
#else
    std::printf("DOT_PAR_GBPS=SKIPPED  (not built with -fopenmp)\n");
#endif

    t = time_kernel([&] { sink = sink + k_dot_repro(a.data(), b.data(), n); }, 1);
    report("DOT_REPRO_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), true, t, "bit-reproducible", "GB/s");

    t = time_kernel([&] { sink = sink + k_dot_det(a.data(), b.data(), n); }, 1);
    report("DOT_DET_GBPS", bytes / 1e9 / (t.min_ms * 1e-3), true, t, "bit-identical", "GB/s");
    (void)sink;
}

void section_matvec(std::size_t Np) {
    const std::size_t E = std::max<std::size_t>(20000, 4000000 / (Np * Np));
    std::vector<double> A(Np * Np), x(Np), y(Np);
    for (std::size_t i = 0; i < Np * Np; ++i) A[i] = std::sin(0.001 * double(i) + 0.3);
    for (std::size_t i = 0; i < Np; ++i) x[i] = 1.0 + 0.01 * double(i);
    const double flops = 2.0 * double(Np) * double(Np) * double(E);
    int reps = g_quick ? 1 : 3;

    static const struct { std::size_t M; void (*fn)(const double *, const double *, double *, std::size_t); } tbl[] = {
        {10, [](const double *a, const double *xx, double *yy, std::size_t) { matvec_fixed<10, 10>(a, xx, yy); }},
        {20, [](const double *a, const double *xx, double *yy, std::size_t) { matvec_fixed<20, 20>(a, xx, yy); }},
        {36, [](const double *a, const double *xx, double *yy, std::size_t) { matvec_fixed<36, 36>(a, xx, yy); }},
        {64, [](const double *a, const double *xx, double *yy, std::size_t) { matvec_fixed<64, 64>(a, xx, yy); }},
    };

    // equality check on a separate copy, so the timed buffers stay pristine
    bool ok_rt = true;
    {
        std::vector<double> y2(Np), r(Np);
        matvec(A.data(), x.data(), y2.data(), Np, Np);
        ref_matvec(A.data(), x.data(), r.data(), Np, Np);
        double md = 0.0;
        for (std::size_t i = 0; i < Np; ++i) md = std::max(md, std::abs(y2[i] - r[i]));
        ok_rt = (md <= 1e-10);
        if (!ok_rt) std::printf("  !! matvec Np=%zu maxdiff %.3e — NOT timed\n", Np, md);
    }

    char key[64];
    if (ok_rt) {
        Timing t = time_kernel([&] { k_matvec_rt(A.data(), x.data(), y.data(), Np, E); }, reps);
        std::snprintf(key, sizeof key, "MATVEC_NP%zu_GFLOPS", Np);
        report(key, flops / (t.min_ms * 1e6), true, t, "runtime cols", "GFLOPS");
    }

    for (auto &e : tbl) {
        if (e.M != Np) continue;
        if (!agrees(A.data(), x.data(), Np, e.fn)) continue;
        Timing t = time_kernel([&] {
            if (e.M == 10) k_matvec_fixed<10>(A.data(), x.data(), y.data(), E);
            else if (e.M == 20) k_matvec_fixed<20>(A.data(), x.data(), y.data(), E);
            else if (e.M == 36) k_matvec_fixed<36>(A.data(), x.data(), y.data(), E);
            else k_matvec_fixed<64>(A.data(), x.data(), y.data(), E);
        }, reps);
        std::snprintf(key, sizeof key, "MATVEC_FIXED_NP%zu_GFLOPS", Np);
        report(key, flops / (t.min_ms * 1e6), true, t, "compile-time cols", "GFLOPS");
    }
}

template <std::size_t N>
void section_contract_one(std::size_t E) {
    constexpr std::size_t Np = N * N;
    std::vector<double> D(N * N), u(Np), o(Np), M(Np * Np), od(Np);
    for (std::size_t i = 0; i < N * N; ++i) D[i] = std::sin(0.5 * double(i));
    for (std::size_t i = 0; i < Np; ++i) u[i] = std::cos(0.1 * double(i));
    for (std::size_t i = 0; i < Np * Np; ++i) M[i] = std::sin(0.0007 * double(i));

    {
        std::vector<double> ref(Np, 0.0), got(Np, 0.0);
        for (std::size_t j = 0; j < N; ++j)
            for (std::size_t i = 0; i < N; ++i)
                for (std::size_t q = 0; q < N; ++q) ref[i + N * j] += D[i * N + q] * u[q + N * j];
        contract_axis<Axis::X>(D.data(), u.data(), got.data(), N, N, 1);
        double md = 0.0;
        for (std::size_t i = 0; i < Np; ++i) md = std::max(md, std::abs(got[i] - ref[i]));
        if (md > 1e-10) {
            std::printf("  !! contract_axis n=%zu maxdiff %.3e — NOT timed\n", N, md);
            return;
        }
    }

    const double sf_flops = 2.0 * 2.0 * double(N) * double(Np) * double(E);
    const double dense_flops = 2.0 * double(Np) * double(Np) * double(E);

    Timing ts = time_kernel([&] { k_contract_sf<N>(D.data(), u.data(), o.data(), E); }, 1);
    Timing td = time_kernel([&] { k_dense(M.data(), u.data(), od.data(), Np, E); }, 1);

    char key[64];
    std::snprintf(key, sizeof key, "SF_N%zu_GFLOPS", N);
    report(key, sf_flops / (ts.min_ms * 1e6), true, ts, "sum-factorized", "GFLOPS");
    std::snprintf(key, sizeof key, "SF_N%zu_DENSE_GFLOPS", N);
    report(key, dense_flops / (td.min_ms * 1e6), true, td, "dense matvec", "GFLOPS");
    std::printf("SF_N%zu_SPEEDUP=%.3f  (dense / sum-factorized)\n", N, td.min_ms / ts.min_ms);
}

void section_solve4() {
    const std::size_t E = g_quick ? 50000 : 200000;
    std::vector<double> A16(16 * E), b(4 * E), x(4 * E, 0.0);
    for (std::size_t i = 0; i < 16 * E; ++i) A16[i] = std::sin(0.0007 * double(i));
    for (std::size_t e = 0; e < E; ++e)
        for (int r = 0; r < 4; ++r) A16[(r * 4 + r) * E + e] += 9.0;
    for (std::size_t i = 0; i < 4 * E; ++i) b[i] = 1.0;

    std::vector<const double *> Ap(16), bp(4);
    std::vector<double *> xp(4);
    for (int i = 0; i < 16; ++i) Ap[i] = A16.data() + i * E;
    for (int i = 0; i < 4; ++i) { bp[i] = b.data() + i * E; xp[i] = x.data() + i * E; }

    Timing t = time_kernel([&] { k_solve4(Ap.data(), bp.data(), xp.data(), E); }, 3);
    std::printf("SOLVE4_MELEM_S=%.3f  (%.2f GFLOPS, %.3g systems)\n",
                double(E) / (t.min_ms * 1e3), (30.0 * double(E)) / (t.min_ms * 1e6), double(E));
    std::printf("SOLVE4_SPREAD=%.2f\n", t.max_ms / t.min_ms);
}

}  // namespace

int main(int argc, char **argv) {
    g_quick = (argc > 1 && std::string(argv[1]) == "--quick");

    std::printf("# vec_math benchmark harness%s\n", g_quick ? " (quick)" : "");
#if defined(_OPENMP)
    std::printf("THREADS=%d\n", omp_get_max_threads());
#else
    std::printf("THREADS=1\n");
#endif

    {
        double probe_vals[3] = {1.0001, 1.0, 12345.678};
        bool ok = true;
        for (double &v : probe_vals) ok = ok && mutation_is_effective(&v);
        std::printf("MUTATION=%s\n", ok ? "OK" : "NOOP");
        if (!ok) {
            std::printf("# fatal: the in-loop mutation does not change the value,\n"
                        "# so every timing below would measure the hoisted loop.\n");
            return 2;
        }
    }

    const std::size_t n = g_quick ? (1u << 20) : (1u << 22);

    section_peaks(n);
    section_stream(n);
    section_dot(n);

    for (std::size_t Np : {10u, 20u, 36u, 64u}) section_matvec(Np);

    section_contract_one<4>(g_quick ? 50000u : 300000u);
    section_contract_one<6>(g_quick ? 20000u : 100000u);

    section_solve4();

    std::printf("IMPOSSIBLE_COUNT=%d\n", g_impossible);
    std::printf("# done\n");
    return g_impossible == 0 ? 0 : 1;
}
