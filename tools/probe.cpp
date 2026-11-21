// tools/probe.cpp — conformance probe for vec_math
// Usage:  probe.exe            (exit 0 = all hard checks pass)
//         probe.exe --verbose  (list every check)
//
// Design rules:
//   * every check is guarded by `requires`, so a REMOVED symbol reports
//     "MISSING" instead of failing to compile. API removal is a finding,
//     not a build error we cannot report.
//   * every numeric check compares against an independent reference
//     implemented here, never against a literal the library would also
//     be wrong about.
//   * FAIL is reserved for "the answer is wrong" or "a documented API is
//     absent". INFO is for design smells that do not invalidate a number.

#include <vec_math/vector_math.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {
int g_pass = 0, g_fail = 0, g_info = 0, g_missing = 0;
bool g_verbose = false;

void pass(const char *what) {
    ++g_pass;
    if (g_verbose) std::printf("PASS  %s\n", what);
}
void fail(const char *what, const char *fmt = "", ...) {
    ++g_fail;
    std::printf("FAIL  %s", what);
    if (*fmt) {
        std::printf("  ");
        va_list ap;
        va_start(ap, fmt);
        std::vprintf(fmt, ap);
        va_end(ap);
    }
    std::printf("\n");
}
void info(const char *what, const char *fmt = "", ...) {
    ++g_info;
    std::printf("INFO  %s", what);
    if (*fmt) {
        std::printf("  ");
        va_list ap;
        va_start(ap, fmt);
        std::vprintf(fmt, ap);
        va_end(ap);
    }
    std::printf("\n");
}
void missing(const char *sym) {
    ++g_missing;
    std::printf("MISSING %s  (API removed or renamed)\n", sym);
}

// independent reference matvec, written the dumb way on purpose
void ref_matvec(const double *A, const double *x, double *y, std::size_t R, std::size_t C) {
    for (std::size_t r = 0; r < R; ++r) {
        double s = 0;
        for (std::size_t c = 0; c < C; ++c) s += A[r * C + c] * x[c];
        y[r] = s;
    }
}

double maxdiff(const double *a, const double *b, std::size_t n) {
    double m = 0;
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

// ---------------------------------------------------------------- sections

void section_basic() {
    using namespace vec_math;
    std::vector<double> a{1, 2, 3, 4}, b{10, 20, 30, 40}, o(4);

    if constexpr (requires { o <<= 2.0 * a + b; }) {
        o <<= 2.0 * a + b;
        (o[0] == 12.0 && o[3] == 48.0) ? pass("assign fused")
                                       : fail("assign fused", "%g", o[0]);
    } else missing("operator<<=");

    if constexpr (requires { dot(a, b); sum(a); }) {
        (std::abs(dot(a, b) - 300.0) < 1e-12) ? pass("dot") : fail("dot", "%g", (double)dot(a, b));
        (std::abs(sum(a) - 10.0) < 1e-12) ? pass("sum") : fail("sum", "%g", (double)sum(a));
    } else missing("dot/sum");

    // aliasing matrix — the batch that should be impossible to skip
    {
        std::vector<double> x{1, 2, 3, 4}, y{5, 6, 7, 8}, r;
        r = x; r <<= r + y;                       // no-alias reference
        std::vector<double> z = x; z <<= z + y;   // self-aliased
        (r == z) ? pass("alias  dst<<=dst+other") : fail("alias  dst<<=dst+other");

        r = x; for (std::size_t i = 0; i < r.size(); ++i) r[i] = r[i] * r[i];
        z = x; z <<= z * z;
        (r == z) ? pass("alias  dst<<=dst*dst") : fail("alias  dst<<=dst*dst");

        r = x; r <<= y - r;
        z = x; z <<= y - z;
        (r == z) ? pass("alias  dst<<=other-dst") : fail("alias  dst<<=other-dst");
    }

    // empty sizes must not crash
    {
        std::vector<double> e, f;
        bool ok = true;
        if constexpr (requires { sum(e); }) ok = ok && (sum(e) == 0.0);
        if constexpr (requires { dot(e, f); }) ok = ok && (dot(e, f) == 0.0);
        ok ? pass("empty inputs") : fail("empty inputs");
    }

    // NaN must propagate, not be scrubbed
    {
        std::vector<double> v{1.0, std::nan(""), 3.0};
        bool ok = true;
        if constexpr (requires { sum(v); }) ok = ok && std::isnan((double)sum(v));
        ok ? pass("NaN propagation") : fail("NaN propagation");
    }

    // span / array compound assignment (was a defect, now a contract)
    {
        std::vector<double> d(4, 1.0), s(4, 0.5);
        bool ok = true;
        if constexpr (requires { std::span<double>(d) += std::span<double>(d); }) {
            std::span<double> dsp(d), ssp(s);
            dsp += ssp;
            ok = ok && (std::abs(d[0] - 1.5) < 1e-15);
        } else ok = false;
        ok ? pass("span +=") : fail("span +=");
    }
}

void section_complex() {
    using namespace vec_math;
    std::vector<std::complex<double>> z{{1, 2}, {3, 4}}, w{{1, 0}, {0, 1}};

    if constexpr (requires { norm(z); norm_sq(z); }) {
        (std::abs(norm_sq(z) - 30.0) < 1e-12) ? pass("complex norm_sq")
                                               : fail("complex norm_sq", "%g", (double)norm_sq(z));
        (std::abs(norm(z) - std::sqrt(30.0)) < 1e-12) ? pass("complex norm")
                                                     : fail("complex norm", "%g", (double)norm(z));
    } else missing("norm/norm_sq on complex");

    if constexpr (requires { dotc(z, w); }) {
        auto dc = dotc(z, w);  // <z,w> = 5 + 1i
        (std::abs(dc - std::complex<double>(5, 1)) < 1e-12)
            ? pass("complex dotc")
            : fail("complex dotc", "(%g,%g) want (5,1)", dc.real(), dc.imag());
    } else missing("dotc");
}

void section_matvec() {
    using namespace vec_math;
    constexpr std::size_t Np = 36;
    double A[Np * Np], x[Np], y[Np], r[Np];
    for (std::size_t i = 0; i < Np * Np; ++i) A[i] = std::sin(0.001 * double(i) + 0.3);
    for (std::size_t i = 0; i < Np; ++i) x[i] = 1.0 + 0.01 * double(i);
    ref_matvec(A, x, r, Np, Np);

    if constexpr (requires { matvec(A, x, y, Np, Np); }) {
        matvec(A, x, y, Np, Np);
        (maxdiff(y, r, Np) < 1e-12) ? pass("matvec 36x36")
                                    : fail("matvec 36x36", "%.3e", maxdiff(y, r, Np));
        double buf[Np];
        std::memcpy(buf, x, sizeof(buf));
        matvec(A, buf, buf, Np, Np);   // aliased
        (maxdiff(buf, r, Np) < 1e-12) ? pass("matvec aliased")
                                      : fail("matvec aliased", "%.3e", maxdiff(buf, r, Np));
    } else missing("matvec");

    if constexpr (requires { matvec_fixed<10, 10>(A, x, y); }) {
        constexpr std::size_t M = 10;
        double A2[M * M], x2[M], y2[M], r2[M];
        for (std::size_t i = 0; i < M * M; ++i) A2[i] = std::cos(0.37 * double(i) + 0.2);
        for (std::size_t i = 0; i < M; ++i) x2[i] = 0.5 + 0.1 * double(i);
        ref_matvec(A2, x2, r2, M, M);
        matvec_fixed<M, M>(A2, x2, y2);
        (maxdiff(y2, r2, M) < 1e-13) ? pass("matvec_fixed<10,10>")
                                     : fail("matvec_fixed<10,10>", "%.3e", maxdiff(y2, r2, M));
        std::memcpy(y2, x2, sizeof(y2));
        matvec_fixed<M, M>(A2, y2, y2);
        (maxdiff(y2, r2, M) < 1e-13) ? pass("matvec_fixed aliased")
                                     : fail("matvec_fixed aliased", "%.3e", maxdiff(y2, r2, M));
    } else missing("matvec_fixed");

    if constexpr (requires { matvec_transposed(A, x, y, Np, Np); }) {
        double At[Np], rt[Np];
        matvec_transposed(A, x, At, Np, Np);
        for (std::size_t c = 0; c < Np; ++c) {
            double s = 0;
            for (std::size_t rr = 0; rr < Np; ++rr) s += A[rr * Np + c] * x[rr];
            rt[c] = s;
        }
        (maxdiff(At, rt, Np) < 1e-12) ? pass("matvec_transposed")
                                      : fail("matvec_transposed", "%.3e", maxdiff(At, rt, Np));
    } else missing("matvec_transposed");
}

void section_contract() {
    using namespace vec_math;
    if constexpr (requires { contract_axis<Axis::X>(static_cast<const double *>(nullptr),
                                                    static_cast<const double *>(nullptr),
                                                    static_cast<double *>(nullptr), 1, 1, 1); }) {
        const std::size_t nx = 3, ny = 4, nz = 2, ntot = nx * ny * nz;
        auto IDX = [&](std::size_t i, std::size_t j, std::size_t k) {
            return i + nx * (j + ny * k);
        };
        std::vector<double> in(ntot), out(ntot), ref(ntot);
        for (std::size_t i = 0; i < ntot; ++i) in[i] = std::sin(0.13 * double(i) + 0.7);

        // axis 0 (compile-time N)
        {
            std::vector<double> D(nx * nx);
            for (std::size_t r = 0; r < nx; ++r)
                for (std::size_t c = 0; c < nx; ++c) D[r * nx + c] = std::cos(0.3 * r + 0.11 * c);
            for (std::size_t k = 0; k < nz; ++k)
                for (std::size_t j = 0; j < ny; ++j)
                    for (std::size_t i = 0; i < nx; ++i) {
                        double s = 0;
                        for (std::size_t q = 0; q < nx; ++q) s += D[i * nx + q] * in[IDX(q, j, k)];
                        ref[IDX(i, j, k)] = s;
                    }
            std::fill(out.begin(), out.end(), 0.0);
            contract_axis<Axis::X, nx>(D.data(), in.data(), out.data(), nx, ny, nz);
            (maxdiff(out.data(), ref.data(), ntot) < 1e-13)
                ? pass("contract_axis X (N known)")
                : fail("contract_axis X (N known)", "%.3e", maxdiff(out.data(), ref.data(), ntot));

            std::vector<double> buf(in);
            contract_axis<Axis::X, nx>(D.data(), buf.data(), buf.data(), nx, ny, nz);
            (maxdiff(buf.data(), ref.data(), ntot) < 1e-13)
                ? pass("contract_axis X aliased")
                : fail("contract_axis X aliased", "%.3e", maxdiff(buf.data(), ref.data(), ntot));
        }
        // axis 1
        {
            std::vector<double> D(ny * ny);
            for (std::size_t r = 0; r < ny; ++r)
                for (std::size_t c = 0; c < ny; ++c) D[r * ny + c] = std::sin(0.27 * r + 0.19 * c);
            for (std::size_t k = 0; k < nz; ++k)
                for (std::size_t j = 0; j < ny; ++j)
                    for (std::size_t i = 0; i < nx; ++i) {
                        double s = 0;
                        for (std::size_t q = 0; q < ny; ++q) s += D[j * ny + q] * in[IDX(i, q, k)];
                        ref[IDX(i, j, k)] = s;
                    }
            std::fill(out.begin(), out.end(), 0.0);
            contract_axis<Axis::Y, ny>(D.data(), in.data(), out.data(), nx, ny, nz);
            (maxdiff(out.data(), ref.data(), ntot) < 1e-13)
                ? pass("contract_axis Y (N known)")
                : fail("contract_axis Y (N known)", "%.3e", maxdiff(out.data(), ref.data(), ntot));
        }
        // axis 2
        {
            std::vector<double> D(nz * nz);
            for (std::size_t r = 0; r < nz; ++r)
                for (std::size_t c = 0; c < nz; ++c) D[r * nz + c] = std::cos(0.41 * r - 0.23 * c);
            for (std::size_t k = 0; k < nz; ++k)
                for (std::size_t j = 0; j < ny; ++j)
                    for (std::size_t i = 0; i < nx; ++i) {
                        double s = 0;
                        for (std::size_t q = 0; q < nz; ++q) s += D[k * nz + q] * in[IDX(i, j, q)];
                        ref[IDX(i, j, k)] = s;
                    }
            std::fill(out.begin(), out.end(), 0.0);
            contract_axis<Axis::Z, nz>(D.data(), in.data(), out.data(), nx, ny, nz);
            (maxdiff(out.data(), ref.data(), ntot) < 1e-13)
                ? pass("contract_axis Z (N known)")
                : fail("contract_axis Z (N known)", "%.3e", maxdiff(out.data(), ref.data(), ntot));
        }
    } else missing("contract_axis");
}

void section_flux() {
    using namespace vec_math;
    if constexpr (requires { gather(static_cast<double *>(nullptr), static_cast<const double *>(nullptr),
                                    static_cast<const int *>(nullptr), std::size_t{0}); }) {
        double src[8];
        for (int i = 0; i < 8; ++i) src[i] = i + 1;
        int idx[5] = {7, 0, 3, 3, 5};
        double dst[5];
        gather(dst, src, idx, 5);
        (dst[0] == 8 && dst[1] == 1 && dst[2] == 4 && dst[3] == 4 && dst[4] == 6)
            ? pass("gather") : fail("gather");
    } else missing("gather");

    if constexpr (requires { scatter_add(static_cast<double *>(nullptr), static_cast<const int *>(nullptr),
                                         static_cast<const double *>(nullptr), std::size_t{0}); }) {
        int idx[5] = {7, 0, 3, 3, 5};
        double vals[5] = {1, 2, 3, 4, 5};
        double big[8] = {10, 20, 30, 40, 50, 60, 70, 80};
        double orig[8];
        std::memcpy(orig, big, sizeof(big));
        scatter_add(big, idx, vals, 5);
        bool ok = (big[7] == 81 && big[0] == 12 && big[3] == 47 && big[5] == 65);
        ok ? pass("scatter_add (incl. duplicate index)") : fail("scatter_add");
        double b2[8];
        std::memcpy(b2, orig, sizeof(b2));   // restart from the *unscattered* input
        scatter_add(b2, idx, vals, 5);
        std::memcmp(b2, big, sizeof(big)) == 0 ? pass("scatter_add repeatable")
                                               : fail("scatter_add repeatable");
    } else missing("scatter_add");

    if constexpr (requires { transpose_aos_to_soa(static_cast<const double *>(nullptr),
                                                  static_cast<double *>(nullptr), 1, 1); }) {
        const std::size_t T = 5, V = 4;
        std::vector<double> aos(T * V), soa(T * V), back(T * V);
        for (std::size_t i = 0; i < T * V; ++i) aos[i] = 10.0 * double(i) + 0.5;
        transpose_aos_to_soa(aos.data(), soa.data(), T, V);
        transpose_soa_to_aos(soa.data(), back.data(), T, V);
        (back == aos) ? pass("transpose aos<->soa roundtrip") : fail("transpose aos<->soa");
    } else missing("transpose_aos_to_soa");
}

void section_solve() {
    using namespace vec_math;
    if constexpr (requires { solve_4x4_batched<double>(static_cast<const double *const *>(nullptr),
                                                       static_cast<const double *const *>(nullptr),
                                                       static_cast<double *const *>(nullptr), 0); }) {
        const std::size_t E = 257;
        std::vector<double> A16(16 * E), b(4 * E), x(4 * E), xref(4 * E);
        std::vector<const double *> Ap(16), bp(4);
        std::vector<double *> xp(4);
        for (int i = 0; i < 16; ++i) Ap[i] = A16.data() + i * E;
        for (int i = 0; i < 4; ++i) { bp[i] = b.data() + i * E; xp[i] = x.data() + i * E; }
        for (std::size_t e = 0; e < E; ++e) {
            double am[4][4], x0[4];
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c) am[r][c] = (r == c ? 8.0 : 0.0) + std::sin(0.31 * e + 1.7 * r + 0.9 * c);
            for (int r = 0; r < 4; ++r) x0[r] = 1.0 + 0.01 * e * (r + 1) - 0.5 * r;
            for (int r = 0; r < 4; ++r) {
                double s = 0;
                for (int c = 0; c < 4; ++c) s += am[r][c] * x0[c];
                b[r * E + e] = s;
                xref[r * E + e] = x0[r];
                for (int c = 0; c < 4; ++c) A16[(r * 4 + c) * E + e] = am[r][c];
            }
        }
        solve_4x4_batched<double>(Ap.data(), bp.data(), xp.data(), E);
        double md = 0;
        for (int r = 0; r < 4; ++r)
            for (std::size_t e = 0; e < E; ++e) md = std::max(md, std::abs(x[r * E + e] - xref[r * E + e]));
        (md < 1e-9) ? pass("solve_4x4_batched") : fail("solve_4x4_batched", "%.3e", md);

        // singular system: silent zero is a wrong answer, not a warning
        std::vector<double> Z(16 * E, 0.0), bz(4 * E, 1.0), xz(4 * E, -999.0);
        std::vector<const double *> Zp(16), bzp(4);
        std::vector<double *> xzp(4);
        for (int i = 0; i < 16; ++i) Zp[i] = Z.data() + i * E;
        for (int i = 0; i < 4; ++i) { bzp[i] = bz.data() + i * E; xzp[i] = xz.data() + i * E; }
        // A singular system has no solution. "Wrote zeros" is not a diagnosis:
        // it is a wrong answer that a solver will silently consume.
        bool diagnosed = false;
        try {
            solve_4x4_batched<double>(Zp.data(), bzp.data(), xzp.data(), E);
        } catch (const std::exception &) { diagnosed = true; }
        diagnosed ? pass("solve_4x4 singular is diagnosed")
                  : fail("solve_4x4 singular returns silently", "x[0]=%.3f (want a throw/status)", xz[0]);
    } else missing("solve_4x4_batched");
}

void section_memory() {
    using namespace vec_math;
    if constexpr (requires { WorkspaceArena(64); }) {
        WorkspaceArena ws(4096);
        auto a = ws.allocate<double>(100);
        auto b = ws.allocate<float>(37);
        bool ok = (reinterpret_cast<std::uintptr_t>(a.data()) % 64 == 0) &&
                  (reinterpret_cast<std::uintptr_t>(b.data()) % 64 == 0) &&
                  (reinterpret_cast<char *>(a.data()) + 100 * sizeof(double) <= reinterpret_cast<char *>(b.data()));
        ok ? pass("WorkspaceArena alignment / no overlap") : fail("WorkspaceArena");
        ws.reset();
        (ws.allocate<double>(100).data() == a.data()) ? pass("WorkspaceArena reset reuses")
                                                      : fail("WorkspaceArena reset");
        bool threw = false;
        try { (void)ws.allocate<double>(100000); } catch (const std::exception &) { threw = true; }
        threw ? pass("WorkspaceArena overflow diagnosed") : fail("WorkspaceArena overflow silent");
    } else missing("WorkspaceArena");

    if constexpr (requires { broadcast_multiply(static_cast<double *>(nullptr), std::size_t{0}, 1.0); }) {
        double v[6] = {1, 2, 3, 4, 5, 6};
        broadcast_multiply(v, 6, 2.5);
        (std::abs(v[5] - 15.0) < 1e-15) ? pass("broadcast_multiply") : fail("broadcast_multiply");
    } else missing("broadcast_multiply");
}

// Must be a template: `if constexpr` inside a non-template function still
// type-checks the discarded branch, which defeats the whole guarding scheme.
// `StridedSpan<T>` stays dependent on T so the `requires` guard can protect it.
template<typename T>
void section_strided_t() {
    using namespace vec_math;
    if constexpr (requires { StridedSpan<T>(static_cast<T *>(nullptr), 1, 1); }) {
        using SpanT = StridedSpan<T>;
        if constexpr (VectorOperand<SpanT>) {
            std::vector<T> base{1, 0, 2, 0, 3, 0, 4, 0};
            SpanT ss(base.data(), 4, 2);
            (std::abs(sum(ss) - 10.0) < 1e-12) ? pass("StridedSpan is a VectorOperand")
                                               : fail("StridedSpan sum wrong", "%g", (double)sum(ss));
        } else {
            info("StridedSpan exists but is not a VectorOperand",
                 "cannot be used with any vec_math op");
        }
    } else {
        info("StridedSpan absent", "no strided views");
    }
}

void section_strided() { section_strided_t<double>(); }

// A documented-but-unimplemented API. This is the D2 "docs truth" check:
// README and api_reference.md both publish sum(v, ReductionPolicy::REPRO).
// A documented function that does not compile is a FAIL, not a warning,
// because it is the one failure a reader cannot detect without a compiler.
// Kept in a template so the `requires` guard is genuinely instantiation-time.
template <typename VecT>
bool documented_policy_sum_compiles(VecT &vec) {
    using namespace vec_math;
    if constexpr (requires { sum(vec, ReductionPolicy::REPRO); }) return true;
    else return false;
}

void section_documented_api() {
    using namespace vec_math;
    std::vector<double> v(8, 1.0);
    if constexpr (requires { ReductionPolicy::REPRO; })
        info("ReductionPolicy enum exists", "documented as a sum() argument");
    documented_policy_sum_compiles(v)
        ? pass("sum(v, ReductionPolicy::REPRO) compiles")
        : fail("sum(v, ReductionPolicy::REPRO)", "documented in README/api_reference but absent");
}

void section_reductions() {
    using namespace vec_math;
    std::vector<double> v(1000003);
    for (std::size_t i = 0; i < v.size(); ++i)
        v[i] = 1.0 / double(i + 1.0) * (i % 2 ? -1 : 1);

    if constexpr (requires { sum_repro(v); }) {
        (sum_repro(v) == sum_repro(v)) ? pass("sum_repro repeatable") : fail("sum_repro repeatable");
        std::printf("      sum_repro = %.17g\n", (double)sum_repro(v));
    } else missing("sum_repro");

    if constexpr (requires { dot_repro(v, v); }) {
        (dot_repro(v, v) == dot_repro(v, v)) ? pass("dot_repro repeatable")
                                             : fail("dot_repro repeatable");
        std::printf("      dot_repro = %.17g\n", (double)dot_repro(v, v));
    } else missing("dot_repro");

    if constexpr (requires { sum_deterministic(v); }) {
        std::printf("      sum_det   = %.17g\n", (double)sum_deterministic(v));
        pass("sum_deterministic present");
    } else missing("sum_deterministic");

    // fp32 accuracy: the number that decides whether you can use float on a GPU
    {
        const std::size_t n = 2000000;
        std::vector<float> f(n);
        for (std::size_t i = 0; i < n; ++i) f[i] = 1.0f / float(i + 1);
        double ref = 0;
        for (std::size_t i = 0; i < n; ++i) ref += double(f[i]);
        auto rel = [&](float got) { return std::abs(double(got) - ref) / ref; };
        double e_det = 0, e_fast = 0, e_rep = 0;
        if constexpr (requires { sum_deterministic(f); }) e_det = rel(sum_deterministic(f));
        if constexpr (requires { sum(f); }) e_fast = rel(sum(f));
        if constexpr (requires { sum_repro(f); }) e_rep = rel(sum_repro(f));
        std::printf("      fp32 n=2e6  rel err: det %.2e | fast %.2e | repro %.2e\n", e_det, e_fast, e_rep);
        if (e_rep > 0 && e_rep < 1e-5) pass("fp32 repro accuracy < 1e-5");
        else if (e_rep > 0) fail("fp32 repro accuracy", "%.2e (want < 1e-5)", e_rep);
        if (e_det > 1e-3) info("fp32 sequential mm is disqualifying", "%.2e — never use for Krylov", e_det);
    }
}

}  // namespace

int main(int argc, char **argv) {
    g_verbose = (argc > 1 && std::string(argv[1]) == "--verbose");
    std::printf("vec_math conformance probe\n");
    section_basic();
    section_complex();
    section_matvec();
    section_contract();
    section_flux();
    section_solve();
    section_memory();
    section_strided();
    section_documented_api();
    section_reductions();

    std::printf("\n---- SUMMARY ----\n");
    std::printf("PASS=%d FAIL=%d MISSING=%d INFO=%d\n", g_pass, g_fail, g_missing, g_info);
    std::printf("RESULT=%s\n", (g_fail == 0 && g_missing == 0) ? "OK" : "FAILED");
    return (g_fail == 0 && g_missing == 0) ? 0 : 1;
}
